import copy
import logging
import os
import sys
import random
import time
from datetime import datetime
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn

from models.model import Net
from models.model import EarlyStopper
from utils.data import load_data, save_data
from utils.grad_clipping import grad_clipping
from utils.metrics import single_loss, single_metric, evaluate_loss
from utils.tools import try_device, mkdir, same_seeds, log_init


def data_iter(states, mode='train'):
    sf_idx = list(range(len(states)))
    if mode == 'train' and shuf == True:  # shuffle
        random.shuffle(sf_idx)
    for i in sf_idx:
        X = torch.tensor(states[i][:-1].reshape(-1, 1), dtype=torch.float32)
        if incre == True:
            y = torch.tensor((states[i][1:] - states[i][:-1]).reshape(-1, 1), dtype=torch.float32)
        else:
            y = torch.tensor((states[i][1:]).reshape(-1, 1), dtype=torch.float32)
        yield X, y, i


def train_epoch_ch3(net, train_iter, loss, updater, lamd, device, kfold):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    for X, y, idx in train_iter:
        # 计算梯度并更新参数
        if cross_val == True:
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            train_index, valid_index = list(kf.split(X))[kfold]
        #         print('train_epoch_ch3 valid index\n', valid_index)
        else:
            train_index = list(range(len(X)))
        random.shuffle(train_index)
        X = X[train_index]
        y = y[train_index]
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X, idx)
        l = single_loss(loss, y_hat, y, lamd, net.E, net.a * net.W, net.s, reg_mod, incre)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            if trloss_mode == 'sum':
                l.sum().backward()
            elif trloss_mode == 'mean':
                l.mean().backward()
            if grad_clip == True:
                grad_clipping(net, grad_thres)
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            if trloss_mode == 'sum':
                l.sum().backward()
            elif trloss_mode == 'mean':
                l.mean().backward()
            updater.step()


def get_net():
    net = Net(cmp_err=cmp_error, cmp_alpha=cmp_alpha, cmp_omega=cmp_omega,
              alpha_in=alpha_in, alpha_rnn=alpha_rnn, alpha_rnn_out=alpha_rnn_out, alpha_hdim1=alpha_hdim1,
              alpha_concatn=concat_n,
              alpha_hdim2=alpha_hdim2, alpha_rnn_layer=alpha_rnn_layer, alpha_mlp_layer=alpha_mlp_layer,
              alpha_bidir=alpha_bidir, alpha_nhead=alpha_nhead, alpha_diff=alpha_diff, alpha_window=alpha_window,
              omega_in=omega_in, omega_func=omega_func, omega_hdim=omega_hdim, omega_initb=omega_initb,
              error_in=error_in, error_hdim=error_hdim, error_outnorm=error_outnorm, err_layer=err_layer,
              sigmaB=cmp_sigmaB, error_dropout=error_dropout,
              norm=norm, incre=incre, error_term=error_term)
    return net


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total:{total_num} Trainable: {trainable_num}')


def train(net, _states_train, trainer, scheduler, loss, epochs, lamd, device, model_path, best_loss,
          kfold, earlystopper):
    train_time = 0
    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    for epoch in range(epochs):
        if epoch == 0:
            net.eval()
            if cross_val:
                train_loss, valid_loss = evaluate_loss(net, data_iter(_states_train, mode='test'),
                                                       cross_val, device, kfold, num_folds, eval_metric, loss, seed)
                train_loss_per_epoch.append(train_loss)
                valid_loss_per_epoch.append(valid_loss)
            else:
                train_loss = evaluate_loss(net, data_iter(_states_train, mode='test'),
                                           cross_val, device, kfold, num_folds, eval_metric, loss, seed)
                valid_loss = train_loss
                train_loss_per_epoch.append(train_loss)
                valid_loss_per_epoch.append(valid_loss)
            print(f'epoch {epoch}, '
                  f'train_loss: {train_loss:.8f} '
                  f'valid_loss: {valid_loss:.8f} ')
            logging.info('epoch: {}, train_loss: {}, valid_loss: {}'.format(epoch, train_loss, valid_loss))
            continue
        time_start = time.time()
        train_epoch_ch3(net, data_iter(_states_train), loss, trainer, lamd, device, kfold)
        time_end = time.time()
        train_time += time_end - time_start
        net.eval()
        if cross_val:
            train_loss, valid_loss = evaluate_loss(net, data_iter(_states_train, mode='test'),
                                                   cross_val, device, kfold, num_folds, eval_metric, loss, seed)
        else:
            train_loss = evaluate_loss(net, data_iter(_states_train, mode='test'),
                                       cross_val, device, kfold, num_folds, eval_metric, loss, seed)
            valid_loss = train_loss

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), model_path)
        scheduler.step(valid_loss) if lr_decay == 'platlr' else scheduler.step()
        train_loss_per_epoch.append(train_loss)
        valid_loss_per_epoch.append(valid_loss)
        if epoch == 1 or (epoch) % 10 == 0:
            print(f'epoch {epoch}, '
                  f'train_loss: {train_loss:.8f} '
                  f'valid_loss: {valid_loss:.8f} ')
            logging.info(
                'epoch: {}, train_loss: {}, valid_loss: {}'.format(epoch, train_loss, valid_loss))
        if earlystopper.early_stop(valid_loss):
            print(f'*************Early stop at validation loss: {valid_loss}*************')
            logging.info(f'*************Early stop at validation loss: {valid_loss}*************')
            break
    return train_time, best_loss, train_loss_per_epoch, valid_loss_per_epoch


@torch.no_grad()  ## 不保留梯度，否则会报OOM
def predict(net, cmp, states_test, hv_data, loss, device):
    test_iter = data_iter(states_test[hv_data], mode='test')
    idx, error_sum, cmp_error_sum, base_error_sum = 0, 0, 0, 0
    net = net.to(device)
    omegas, errors, alphas, bounds, tests, test_errs, cmp_test_errs, base_test_errs = [], [], [], [], [], [], [], []
    step_test_errs, step_cmp_test_errs, step_base_test_errs = [], [], []
    cpu = try_device(-1)
    test_time = 0

    for X, y, idx in test_iter:
        # torch.cuda.empty_cache()
        X = X.to(device)
        y = y.to(cpu)
        time_start = time.time()
        onestep_preds = net(X, idx, predict=True)
        time_end = time.time()
        test_time += (time_end - time_start)
        onestep_preds = onestep_preds.to(cpu)
        ith_cmp = copy.deepcopy(cmp[idx])
        ith_cmp = ith_cmp[(ith_cmp.shape[0] - onestep_preds.shape[0]):].reshape(onestep_preds.shape)
        omegas.append(net.W)
        errors.append(net.E)
        alphas.append(net.a)
        bounds.append(net.error.sigmaB)
        tests.append(onestep_preds)
        prev_s = copy.deepcopy(net.s).to(cpu)
        if not incre:
            onestep_preds += prev_s
            y += prev_s
            ith_cmp += prev_s.numpy()
        else:
            ith_cmp -= prev_s.numpy()
            prev_s -= prev_s
        ith_cmp = torch.tensor(ith_cmp)
        e, e_list = single_metric(onestep_preds, y, eval_metric, loss)
        step_test_errs.extend(e_list.detach().cpu().numpy().tolist())
        cmp_e, cmp_e_list = single_metric(ith_cmp, y, eval_metric, loss)
        step_cmp_test_errs.extend(cmp_e_list.detach().cpu().numpy().tolist())
        base_e, base_e_list = single_metric(prev_s, y, eval_metric, loss)
        step_base_test_errs.extend(base_e_list.detach().cpu().numpy().tolist())

        test_errs.append(e.detach().numpy().item())
        cmp_test_errs.append(cmp_e.item())
        base_test_errs.append(base_e.item())
        # logging.info('sample: {}, error: {}, weiner: {}, base: {}'.format(idx + 1, e, cmp_e, base_e))
        error_sum += e
        cmp_error_sum += cmp_e
        base_error_sum += base_e

    print('Evaluation metrics: {}'.format(eval_metric))
    print('error sum: {}'.format(error_sum))
    print('weiner sum: {}'.format(cmp_error_sum))
    print('base sum: {}'.format(base_error_sum))
    logging.info('Evaluation metrics: {}'.format(eval_metric))
    logging.info('error sum: {}'.format(error_sum))
    logging.info('weiner sum: {}'.format(cmp_error_sum))
    logging.info('base sum: {}'.format(base_error_sum))

    logging.info('alphas: {}'.format(torch.tensor(alphas).detach().numpy()))
    if omega_in == 'fixed':
        logging.info('b: {}'.format(cmp_paramb))
    else:
        logging.info('b: {}'.format(net.omega.paramb.to(cpu).detach().numpy()))

    return omegas, errors, alphas, tests, np.array(test_errs), np.array(cmp_test_errs), base_test_errs, test_time, \
           step_test_errs, step_cmp_test_errs, step_base_test_errs, bounds


device = try_device(int(sys.argv[2]))  # <0: cpu, >=0: gpu

if __name__ == '__main__':
    # datasets = ['XJTU', 'NASA']
    datasets = ['NASA']
    deepwiener = 'dw'
    remove_xjtu_outlier = False

    for dataset in datasets:
        hv_datas = ['v'] if dataset == 'NASA' else ['v', 'h']
        # hv_datas = ['v']
        data_folder = 'CMAPSSData' if dataset == 'NASA' else 'XJTU-SY_Bearing_Datasets'
        split_folder = 'Train_Test/'
        folder = os.path.join('Datasets', data_folder, split_folder)
        v_state_train_raw = load_data(folder + 'v_state_train')
        v_state_test_raw = load_data(folder + 'v_state_test')
        v_state_train = []
        v_state_test = []
        h_state_train = []
        h_state_test = []
        hv_state_train = []
        hv_state_test = []
        if data_folder == 'XJTU-SY_Bearing_Datasets':
            h_state_train_raw = load_data(folder + 'h_state_train')
            h_state_test_raw = load_data(folder + 'h_state_test')
            if remove_xjtu_outlier:
                for i in range(15):
                    if i != 11 and i != 14:
                        v_state_train.append(v_state_train_raw[i])
                        v_state_test.append(v_state_test_raw[i])
                        h_state_train.append(h_state_train_raw[i])
                        h_state_test.append(h_state_test_raw[i])
            else:
                v_state_train = v_state_train_raw
                v_state_test = v_state_test_raw
                h_state_train = h_state_train_raw
                h_state_test = h_state_test_raw
            hv_state_train = v_state_train + h_state_train
            hv_state_test = v_state_test + h_state_test
            v_state_train_len = [len(x) for x in v_state_train]
            v_state_test_len = [len(x) for x in v_state_test]
            h_state_train_len = [len(x) for x in h_state_train]
            h_state_test_len = [len(x) for x in h_state_test]
            states_len = {'v_train': v_state_train_len, 'v_test': v_state_test_len, 'h_train': h_state_train_len,
                          'h_test':
                              h_state_test_len}
            pd.DataFrame(states_len).to_csv('./xjtu_state_len.csv')
        else:
            v_state_train = v_state_train_raw
            v_state_test = v_state_test_raw
            v_state_train_len = [len(x) for x in v_state_train]
            v_state_test_len = [len(x) for x in v_state_test]
            states_len = {'v_train': v_state_train_len, 'v_test': v_state_test_len}
            pd.DataFrame(states_len).to_csv('./nasa_state_len.csv')
        states_train = {'v': v_state_train, 'h': h_state_train, 'hv': hv_state_train}
        states_test = {'v': v_state_test, 'h': h_state_test, 'hv': hv_state_test}

        # 超参数搜索范围
        alpha_rnn_layers = [2]
        alpha_mlp_layers = [1]
        # alpha_bidirs = [True]
        alpha_bidir = False
        alpha_dropouts = [0]
        # alpha_diffs = [2]
        alpha_diff = 0
        err_layers = [2]
        omega_initbs = [.5]
        error_hdims = [16]
        alpha_hdim1s = [16]
        alpha_hdim2s = [16]
        alpha_rnn = 'gru'  # gru, lstm, rnn, transformer
        alpha_rnn_out = 'mean'  # last, mean
        alpha_nheads = [1]
        alpha_windows = [False]
        lamds = [10]
        error_dropouts = [0]
        # lr_desteps = [10]
        # lr_derates = [0.5]
        # init_lrs = [0.02]
        lr_destep = 10
        lr_derate = 0.5
        init_lr = 0.02
        concat_ns = [1]
        patience = 10
        min_delta = .1 if dataset == "XJTU" else .01
        seed = int(sys.argv[1])  # random seed

        num_folds = 5
        kfolds = list(range(num_folds))
        betas = [0.5]  ## init_sigma

        # 网络配置
        # alpha_rnn_layer = 1
        # alpha_mlp_layer = 1
        # err_layer = 1
        # omega_initb = 1
        # error_hdim = 16
        # alpha_hdim1 = 32
        # alpha_hdim2 = 16
        # lamd = 10
        for hv_data in hv_datas:
            train_best_loss_all = 2e9
            train_best_time_all = 2e9
            for beta in betas:
                for alpha_rnn_layer in alpha_rnn_layers:
                    for alpha_mlp_layer in alpha_mlp_layers:
                        for alpha_nhead in alpha_nheads:
                            for alpha_window in alpha_windows:
                                for err_layer in err_layers:
                                    for omega_initb in omega_initbs:
                                        for error_hdim in error_hdims:
                                            for alpha_hdim1 in alpha_hdim1s:
                                                for alpha_hdim2 in alpha_hdim2s:
                                                    for lamd in lamds:
                                                        # for alpha_bidir in alpha_bidirs:
                                                        for alpha_dropout in alpha_dropouts:
                                                            # for alpha_diff in alpha_diffs:
                                                            for error_dropout in error_dropouts:
                                                                # for lr_derate in lr_derates:
                                                                #     for lr_destep in lr_desteps:
                                                                #         for init_lr in init_lrs:
                                                                for concat_n in concat_ns:
                                                                    alpha_in = 'state'  # state, stts, vibr, vibr_fcn, fixed
                                                                    omega_in = 'fixed'  # state, vs, fixed
                                                                    error_term = True
                                                                    if 'na' in deepwiener:
                                                                        alpha_in = 'fixed'
                                                                    if 'nb' in deepwiener:
                                                                        omega_in = 'fixed'
                                                                    if 'ne' in deepwiener:
                                                                        error_term = False
                                                                    omega_func = 'exp'  # exp, mlp
                                                                    omega_hdim = 32  # only when omega_func==mlp
                                                                    error_in = 'axw'  # axw, a_w, vs, fixed, axw+ori
                                                                    error_outnorm = True

                                                                    norm = True
                                                                    incre = True

                                                                    ## 训练配置
                                                                    shuf = True
                                                                    if alpha_in == 'fixed':
                                                                        grad_clip = False
                                                                    else:
                                                                        grad_clip = True
                                                                    cross_val = True
                                                                    grad_thres = 1
                                                                    reg_mod = 'axw'  # axw, errorabs, errorsqr, none
                                                                    loss = nn.MSELoss(reduction='none')
                                                                    trloss_mode = 'mean'  # sum, mean
                                                                    eval_metric = 'rmse'  # mse, rmse
                                                                    lr, num_epochs = init_lr, 100
                                                                    opt = 'adam'  # sgd, adam
                                                                    lr_decay = 'platlr'  # platlr, steplr, explr
                                                                    rc = 0
                                                                    mod = 2
                                                                    errtm = True if error_term else False

                                                                    compare_folder = f'Results/{dataset}/wiener/jili_mod{mod}_usedeep_errtm{errtm}_seed{seed}'
                                                                    cmp_omega_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'omega')
                                                                    cmp_error_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'error')
                                                                    cmp_alpha_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'alpha')
                                                                    cmp_b_pickles = os.path.join(
                                                                        compare_folder, 'b')
                                                                    cmp_sigmaB_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'sigmaB')
                                                                    cmp_train_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'train_est')
                                                                    cmp_test_pickles = os.path.join(
                                                                        compare_folder,
                                                                        'test_est')

                                                                    cmp_omega = \
                                                                        load_data(cmp_omega_pickles)[
                                                                            hv_data]
                                                                    cmp_error = \
                                                                        load_data(cmp_error_pickles)[
                                                                            hv_data]
                                                                    cmp_alpha = \
                                                                        load_data(cmp_alpha_pickles)[
                                                                            hv_data]
                                                                    cmp_paramb = load_data(cmp_b_pickles)[
                                                                        hv_data]
                                                                    # cmp_sigmaB = load_data(cmp_sigmaB_pickles)[hv_data]
                                                                    cmp_sigmaB = beta
                                                                    cmp_train_raw = \
                                                                        load_data(cmp_train_pickles)[
                                                                            hv_data]
                                                                    cmp_test_raw = load_data(cmp_test_pickles)[
                                                                        hv_data]

                                                                    cmp_train = []
                                                                    cmp_test = []
                                                                    if data_folder == 'XJTU-SY_Bearing_Datasets':
                                                                        if remove_xjtu_outlier:
                                                                            for i in range(15):
                                                                                if i != 11 and i != 14:
                                                                                    cmp_train.append(
                                                                                        cmp_train_raw[i])
                                                                                    cmp_test.append(
                                                                                        cmp_test_raw[i])
                                                                        else:
                                                                            cmp_train = cmp_train_raw
                                                                            cmp_test = cmp_test_raw
                                                                    else:
                                                                        cmp_train = cmp_train_raw
                                                                        cmp_test = cmp_test_raw
                                                                    alpha_rnn_layer_str = ''
                                                                    alpha_mlp_layer_str = ''
                                                                    err_layer_str = ''
                                                                    for i in range(alpha_rnn_layer):
                                                                        alpha_rnn_layer_str += f'@{alpha_hdim1}'
                                                                    for i in range(alpha_mlp_layer):
                                                                        alpha_mlp_layer_str += f'@{alpha_hdim2}'
                                                                    for i in range(err_layer):
                                                                        err_layer_str += f'@{error_hdim}'

                                                                    now_time = datetime.now().strftime(
                                                                        '%Y%m%d')
                                                                    tmp_folder_path_prefix = f'Results_{now_time}_runs/{dataset}_remove{remove_xjtu_outlier}/Deep_Wiener/data{hv_data}_log_old'
                                                                    # tmp_folder_path_suffix = f'{deepwiener}_wiener_alpin{alpha_in}_alptype{alpha_rnn}_alprnn{alpha_rnn_layer_str}_' \
                                                                    #                          f'alpmlp{alpha_mlp_layer_str}_alpbid{alpha_bidir}_alpdrop{alpha_dropout}_alpdiff{alpha_diff}_' \
                                                                    #                          f'omgin{omega_in}_omginitb{omega_initb}_' \
                                                                    #                          f'errmlp{err_layer_str}_errOnorm{error_outnorm}_errdrop{error_dropout}_initb{beta}_' \
                                                                    #                          f'lrdecay{lr_decay}_lrds{lr_destep}_lrdr{lr_derate}_lr{lr}_' \
                                                                    #                          f'concatn{concat_n}_lamd{lamd}_norm{norm}_shuf{shuf}_gradclip{grad_clip}@{grad_thres}_errtm{error_term}_cv{cross_val}_seed{seed}'
                                                                    tmp_folder_path_suffix = f'{deepwiener}_wiener_alpin{alpha_in}_alptype{alpha_rnn}_alprnn{alpha_rnn_layer_str}_' \
                                                                                             f'alpmlp{alpha_mlp_layer_str}_alpbid{alpha_bidir}_alpdrop{alpha_dropout}_alpdiff{alpha_diff}_alpnhead{alpha_nhead}_alpwnd{alpha_window}_' \
                                                                                             f'omgin{omega_in}_omginitb{omega_initb}_' \
                                                                                             f'errmlp{err_layer_str}_errOnorm{error_outnorm}_errdrop{error_dropout}_initsgm{beta}_' \
                                                                                             f'concatn{concat_n}_lamd{lamd}_norm{norm}_shuf{shuf}_gradclip{grad_clip}@{grad_thres}_errtm{error_term}_cv{cross_val}_seed{seed}'

                                                                    folder_path = f'{tmp_folder_path_prefix}/{tmp_folder_path_suffix}'  # 定义要创建的目录

                                                                    # 调用函数
                                                                    mkdir(folder_path)
                                                                    imgfmt = '.png'
                                                                    plot_file = os.path.join(folder_path,
                                                                                             'predict' + imgfmt)
                                                                    log_file = os.path.join(folder_path,
                                                                                            'logging.txt')
                                                                    log_init(log_file)

                                                                    # train_plot_file = os.path.join(folder_path,
                                                                    #                                'train_predict' + imgfmt)
                                                                    # train_res = os.path.join(folder_path, 'train_res.txt')

                                                                    kfold_train_best_loss = 0
                                                                    kfold_train_best_time = 0
                                                                    kfold_train_loss_per_epoch = None
                                                                    kfold_valid_loss_per_epoch = None
                                                                    net = None
                                                                    for kfold in kfolds:
                                                                        logging.info(
                                                                            f"========kfold{kfold}========")
                                                                        print(
                                                                            f"========kfold{kfold}========")
                                                                        same_seeds(seed)
                                                                        net = get_net()
                                                                        net = net.to(device)
                                                                        get_parameter_number(net)
                                                                        earlystopper = EarlyStopper(
                                                                            patience=patience,
                                                                            min_delta=min_delta)

                                                                        if opt == 'sgd':
                                                                            trainer = torch.optim.SGD(
                                                                                net.parameters(),
                                                                                lr=lr)
                                                                        elif opt == 'adam':
                                                                            trainer = torch.optim.AdamW(
                                                                                net.parameters(),
                                                                                lr=lr)
                                                                        else:
                                                                            trainer = torch.optim.AdamW(
                                                                                net.parameters(),
                                                                                lr=lr)

                                                                        if lr_decay == 'steplr':
                                                                            scheduler = torch.optim.lr_scheduler.StepLR(
                                                                                trainer,
                                                                                step_size=lr_destep,
                                                                                gamma=lr_derate,
                                                                                verbose=True)
                                                                        elif lr_decay == 'explr':
                                                                            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                                                                                trainer,
                                                                                gamma=lr_derate,
                                                                                verbose=True)
                                                                        elif lr_decay == 'platlr':
                                                                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                                trainer,
                                                                                factor=lr_derate,
                                                                                patience=lr_destep,
                                                                                min_lr=0.0001,
                                                                                verbose=True
                                                                            )
                                                                        else:
                                                                            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                                                                                trainer,
                                                                                gamma=lr_derate,
                                                                                verbose=True)

                                                                        model_path = os.path.join(
                                                                            folder_path,
                                                                            f'model_fold{kfold}.pth')

                                                                        train_time, train_best_loss, train_loss_per_epoch, valid_loss_per_epoch = train(
                                                                            net, states_train[hv_data],
                                                                            trainer,
                                                                            scheduler, loss, num_epochs,
                                                                            lamd,
                                                                            device, model_path,
                                                                            float('inf'), kfold,
                                                                            earlystopper)
                                                                        print(
                                                                            f'training_time: {train_time}, train_best_loss: {train_best_loss}')
                                                                        logging.info(
                                                                            'training_time: {}, train_best_loss: {}'.format(
                                                                                train_time,
                                                                                train_best_loss))
                                                                        kfold_train_best_loss += train_best_loss
                                                                        kfold_train_best_time += train_time

                                                                        if kfold_train_loss_per_epoch is None:
                                                                            kfold_train_loss_per_epoch = train_loss_per_epoch
                                                                        else:
                                                                            kfold_train_loss_per_epoch = list(
                                                                                map(lambda x, y: x + y,
                                                                                    train_loss_per_epoch,
                                                                                    kfold_train_loss_per_epoch))

                                                                        if kfold_valid_loss_per_epoch is None:
                                                                            kfold_valid_loss_per_epoch = valid_loss_per_epoch
                                                                        else:
                                                                            kfold_valid_loss_per_epoch = list(
                                                                                map(lambda x, y: x + y,
                                                                                    valid_loss_per_epoch,
                                                                                    kfold_valid_loss_per_epoch))
                                                                        torch.cuda.empty_cache()

                                                                    kfold_train_best_loss /= num_folds
                                                                    kfold_train_best_time /= num_folds

                                                                    if kfold_train_best_loss < train_best_loss_all:
                                                                        train_best_loss_all = kfold_train_best_loss
                                                                        train_best_time_all = kfold_train_best_time
                                                                        best_folder_path_prefix = f'Results_{now_time}_runs/{dataset}_remove{remove_xjtu_outlier}/Deep_Wiener/data{hv_data}_log_best'
                                                                        folder_path_best = f'{best_folder_path_prefix}/{tmp_folder_path_suffix}'
                                                                        # folder_path_best = f'Results/{dataset}/Deep_Wiener/data{hv_data}_log_best/beta{beta}/{deepwiener}_wiener_alpin{alpha_in}_alptype{alpha_rnn}_alprnn{alpha_rnn_layer_str}_' \
                                                                        #                    f'alpmlp{alpha_mlp_layer_str}_alpbid{alpha_bidir}_alpdrop{alpha_dropout}_alpdiff{alpha_diff}_' \
                                                                        #                    f'omgin{omega_in}_omginitb{omega_initb}_' \
                                                                        #                    f'errmlp{err_layer_str}_errOnorm{error_outnorm}_errdrop{error_dropout}' \
                                                                        #                    f'lrdecay{lr_decay}_lrds{lr_destep}_lrdr{lr_derate}_lr{lr}_' \
                                                                        #                    f'lamd{lamd}_norm{norm}_shuf{shuf}_gradclip{grad_clip}@{grad_thres}_errtm{error_term}_cv{cross_val}_seed{seed}'
                                                                        mkdir(folder_path_best)
                                                                        with open(
                                                                                f'Results_{now_time}_runs/{dataset}_remove{remove_xjtu_outlier}/Deep_Wiener/data{hv_data}_log_best/best_hp.txt',
                                                                                'w') as f:
                                                                            f.write(f"{folder_path_best}")
                                                                        eval_metric = 'rmse'
                                                                        test_res = os.path.join(
                                                                            folder_path_best,
                                                                            'test_res_' + eval_metric + '.txt')
                                                                        log_init(test_res)
                                                                        kfold_test_errs = None
                                                                        kfold_step_test_errs = None
                                                                        kfold_test_time = 0
                                                                        alphas = None
                                                                        omegas = None
                                                                        errors = None
                                                                        tests = None
                                                                        bounds = None
                                                                        cmp_test_errs = None
                                                                        base_test_errs = None
                                                                        step_cmp_test_errs = None

                                                                        for kfold in kfolds:
                                                                            model_path = os.path.join(
                                                                                folder_path,
                                                                                f'model_fold{kfold}.pth')
                                                                            net.load_state_dict(
                                                                                torch.load(model_path))
                                                                            net.eval()
                                                                            logging.info(
                                                                                f"========kfold{kfold}========")
                                                                            print(
                                                                                f"========kfold{kfold}========")
                                                                            omegas, errors, alphas, tests, test_errs, cmp_test_errs, base_test_errs, test_time, \
                                                                            step_test_errs, step_cmp_test_errs, step_base_test_errs, bounds \
                                                                                = predict(net, cmp_test,
                                                                                          states_test,
                                                                                          hv_data,
                                                                                          loss, device)
                                                                            if kfold_test_errs is None:
                                                                                kfold_test_errs = np.array(
                                                                                    test_errs)
                                                                            else:
                                                                                kfold_test_errs += np.array(
                                                                                    test_errs)
                                                                            if kfold_step_test_errs is None:
                                                                                kfold_step_test_errs = np.array(
                                                                                    step_test_errs)
                                                                            else:
                                                                                kfold_step_test_errs += np.array(
                                                                                    step_test_errs)
                                                                            print(f'test_time: {test_time}')
                                                                            logging.info(
                                                                                'predict time: {}'.format(
                                                                                    test_time))
                                                                            kfold_test_time += test_time
                                                                            torch.cuda.empty_cache()

                                                                        kfold_test_errs /= num_folds
                                                                        kfold_step_test_errs /= num_folds
                                                                        kfold_test_time /= num_folds
                                                                        kfold_test_errs_sum = np.sum(
                                                                            kfold_test_errs)
                                                                        kfold_step_test_errs_sum = np.sum(
                                                                            kfold_step_test_errs)
                                                                        cmp_test_errs_sum = np.sum(
                                                                            cmp_test_errs)
                                                                        step_cmp_test_errs_sum = np.sum(
                                                                            step_cmp_test_errs)

                                                                        test_errs_pickles = os.path.join(
                                                                            folder_path_best,
                                                                            'test_errs')
                                                                        cmp_test_errs_pickles = os.path.join(
                                                                            folder_path_best,
                                                                            'cmp_test_errs')
                                                                        base_test_errs_pickles = os.path.join(
                                                                            folder_path_best,
                                                                            'base_test_errs')
                                                                        omega_pickles = os.path.join(
                                                                            folder_path_best, 'omega')
                                                                        error_pickles = os.path.join(
                                                                            folder_path_best, 'error')
                                                                        alpha_pickles = os.path.join(
                                                                            folder_path_best, 'alpha')
                                                                        b_pickles = os.path.join(
                                                                            folder_path_best, 'b')
                                                                        bound_pickles = os.path.join(
                                                                            folder_path_best, 'bounds')
                                                                        test_pickles = os.path.join(
                                                                            folder_path_best, 'test_est')
                                                                        train_pickles = os.path.join(
                                                                            folder_path_best, 'train_est')
                                                                        save_data(omegas, omega_pickles)
                                                                        save_data(alphas, alpha_pickles)
                                                                        save_data(errors, error_pickles)
                                                                        save_data(bounds, bound_pickles)
                                                                        if omega_in == 'fixed':
                                                                            save_data(cmp_paramb, b_pickles)
                                                                        else:
                                                                            save_data(net.omega.paramb,
                                                                                      b_pickles)
                                                                        save_data(tests, test_pickles)
                                                                        save_data(kfold_test_errs,
                                                                                  test_errs_pickles)
                                                                        save_data(cmp_test_errs,
                                                                                  cmp_test_errs_pickles)
                                                                        save_data(base_test_errs,
                                                                                  base_test_errs_pickles)
                                                                        cd_df = pd.DataFrame(
                                                                            {'DeepWiener': kfold_test_errs,
                                                                             'Wiener': cmp_test_errs})
                                                                        step_cd_df = pd.DataFrame(
                                                                            {
                                                                                'DeepWiener': kfold_step_test_errs,
                                                                                'Wiener': step_cmp_test_errs})
                                                                        err_sum_df = pd.DataFrame(
                                                                            {'DeepWiener': [
                                                                                kfold_test_errs_sum,
                                                                                np.sqrt(
                                                                                    kfold_step_test_errs_sum)],
                                                                                'Wiener': [
                                                                                    cmp_test_errs_sum,
                                                                                    np.sqrt(
                                                                                        step_cmp_test_errs_sum)]},
                                                                            index=['Per Unit',
                                                                                   'Per Step'])
                                                                        time_df = pd.DataFrame(
                                                                            {'DeepWiener': [
                                                                                train_best_time_all,
                                                                                kfold_test_time],
                                                                            },
                                                                            index=['Training',
                                                                                   'Test'])
                                                                        train_loss_per_epoch_list = [
                                                                            x.item() for x in
                                                                            kfold_train_loss_per_epoch]
                                                                        valid_loss_per_epoch_list = [
                                                                            x.item() for x in
                                                                            kfold_valid_loss_per_epoch]

                                                                        trainloss_df = pd.DataFrame(
                                                                            {
                                                                                'Train Loss': train_loss_per_epoch_list})
                                                                        validloss_df = pd.DataFrame(
                                                                            {
                                                                                'Valid Loss': valid_loss_per_epoch_list})

                                                                        cd_df.to_csv(
                                                                            folder_path_best + f'/err_per_unit.csv',
                                                                            index=False)
                                                                        step_cd_df.to_csv(
                                                                            folder_path_best + f'/err_per_step.csv',
                                                                            index=False)
                                                                        err_sum_df.to_csv(
                                                                            folder_path_best + f'/err_sum.csv'
                                                                        )
                                                                        time_df.to_csv(
                                                                            folder_path_best + f'/time.csv')

                                                                        trainloss_df.to_csv(
                                                                            folder_path_best + f'/trainloss.csv',
                                                                            index=False)
                                                                        validloss_df.to_csv(
                                                                            folder_path_best + f'/validloss.csv',
                                                                            index=False)
                                                                        shutil.copy(
                                                                            folder_path + '/logging.txt',
                                                                            folder_path_best + '/logging.txt')
                                                                        torch.cuda.empty_cache()
                                                                        # sys.exit(0)

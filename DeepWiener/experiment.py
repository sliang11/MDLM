import copy
import logging
import os
import random
import sys
import time

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
import numpy as np

from utils.data import load_data, save_data
from utils.grad_clipping import grad_clipping
from utils.metrics import single_loss, single_metric, evaluate_loss
from utils.tools import try_device, mkdir, same_seeds, log_init

from models.model import Net

def data_iter(vibrs, states, mode='train'):
    sf_idx = list(range(len(vibrs)))
    if mode == 'train' and shuf == True:  # shuffle
        random.shuffle(sf_idx)
    for i in sf_idx:
        X = torch.tensor(np.concatenate((vibrs[i][:-1], states[i][:-1].reshape(-1, 1)), axis=1), dtype=torch.float32)
        if incre == True:
            y = torch.tensor((states[i][1:] - states[i][:-1]).reshape(-1, 1), dtype=torch.float32)
        else:
            y = torch.tensor((states[i][1:]).reshape(-1, 1), dtype=torch.float32)
        yield X, y, i

def train_epoch_ch3(net, train_iter, loss, updater, lamd, device):
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
              alpha_hdim2=alpha_hdim2, alpha_rnn_layer=alpha_rnn_layer, alpha_mlp_layer=alpha_mlp_layer,
              alpha_bidir=alpha_bidir, alpha_nhead=alpha_nhead,
              omega_in=omega_in, omega_func=omega_func, omega_hdim=omega_hdim, omega_initb=omega_initb,
              error_in=error_in, error_hdim=error_hdim, error_outnorm=error_outnorm, err_layer=err_layer,
              sigmaB=cmp_sigmaB,
              norm=norm, incre=incre, error_term=error_term)
    return net


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total:{total_num} Trainable: {trainable_num}')


def train(net, _vibrs_train, _states_train, trainer, scheduler, loss, epochs, lamd, device, model_path, best_loss):
    train_time = 0
    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    for epoch in range(epochs):
        lr = scheduler.get_last_lr()
        if epoch == 0:
            net.eval()
            if cross_val:
                train_loss, valid_loss = evaluate_loss(net, data_iter(_vibrs_train, _states_train, mode='test'),
                                                       cross_val, device, kfold, num_folds, eval_metric, loss, seed)
                train_loss_per_epoch.append(train_loss)
                valid_loss_per_epoch.append(valid_loss)
            else:
                train_loss = evaluate_loss(net, data_iter(_vibrs_train, _states_train, mode='test'),
                                           cross_val, device, kfold, num_folds, eval_metric, loss, seed)
                valid_loss = train_loss
                train_loss_per_epoch.append(train_loss)
                valid_loss_per_epoch.append(valid_loss)
            print(f'epoch {epoch}, '
                  f'train_loss: {train_loss:.8f} '
                  f'valid_loss: {valid_loss:.8f} '
                  f'lr: {lr}')
            logging.info('epoch: {}, train_loss: {}, valid_loss: {}, lr: {}'.format(epoch, train_loss, valid_loss, lr))
        time_start = time.time()
        train_epoch_ch3(net, data_iter(_vibrs_train, _states_train), loss, trainer, lamd, device)
        time_end = time.time()
        train_time += time_end - time_start
        net.eval()
        if cross_val:
            train_loss, valid_loss = evaluate_loss(net, data_iter(_vibrs_train, _states_train, mode='test'),
                                                   cross_val, device, kfold, num_folds, eval_metric, loss, seed)
        else:
            train_loss = evaluate_loss(net, data_iter(_vibrs_train, _states_train, mode='test'),
                                       cross_val, device, kfold, num_folds, eval_metric, loss, seed)
            valid_loss = train_loss

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), model_path)
        scheduler.step()
        train_loss_per_epoch.append(train_loss)
        valid_loss_per_epoch.append(valid_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, '
                  f'train_loss: {train_loss:.8f} '
                  f'valid_loss: {valid_loss:.8f} '
                  f'lr: {lr}')
            logging.info(
                'epoch: {}, train_loss: {}, valid_loss: {}, lr: {}'.format(epoch + 1, train_loss, valid_loss, lr))
    return train_time, best_loss, train_loss_per_epoch, valid_loss_per_epoch

def predict(net, cmp, vibrs_test, states_test, hv_data, loss, device):
    test_iter = data_iter(vibrs_test[hv_data], states_test[hv_data], mode='test')
    idx, error_sum, cmp_error_sum, base_error_sum = 0, 0, 0, 0
    net = net.to(device)
    omegas, errors, alphas, tests, test_errs, cmp_test_errs, base_test_errs = [], [], [], [], [], [], []
    step_test_errs, step_cmp_test_errs, step_base_test_errs = [], [], []
    cpu = try_device(-1)
    test_time = 0

    for X, y, idx in test_iter:
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
        logging.info('sample: {}, error: {}, weiner: {}, base: {}'.format(idx + 1, e, cmp_e, base_e))
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

    return omegas, errors, alphas, tests, test_errs, cmp_test_errs, base_test_errs, test_time, \
           step_test_errs, step_cmp_test_errs, step_base_test_errs


if __name__ == '__main__':
    dataset = 'NASA'
    hv_data = 'v'
    deepwiener = 'dw'

    data_folder = 'CMAPSSData' if dataset == 'NASA' else 'XJTU-SY_Bearing_Datasets'
    split_folder = 'Train_Test/'
    folder = os.path.join('Datasets', data_folder, split_folder)
    v_state_train = load_data(folder + 'v_state_train')
    v_state_test = load_data(folder + 'v_state_test')
    v_vibr_train = load_data(folder + 'v_vibr_train')
    v_vibr_test = load_data(folder + 'v_vibr_test')
    h_state_train = []
    h_state_test = []
    h_vibr_train = []
    h_vibr_test = []
    hv_state_train = []
    hv_state_test = []
    hv_vibr_train = []
    hv_vibr_test = []
    if data_folder == 'XJTU-SY_Bearing_Datasets':
        h_state_train = load_data(folder + 'h_state_train')
        h_state_test = load_data(folder + 'h_state_test')
        hv_state_train = v_state_train + h_state_train
        hv_state_test = v_state_test + h_state_test

    vibrs_train = {'v': v_vibr_train, 'h': h_vibr_train, 'hv': hv_vibr_train}
    states_train = {'v': v_state_train, 'h': h_state_train, 'hv': hv_state_train}
    vibrs_test = {'v': v_vibr_test, 'h': h_vibr_test, 'hv': hv_vibr_test}
    states_test = {'v': v_state_test, 'h': h_state_test, 'hv': hv_state_test}

    # 网络配置
    alpha_rnn_layer = 2
    alpha_mlp_layer = 2
    err_layer = 2
    omega_initb = 1
    lamd = 10
    device = try_device(0)  # <0: cpu, >=0: gpu
    alpha_in = 'state'  # state, stts, vibr, vibr_fcn, fixed
    omega_in = 'state'  # state, vs, fixed
    error_term = True
    if 'na' in deepwiener:
        alpha_in = 'fixed'
    if 'nb' in deepwiener:
        omega_in = 'fixed'
    if 'ne' in deepwiener:
        error_term = False

    alpha_rnn = 'gru'  # gru, lstm, rnn, transformer
    alpha_rnn_out = 'mean'  # last, mean
    alpha_hdim1 = 32
    alpha_hdim2 = 16
    alpha_bidir = False
    alpha_nhead = 1
    omega_func = 'exp'  # exp, mlp
    omega_hdim = 32  # only when omega_func==mlp
    error_in = 'axw'  # axw, a_w, vs, fixed
    error_hdim = 16
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
    lr, num_epochs = 0.002, 100
    kfold = 1
    num_folds = 5
    opt = 'adam'  # sgd, adam
    lr_decay = 'steplr'
    lr_destep = 20
    lr_derate = 0.9
    seed = 0  # random seed
    rc = 0
    mod = 2
    errtm = True if error_term else False

    compare_folder = f'Results/{dataset}/wiener/jili_mod{mod}_errtm{errtm}'
    cmp_omega_pickles = os.path.join(compare_folder, 'omega')
    cmp_error_pickles = os.path.join(compare_folder, 'error')
    cmp_alpha_pickles = os.path.join(compare_folder, 'alpha')
    cmp_b_pickles = os.path.join(compare_folder, 'b')
    cmp_sigmaB_pickles = os.path.join(compare_folder, 'sigmaB')
    cmp_train_pickles = os.path.join(compare_folder, 'train_est')
    cmp_test_pickles = os.path.join(compare_folder, 'test_est')

    cmp_omega = load_data(cmp_omega_pickles)[hv_data]
    cmp_error = load_data(cmp_error_pickles)[hv_data]
    cmp_alpha = load_data(cmp_alpha_pickles)[hv_data]
    cmp_paramb = load_data(cmp_b_pickles)[hv_data]
    cmp_sigmaB = load_data(cmp_sigmaB_pickles)[hv_data]
    cmp_train = load_data(cmp_train_pickles)[hv_data]
    cmp_test = load_data(cmp_test_pickles)[hv_data]

    alpha_rnn_layer_str = ''
    alpha_mlp_layer_str = ''
    err_layer_str = ''
    for i in range(alpha_rnn_layer):
        alpha_rnn_layer_str += f'@{alpha_hdim1}'
    for i in range(alpha_mlp_layer):
        alpha_mlp_layer_str += f'@{alpha_hdim2}'
    for i in range(err_layer):
        err_layer_str += f'@{error_hdim}'
    folder_path = f'Results/{dataset}/Deep_Wiener/data{hv_data}_log/{deepwiener}_wiener_alpin{alpha_in}_alptype{alpha_rnn}_alprnn{alpha_rnn_layer_str}_' \
                  f'alpmlp{alpha_mlp_layer_str}_alpbid{alpha_bidir}_alpnhead{alpha_nhead}_' \
                  f'omgin{omega_in}_omginitb{omega_initb}_' \
                  f'errmlp{err_layer_str}_errOnorm{error_outnorm}_' \
                  f'lrdecay{lr_decay}_lrds{lr_destep}_lrdr{lr_derate}_lr{lr}_kfold{kfold}_' \
                  f'lamd{lamd}_norm{norm}_shuf{shuf}_gradclip{grad_clip}@{grad_thres}_errtm{error_term}_cv{cross_val}_seed{seed}'
    # 定义要创建的目录
    # 调用函数
    mkdir(folder_path)
    imgfmt = '.png'
    plot_file = os.path.join(folder_path, 'predict' + imgfmt)
    log_file = os.path.join(folder_path, 'logging.txt')
    log_init(log_file)
    omega_pickles = os.path.join(folder_path, 'omega')
    error_pickles = os.path.join(folder_path, 'error')
    alpha_pickles = os.path.join(folder_path, 'alpha')
    b_pickles = os.path.join(folder_path, 'b')
    test_pickles = os.path.join(folder_path, 'test_est')
    train_pickles = os.path.join(folder_path, 'train_est')

    train_plot_file = os.path.join(folder_path, 'train_predict' + imgfmt)
    train_res = os.path.join(folder_path, 'train_res.txt')
    model_path = os.path.join(folder_path, 'model.pth')

    same_seeds(seed)
    net = get_net()
    net = net.to(device)
    get_parameter_number(net)

    if opt == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=lr)
    elif opt == 'adam':
        trainer = torch.optim.AdamW(net.parameters(), lr=lr)
    else:
        trainer = torch.optim.AdamW(net.parameters(), lr=lr)

    if lr_decay == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=lr_destep, gamma=lr_derate)
    elif lr_decay == 'explr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=lr_derate)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=lr_derate)

    train_time, train_best_loss, train_loss_per_epoch, valid_loss_per_epoch = train(net, vibrs_train[hv_data],
                                                                                    states_train[hv_data], trainer,
                                                                                    scheduler, loss, num_epochs, lamd,
                                                                                    device, model_path, float('inf'))
    print(f'training_time: {train_time}, train_best_loss: {train_best_loss}')
    logging.info('training_time: {}, train_best_loss: {}'.format(train_time, train_best_loss))
    torch.cuda.empty_cache()

    eval_metric = 'rmse'
    test_res = os.path.join(folder_path, 'test_res_' + eval_metric + '.txt')
    log_init(test_res)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    omegas, errors, alphas, tests, test_errs, cmp_test_errs, base_test_errs, test_time, \
    step_test_errs, step_cmp_test_errs, step_base_test_errs \
        = predict(net, cmp_test, vibrs_test, states_test, hv_data, loss, device)

    print(f'test_time: {test_time}')
    logging.info('predict time: {}'.format(test_time))
    torch.cuda.empty_cache()
    test_errs_pickles = os.path.join(folder_path, 'test_errs')
    cmp_test_errs_pickles = os.path.join(folder_path, 'cmp_test_errs')
    base_test_errs_pickles = os.path.join(folder_path, 'base_test_errs')
    save_data(omegas, omega_pickles)
    save_data(alphas, alpha_pickles)
    save_data(errors, error_pickles)
    if omega_in == 'fixed':
        save_data(cmp_paramb, b_pickles)
    else:
        save_data(net.omega.paramb, b_pickles)
    save_data(tests, test_pickles)
    save_data(test_errs, test_errs_pickles)
    save_data(cmp_test_errs, cmp_test_errs_pickles)
    save_data(base_test_errs, base_test_errs_pickles)
    cd_df = pd.DataFrame({'Ours': test_errs, 'Wiener': cmp_test_errs})
    step_cd_df = pd.DataFrame({'Ours': step_test_errs, 'Wiener': step_cmp_test_errs})
    train_loss_per_epoch_list = [x.item() for x in train_loss_per_epoch]
    valid_loss_per_epoch_list = [x.item() for x in valid_loss_per_epoch]

    trainloss_df = pd.DataFrame({'Train Loss': train_loss_per_epoch_list})
    validloss_df = pd.DataFrame({'Valid Loss': valid_loss_per_epoch_list})

    cd_df.to_csv(folder_path + f'/{hv_data}_cd_diagram.csv', index=False)
    step_cd_df.to_csv(folder_path + f'/{hv_data}_step_cd_diagram.csv', index=False)

    trainloss_df.to_csv(folder_path + f'/{hv_data}_trainloss.csv', index=False)
    validloss_df.to_csv(folder_path + f'/{hv_data}_validloss.csv', index=False)
    torch.cuda.empty_cache()
    sys.exit(0)

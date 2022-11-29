import torch
from sklearn.model_selection import KFold


def single_loss(loss, y_hat, y, lamd, error, axw, prev_s, mod, incre):
    if mod == 'errorabs':
        return loss(y_hat, y) + lamd * (torch.log(abs(error / y_hat)) ** 2)
    elif mod == 'errorsqr':
        return loss(y_hat, y) + lamd * (torch.log((error / y_hat) ** 2) ** 2)
    elif mod == 'axw':
        if incre == True:
            return loss(y_hat, y) + lamd * ((y_hat - axw) ** 2)
        else:
            return loss(y_hat, y) + lamd * ((y_hat - prev_s - axw) ** 2)
    else:
        return loss(y_hat, y)


def single_metric(y_hat, y, eval_metric, loss):
    if eval_metric == 'rmse':
        return torch.sqrt(loss(y_hat, y).mean()), loss(y_hat, y).flatten()
    elif eval_metric == 'mse':
        return loss(y_hat, y).mean()


def evaluate_loss(net, data_iter, cross_val, device, kfold, num_folds, eval_metric, loss, seed):
    """评估给定数据集上模型的损失"""
    train_loss = 0
    valid_loss = 0
    if cross_val == False:
        for X, y, idx in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X, idx, predict=False)
            y = y.reshape(y_hat.shape)
            train_loss_unit, _ = single_metric(y_hat, y, eval_metric, loss)
            train_loss += train_loss_unit
        return train_loss
    else:
        for X, y, idx in data_iter:
            X = X.to(device)
            y = y.to(device)
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            train_index, valid_index = list(kf.split(X))[kfold]
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            y_train_hat = net(X_train, idx, predict=False)
            y_valid_hat = net(X_valid, idx, predict=False)
            y_train = y_train.reshape(y_train_hat.shape)
            y_valid = y_valid.reshape(y_valid_hat.shape)
            train_loss_unit, _ = single_metric(y_train_hat, y_train, eval_metric, loss)
            valid_loss_unit, _ = single_metric(y_valid_hat, y_valid, eval_metric, loss)
            train_loss += train_loss_unit
            valid_loss += valid_loss_unit
        return train_loss, valid_loss

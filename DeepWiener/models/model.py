from torch import nn, Tensor
import torch
import numpy as np
import math
import sys
# from experiment import device

def try_device(i=0):
    """<0返回cpu(); 如果存在，则返回gpu(i)，否则返回cpu()"""
    if i < 0: return torch.device('cpu')
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


device = try_device(int(sys.argv[2]))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1] = torch.cos(position * div_term[1]).squeeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Alpha(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, out_dim=1, rnn_layer=1, nhead=1, mlp_layer=2,
                 rnn_type='gru', rnn_out='last', norm=True, bidir=False, dropout=0.2, diff=1, concat_nframe=3):
        super(Alpha, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.rnn_out = rnn_out
        self.rnn_output = None
        self.diff = diff
        self.concat_nframe = concat_nframe
        self.pos_encoder = None

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True,
                               dropout=dropout)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True, dropout=dropout)
        elif rnn_type == 'transformer':
            self.tel = nn.TransformerEncoderLayer(d_model=concat_nframe, nhead=nhead, dim_feedforward=hidden_dim1,
                                                  batch_first=True)
            self.rnn = nn.TransformerEncoder(self.tel, num_layers=rnn_layer)
            self.pos_encoder = PositionalEncoding(concat_nframe, dropout=dropout)

        else:
            self.rnn = nn.GRU(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True, dropout=dropout)

        if bidir:
            hidden_dim1 = 2 * hidden_dim1
        if rnn_type == 'transformer':
            hidden_dim1 = concat_nframe
        if norm:
            self.mlp = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.LayerNorm(hidden_dim2), nn.PReLU()
                                     )
            for i in range(mlp_layer):
                self.mlp = nn.Sequential(self.mlp,
                                         nn.Linear(hidden_dim2, hidden_dim2), nn.LayerNorm(hidden_dim2), nn.PReLU(),
                                         )
            self.mlp = nn.Sequential(self.mlp,
                                     nn.Linear(hidden_dim2, out_dim)
                                     )
        else:
            self.mlp = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.PReLU(),
                                     nn.Linear(hidden_dim2, out_dim))

    def forward(self, x):
        if self.diff != 0:
            x = x.diff(n=self.diff, dim=0, prepend=torch.Tensor([[0] for _ in range(self.diff)]).to(device))
        if self.rnn_type == 'gru':
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))
        elif self.rnn_type == 'lstm':
            out, (_, _) = self.rnn(x.reshape(1, -1, self.input_dim))
        elif self.rnn_type == 'rnn':
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))
        elif self.rnn_type == 'transformer':
            x = x.reshape(-1, self.input_dim)
            x = concat_feat(x, self.concat_nframe)
            x = self.pos_encoder(x.reshape(-1, 1, self.concat_nframe))
            out = self.rnn(x.reshape(1, -1, self.concat_nframe))
        else:
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))

        self.rnn_output = out.mean(dim=1).reshape(1, -1)
        return self.mlp(self.rnn_output).squeeze(dim=0)


class Omega(nn.Module):
    def __init__(self, in_dim=1, h_dim=0, norm=True, initb=1.):
        super(Omega, self).__init__()

        # 将自定义参数b加入网络进行学习
        self.register_parameter(name='paramb',
                                param=torch.nn.Parameter(torch.Tensor([[initb]])))
        self.h_dim = h_dim
        self.in_dim = in_dim
        if h_dim > 0 and in_dim > 1:
            if norm:
                self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.PReLU(),
                                         nn.Linear(h_dim, 1))
            else:
                self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim), nn.PReLU(),
                                         nn.Linear(h_dim, 1))
        elif in_dim > 1:
            self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.h_dim > 0 and self.in_dim > 1:
            omg_in = torch.cat((x, self.paramb.expand((x.shape[0], 1))), dim=1)
            return self.mlp(omg_in)
        elif self.in_dim > 1:
            return self.out(x.pow(self.paramb))
        return x.pow(self.paramb)


class Error(nn.Module):
    def __init__(self, in_dim, h_dim=16, norm=True, out_norm=False, sigmaB=0.02, mlp_layer=1, dropout=0.2):
        super(Error, self).__init__()
        self.out_norm = out_norm
        # 将自定义参数sigmaB加入网络进行学习
        self.register_parameter(name='sigmaB',
                                param=torch.nn.Parameter(torch.Tensor([[sigmaB]])))
        # self.sigmaB = sigmaB
        self.mlp_layer = mlp_layer
        self.in_dim = in_dim
        if norm:
            self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.PReLU()
                                     )

            for i in range(mlp_layer):
                self.mlp = nn.Sequential(self.mlp,
                                         nn.Dropout(dropout), nn.Linear(h_dim, h_dim), nn.BatchNorm1d(h_dim),
                                         nn.PReLU(),
                                         )
            self.mlp = nn.Sequential(self.mlp,
                                     nn.Dropout(dropout), nn.Linear(h_dim, 1)
                                     )
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim), nn.PReLU(),
                                     nn.Linear(h_dim, 1))

    def forward(self, x):
        if self.out_norm:
            # bounding = (x[:, 0] * self.sigmaB).reshape(x.shape[0], 1)
            unbounded = torch.tanh(self.mlp(x))
            return (self.sigmaB * unbounded).squeeze(dim=0)
        else:
            return self.mlp(x).squeeze(dim=0)


class Net(nn.Module):
    """
    整体网络
    """

    def __init__(self, cmp_err, cmp_alpha, cmp_omega,
                 alpha_in='vibr_fcn', alpha_rnn='gru', alpha_rnn_out='last', alpha_hdim1=16, alpha_hdim2=32,
                 alpha_rnn_layer=1, alpha_mlp_layer=2, alpha_bidir=False, alpha_nhead=1, alpha_dropout=0.2,
                 alpha_diff=1, alpha_concatn=3, alpha_window=True,
                 omega_in='state', omega_func='exp', omega_hdim=0, omega_initb=1.,
                 error_in='axw', error_hdim=16, error_outnorm=False, err_layer=1, sigmaB=0.02,
                 norm=True, incre=True, error_term=True, error_dropout=0.2):
        super(Net, self).__init__()
        self.E = None
        self.a = None
        self.W = None
        self.s = None
        self.ret = None
        self.incre = incre
        self.alpha_in = alpha_in
        self.error_in = error_in
        self.omega_in = omega_in
        self.omega_func = omega_func
        self.error_term = error_term
        self.alpha_trained = [0] * len(cmp_err)
        self.history = [torch.ones(1, 1).to(device) for _ in range(len(cmp_err))]
        self.an = []
        self.alpha_window = alpha_window
        if alpha_in == 'state':
            self.alpha = Alpha(input_dim=1, hidden_dim1=alpha_hdim1, hidden_dim2=alpha_hdim2,
                               rnn_type=alpha_rnn, rnn_out=alpha_rnn_out, norm=norm, nhead=alpha_nhead,
                               rnn_layer=alpha_rnn_layer, mlp_layer=alpha_mlp_layer, bidir=alpha_bidir,
                               dropout=alpha_dropout, diff=alpha_diff, concat_nframe=alpha_concatn)
        if alpha_in == 'stts':
            self.alpha = Alpha(input_dim=4, hidden_dim1=alpha_hdim1, hidden_dim2=alpha_hdim2,
                               rnn_type=alpha_rnn, rnn_out=alpha_rnn_out, norm=norm,
                               rnn_layer=alpha_rnn_layer, mlp_layer=alpha_mlp_layer, bidir=alpha_bidir,
                               dropout=alpha_dropout, concat_nframe=alpha_concatn)
        elif self.alpha_in == 'fixed':
            self.alpha = cmp_alpha

        if omega_in == 'state':
            self.omega = Omega(initb=omega_initb)
        elif omega_in == 'fixed':
            self.omega = cmp_omega

        if error_in == 'axw' or error_in == 'a_w':
            self.error = Error(in_dim=1, h_dim=error_hdim, norm=norm, out_norm=error_outnorm, sigmaB=sigmaB,
                               mlp_layer=err_layer)
        elif self.error_in == 'fixed':
            self.error = cmp_err
        elif self.error_in == 'axw+ori':
            if alpha_rnn == 'transformer':
                error_indim = alpha_concatn + 2
            elif alpha_bidir:
                error_indim = 2 + 2 * alpha_hdim1
            else:
                error_indim = 2 + alpha_hdim1

            self.error = Error(in_dim=error_indim, h_dim=error_hdim, norm=norm, out_norm=error_outnorm,
                               sigmaB=sigmaB,
                               mlp_layer=err_layer, dropout=error_dropout)

    def forward(self, x, idx, predict=False):
        # v = x[:, :-1]
        seqs = x.shape[0]
        s = x[:, -1].reshape(seqs, 1)
        self.s = s

        if not predict:
            self.history[idx] = s
            if self.alpha_in == 'state':
                self.a = self.alpha(s)
            # elif self.alpha_in == 'vibr_fcn':
            #     self.a = self.alpha(self.v_fcn)
            # elif self.alpha_in == 'stts':
            #     std = torch.std(v, 1, keepdim=True)
            #     mean = torch.mean(v, 1, keepdim=True)
            #     diffs = v - mean
            #     zscores = diffs / std
            #     skew = torch.mean(zscores ** 3, dim=1, keepdim=True)
            #     kurt = torch.mean(zscores ** 4, dim=1, keepdim=True) - 3.0
            #     stts = torch.cat((s, std, kurt, skew), dim=1)
            #     self.a = self.alpha(stts)
            elif self.alpha_in == 'fixed':
                self.a = torch.tensor(self.alpha[idx]).to(device)
            self.alpha_trained[idx] = self.a

            if self.omega_in == 'state':
                self.W = self.omega(s)
            elif self.omega_in == 'fixed':
                omg = torch.tensor(self.omega[idx], dtype=torch.float32)
                self.W = omg[(omg.shape[0] - s.shape[0]):].reshape(s.shape).to(device)

            if self.error_in == 'axw':
                self.E = self.error(self.a * self.W)
            elif self.error_in == 'axw+ori':
                alpha_rnn_output_for_concat = self.alpha.rnn_output.repeat(self.s.shape[0], 1)
                error_in_ = torch.cat((self.a * self.W, alpha_rnn_output_for_concat, self.s), dim=1)
                self.E = self.error(error_in_)

            if self.error_term:
                self.ret = self.a * self.W + self.E
            else:
                self.ret = self.a * self.W

        else:  ## predict
            self.ret = torch.ones(s.shape).to(device)
            if self.omega_in == 'state':
                self.W = self.omega(s)
            elif self.omega_in == 'fixed':
                omg = torch.tensor(self.omega[idx], dtype=torch.float32)
                self.W = omg[(omg.shape[0] - s.shape[0]):].reshape(s.shape).to(device)
            for i in range(len(s)):
                if self.alpha_window:
                    self.history[idx] = torch.cat((self.history[idx][1:], s[i].unsqueeze(dim=0)), dim=0)
                else:
                    self.history[idx] = torch.cat((self.history[idx], s[i].unsqueeze(dim=0)), dim=0)
                #                 self.an.append(self.alpha(self.history[idx]))
                if self.alpha_in == 'fixed':
                    self.a = torch.tensor(self.alpha[idx]).to(try_device(-1))
                else:
                    self.a = self.alpha(self.history[idx])
                if self.error_in == 'axw':
                    self.E = self.error(self.a * self.W)
                elif self.error_in == 'axw+ori':
                    alpha_rnn_output_for_concat = self.alpha.rnn_output.repeat(self.s.shape[0], 1)
                    error_in_ = torch.cat((self.a * self.W, self.s, alpha_rnn_output_for_concat), dim=1)
                    self.E = self.error(error_in_)
                if self.error_term:
                    self.ret[i] = self.a * self.W[i] + self.E[i]
                else:
                    self.ret[i] = self.a * self.W[i]

        return self.ret


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

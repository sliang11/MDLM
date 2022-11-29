from torch import nn
import torch
import math

def try_device(i=0):
    """<0返回cpu(); 如果存在，则返回gpu(i)，否则返回cpu()"""
    if i < 0: return torch.device('cpu')
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

device = try_device(0)

class Alpha(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, out_dim=1, rnn_layer=1, nhead=1, mlp_layer=2,
                 rnn_type='gru', rnn_out='last', norm=True, bidir=False):
        super(Alpha, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.rnn_out = rnn_out
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim1, rnn_layer, batch_first=True)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim1, rnn_layer, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim1, rnn_layer, bidirectional=bidir, batch_first=True)

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
        if self.rnn_type == 'gru':
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))
        elif self.rnn_type == 'lstm':
            out, (_, _) = self.rnn(x.reshape(1, -1, self.input_dim))
        elif self.rnn_type == 'rnn':
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))
        else:
            out, hn = self.rnn(x.reshape(1, -1, self.input_dim))

        rnn_output = out.mean(dim=1).reshape(1, -1)
        return self.mlp(rnn_output).squeeze(dim=0)


class Omega(nn.Module):
    def __init__(self, in_dim=1, h_dim=0, norm=True, initb=1.):
        super(Omega, self).__init__()

        # 将自定义参数b加入网络进行学习
        self.register_parameter(name='paramb',
                                param=torch.nn.Parameter(initb * torch.ones(1, 1)))
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
            return self.mlp(torch.cat((x, self.paramb.expand((x.shape[0], 1))), dim=1))
        elif self.in_dim > 1:
            return self.out(x.pow(self.paramb))
        return x.pow(self.paramb)


class Error(nn.Module):
    def __init__(self, in_dim, h_dim=16, norm=True, out_norm=False, sigmaB=0.02, mlp_layer=1):
        super(Error, self).__init__()
        self.out_norm = out_norm
        self.sigmaB = sigmaB
        if norm:
            self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.PReLU()
                                     )

            for i in range(mlp_layer):
                self.mlp = nn.Sequential(self.mlp,
                                         nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.PReLU(),
                                         )
            self.mlp = nn.Sequential(self.mlp,
                                     nn.Linear(h_dim, 1)
                                     )
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, h_dim), nn.PReLU(),
                                     nn.Linear(h_dim, 1))

    def forward(self, x):
        if self.out_norm:
            m = (1 / (2 * math.pi) ** 0.5) * self.sigmaB
            return (m * torch.tanh(self.mlp(x))).squeeze(dim=0)
        else:
            return self.mlp(x).squeeze(dim=0)


class Net(nn.Module):
    """
    整体网络
    """

    def __init__(self, cmp_err, cmp_alpha, cmp_omega,
                 alpha_in='vibr_fcn', alpha_rnn='gru', alpha_rnn_out='last', alpha_hdim1=16, alpha_hdim2=32,
                 alpha_rnn_layer=1, alpha_mlp_layer=2, alpha_bidir=False, alpha_nhead=1,
                 omega_in='state', omega_func='exp', omega_hdim=0, omega_initb=1.,
                 error_in='axw', error_hdim=16, error_outnorm=False, err_layer=1, sigmaB=0.02,
                 norm=True, incre=True, error_term=True):
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
        if alpha_in == 'state':
            self.alpha = Alpha(input_dim=1, hidden_dim1=alpha_hdim1, hidden_dim2=alpha_hdim2,
                               rnn_type=alpha_rnn, rnn_out=alpha_rnn_out, norm=norm, nhead=alpha_nhead,
                               rnn_layer=alpha_rnn_layer, mlp_layer=alpha_mlp_layer, bidir=alpha_bidir)
        if alpha_in == 'stts':
            self.alpha = Alpha(input_dim=4, hidden_dim1=alpha_hdim1, hidden_dim2=alpha_hdim2,
                               rnn_type=alpha_rnn, rnn_out=alpha_rnn_out, norm=norm,
                               rnn_layer=alpha_rnn_layer, mlp_layer=alpha_mlp_layer, bidir=alpha_bidir)
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

    def forward(self, x, idx, predict=False):
        v = x[:, :-1]
        seqs = v.shape[0]
        s = x[:, -1].reshape(seqs, 1)
        self.s = s

        if not predict:
            self.history[idx] = s
            if self.alpha_in == 'state':
                self.a = self.alpha(s)
            elif self.alpha_in == 'vibr_fcn':
                self.a = self.alpha(self.v_fcn)
            elif self.alpha_in == 'stts':
                std = torch.std(v, 1, keepdim=True)
                mean = torch.mean(v, 1, keepdim=True)
                diffs = v - mean
                zscores = diffs / std
                skew = torch.mean(zscores ** 3, dim=1, keepdim=True)
                kurt = torch.mean(zscores ** 4, dim=1, keepdim=True) - 3.0
                stts = torch.cat((s, std, kurt, skew), dim=1)
                self.a = self.alpha(stts)
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
                self.history[idx] = torch.cat((self.history[idx], s[i].unsqueeze(dim=0)), dim=0)
                #                 self.an.append(self.alpha(self.history[idx]))
                if self.alpha_in == 'fixed':
                    self.a = torch.tensor(self.alpha[idx]).to(try_device(-1))
                else:
                    self.a = self.alpha(self.history[idx])
                if self.error_in == 'axw':
                    self.E = self.error(self.a * self.W)
                if self.error_term:
                    self.ret[i] = self.a * self.W[i] + self.E[i]
                else:
                    self.ret[i] = self.a * self.W[i]

        return self.ret
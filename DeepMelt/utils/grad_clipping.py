from torch import nn
import torch
import re


def grad_clipping(net, theta=1):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for n, p in net.named_parameters() if p.requires_grad and re.match(r'(.*)rnn(.*)', n)]  #
        # re.match(r'(.*)rnn(.*)', n) / p.grad is not None
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
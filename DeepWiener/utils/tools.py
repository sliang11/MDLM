import os
import torch
import logging
import random
import numpy as np


def try_device(i=0):
    """<0返回cpu(); 如果存在，则返回gpu(i)，否则返回cpu()"""
    if i < 0: return torch.device('cpu')
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def mkdir(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def log_init(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    if log_file is None:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

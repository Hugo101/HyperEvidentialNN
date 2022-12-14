import torch
import numpy as np
import random
import os
import sys


def set_device(gpu_id):
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available() and gpu_id >= 0:
        # cmd_args.device = torch.device('cuda:'+str(gpu_id))
        print('use gpu indexed: %d' % gpu_id)
        device = 'cuda:' + str(gpu_id)
        return device

    else:
        # cmd_args.device = torch.device('cpu')
        print('use cpu')
        device = 'cpu'
        return device


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def create_path(dir):
    if dir is not None:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    print(dir)


# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_logger(log_file, mode):
    if mode == 'out':
        sys.stdout = Logger(log_file, 'out')
    if mode == 'err':
        sys.stderr = Logger(log_file, 'err')


def set_random_seeds(seed=1):
    print('Random Seed is set: {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all， # if you are using multi-GPU.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d
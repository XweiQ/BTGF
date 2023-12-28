import torch
import random
import argparse
import numpy as np


def get_config():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-dataset', type=str, default='ACM')

    # filter
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-a', type=float, default=10)

    # training settings
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-lr', type=float, default=1e-2, help='learning_rate')
    parser.add_argument('-wd', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('-p', type=float, default=0.5, help='probability of dropout')

    args = parser.parse_args()

    return args


def init_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
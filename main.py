import os
import load_data

from graph_filtering import *
from load_data import *
from train import Training
from util import *


if __name__ == '__main__':

    args = get_config()

    # X_raw: ndarray[nnodes, features]
    # A_list: list[adj1(ndarray[nnodes, nnodes]), adj2]
    # Y: ndarray[nnodes, ]
    X_raw, A_list, Y = getattr(load_data, args.dataset)()

    len_X, len_A = len(X_raw), len(A_list)
    if len_X >= 10:
        print("multi-relational dataset")
        X = []
        A = A_list.copy()
        for nn in range(len_A):
            X.append(X_raw)

    elif len_A >= 10:
        print("multi-attribute dataset")
        A = []
        X = X_raw.copy()
        for nn in range(len_X) :
            A.append(A_list)
    else:
        print("Input Error!!\n")

    log_path = './record/' + args.dataset + '/result.txt'    
    if os.path.exists(log_path):
        os.makedirs(log_path)

    init_seed(4396)
    X_ = fgc_multi(A, X, args.device, args.k, args.a, args)
    max_epoch, ac_best, nm_best, f1_best, ari_best = Training(A, X_, Y, args)
    with open(log_path, 'a') as f:
        result = 'k: {:d}, a: {:.4f}, epoch: {:d}, ac_best: {:.4f}, nm_best: {:.4f}, f1_best: {:.4f}, ari_best: {:.4f}'.format(args.k, args.a, max_epoch, ac_best, nm_best, f1_best, ari_best)
        f.write(result + '\n')

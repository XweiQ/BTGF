import numpy as np
import traceback
import matplotlib.pyplot as plt
import torch
import scipy.sparse as sp


def fgc_filter(A, X, device, f, k, a):
    # Convert A and X to PyTorch tensors
    if type(A) is np.ndarray:
        A = torch.from_numpy(A).to(device)
    if type(X) is np.ndarray:
        X = torch.from_numpy(X).to(device)
    # Convert X to float
    X = X.float().to(device)
    A = A.float().to(device)
    
    I_n = torch.eye(A.shape[0]).to(device)
    I_d = torch.eye(X.shape[1]).to(device)
    # Normalize A
    A = A + I_n
    D = torch.sum(A, 1)
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0
    D = torch.diag(D)
    A = D.matmul(A).matmul(D)
	# Get filter G	
    Ls = I_n - A
    G = I_n - 0.5*Ls
    # Set f(A)
    A_ = I_n
    for _ in range(f):
        A_ = G.matmul(A_)
	# Set the order of filter
    G_ = G
    kk = 1
    while(kk <= k):
		#compute
        X_bar = G_.matmul(X)
        XtX_bar = X_bar.matmul(X_bar.T)
        XXt_bar = X_bar.t().matmul(X_bar)
        tmp = torch.inverse(I_d + XXt_bar/a)
        tmp = X_bar.matmul(tmp).matmul((X_bar.t()))
        tmp = I_n/a -tmp/(a*a)
        S = tmp.matmul(a * A_ + XtX_bar)  
        kk += 1
        G_ = G_.matmul(G)
    X_ = S.matmul(X)
    return X_


def fgc_multi(A, X, device, k, a, f):
    X_ = X.copy()
    for idx, x in enumerate(X):
        X_[idx] = fgc_filter(A[idx], x, device, f, k, a)
    return X_

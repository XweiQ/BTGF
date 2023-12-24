import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def ACM(dataname='ACM'):
    """load dataset ACM

    Returns:
        gnd(ndarray): [nnodes,]
    """
    if dataname == "ACM":
        # Load data
        dataset = "./mat/" + 'ACM3025'
        data = sio.loadmat('{}.mat'.format(dataset))
        if (dataset == 'large_cora'):
            X = data['X']
            A = data['G']
            gnd = data['labels']
            gnd = gnd[0, :]
        else:
            X = data['feature']
            A = data['PAP']
            B = data['PLP']
            # C = data['PMP']
            # D = data['PTP']
    if sp.issparse(X):
        X = X.todense()

    A = np.array(A)
    B = np.array(B)
    X = np.array(X)

    Adj = []
    Adj.append(A)
    Adj.append(B)

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X, Adj, gnd


def amazon():
    data = pkl.load(open("data/amazon.pkl", "rb"))
    label = data['label'].argmax(1)

    # dense
    ivi = torch.from_numpy(data["IVI"]).float()
    ibi = torch.from_numpy(data["IBI"]).float()
    ioi = torch.from_numpy(data["IOI"]).float()
    adj = []
    adj.append(ivi)
    adj.append(ibi)
    adj.append(ioi)
        
    features = torch.from_numpy(data['feature']).float()

    return features, adj, label


def DBLP_L():
    data = pkl.load(open("./data/dblp.pkl", "rb"))
    label = data['label'].argmax(1)
 
    pap = np.array(data["PAP"])
    ppp = np.array(data["PPrefP"])

    A = []
    A.append(pap)
    A.append(ppp)
    
    features = np.array(data['feature'])

    return features, A, label


def aminer(ratio=[20, 40, 60], type_num=[6564, 13329, 35890]):
    """load aminer

    Args:
        ratio (list, optional): _description_. Defaults to [20, 40, 60].
        type_num (list, optional): _description_. Defaults to [6564, 13329, 35890].

    Returns:
        label(ndarray): [nnodes, ]
    """
    path = "./data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")

    adj_pap = pap.todense().astype(int)
    adj_prp = prp.todense().astype(int)
    adj_pos = pos.todense().astype(int)
    adj = []
    adj.append(torch.from_numpy(adj_pap))
    adj.append(torch.from_numpy(adj_prp))

    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    feat_p = torch.FloatTensor(preprocess_features(feat_p))
    feat_a = torch.FloatTensor(preprocess_features(feat_a))
    feat_r = torch.FloatTensor(preprocess_features(feat_r))

    return feat_p, adj, label


if __name__ == '__main__':
    aminer()

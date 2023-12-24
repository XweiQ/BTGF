import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from itertools import combinations
from math import comb
from kmeans_pytorch import kmeans

def target_distribution(q) :
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def soft_assignment(device, embeddings, n_clusters, alpha=1):

    cluster_ids_x, cluster_centers = kmeans(X=embeddings, num_clusters=n_clusters, distance='euclidean', device=device)
    cluster_layer = Parameter(torch.Tensor(n_clusters, 64))

    cluster_layer.data = torch.tensor(cluster_centers).to(device)
    q = 1.0 / (1.0 + torch.sum(torch.pow(embeddings.unsqueeze(1) - cluster_layer, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return q


def KL_clustering(device, z_list, h, num_classes):

    q = soft_assignment(device, h, num_classes, alpha=1)
    p = target_distribution(q)

    clu_loss = torch.tensor(0).to(device).float()
    for z in z_list:
        qz = soft_assignment(device, z, num_classes, alpha=1)
        pz = target_distribution(qz)
        clu_loss +=  F.mse_loss(qz.float(), pz.float())
    clu_loss += F.mse_loss(q.float(), p.float())

    return clu_loss, p , q


def sce_loss(x, y, alpha):

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sce_loss_multi(device, x_list, x_bar_list, alpha = 2):

    rec_loss = torch.tensor(0).to(device).float()
    for x, x_bar in zip(x_list, x_bar_list):
        rec_loss += sce_loss(x.float(), x_bar.float(), alpha)
    
    return rec_loss


def off_diagonal(x):

    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def BarlowTwins_multi(device, z_list, Batch_size, n_z=10):

    bn = nn.BatchNorm1d(n_z, affine=False).to(device)
    Bl = torch.tensor([0]).to(device).float()
    num_view = z_list.size(0)
    V = range(num_view)

    if num_view != 0:
        for combn in combinations(V, 2):

            c = bn(z_list[combn[0]]).t().mm(bn(z_list[combn[1]]))
            c.div_(Batch_size)

            lambd = 0.0051
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            Bl += on_diag + lambd * off_diag

        Bl = Bl / comb(num_view, 2)
        
    return Bl

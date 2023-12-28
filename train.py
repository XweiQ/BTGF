import numpy as np
import torch
import warnings
import traceback

from torch.optim import Adam
from model import *
from losses import *
from metrics import clustering_metrics
warnings.filterwarnings('ignore')


def Training(A, X_, Y, args) :

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if not torch.cuda.is_available() :
        print("use GPU machine for not demaging your PC")

    for idxx, XX in enumerate(X_):
        if type(XX) is np.ndarray:
            X1 = torch.from_numpy(XX)
        else:
            X1 = XX
        if idxx == 0:
            X = torch.unsqueeze(X1, 0)
            X = X.to(torch.float32)
        else:
            X1 = torch.unsqueeze(X1, 0)
            X1 = X1.to(torch.float32)
            X = torch.cat((X, X1), 0)

    Batch_size = X[0].size(0)
    nfeature = X[0].size(1)
    num_classes = len(np.unique(Y))

    Net = model(args, Batch_size, nfeature).to(device)
    Optimizer = Adam(Net.parameters(), lr=args.lr, weight_decay=args.wd)

    ac_list = []
    nm_list = []
    f1_list = []
    ari_list = []
    loss_list = []
    for epoch in range(args.epoch) :

        X = X.to(device)

        z_list, x_bar_list, h = Net(A, X)
        
        Bl = BarlowTwins_multi(args.device, z_list, Batch_size)
        clu_loss, p, q = KL_clustering(args.device, z_list, h, num_classes)
        rec_loss = sce_loss_multi(args.device, X, x_bar_list)
        lss = 1e0 * rec_loss + 1e0 * Bl + 1e0 * clu_loss
        
        Optimizer.zero_grad()

        try:
            with torch.autograd.detect_anomaly():
                lss.backward()
        except:
            traceback.print_stack()

        Optimizer.step()
        rloss, closs, bloss = rec_loss.item(), clu_loss.item(), Bl.item()

        with torch.no_grad():
            Y_bar = torch.argmax(q.cpu(), 1)
            Y_pd = Y_bar.numpy()
            
            cm = clustering_metrics(Y, Y_pd)
            ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()
            ac_list.append(ac)
            nm_list.append(nm)
            f1_list.append(f1)  
            ari_list.append(ari)
            
            nxia = np.argmax(ac_list)
            
            ac_best = ac_list[nxia]    
            nm_best = nm_list[nxia]                    
            f1_best = f1_list[nxia]
            ari_best = ari_list[nxia]
            
            if((epoch+1)%10 == 0):
                print("Epoch=======>{} ACC=======>{} Re_loss======>{} Bl_loss====>{} CluLoss=====>{}".format(epoch+1, ac, rloss, bloss, closs))

    print("acc_max, epoch: ", ac_best, nxia+1)

    return nxia+1, ac_best, nm_best, f1_best, ari_best



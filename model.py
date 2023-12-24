import torch
import torch.nn as nn
from torch.nn import Linear


class mlp(nn.Module):
    def __init__(self, n_input, n_z):
        super(mlp, self).__init__()
        self.Encder1 = nn.Sequential \
                (
                Linear(n_input, n_z))
        self.Decoder1 = nn.Sequential \
                (
                Linear(n_z, n_input))
                
    def forward(self, x):
        z1 = self.Encder1(x)

        x_bar_1 = self.Decoder1(z1)

        return z1, x_bar_1


class MvRCN(nn.Module):

    def __init__(self, args, nodes, in_dims_1, size_model="mlp"):
        super(MvRCN, self).__init__()
        if size_model == 'mlp':
            self.AE = mlp(args.p, in_dims_1, n_z=10)

    def forward(self, A, X_raw_list):
        for idx_x, x in enumerate(X_raw_list):
            z, x_bar = self.AE(x.float())
            if idx_x == 0:
                h = z.clone()
                z_list = torch.unsqueeze(z, 0)
                x_bar_list = torch.unsqueeze(x_bar, 0)
            else:
                h = torch.cat((h, z), 1)
                z = torch.unsqueeze(z, 0)
                z_list = torch.cat((z_list, z), 0)
                x_bar = torch.unsqueeze(x_bar, 0)
                x_bar_list = torch.cat((x_bar_list, x_bar), 0)

        return z_list, x_bar_list, h


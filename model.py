import torch
import torch.nn as nn
import torch.nn.functional as F


class my_model(nn.Module):
    def __init__(self, dims, act="ident"):
        super(my_model, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[0], dims[1])
        self.bn = nn.BatchNorm1d(dims[1])
        self.reset_parameters()

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        out1 = self.activate(self.lin1(x))
        out2 = self.activate(self.lin2(x))

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2


class my_Q_net(nn.Module):
    def __init__(self, dims):
        super(my_Q_net, self).__init__()
        self.lin1 = nn.Linear(dims[0], dims[1])
        self.lin_cluster = nn.Linear(dims[0], dims[1])
        self.lin2 = nn.Linear(dims[1], dims[2])
        self.reset_parameters()
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, cluster):
        x = self.act(F.normalize(self.lin1(x), dim=1, p=2))
        cluster = self.act(F.normalize(self.lin_cluster(cluster), dim=1, p=2))
        x = F.softmax(self.lin2(torch.cat([x, cluster], dim=0)), dim=-1)
        return x

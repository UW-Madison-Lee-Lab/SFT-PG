import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
        torch.nn.init.xavier_normal_(self.lin.weight)


    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 2)

    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

class ResConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ResConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 64, n_steps)
        self.lin2 = ConditionalLinear(64, 64, n_steps)
        self.lin3 = ConditionalLinear(64, 64, n_steps)
        self.lin4 = nn.Linear(64, 2)

    def forward(self, x, y):
        res = x
        x = F.relu(self.lin1(x, y))
        x = F.relu(self.lin2(x, y))
        x = F.relu(self.lin3(x, y))
        return self.lin4(x)+res


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.lin1 = nn.Linear(2, 256)
        self.lin2 = nn.Linear(256, 256)
        # self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, 2)
        # torch.nn.init.xavier_normal_(self.lin1.weight)
        # torch.nn.init.xavier_normal_(self.lin2.weight)
        # # # torch.nn.init.xavier_normal_(self.lin3.weight)
        # torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, x, t):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        return self.scale * self.lin4(x)

class classifier_new(nn.Module):
    def __init__(self, n_steps):
        super(classifier_new, self).__init__()
        # self.scale = torch.nn.Parameter(torch.ones(1))
        self.lin1 = ConditionalLinear(2, 256, n_steps)
        self.lin2 = ConditionalLinear(256, 256, n_steps)
        # self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, 2)
    
    def forward(self, x, t):
        x = F.relu(self.lin1(x, t))
        x = F.relu(self.lin2(x, t))
        # x = F.relu(self.lin3(x))
        return self.lin4(x)

class Classifier9(nn.Module):
    def __init__(self):
        super(Classifier9, self).__init__()
        self.lin1 = nn.Linear(2, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)

class fstar(nn.Module):
    def __init__(self):
        super(fstar, self).__init__()
        self.lin1 = spectral_norm(nn.Linear(2, 256))
        self.lin2 =  spectral_norm(nn.Linear(256, 256))
        # self.lin3 = spectral_norm(nn.Linear(256, 256))
        self.lin4 =  spectral_norm(nn.Linear(256, 1))
        # torch.nn.init.xavier_normal_(self.lin1.weight)
        # torch.nn.init.xavier_normal_(self.lin2.weight)
        # torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        return self.lin4(x)

class fstar_tanh(nn.Module):
    def __init__(self):
        super(fstar_tanh, self).__init__()
        self.lin1 = nn.Linear(2, 256)
        self.lin2 = nn.Linear(256, 256)
        # self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 1)
        # self.bn1 = torch.nn.BatchNorm1d(256)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        # self.bn3 = torch.nn.BatchNorm1d(1)

        # torch.nn.init.xavier_normal_(self.lin1.weight)
        # torch.nn.init.xavier_normal_(self.lin2.weight)
        # torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, x):
        x = F.relu((self.lin1(x)))
        x = F.relu((self.lin2(x)))
        # x = F.relu(self.lin3(x))
        return self.lin4(x)

class value(nn.Module):
    def __init__(self, n_steps):
        super(value, self).__init__()
        self.lin1 = ConditionalLinear(2, 256, n_steps)
        self.lin2 = ConditionalLinear(256, 256, n_steps)
        # self.lin3 = ConditionalLinear(256, 256, n_steps)
        self.lin4 = ConditionalLinear(256, 1, n_steps)

    def forward(self, x, t):
        x = F.relu(self.lin1(x, t))
        x = F.relu(self.lin2(x, t))
        # x = F.relu(self.lin3(x, t))
        return self.lin4(x, t)

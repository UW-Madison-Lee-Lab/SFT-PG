import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# import math
# img_shape = (3, 32, 32)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

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

class Value(nn.Module):
    def __init__(self, num_steps, img_shape):
        super(Value, self).__init__()
        self.lin1 = ConditionalLinear(int(np.prod(img_shape)), 1024, num_steps)
        self.lin2 = ConditionalLinear(1024, 1024, num_steps)
        self.lin3 = ConditionalLinear(1024, 256, num_steps)
        self.lin4 = nn.Linear(256, 1)
        torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, img, t):
        x = img.view(img.shape[0], -1)
        x = F.relu(self.lin1(x, t))
        x = F.relu(self.lin2(x, t))
        x = F.relu(self.lin3(x, t))
        return self.lin4(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels=3, features = 64):
        super().__init__()

        self.layer1 = nn.Conv2d(in_channels=channels, out_channels=features*2, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # [batch_size, 256, 16, 16]
        self.layer2 = nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=4, stride=2, padding=1)
        # [512, 8, 8]
        self.layer3 = nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=4, stride=2, padding=1)
        # [1024, 4, 4]
        self.layer4 = nn.Conv2d(in_channels=features*8, out_channels=1, kernel_size=4, stride=1, padding=0)

        # self.layer4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0)

        # self.linear = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        # torch.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x) 
        x = self.relu(x)
        x = self.layer3(x) 
        x = self.relu(x)
        x = self.layer4(x) 
        x = x.mean(dim=3).mean(dim=2)

        return x


class ValueCelebA(nn.Module):
    def __init__(self, num_steps, img_shape):
        super().__init__()
        self.lin1 = ConditionalLinear(int(np.prod(img_shape)), 2048, num_steps)
        self.lin2 = ConditionalLinear(2048, 1024, num_steps)
        self.lin3 = ConditionalLinear(1024, 256, num_steps)
        self.lin4 = nn.Linear(256, 1)
        torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, img, t):
        x = img.view(img.shape[0], -1)
        x = F.relu(self.lin1(x, t))
        x = F.relu(self.lin2(x, t))
        x = F.relu(self.lin3(x, t))
        return self.lin4(x)


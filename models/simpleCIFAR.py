import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3*32*32, 92)
        self.linear2 = nn.Linear(92, 46)
        self.linear3 = nn.Linear(46, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out
import torch
import torch.nn as nn


class simpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, 28 * 28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a
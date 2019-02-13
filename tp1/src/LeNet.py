#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(30, 60, kernel_size=(5,5))
        self.fc1 = nn.Linear(60 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu( self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu( self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # Equivalent to 1x1 convolution.
        x = x.view(-1, 60*4*4)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

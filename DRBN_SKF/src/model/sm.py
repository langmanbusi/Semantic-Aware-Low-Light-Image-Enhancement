# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

def make_model(args, parent=False):
    sm_module = SM(args)
    sm_module.apply(init_weights)
    return sm_module

class SM_uint(nn.Module):
    def __init__(self):
        super(SM_uint, self).__init__()

        ch = 8
        self.E = nn.Conv2d(1, ch, 3, padding=1, dilation=1, stride=1)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1, stride=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2, stride=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=3, dilation=3, stride=1)
        self.G = nn.Conv2d(ch, 1, 3, padding=1, dilation=1, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.E(x))
        out = self.relu(self.conv1(out)) + out
        out = self.relu(self.conv2(out)) + out
        out = self.relu(self.conv3(out)) + out
        out = self.relu(self.G(out))
        return out

class SM(nn.Module):
    def __init__(self, args):
        super(SM, self).__init__()

        self.u1 = SM_uint()
        self.u2 = SM_uint()
        self.u3 = SM_uint()
        self.u4 = SM_uint()
        self.u5 = SM_uint()
        self.u6 = SM_uint()
        self.u7 = SM_uint()

    def forward(self, x):
        x1 = self.u1(x.detach())
        x2 = self.u2(x1.detach())
        x3 = self.u3(x2.detach())
        x4 = self.u4(x3.detach())
        x5 = self.u5(x4.detach())
        x6 = self.u6(x5.detach())
        x7 = self.u7(x6.detach())

        return x1, x2, x3, x4, x5, x6, x7

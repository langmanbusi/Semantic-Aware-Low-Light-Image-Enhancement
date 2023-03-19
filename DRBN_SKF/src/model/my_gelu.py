"""
Name    : my_gelu.py
Author  : xxxxxx
Time    : 2022/9/13 9:40
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))




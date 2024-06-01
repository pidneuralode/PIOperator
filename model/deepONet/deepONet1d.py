"""
@author: lulu
This file is the version of Pytorch used in 1d deepONet(https://github.com/lululxvi/deepxde/issues)
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.deepONet.FNN import FNN
from utils.constants import ACTIVATION


################################################################
#  DeepONet for 1d data (sequence data)
################################################################
class DeepONet(nn.Module):
    def __init__(self,
                 branch_layers,
                 trunk_layers,
                 activation_type
                 ):
        super().__init__()
        self.branch = FNN(branch_layers, activation_type)
        self.trunk = FNN(trunk_layers, activation_type)
        self.b = Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        here we use Dot product to combine the branch network and the trunk network
        :param x: (x_func, x_loc)
        :return:
        """
        x_func = x[0]
        x_loc = x[0]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.trunk(x_loc)
        # Dot product
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        # Add bias
        x += self.b

        return x

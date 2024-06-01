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
#  DeepONet for Custom data
################################################################
class CustomDeepONet(nn.Module):
    def __init__(self,
                 branch_layers,
                 trunk_layers,
                 activation_type
                 ):
        super().__init__()
        if callable(branch_layers[1]):
            # User-defined network
            self.branch = branch_layers[1]
        else:
            # Fully connected network
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
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b

        return x


################################################################
#  A simple implementation of CustomDeepONet for processing darcy
################################################################
class SimpleDarcyDeepONet(nn.Module):
    def __init__(self,
                 num_points,
                 branch_layers=None,
                 trunk_layers=None,
                 activation_type="relu"
                 ):
        super().__init__()

        if trunk_layers is None:
            trunk_layers = [2, 128, 128, 128, 128]
        """
        here we create a simple version for (a(x,y), x, y) darcy data, in deeponet, we usually stack the input
        thus x_func -> (batchsize, num_points) x_loc -> (num_points, 2)
        """
        if activation_type not in ACTIVATION.keys():
            raise ValueError("activation_type is not in ACTIVATION")
        self.activation = ACTIVATION[activation_type]

        # custom branch layers
        self.branch = None
        if branch_layers is None:
            self.branch = nn.Sequential(
                nn.Flatten(start_dim=1),  # Flatten the input
                nn.Unflatten(1, (1, 29, 29)),  # Reshape to (batch_size, 1, 29, 29)
                nn.Conv2d(1, 64, kernel_size=5, stride=2),
                self.activation,
                nn.Conv2d(64, 128, kernel_size=5, stride=2),
                self.activation,
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 128),  # Assuming the input size `m` leads to 6x6 feature maps after conv layers
                self.activation,
                nn.Linear(128, 128)
            )
        else:
            self.branch = branch_layers

        if trunk_layers is None:
            trunk_layers = [2, 128, 128, 128, 128]

        self.model = CustomDeepONet([num_points, self.branch], trunk_layers, activation_type)

    def forward(self, x):
        return self.model(x)

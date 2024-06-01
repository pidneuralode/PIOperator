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
#  POD-DeepONet for Custom data
################################################################
class CustomPODDeepONet(nn.Module):
    def __init__(self,
                 # pod_basis,
                 branch_layers,
                 trunk_layers,
                 activation_type
                 ):
        super().__init__()

        # self.pod_basis = torch.tensor(pod_basis, dtype=torch.float32)

        if callable(branch_layers[1]):
            # User-defined network
            self.branch = branch_layers[1]
        else:
            # Fully connected network
            self.branch = FNN(branch_layers, activation_type)
        self.trunk = None
        if trunk_layers is not None:
            self.trunk = FNN(trunk_layers, activation_type)
            self.b = Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        here we use Dot product to combine the branch network and the trunk network
        :param x: (x_func, x_loc, pod_basis)
        :return:
        """
        x_func = x[0]
        x_loc = x[1]
        pod_basis = x[2]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self.trunk is None:
            # POD only
            x = torch.einsum('bi,ni->bn', x_func, pod_basis)
        else:
            x_loc = self.activation(self.trunk(x_loc))
            x = torch.einsum('bi,ni->bn', x_func, torch.concat((pod_basis, x_loc), 1))
            x += self.b

        return x


################################################################
#  A simple implementation of CustomPODDeepONet for processing darcy
################################################################
class SimpleDarcyDeepONet(nn.Module):
    def __init__(self,
                 num_points,
                 modes,
                 pod_basis,
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

        self.modes = modes

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
                nn.Linear(128, modes)
            )
        else:
            self.branch = branch_layers

        if trunk_layers is None:
            trunk_layers = [2, 128, 128, 128, 128]

        self.model = CustomPODDeepONet(pod_basis, [num_points, self.branch], None, activation_type)

    def forward(self, x):
        return self.model(x)

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
#  POD-DeepONet for 1d data (sequence data)
################################################################
class PODDeepONet(nn.Module):
    def __init__(self,
                 # pod_basis,
                 branch_layers,
                 activation_type,
                 trunk_layers=None):
        super().__init__()

        if activation_type not in ACTIVATION.keys():
            raise ValueError("activation_type is not in ACTIVATION")
        self.activation = ACTIVATION[activation_type]

        # self.pod_basis = torch.as_tensor(pod_basis, dtype=np.float32)
        self.branch = FNN(branch_layers, activation_type)
        self.trunk = None
        if trunk_layers is not None:
            self.trunk = FNN(trunk_layers, activation_type)
            self.b = Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
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

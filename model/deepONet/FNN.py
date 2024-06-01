import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.constants import ACTIVATION


class FNN(nn.Module):
    def __init__(self,
                 layer_sizes,
                 activation_type
                 ):
        """
        flexible constructor for full connected network with different layer sizes
        :param layer_sizes: different layer sizes, such as [2, 3, 4, 5]
        :param activation_type: type of activation, you can see it in ACTIVATION
        """
        super().__init__()

        if activation_type not in ACTIVATION.keys():
            raise ValueError("activation_type is not in ACTIVATION")
        self.activation = ACTIVATION[activation_type]

        self.linear_layers = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linear_layers.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )

    def forward(self, x: torch.Tensor):
        for layer in self.linear_layers:
            x = layer(x)
            x = self.activation(x)

        return x

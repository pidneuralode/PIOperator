"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################
#  2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes1,
                 modes2
                 ):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1. / (in_channels * out_channels))
        self.weights1 = Parameter(self.scale * torch.rand(
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            dtype=torch.cfloat
        ))
        self.weights2 = Parameter(self.scale * torch.rand(
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            dtype=torch.cfloat
        ))

    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum('bixy,ioxy->boxy', a, b)

    def forward(self, x: torch.Tensor):
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        batchsize = x.shape[0]
        # rfft will process the last two dimensions in default configuration
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize,
                             self.out_channels,
                             x.size(-2),
                             x.size(-1) // 2 + 1,
                             device=x.device,
                             dtype=torch.cfloat
                             )
        # here we get the forward and backward of frequenciy domains in x space
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                                                                    self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                                                                     self.weights2)

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


################################################################
#  FNO for 2d data, we give the simple useful version here
################################################################
class SimpleFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y), you can make input_features(3) as input if you want
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # output layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]

        # x->(batchsize, size_x, size_y, 3)
        x = self.fc0(x)
        # x->(batchsize, 3, size_x, size_y)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.relu(x1 + x2)

        # output transformations
        x = x.permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class DataNormalizer:
    def __init__(self, dataloader):
        """
        Initializes the DataNormalizer with the given DataLoader.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.
        """
        self.mean, self.std = self.compute_mean_std(dataloader)
        self.normalize_transform = transforms.Normalize(self.mean, self.std)
        self.denormalize_transform = transforms.Normalize(-self.mean / self.std, 1 / self.std)

    def compute_mean_std(self, dataloader):
        """
        Computes the mean and standard deviation of the dataset.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.

        Returns:
            tuple: Mean and standard deviation of the dataset.
        """
        mean = 0.0
        std = 0.0
        total_samples = 0

        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

        return mean, std

    def normalize(self, tensor):
        """
        Normalizes the input tensor.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        return self.normalize_transform(tensor)

    def denormalize(self, tensor):
        """
        Denormalizes the input tensor.

        Args:
            tensor (Tensor): The normalized tensor.

        Returns:
            Tensor: The denormalized tensor.
        """
        return self.denormalize_transform(tensor)


class UnitGaussianNormalizer(nn.Module):
    def __init__(self, mean, std, eps=1e-05):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.eps = torch.tensor(eps)

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = x * (self.std + self.eps) + self.mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)


class MinMaxNormalizer(nn.Module):
    def __init__(self, min, max, eps=1e-05):
        super().__init__()
        self.max = torch.tensor(max)
        self.min = torch.tensor(min)
        self.eps = torch.tensor(eps)

    def encode(self, x):
        x = (x - self.min.to(x)) / (self.max.to(x) - self.min.to(x) + self.eps)
        return x

    def decode(self, x):
        x = x * (self.max.to(x) - self.min.to(x) + self.eps) + self.min.to(x)
        return x

    def to(self, device):
        self.max = self.max.to(device)
        self.min = self.min.to(device)
        self.eps = self.eps.to(device)

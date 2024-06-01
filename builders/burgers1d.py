import torch
import os
import numpy as np

from scipy import io
from torch.utils.data import DataLoader, Dataset
from utils.constants import MODEL_TYPE


# POD operations
def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    return y_mean, v


class Burgers1dBuilder:
    """Load the Burgers1d dataset.
    """

    def __init__(self, n_train, n_test, resolution, path: str, model_type: str, **kwargs):
        """
        init the burgers-1d dataset
        :param n_train: number of training
        :param n_test: number of testing
        :param resolution: the number of selected points in space domain, such as 128 in [0,1]
        :param path: file path
        :param model_type: MODEL_TYPE
        :param kwargs: dataset configuration
        """

        self.kwargs = kwargs

        # here data is of the shape (number of samples = 2048, grid size = 2^13)
        self.grid_size = 2 ** 13

        # help path expand to absolute style in different platforms
        data_path = os.path.expanduser(path)
        data_mat = io.loadmat(data_path)

        if model_type not in MODEL_TYPE:
            raise ValueError('the type of selected model is not correct')

        x_train, y_train, x_test, y_test = None, None, None, None
        if model_type == 'FNO':
            x_train, y_train, x_test, y_test = self.process_data_FNO(data_mat, n_train, n_test, resolution)
        elif model_type == 'DeepONet':
            x_train, y_train, x_test, y_test = self.process_data_DeepONet(data_mat, n_train, n_test, resolution)
        elif model_type == 'PODDeepONet':
            x_train, y_train, x_test, y_test = self.process_data_PODDeepONet(data_mat, n_train, n_test, resolution)

        self.train_dataset = Burgers1dDataset(x_train, y_train)
        self.test_dataset = Burgers1dDataset(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            shuffle=True,
                            **self.kwargs)
        return loader

    # def val_dataloader(self) -> DataLoader:
    #     loader = DataLoader(self.valid_dataset,
    #                         shuffle=False,
    #                         **self.kwargs)
    #     return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            **self.kwargs)
        return loader

    def process_data_FNO(self, data, n_train, n_test, resolution):
        inter_select = self.grid_size // resolution
        x_data = data['a'][:, ::inter_select].astype(np.float32)
        y_data = data['u'][:, ::inter_select].astype(np.float32)

        # select data series
        x_train = x_data[:n_train, :]
        y_train = y_data[:n_train, :]
        x_test = x_data[-n_test:, :]
        y_test = y_data[-n_test:, :]

        # cat the locations information
        grid_all = np.linspace(0, 1, 2 ** 13).reshape(2 ** 13, 1).astype(np.float64)
        grid = grid_all[::inter_select, :]
        grid = torch.tensor(grid, dtype=torch.float)
        x_train = torch.cat([x_train.reshape(n_train, resolution, 1), grid.repeat(n_train, 1, 1)],
                            dim=2)
        x_test = torch.cat([x_test.reshape(n_test, resolution, 1), grid.repeat(n_train, 1, 1)],
                           dim=2)

        return x_train, y_train, x_test, y_test

    def process_data_DeepONet(self, data, n_train, n_test, resolution):
        inter_select = self.grid_size // resolution
        x_data = data['a'][:, ::inter_select].astype(np.float32)
        y_data = data['u'][:, ::inter_select].astype(np.float32)

        # select data series
        x_branch_train = x_data[:n_train, :]
        y_train = y_data[:n_train, :]
        x_branch_test = x_data[-n_test:, :]
        y_test = y_data[-n_test:, :]

        # gird x as trunk
        grid = np.linspace(0, 1, self.grid_size)[::inter_select, None]

        # stack branch input and trunk input
        x_train = (x_branch_train, grid, None)
        x_test = (x_branch_test, grid, None)

        return x_train, y_train, x_test, y_test

    def process_data_PODDeepONet(self, data, n_train, n_test, resolution):
        x_train, y_train, x_test, y_test = self.process_data_DeepONet(data, n_train, n_test, resolution)
        y_mean, v = pod(y_test)

        # stack branch input and trunk input
        modes = 32
        x_train = (x_train[0], x_train[1], v[:, :modes])
        x_test = (x_test[0], x_test[1], v[:, :modes])

        return x_train, y_train, x_test, y_test


class Burgers1dDataset(Dataset):
    def __init__(self, x_data, y_data):
        # transform the numpy data to torch data
        self.x = torch.Tensor(x_data)
        self.y = torch.Tensor(y_data)

    def __len__(self, item):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

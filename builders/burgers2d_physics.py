import random
from typing import Tuple, Any

import torch
import os
import numpy as np

from scipy import io
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from utils.constants import MODEL_TYPE


class Burgers2dBuilder:
    """Load the Burgers2d_physics dataset.
    """

    def __init__(self, n_train, n_test, path: str, device, **kwargs):
        self.kwargs = kwargs

        # here data is of the shape (number of samples = 2048, grid size = 2^13)
        self.grid_size = 101

        # help path expand to absolute style in different platforms
        data_path = os.path.expanduser(path)
        data_mat = io.loadmat(data_path)
        # (samples, t, x)
        usol = np.array(data_mat['output'])

        # some fixed parameters for the number of training examples
        # number of locations for evulating the initial condition
        self.P_ics_train = 101
        # number of locations for evulating the boundary condition
        self.P_bcs_train = 100
        # number of locations for evulating the PDE residual
        self.P_res_train = 2500
        # resolution of uniform grid for the test data
        self.P_test = 101

        # train and test numbers
        self.n_train = n_train
        self.n_test = n_test

        # input samples shape(num_samples, t, x)
        u0_train = usol[:n_train, 0, :]

        # generate training data with pde equation
        # Generate training data for inital condition
        u_ics_train, y_ics_train, s_ics_train = self.generate_training_data(self.generate_one_ics_training_data,
                                                                            u0_train,
                                                                            self.P_ics_train
                                                                            )
        # Generate training data for boundary condition
        u_bcs_train, y_bcs_train, s_bcs_train = self.generate_training_data(self.generate_one_bcs_training_data,
                                                                            u0_train,
                                                                            self.P_bcs_train
                                                                            )
        # Generate training data for PDE residual
        u_res_train, y_res_train, s_res_train = self.generate_training_data(self.generate_one_res_training_data,
                                                                            u0_train,
                                                                            self.P_res_train)

        # create the training dataset
        self.ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, device)
        self.bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, device)
        self.res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, device)

        # pre-process the test data
        in_test = u0_train[-n_test:, :]
        out_test = usol[-n_test:, :]
        u_test, y_test, s_test = self.generate_test_data(in_test, out_test)

        # create the test dataset
        self.test_dataset = DataGenerator(u_test, y_test, s_test, device)

    def generate_training_data(self, generate_function, u0_train, P_train):
        u_train_list = []
        y_train_list = []
        s_train_list = []

        # 生成训练数据
        for i in range(self.n_train):
            u_train_element, y_train_element, s_train_element = generate_function(u0_train[i])
            u_train_list.append(u_train_element)
            y_train_list.append(y_train_element)
            s_train_list.append(s_train_element)

        # 将列表转换为 NumPy 数组
        u_train = np.stack(u_train_list).reshape((self.n_train * P_train, -1))
        y_train = np.stack(y_train_list).reshape((self.n_train * P_train, -1))
        s_train = np.stack(s_train_list).reshape((self.n_train * P_train, -1))

        return u_train, y_train, s_train

    def generate_one_ics_training_data(self, u0_train):
        """
        generate ics training data corresponding to one input sample

        :return:
        """
        t_0 = np.zeros((self.P_ics_train, 1))
        x_0 = np.linspace(0, 1, self.P_ics_train)[:, None]

        y = np.hstack([t_0, x_0])
        u = np.tile(u0_train, (self.P_ics_train, 1))
        s = u0_train

        return u, y, s

    def generate_one_bcs_training_data(self, u0_train):
        """
        generate bcs training data corresponding to one input sample
        :param u0_train:
        :return:
        """
        t_bc = np.random.rand(self.P_bcs_train).reshape((self.P_bcs_train, 1))
        # u(0, t)
        x_bc1 = np.zeros((self.P_bcs_train, 1))
        # u(1, t)
        x_bc2 = np.zeros((self.P_bcs_train, 1))

        y1 = np.hstack([t_bc, x_bc1])
        y2 = np.hstack([t_bc, x_bc2])

        u = np.tile(u0_train, (self.P_bcs_train, 1))
        y = np.hstack([y1, y2])
        s = np.zeros((self.P_bcs_train, 1))

        return u, y, s

    def generate_one_res_training_data(self, u0_train):
        """
        generate residual training data corresponding to one input sample
        :param u0_train:
        :return:
        """
        t_res = np.random.rand(self.P_res_train).reshape((self.P_res_train, 1))
        x_res = np.random.rand(self.P_res_train).reshape((self.P_res_train, 1))

        u = np.tile(u0_train, (self.P_res_train, 1))
        y = np.hstack([t_res, x_res])
        s = np.zeros((self.P_res_train, 1))

        return u, y, s

    def train_dataloader(self) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        ics_loader = DataLoader(self.ics_dataset,
                                shuffle=True,
                                **self.kwargs)
        bcs_loader = DataLoader(self.bcs_dataset,
                                shuffle=True,
                                **self.kwargs)
        res_loader = DataLoader(self.res_dataset,
                                shuffle=True,
                                **self.kwargs)
        return ics_loader, bcs_loader, res_loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            **self.kwargs)
        return loader

    def generate_test_data(self, in_test: np.ndarray, out_test: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate test data.

        Parameters:
        - in_test: Input test data, shape (n_test, ...)
        - out_test: Output test data, shape (n_test, 2, ...)

        Returns:
        - u_test: Reshaped input test data, shape (n_test, ...)
        - y_test: Generated test labels, shape (n_test, t_scan * x_scan, 2)
        - s_test: Generated test states, shape (n_test, t_scan * x_scan, 1)
        """

        # Reshape input test data
        u_test = in_test.reshape((self.n_test, -1))

        # Extract sizes for t_scan and x_scan
        t_scan, x_scan = out_test[0].shape[0], out_test[1].shape[1]

        # Generate linear interpolations for time and space coordinates
        t_inter = np.linspace(0, 1, t_scan)
        x_inter = np.linspace(0, 1, x_scan)

        # Generate grid coordinates using meshgrid for vectorized operations
        t_grid, x_grid = np.meshgrid(t_inter, x_inter, indexing='ij')

        # Flatten the coordinate grids
        y_coords = np.stack([t_grid.flatten(), x_grid.flatten()], axis=-1)

        # Initialize output arrays
        y_test = np.broadcast_to(y_coords[None, :, :], (self.n_test, y_coords.shape[0], y_coords.shape[1]))
        s_test = np.zeros((self.n_test, t_scan * x_scan, 1))

        # Reshape the output test data
        out_test_flat = out_test.reshape((self.n_test, t_scan * x_scan))

        # Assign the flattened output test data to s_test
        s_test[:, :, 0] = out_test_flat

        y_test = y_test.reshape((self.n_test * x_scan * t_scan, -1))
        s_test = s_test.reshape((self.n_test * x_scan * t_scan, -1))
        u_test = np.repeat(u_test, t_scan * x_scan, axis=0)

        return u_test, y_test, s_test


# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, device,
                 batch_size=64):
        """Initialization"""
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.s = torch.tensor(s, dtype=torch.float32).to(device)

        self.N = u.shape[0]
        self.batch_size = batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        s = self.s[index]
        y = self.y[index]
        u = self.u[index]
        # construct batch
        inputs = (u, y)
        outputs = s

        return inputs, outputs

    def __len__(self):
        return self.N


if __name__ == '__main__':
    test_builder = Burgers1dBuilder(100,
                                    100,
                                    "E:\\dfno\\PIOperator\\data\\burgers\\2d\\burgers_2d.mat"
                                    )

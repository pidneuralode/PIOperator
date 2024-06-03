import random

import torch
import os
import numpy as np

from scipy import io
from torch.utils.data import DataLoader, Dataset
from utils.constants import MODEL_TYPE


class Burgers1dBuilder:
    """Load the Burgers2d_physics dataset.
    """

    def __init__(self, n_train, n_test, path: str, **kwargs):
        self.kwargs = kwargs

        # here data is of the shape (number of samples = 2048, grid size = 2^13)
        self.grid_size = 101

        # help path expand to absolute style in different platforms
        data_path = os.path.expanduser(path)
        data_mat = io.loadmat(data_path)
        # (samples, t, x)
        u_0_inputs = np.array(data_mat['input'])
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

        # input samples shape(num_samples, t, x)
        u0_train = usol[:n_train, 0, :]

        # generate training data with pde equation
        # Generate training data for inital condition
        u_ics_train, y_ics_train, s_ics_train = self.generate_one_ics_training_data(u0_train)
        # Generate training data for boundary condition
        u_bcs_train, y_bcs_train, s_bcs_train = self.generate_one_bcs_training_data(u0_train)
        # Generate training data for PDE residual
        u_res_train, y_res_train, s_res_train = self.generate_one_res_training_data(u0_train)

        # create the training dataset

        # pre-process the test data
        x_test = torch.tensor(u0_train[-n_test:, :], dtype=torch.float)
        y_test = torch.tensor(usol[-n_test:, :], dtype=torch.float)

        # create the test dataset

    def generate_one_ics_training_data(self, u0_train):
        """
        generate ics training data corresponding to one input sample

        :return:
        """
        t_0 = np.zeros((self.P_ics_train, 1))
        x_0 = np.linspace(0, 1, self.P_ics_train)[:, None]

        y = np.stack([t_0, x_0])
        u = np.tile(u0_train, (self.P_ics_train, 1))
        s = u0_train

        return u, y, s

    def generate_one_bcs_training_data(self, u0_train):
        """
        generate bcs training data corresponding to one input sample
        :param u0_train:
        :param P_bcs_train:
        :return:
        """
        t_bc = random.uniform((self.P_bcs_train, 1))
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
        :param P_res_train:
        :return:
        """
        t_res = random.uniform((self.P_res_train, 1))
        x_res = random.uniform((self.P_res_train, 1))

        u = np.tile(u0_train, (self.P_res_train, 1))
        y = np.hstack([t_res, x_res])
        s = np.zeros((self.P_res_train, 1))

        return u, y, s


if __name__ == '__main__':
    test_builder = Burgers1dBuilder(100,
                                    100,
                                    "E:\\dfno\\PIOperator\\data\\burgers\\2d\\burgers_2d.mat"
                                    )



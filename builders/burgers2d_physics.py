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
        usol = np.array(data_mat['output'])

        # some fixed parameters for the number of training examples
        # number of locations for evulating the initial condition
        P_ics_train = 101
        # number of locations for evulating the boundary condition
        P_bcs_train = 100
        # number of locations for evulating the PDE residual
        P_res_train = 2500
        # resolution of uniform grid for the test data
        P_test = 101

        # input samples shape(num_samples, t, x)
        u0_train = usol[:n_train, 0, :]

        # generate training data with pde equation
        # Generate training data for inital condition
        u_ics_train, y_ics_train, s_ics_train = self.generate_one_ics_training_data()
        # Generate training data for boundary condition
        u_bcs_train, y_bcs_train, s_bcs_train = self.generate_one_bcs_training_data()
        # Generate training data for PDE residual
        u_res_train, y_res_train, s_res_train = self.generate_one_res_training_data()

    def generate_one_ics_training_data(self):
        return 0

    def generate_one_bcs_training_data(self):
        return 0

    def generate_one_res_training_data(self):
        return 0

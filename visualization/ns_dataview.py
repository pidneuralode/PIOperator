import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import h5py

# view the ns equation and print some useful information
file_path = 'E:\\dfno\\PIOperator\\data\\ns\\ns_V1e-3_N5000_T50.mat'
with h5py.File(file_path, 'r') as file:
    # list the keys in the top file
    print(list(file.keys()))
    for key in {'a', 'u', 't'}:
        print(f'{key} shape:{np.array(file[key]).shape}')

    # begin record the value from the file
    u = np.array(file['u'])  # shape:(50, 64, 64, 5000) the vorticity at time:1 to time:50
    a = np.array(file['a'])  # shape:(64, 64, 5000) the vorticity at time:0
    t = np.array(file['t'])  # shape:(50, 1) 1->50

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    cax1 = ax[0].imshow(u[0, :, :, 0], cmap='viridis')
    cax2 = ax[1].imshow(u[19, :, :, 0], cmap='viridis')

    plt.show()
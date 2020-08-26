# from constants import DATA_PATH
# from h5py import File
#
#
# with File(DATA_PATH / 'scsn_ps_2000_2017_shuf.hdf5', 'r') as f:
#     data_x = f['X']
#     data_y = f['Y']
#
#     print(data_x[0])
#     print("----")
#     print(data_y[0])

from kymatio.torch import Scattering1D
import os
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

scattering = Scattering1D(J=2, Q=8, shape=20000)
x = torch.rand((1, 1, 20000))

Sx = scattering(x)
print(Sx.shape)
meta = scattering.meta()
print(meta)
order0 = np.where(meta['order'] == 0)[0]
order1 = np.where(meta['order'] == 1)[0]
order2 = np.where(meta['order'] == 2)[0]
print("-----")
print(order0)
print("-----")
print(order1)
print("----")
print(order2)
print(len(order0), len(order1), len(order2))
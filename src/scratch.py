from constants import DATA_PATH
from h5py import File


with File(DATA_PATH / 'scsn_ps_2000_2017_shuf.hdf5', 'r') as f:
    data_x = f['X']
    data_y = f['Y']

    print(data_x[0])
    print("----")
    print(data_y[0])
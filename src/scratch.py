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


from constants import DATA_PATH
# # Training labels
# label_dict = {0: 'Earthquake',
#               1: 'Noise',
#               2: 'Tremor',
#               3: 'Unclear Event',
#               }
#
# with open(DATA_PATH / 'Golden' / 'golden_classified.txt', 'a') as f:
#     with open(DATA_PATH / 'Golden' / 'golden_classified_suzan.txt', 'r') as f2:
#         for line in f2.readlines():
#             info = line.split()
#             info[-1] = label_dict[int(info[-1])]
#             new_info = " ".join(info)
#             f.write(new_info + '\n')
from pathlib import Path
a = "Hello"
b = "Mello"
print(Path(a) / b)
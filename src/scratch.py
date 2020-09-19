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

file = open(DATA_PATH / 'classification_data_Vivitang.txt', 'r')
data = file.readlines()

with open(DATA_PATH / 'V_golden_new.txt', 'w') as f:
    with open(DATA_PATH / 'V_golden.txt', 'r') as f2:
        for line in f2.readlines():
            line = " ".join(line.split())
            for l in data:
                info = l.split()
                # print(info)
                info = " ".join(info[2: ])
                # print(line.strip(), info.strip())
                if line == info:
                    f.write(" ".join(l.split()) + '\n')



import ml.models
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train, test
from constants import SAVE_PATH
from utils import save_file, load_file
from ml.wavelet import scatter_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 300
batch_size = 50
learning_rate = 1e-5

# # Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/classification_data_Vivitang.txt',
             'training_folder': 'Vivian_Set',
             'folder_type': 'trimmed_data', 'avg': True},

            {'file_name': '../../data/classification_data_suzanv.txt',
                         'training_folder': 'Testing_Set_Suzan',
            'folder_type': 'trimmed_data', 'avg': True
             },

            {'file_name': '../../data/classification_data_ElisabethB.txt',
             'training_folder': 'ElisabethB_set',
             'folder_type': 'trimmed_data', 'avg': True
             },

            {'file_name': '../../data/classification_data_Jeff503.txt',
             'training_folder': 'Jeff_Set',
             'folder_type': 'trimmed_data', 'avg': True
             }
            ]

ld_folders = [{'training_folder': 'Training_Set_Tremor', 'folder_type': 'positive',
              'data_type': 'tremor'},
              {'training_folder': 'Training_Set_Prem', 'folder_type': 'positive',
               'data_type': 'earthquake'}
              ]

#
# ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
# print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))
# save_file(ds, 'ds_large1')

transform_and_save = True

if transform_and_save:
    # Save the dataset object for quicker loading
    # ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
    # save_file(ds, 'ds_exp2')
    ds = load_file('ds_large1')
    # Split data into train and test
    train_indices, test_indices = ds.get_indices_split(train_percent=0.8)
    print(len(train_indices), len(test_indices))
    # print(train_indices)
    # print(test_indices)
    train_set = torch.utils.data.Subset(ds, indices=train_indices)
    test_set = torch.utils.data.Subset(ds, indices=test_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    # Transform the data
    combined_train = []
    combined_test = []

    for index, data in enumerate(train_loader):
        coeffs = scatter_transform(data['data'], excerpt_len=20000, cuda=True)
        # Make sure to pass both seismic and coeffs as we will be training on both
        combined_train.append({'data': [data['data'], coeffs.cpu()], 'label': data['label'],
                               'user': data['user'], 'sub_id': data['sub_id']})

    save_file(combined_train, 'combined_train_exp4')

    for index, data in enumerate(test_loader):
        coeffs = scatter_transform(data['data'], excerpt_len=20000, cuda=True)
        combined_test.append({'data': [data['data'], coeffs.cpu()], 'label': data['label']})

    save_file(combined_test, 'combined_test_exp4')

    # Empty GPU memory
    torch.cuda.empty_cache()

'''
Exp 4: Only Earthquake detective data + (extra tremor data) -> Excerpt len set to 20,000 samples

Outcome -> Doesn't work well as actual excerpts are larger and data contains user bias
'''

import sys
sys.path.append('../../src/')
import ml.models
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train, test
from constants import SAVE_PATH
from utils import save_file, load_file
from kymatio.torch import Scattering1D
from ml.wavelet import scatter_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 100
batch_size = 100
learning_rate = 1e-5

# # Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/classification_data_Vivitang.txt',
             'training_folder': 'Vivian_Set',
             'folder_type': 'trimmed_data', 'avg': False},

            {'file_name': '../../data/classification_data_suzanv.txt',
                         'training_folder': 'Testing_Set_Suzan',
            'folder_type': 'trimmed_data', 'avg': False
             },

            {'file_name': '../../data/classification_data_ElisabethB.txt',
             'training_folder': 'ElisabethB_set',
             'folder_type': 'trimmed_data', 'avg': False
             },

            {'file_name': '../../data/classification_data_Jeff503.txt',
             'training_folder': 'Jeff_Set',
             'folder_type': 'trimmed_data', 'avg': False
             }
            ]

ld_folders = [{'training_folder': 'Training_Set_Tremor', 'folder_type': 'positive',
              'data_type': 'tremor'},
              ]

#
# ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
# print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))
# save_file(ds, 'ds_large2')

transform_and_save = False
load_transformed = True
train_mod = True

J, Q = 10, 32

if transform_and_save:

    ds = load_file('ds_large2')

    # Split data into train and test
    train_indices, test_indices = ds.get_indices_split(train_percent=0.8, seed=True)
    print(len(train_indices), len(test_indices))

    train_set = torch.utils.data.Subset(ds, indices=train_indices)
    test_set = torch.utils.data.Subset(ds, indices=test_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Transform the data
    combined_train = []
    combined_test = []
    for index, data in enumerate(train_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=20000, cuda=True)
        # Make sure to pass both seismic and coeffs as we will be training on both
        combined_train.append({'data': [data['data'], coeffs.cpu()], 'label': data['label'],
                               'user': data['user'], 'sub_id': data['sub_id']})
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(0))

    save_file(combined_train, 'combined_train_exp4_J{}_Q{}'.format(J, Q))

    print("Starting test set: ")
    for index, data in enumerate(test_loader):
        print("Processing batch: {}".format(index))
        coeffs = scatter_transform(data['data'], J=J, Q=Q, excerpt_len=20000, cuda=True)
        combined_test.append({'data': [data['data'], coeffs.cpu()], 'label': data['label'],
                               'user': data['user'], 'sub_id': data['sub_id']})
        torch.cuda.empty_cache()

    save_file(combined_test, 'combined_test_exp4_J{}_Q{}'.format(J, Q))


if load_transformed:
    # Load file
    transformed_train = load_file('combined_train_exp4_J{}_Q{}'.format(J, Q))
    transformed_test = load_file('combined_test_exp4_J{}_Q{}'.format(J, Q))


# Train the model
if train_mod:
    # Initialize the model
    model = ml.models.WavNet().to(device)
    # Load existing model to continue training
    # model_name = 'model_Exp2_02_09_2020-21_57_32_270.pt'
    # model.load_state_dict(torch.load(SAVE_PATH / model_name))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Tensorboard
    exp_name = "Exp4"
    exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(exp_id))

    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=transformed_train, test_set=transformed_test,
          exp_id=exp_id, writer=writer,  print_freq=20, test_freq=30)

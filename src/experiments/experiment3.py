"""
Experiment details: Model which combines convolution and wavelet scattering
"""

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

# Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/V_golden.txt', 'training_folder': 'Training_Set_Vivian',
             'folder_type': 'trimmed_data', 'avg': True},
            {'file_name': '../../data/classification_data.txt', 'training_folder':
                'Testing_Set_Suzan', 'folder_type': 'trimmed_data', 'avg': True}
            ]

ld_folders = [{'training_folder': 'Training_Set_Prem', 'folder_type': 'positive',
               'data_type': 'earthquake'},
              {'training_folder': 'Training_Set_Prem', 'folder_type': 'negative',
               'data_type': 'earthquake'},
              {'training_folder': 'Training_Set_Tremor', 'folder_type': 'positive',
               'data_type': 'tremor'}, ]


# Quick bool values to save data or directly load saved data
transform_and_save = False
load_transformed = True

if transform_and_save:
    # Save the dataset object for quicker loading
    # ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)
    # save_file(ds, 'ds_exp2')
    ds = load_file('ds_exp2')
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
        combined_train.append({'data': [data['data'], coeffs.cpu()], 'label': data['label']})

    save_file(combined_train, 'combined_train_2')

    for index, data in enumerate(test_loader):
        coeffs = scatter_transform(data['data'], excerpt_len=20000, cuda=True)
        combined_test.append({'data': [data['data'], coeffs.cpu()], 'label': data['label']})

    save_file(combined_test, 'combined_test_2')

    # Empty GPU memory
    torch.cuda.empty_cache()

if load_transformed:
    # Load file
    combined_train = load_file('combined_train')
    combined_test = load_file('combined_test')


train_mod = True

# Train the model
if train_mod:
    # Initialize the model
    model = ml.models.WavCon().to(device)
    # Load existing model to continue training
    # model_name = 'model_Exp2_02_09_2020-21_57_32_270.pt'
    # model.load_state_dict(torch.load(SAVE_PATH / model_name))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Tensorboard
    exp_name = "Exp3"
    exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(exp_id))

    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=combined_train, test_set=combined_test,
          exp_id=exp_id, writer=writer, save_freq=50, print_freq=20, test_freq=30)

    # Get the confusion matrix for the training data
    conf_mat, acc, loss = test(model=model, test_set=combined_train, loss_func=loss_func)
    print("Training confusion matrix:")
    print(conf_mat)
"""
Model with wavelet scattering transform
"""

import ml.models
from ml.data_processing import load_data
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train, test, train_wavelets
from constants import SAVE_PATH
from utils import save_file, load_file
from ml.wavelet import scatter_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 100
batch_size = 50
learning_rate = 1e-4

# Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/V_golden.txt', 'training_folder': 'Training_Set_Vivian',
             'folder_type': 'trimmed_data', 'avg': True}]

ld_folders = [{'training_folder': 'Training_Set_Prem', 'folder_type': 'positive',
               'data_type': 'earthquake'},
              {'training_folder': 'Training_Set_Prem', 'folder_type': 'negative',
               'data_type': 'earthquake'},
              {'training_folder': 'Training_Set_Tremor', 'folder_type': 'positive',
               'data_type': 'tremor'},
             ]

# Save the dataset object for quicker loading
# ds = QuakeDataSet(ld_files=ld_files, ld_folders=ld_folders, excerpt_len=20000)

# save_file(ds, 'ds_exp2')

# Load the dataset object
ds = load_file('ds_avg')

dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

X_trans = []

# #
# # Calculate wavelet coefficients once
# for index, data in enumerate(dataloader):
#     wav_coeffs = scatter_transform(data['data'], excerpt_len=20000)
#     wav_coeffs = wav_coeffs.reshape(data['data'].shape[0], 3, -1)
#     X_trans.append([wav_coeffs, data['label']])
#
# save_file(X_trans, 'X_trans')

# Load file
X_trans = load_file('X_trans')
print(X_trans[0][0].shape)
# Initialize the model
model = ml.models.WavNet().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize Tensorboard
exp_name = "Exp2"
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))


train_mod = True
test_mod = False

# Train the model
if train_mod:
    train_wavelets(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, dataloader=X_trans, exp_id=exp_id, writer=writer, save_freq=30,
          print_freq=20, val_freq=20)


# Test the model
if test_mod:
    model_name = 'model_Exp1_24_08_2020-20_45_52_90.pt'
    model.load_state_dict(torch.load(SAVE_PATH / model_name))
    model.eval()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True)
    test(model, dataloader=dataloader)
"""
Experiment details: Running with a simple convolution architecture

"""
import sys
sys.path.append('../../src/')
import ml.models
from ml.data_processing import load_data
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train, test
from constants import SAVE_PATH
from utils import save_file, load_file


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X, Y, X_names = load_data('../../data/V_golden.txt', training_folder='Training_Set_Vivian')


# Hyper parameters
num_epochs = 100
batch_size = 500
learning_rate = 1e-4

# Load the data as PyTorch tensors
ld_files = [{'file_name': '../../data/V_golden_new.txt', 'training_folder': 'Training_Set_Vivian',
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
# save_file(ds, 'ds_avg')

# Load the dataset object
ds = load_file('ds_avg')


dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


# Initialize the model
model = ml.models.FeatureExtractor().to(device)
# Load existing model to continue training
model_name = 'model_Exp1_25_08_2020-15_04_46_90.pt'
model.load_state_dict(torch.load(SAVE_PATH / model_name))

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# Initialize Tensorboard
exp_name = "Exp1"
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))


train_mod = False
test_mod = True

# Train the model
if train_mod:
    train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
          optimizer=optimizer, train_set=dataloader, exp_id=exp_id, writer=writer, save_freq=30,
          print_freq=20, test_freq=20)

# Test the model
if test_mod:
    model_name = 'model_Exp1_25_08_2020-15_39_43_30.pt'
    model.load_state_dict(torch.load(SAVE_PATH / model_name))
    model.eval()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True)
    test(model, test_set=dataloader, loss_func=loss_func)


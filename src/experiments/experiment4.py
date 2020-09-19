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
ld_files = [{'file_name': '../../data/V_golden_new.txt', 'training_folder': 'Training_Set_Vivian',
             'folder_type': 'trimmed_data', 'avg': True}]


ds = QuakeDataSet(ld_files=ld_files, ld_folders=[], excerpt_len=20000)

print(len(ds.X), len(ds.X_users), len(ds.X_ids), len(ds.X_names), len(ds.Y))

ds = torch.utils.data.DataLoader(ds, batch_size=100)


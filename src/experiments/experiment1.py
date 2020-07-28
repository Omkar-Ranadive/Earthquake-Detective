import ml.models
from ml.data_processing import load_data
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from ml.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, Y, X_names = load_data('../../data/V_golden.txt', training_folder='Training_Set_Vivian')

# Hyper parameters
num_epochs = 100
batch_size = X.shape[0]
learning_rate = 1e-3

# Load the data as PyTorch tensors
ds = QuakeDataSet(X, Y, X_names)
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

# Initialize the model
model = ml.models.FeatureExtractor().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize Tensorboard
exp_name = "Exp1"
exp_id = '{}_{}'.format(exp_name, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
writer = SummaryWriter('runs/{}'.format(exp_id))


# Train the model
train(num_epochs=num_epochs, batch_size=batch_size, model=model, loss_func=loss_func,
      optimizer=optimizer, dataloader=dataloader, exp_id=exp_id, writer=writer, save_freq=30)
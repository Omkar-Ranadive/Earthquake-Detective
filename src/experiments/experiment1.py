import ml.models
from ml.data_processing import load_data
import torch
from ml.dataset import QuakeDataSet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, Y, X_names = load_data('../../data/V_golden.txt', training_folder='Training_Set_Vivian')

# Hyper parameters
num_epochs = 50
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
for epoch in range(num_epochs):
    total_loss = []
    for index, data in enumerate(dataloader):
        batch_in = data['data'].to(device)
        batch_out = data['label'].to(device)
        output = model(batch_in)

        # Calculate loss
        loss = loss_func(output, batch_out)

        print("Loss: {}".format(loss))
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Avg Loss', np.mean(total_loss), epoch)

writer.flush()
writer.close()

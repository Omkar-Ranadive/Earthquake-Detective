import torch
import numpy as np
from src.constants import SAVE_PATH
from sklearn.metrics import confusion_matrix


def train(num_epochs,
          batch_size,
          model,
          loss_func,
          optimizer,
          dataloader,
          exp_id,
          writer=None,
          save_freq=None,
          print_freq=50,
          val_freq=50,
          ):

    """
    A generalized train function to experiment with different model types and hyperparameters
    Args:
        num_epochs (int): The number of epochs for which the training should be repeated
        batch_size (int): Number of examples per batch
        model (Pytorch model): Instance of Pytorch model
        loss_func (torch.nn object): PyTorch loss object
        optimizer (torch.nn object): Optimizer to use for backprop
        dataloader (Pytorch Dataloader): PyTorch Dataloader object
        exp_id (str): Experiment ID for logging
        writer (Tensorboard Writer): Tensorboard summary writer to visualize data
        save_freq (int): Model saving frequency
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss = []
        accuracies = []
        for index, data in enumerate(dataloader):
            batch_in = data['data'].to(device)
            batch_out = data['label'].to(device)
            output = model(batch_in)

            # Calculate loss
            loss = loss_func(output, batch_out)

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_mat = (predicted == batch_out).squeeze()
            correct_count = torch.sum(correct_mat).item()
            accuracies.append((correct_count/len(predicted)))

            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % print_freq == 0:
            print("Epoch: {}  Avg Loss: {}: loss, Avg Acc: {}".format(epoch, np.mean(
                total_loss), np.mean(accuracies)))

        if writer:
            writer.add_scalar('Avg Loss', np.mean(total_loss), epoch)
            writer.add_scalar('Avg Acc', np.mean(accuracies), epoch)

        # Save the model
        if save_freq is not None and epoch % save_freq == 0:
            model_name = "model_{}_{}.pt".format(exp_id, epoch)
            torch.save(model.state_dict(), str(SAVE_PATH / model_name))

    if writer:
        writer.flush()
        writer.close()


def train_wavelets(num_epochs,
          batch_size,
          model,
          loss_func,
          optimizer,
          dataloader,
          exp_id,
          writer=None,
          save_freq=None,
          print_freq=50,
          val_freq=50,
          ):

    """
    A generalized train function to experiment with different model types and hyperparameters
    Args:
        num_epochs (int): The number of epochs for which the training should be repeated
        batch_size (int): Number of examples per batch
        model (Pytorch model): Instance of Pytorch model
        loss_func (torch.nn object): PyTorch loss object
        optimizer (torch.nn object): Optimizer to use for backprop
        dataloader (List): List containing tensors of wavlet coefficients and labels
        exp_id (str): Experiment ID for logging
        writer (Tensorboard Writer): Tensorboard summary writer to visualize data
        save_freq (int): Model saving frequency
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss = []
        accuracies = []
        for index, data in enumerate(dataloader):
            batch_in = data[0].to(device)
            batch_out = data[1].to(device)
            output = model(batch_in)

            # Calculate loss
            loss = loss_func(output, batch_out)

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_mat = (predicted == batch_out).squeeze()
            correct_count = torch.sum(correct_mat).item()
            accuracies.append((correct_count/len(predicted)))

            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % print_freq == 0:
            print("Epoch: {}  Avg Loss: {}: loss, Avg Acc: {}".format(epoch, np.mean(
                total_loss), np.mean(accuracies)))

        if writer:
            writer.add_scalar('Avg Loss', np.mean(total_loss), epoch)
            writer.add_scalar('Avg Acc', np.mean(accuracies), epoch)

        # Save the model
        if save_freq is not None and epoch % save_freq == 0:
            model_name = "model_{}_{}.pt".format(exp_id, epoch)
            torch.save(model.state_dict(), str(SAVE_PATH / model_name))

    if writer:
        writer.flush()
        writer.close()


def test(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for index, data in enumerate(dataloader):
        batch_in = data['data'].to(device)
        batch_out = data['label'].to(device)
        output = model(batch_in)

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct_mat = (predicted == batch_out).squeeze()
        correct_count = torch.sum(correct_mat).item()
        print(predicted)
        # Print the confusion matrix
        print("Confusion Matrix:")
        conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(), y_pred=predicted.cpu().numpy())
        print(conf_mat)
        # Print the accuracy
        print("Accuracy: {}".format(correct_count/len(predicted)))



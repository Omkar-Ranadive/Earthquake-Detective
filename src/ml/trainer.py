import torch
import numpy as np
from constants import DATA_PATH, SAVE_PATH, n_classes
from sklearn.metrics import confusion_matrix


def train(num_epochs,
          batch_size,
          model,
          loss_func,
          optimizer,
          exp_id,
          train_set,
          test_set=None,
          test_set_gold=None,
          writer=None,
          save_freq=None,
          print_freq=50,
          test_freq=50,
          ):

    """
    A generalized train function to experiment with different model types and hyperparameters
    Args:
        num_epochs (int): The number of epochs for which the training should be repeated
        batch_size (int): Number of examples per batch
        model (Pytorch model): Instance of Pytorch model
        loss_func (torch.nn object): PyTorch loss object
        optimizer (torch.nn object): Optimizer to use for backprop
        train_set (Pytorch Dataloader): PyTorch Dataloader object
        test_set (Pytorch Dataloader): PyTorch Dataloader object for test data
        test_set_gold (Pytorch Dataloader): PyTorch Dataloader object for test data with full
                                            reliability
        exp_id (str): Experiment ID for logging
        writer (Tensorboard Writer): Tensorboard summary writer to visualize data
        save_freq (int): Model saving frequency
        print_freq (int): Frequency of printing accuracy and loss to console
        test_freq (int): Frequency of testing on test set
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss = []
        accuracies = []
        combined_mat = np.zeros((n_classes, n_classes))

        for index, data in enumerate(train_set):
            # Move data to device individually if data consists of multi sub-data components
            if isinstance(data['data'], list):
                batch_in = [d.to(device) for d in data['data']]
            else:
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

            # Calculate confusion matrix
            conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(),
                                        y_pred=predicted.cpu().numpy(), labels=range(n_classes))
            combined_mat += conf_mat

            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % test_freq == 0 and test_set is not None:
            conf_mat, test_acc, test_loss = test(model, test_set, loss_func)
            print("Testing Accuracy {} Testing Loss {}".format(test_acc, test_loss))
            print("Confusion Matrix for testing set:  ")
            print(conf_mat)
            if writer:
                writer.add_scalar('Avg Test Acc', test_acc, epoch)

            if test_set_gold is not None:
                # Also calculate over "clean" test set
                conf_mat, test_acc, test_loss = test(model, test_set_gold, loss_func)
                print("Gold Test Accuracy {} Gold Test Loss {}".format(test_acc, test_loss))
                print("Confusion Matrix for gold testing set:  ")
                print(conf_mat)
                if writer:
                    writer.add_scalar('Avg Test Gold Acc', test_acc, epoch)

        if epoch % print_freq == 0:
            print("Epoch: {}  Avg Train Loss: {}: loss, Avg Train Acc: {}".format(epoch, np.mean(
                total_loss), np.mean(accuracies)))
            print("Confusion matrix for training set")
            print(combined_mat)

        if writer:
            writer.add_scalar('Avg Loss', np.mean(total_loss), epoch)
            writer.add_scalar('Avg Acc', np.mean(accuracies), epoch)

        # Save the model
        if save_freq is not None and epoch % save_freq == 0 or (epoch == num_epochs-1):
            model_name = "model_{}_{}.pt".format(exp_id, epoch)
            print("Modal saved: {}".format(model_name))
            torch.save(model.state_dict(), str(SAVE_PATH / model_name))

    if writer:
        writer.flush()
        writer.close()


def test(model, test_set, loss_func):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    combined_mat = np.zeros((n_classes, n_classes))
    total_loss = []
    with torch.no_grad():
        for index, data in enumerate(test_set):
            if isinstance(data['data'], list):
                batch_in = [d.to(device) for d in data['data']]
            else:
                batch_in = data['data'].to(device)

            batch_out = data['label'].to(device)
            output = model(batch_in)

            # Calculate loss
            total_loss.append(loss_func(output, batch_out).item())

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_mat = (predicted == batch_out).squeeze()
            correct_count = torch.sum(correct_mat).item()
            accuracies.append(correct_count/len(predicted))
            conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(),
                                        y_pred=predicted.cpu().numpy(), labels=range(n_classes))
            combined_mat += conf_mat

    return combined_mat, np.mean(accuracies), np.mean(total_loss)


def generate_model_log(model, model_name, sample_set, names, set='test'):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    combined_mat = np.zeros((n_classes, n_classes))
    total_loss = []

    file_name = model_name[:-2] + '_' + set + '.txt'
    with open(SAVE_PATH / file_name, 'w') as f:
        with torch.no_grad():
            for index, data in enumerate(sample_set):
                f.write("Batch {}\n".format(index))
                if isinstance(data['data'], list):
                    batch_in = [d.to(device) for d in data['data']]
                else:
                    batch_in = data['data'].to(device)

                batch_out = data['label'].to(device)
                output = model(batch_in)

                # Calculate loss
                #  total_loss.append(loss_func(output, batch_out).item())

                # Calculate accuracy
                _, predicted = torch.max(output, 1)

                predicted_labels = predicted.cpu()
                true_labels = data['label']
                users = data['user']
                subject_ids = data['sub_id']
                indices = data['index']

                for i, probs in enumerate(output.cpu()):

                    outcome = 'C' if predicted_labels[i].item() == true_labels[i].item() else 'I'

                    f.write("{} Prob(E/N/T): {} P: {}  T: {} UID: {} Path: {}\n\n".format(outcome,
                        probs, predicted_labels[i].item(), true_labels[i].item(),
                        users[i].item(), names[indices[i]]))

                f.write("--"*20 + "\n")
                correct_mat = (predicted == batch_out).squeeze()
                correct_count = torch.sum(correct_mat).item()
                accuracies.append(correct_count / len(predicted))
                conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(),
                                            y_pred=predicted.cpu().numpy(), labels=range(n_classes))
                combined_mat += conf_mat

        f.write("Confusion Matrix: \n")
        np.savetxt(f, combined_mat, fmt='%i')

        f.write("\n Accuracy: {} ".format(np.mean(accuracies)))

        f.close()


def train_v2(num_epochs,
          batch_size,
          model,
          loss_func,
          optimizer,
          exp_id,
          rel_dict,
          train_set,
          test_set=None,
          test_set_gold=None,
          writer=None,
          save_freq=None,
          print_freq=50,
          test_freq=50,
        ):
    """
      Train v2 - Takes reliability scores into account while updating gradients
      Args:
          num_epochs (int): The number of epochs for which the training should be repeated
          batch_size (int): Number of examples per batch
          model (Pytorch model): Instance of Pytorch model
          loss_func (torch.nn object): PyTorch loss object
          optimizer (torch.nn object): Optimizer to use for backprop
          train_set (Pytorch Dataloader): PyTorch Dataloader object
          test_set (Pytorch Dataloader): PyTorch Dataloader object for test data
          exp_id (str): Experiment ID for logging
          rel_dict (dict): Dictionary containing reliability scores
          writer (Tensorboard Writer): Tensorboard summary writer to visualize data
          save_freq (int): Model saving frequency
          print_freq (int): Frequency of printing accuracy and loss to console
          test_freq (int): Frequency of testing on test set
      """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss = []
        accuracies = []
        combined_mat = np.zeros((n_classes, n_classes))

        for index, data in enumerate(train_set):
            # Move data to device individually if data consists of multi sub-data components
            if isinstance(data['data'], list):
                batch_in = [d.to(device) for d in data['data']]
            else:
                batch_in = data['data'].to(device)
            batch_out = data['label'].to(device)
            output = model(batch_in)
            users = data['user'].to(device)

            # Calculate loss
            loss = loss_func(model=model, predicted=output, actual=batch_out, users=users,
                             rel_dict=rel_dict)

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_mat = (predicted == batch_out).squeeze()
            correct_count = torch.sum(correct_mat).item()
            accuracies.append((correct_count / len(predicted)))

            # Calculate confusion matrix
            conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(),
                                        y_pred=predicted.cpu().numpy(), labels=range(n_classes))
            combined_mat += conf_mat

            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % test_freq == 0 and test_set is not None:
            conf_mat, test_acc, test_loss = test_v2(model, test_set, loss_func, rel_dict)
            print("Testing Accuracy {} Testing Loss {}".format(test_acc, test_loss))
            print("Confusion Matrix for testing set:  ")
            print(conf_mat)
            if writer:
                writer.add_scalar('Avg Test Acc', test_acc, epoch)
            if test_set_gold is not None:
                # Also calculate over "clean" test set
                print("Gold Test Accuracy {} Gold Test Loss {}".format(test_acc, test_loss))
                print("Confusion Matrix for gold testing set:  ")
                conf_mat, test_acc, test_loss = test_v2(model, test_set_gold, loss_func, rel_dict)
                print(conf_mat)
                if writer:
                    writer.add_scalar('Avg Test Gold Acc', test_acc, epoch)

        if epoch % print_freq == 0:
            print("Epoch: {}  Avg Train Loss: {}: loss, Avg Train Acc: {}".format(epoch, np.mean(
                total_loss), np.mean(accuracies)))
            print("Confusion matrix for training set")
            print(combined_mat)

        if writer:
            writer.add_scalar('Avg Loss', np.mean(total_loss), epoch)
            writer.add_scalar('Avg Acc', np.mean(accuracies), epoch)

        # Save the model
        if save_freq is not None and epoch % save_freq == 0 or (epoch == num_epochs - 1):
            model_name = "model_{}_{}.pt".format(exp_id, epoch)
            torch.save(model.state_dict(), str(SAVE_PATH / model_name))

    if writer:
        writer.flush()
        writer.close()


def test_v2(model, test_set, loss_func, rel_dict):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    combined_mat = np.zeros((n_classes, n_classes))
    total_loss = []
    with torch.no_grad():
        for index, data in enumerate(test_set):
            if isinstance(data['data'], list):
                batch_in = [d.to(device) for d in data['data']]
            else:
                batch_in = data['data'].to(device)

            batch_out = data['label'].to(device)
            output = model(batch_in)
            users = data['user'].to(device)

            # Calculate loss
            total_loss.append(loss_func(model=model, predicted=output, actual=batch_out,
                                        users=users, rel_dict=rel_dict).item())

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_mat = (predicted == batch_out).squeeze()
            correct_count = torch.sum(correct_mat).item()
            accuracies.append(correct_count/len(predicted))
            conf_mat = confusion_matrix(y_true=batch_out.cpu().numpy(),
                                        y_pred=predicted.cpu().numpy(), labels=range(n_classes))
            combined_mat += conf_mat

    return combined_mat, np.mean(accuracies), np.mean(total_loss)
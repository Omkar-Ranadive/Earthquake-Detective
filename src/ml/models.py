import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Define the architecture
        self.conv1 = nn.Conv1d(3, 6, kernel_size=100, stride=5)
        self.pool1 = nn.MaxPool1d(kernel_size=5)
        self.conv2 = nn.Conv1d(6, 8, kernel_size=100, stride=5)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.fc1 = nn.Linear(8*206, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # print("Input shape: {}".format(x.shape))
        x = F.relu(self.conv1(x))
        # print("After first conv1: {}".format(x.shape))
        x = self.pool1(x)
        # print("After first pool: {}".format(x.shape))
        x = F.relu(self.conv2(x))
        # print("After second conv2 {}".format(x.shape))
        x = self.pool2(x)
        # print("after pool: {}".format(x.shape))
        x = x.view(-1, self.flatten_features(x))
        # print("After flattening {}".format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        # print("Final {}".format(x.shape))
        return x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features

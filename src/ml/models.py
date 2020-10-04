import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D
from constants import n_classes


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Define the architecture
        self.conv1 = nn.Conv1d(3, 6, kernel_size=75, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(6, 8, kernel_size=50, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.fc1 = nn.Linear(1264, 128)
        self.fc2 = nn.Linear(128, 3)

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


class WavNet(nn.Module):
    def __init__(self):
        super(WavNet, self).__init__()

        # self.pool1 = nn.MaxPool1d(kernel_size=20)
        self.fc1 = nn.Linear(4113, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, batch_in):
        # x = self.pool1(x)

        if isinstance(batch_in, list):
            x, transformed_x = batch_in[0], batch_in[1]
        else:
            transformed_x = batch_in
        transformed_x = transformed_x.view(-1, self.flatten_features(transformed_x))
        transformed_x = F.relu(self.fc1(transformed_x))
        transformed_x = F.relu(self.fc2(transformed_x))
        transformed_x = F.softmax(self.fc3(transformed_x), dim=1)

        return transformed_x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class WavCon(nn.Module):
    def __init__(self):
        super(WavCon, self).__init__()
        self.conv1 = nn.Conv1d(3, 6, kernel_size=75, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(6, 8, kernel_size=50, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.fc1 = nn.Linear(4113, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, batch_in):
        """
        Args:
            batch_in (list): Consists of two tensors:
                            - Seismic data in form of [batch size, features, excerpt_len]
                            - Wavelet coefficients in form of [batch_size, features, num_coeffs]
        """
        x, transformed_x = batch_in[0], batch_in[1]

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten out both of them
        x = x.view(-1, self.flatten_features(x))
        transformed_x = transformed_x.view(-1, self.flatten_features(transformed_x))
        # Combine both wavelet and convolution information before passing it through fcn
        x_concat = torch.cat((x, transformed_x), dim=-1)
        x = F.relu(self.fc1(x_concat))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class WavNetV2(nn.Module):
    def __init__(self):
        super(WavNetV2, self).__init__()

        # self.pool1 = nn.MaxPool1d(kernel_size=20)
        self.fc1 = nn.Linear(4350, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, batch_in):
        coeffs_1, coeffs_2 = batch_in[0], batch_in[1]

        t1 = coeffs_1.view(-1, self.flatten_features(coeffs_1))
        t2 = coeffs_2.view(-1, self.flatten_features(coeffs_2))
        transformed_x = torch.cat((t1, t2), dim=-1)
        transformed_x = F.relu(self.fc1(transformed_x))
        transformed_x = F.relu(self.fc2(transformed_x))
        transformed_x = F.softmax(self.fc3(transformed_x), dim=1)

        return transformed_x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class WavImg(nn.Module):
    def __init__(self, h, w):
        super(WavImg, self).__init__()
        self.h = h
        self.w = w
        # For image processing
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(9233, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, batch_in):

        coeffs, x_img = batch_in[0], batch_in[1]

        x_img = F.relu(self.conv1(x_img))
        x_img = self.pool1(x_img)
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.pool2(x_img))
        x_img = x_img.view(-1, self.flatten_features(x_img))

        coeffs = coeffs.view(-1, self.flatten_features(coeffs))
        x = torch.cat((coeffs, x_img), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class ImgNet(nn.Module):
    def __init__(self, h, w):
        super(ImgNet, self).__init__()
        self.h = h
        self.w = w
        # For image processing
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, batch_in):
        coeffs, x_img = batch_in[0], batch_in[1]

        x_img = F.relu(self.conv1(x_img))
        x_img = self.pool1(x_img)
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.pool2(x_img))
        x_img = x_img.view(-1, self.flatten_features(x_img))
        x = F.relu(self.fc1(x_img))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


if __name__ == '__main__':

    # Use the following code to understand the changing dimensions after applying each layer
    batch, features, excerpt = 500, 3, 20000
    dummy_x = torch.rand((50, 3, 787))
    dummy_x2 = torch.rand((50, 3, 20000))
    dummy_x3 = torch.rand((100, 3, 433))
    dummy_x4 = torch.rand((100, 3, 1884))
    c1, c2 = torch.rand((100, 3, 1371)), torch.rand((100, 3, 79))

    ''' For simple Feature Extractor Net '''
    # model = FeatureExtractor()
    # model(dummy_x)

    # ''' For WavNet'''
    # model = WavNet()
    # model(dummy_x4)

    # '''For WavCon Net'''
    # model = WavCon()
    # model([dummy_x4, dummy_x])
    #
    # ''' For WavNetV2'''
    # model = WavNetV2()
    # model([c1, c2])
    #

    '''For WavImg Net'''
    img = torch.rand((100, 3, 200, 320))
    # model = WavImg(h=200, w=300)
    model = ImgNet(h=200, w=300)
    model([c1, img])

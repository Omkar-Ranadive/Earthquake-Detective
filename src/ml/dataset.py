from torch.utils.data import Dataset


class QuakeDataSet(Dataset):
    def __init__(self, X, Y, X_names):
        self.X = X
        self.Y = Y
        self.X_names = X_names

    def __getitem__(self, item):
        return {'data': self.X[item], 'label': self.Y[item]}

    def __len__(self):
        return self.X.shape[0]


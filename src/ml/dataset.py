from torch.utils.data import Dataset
import numpy as np
from src.utils import generate_file_name_from_labels
from constants import DATA_PATH, label_dict, folder_labels
from obspy import read
import os
import warnings


class QuakeDataSet(Dataset):
    def __init__(self, ld_files, ld_folders, excerpt_len, mode='train'):
        self.X, self.Y, self.X_names = [], [], []
        self.excerpt_len = excerpt_len
        self.mode = mode

        if ld_files is not None:
            for ld_file in ld_files:
                x, y, x_names = self._load_data(**ld_file)
                self.X.extend(x)
                self.Y.extend(y)
                self.X_names.extend(x_names)
        if ld_folders is not None:
            for ld_folder in ld_folders:
                x, y, x_names = self._load_data_from_folder(**ld_folder)
                self.X.extend(x)
                self.Y.extend(y)
                self.X_names.extend(x_names)

        # Convert to numpy
        self.Y = np.array(self.Y, dtype='int64')

    def __getitem__(self, item):

        # Pad the data to fixed excerpt length
        cur_item = self.X[item]
        t_item = []
        for feature in cur_item:
            transformed_data = self._pad_data(feature)
            t_item.append(transformed_data)

        return {'data': np.array(t_item), 'label': self.Y[item]}

    def __len__(self):
        return len(self.X)

    def _pad_data(self, x):
        length = x.shape[-1]

        if length > self.excerpt_len:
            if self.mode == 'train':
                offset = np.random.randint(0, length - self.excerpt_len)
            else:
                offset = 0
        else:
            offset = 0

        pad_length = max(self.excerpt_len - length, 0)
        pad_tuple = [(int(pad_length / 2), int(pad_length / 2) + (length % 2))]
        data = np.pad(x, pad_tuple, mode='constant')
        data = data[offset:offset + self.excerpt_len]

        return data

    @staticmethod
    def _load_data(file_name, training_folder, folder_type="trimmed_data"):
        """
        Function to load data from text file
        Args:
            file_name (str): Path to text file from which contains info on data. Assumption:
                           Lines in file  are of the form: Time_stamp network station component label
            training_folder (str): Folder name which contains the actual training data
            folder_type (str): Specify what kind of data to load (processed_data, raw_data, audio,
                               plots). Default = trimmed_data

        Returns (np arrays): Two arrays X and Y where X = training data, Y = training labels and
                            X_names = file name associated with the data in X

        """
        fl_map = generate_file_name_from_labels(file_name)
        X, Y, X_names = [], [], []
        train_path = DATA_PATH / training_folder

        for folder, files in fl_map.items():
            folder_path = train_path / folder / folder_type
            for file in files:
                # File is a list of following form = [file_name, label]
                # Also, load the BHE and BHN components along with the Z component
                # NOTE: Its assumed that the labelled data only contains BHZ components. We are
                # assuming the labels of other components to be the same
                f_bhe = file[0].replace('BHZ', 'BHE')
                f_bhn = file[0].replace('BHZ', 'BHN')
                file_path_z = str(folder_path / (file[0] + '.sac'))
                file_path_e = str(folder_path / (f_bhe + '.sac'))
                file_path_n = str(folder_path / (f_bhn + '.sac'))

                if os.path.exists(file_path_z) and os.path.exists(file_path_e) and os.path.exists(
                        file_path_n):
                    st1 = read(file_path_z)
                    st2 = read(file_path_e)
                    st3 = read(file_path_n)
                    X.append([st1[0].data, st2[0].data, st3[0].data])
                    X_names.append(file[0])
                    Y.append(label_dict[file[1]])
                else:
                    # Warn users if some file is not found
                    warnings.warn("File not found: {}".format(file[0]))

        return X, Y, X_names

    @staticmethod
    def _load_data_from_folder(training_folder, folder_type):
        """
        Function to load data directly from folder.
        Assumes the following folder structure:
        Training Folder name
            - Data Folder 1
                - positive
                - negative
                - etc
            - Data Folder 2
                - positive
                - negative
                - etc
        Args:
            training_folder (str): Name of parent training folder
            folder_type (str): Type of examples to load (i.e positive, negative etc)

       Returns (np arrays): Two arrays X and Y where X = training data, Y = training labels and
                            X_names = file name associated with the data in X
        """

        X, Y, X_names = [], [], []
        train_path = DATA_PATH / training_folder
        for folder in os.listdir(train_path):
            folder_path = train_path / folder
            if os.path.isdir(folder_path):
                # Each earthquake data has a different folder, so loop through this inner folder
                for inner_folder in os.listdir(folder_path):
                    if folder_type == inner_folder:
                        # Get unique file names (ignore the components for now)
                        files = []

                        for file in os.listdir(folder_path / inner_folder):
                            if '.SAC' in file or '.sac' in file:
                                # Remove component info (BHE,BHZ etc) and ".sac" before appending
                                # File names are assumed to be name_BH{}.sac
                                files.append(file[:-7])  # Hence remove last 7 chars

                        # Now, load three component data for each unique file name
                        for file in files:
                            file_path_z = str(folder_path / inner_folder / (file + 'BHZ' + '.SAC'))
                            file_path_e = str(folder_path / inner_folder / (file + 'BHE' + '.SAC'))
                            file_path_n = str(folder_path / inner_folder / (file + 'BHN' + '.SAC'))

                            if os.path.exists(file_path_z) and os.path.exists(
                                    file_path_e) and os.path.exists(
                                file_path_n):
                                st1 = read(file_path_z)
                                st2 = read(file_path_e)
                                st3 = read(file_path_n)
                                X.append([st1[0].data, st2[0].data, st3[0].data])
                                X_names.append(file)
                                Y.append(folder_labels[folder_type])

        return X, Y, X_names

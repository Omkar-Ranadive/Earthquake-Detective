from torch.utils.data import Dataset
import numpy as np
from src.utils import generate_file_name_from_labels
from constants import DATA_PATH, label_dict, folder_labels
from obspy import read
import os
import warnings
from kymatio.numpy import Scattering1D


class QuakeDataSet(Dataset):
    def __init__(self, ld_files, ld_folders, excerpt_len, transforms=[], mode='train'):
        """
        Args:
            ld_files (list): A list of dictionaries where the dictionary contains parameters for
                            _load_data func
            ld_folders (list): List of dictionaries where each dictionary contains parameters
                            for _load_data_from_folder func
            excerpt_len (int): Length of each trace in final training data. Each signal will be
                                padded/trimmed to this len.
            transforms (list): A list of strings listing transforms to apply.
                               Supported transforms:
                               'wavelet' - Applies wavelet scattering transform
            mode (str): mode can be 'train' or any other str. For train, random offset sampling
                        is performed.
        """
        self.X, self.Y, self.X_names = [], [], []
        self.excerpt_len = excerpt_len
        self.mode = mode
        self.transforms = transforms

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

        # Print info on class distribution
        print("Total training examples: {}".format(len(self.Y)))
        print("-------------------------------------------")
        print("Class distribution: ")
        for k, v in label_dict.items():
            print("Number of {} examples: {}".format(k, np.count_nonzero(self.Y == v)))
        print("-------------------------------------------")

        # Initialize the transforms
        # Note: This is too slow to perform dynamically; for now don't use inside dataset class.
        for transform in self.transforms:
            if transform == 'wavelet':
                self.scattering = Scattering1D(J=6, Q=16, shape=self.excerpt_len)

    def __getitem__(self, item):

        # Pad the data to fixed excerpt length
        cur_item = self.X[item]
        t_item = []
        for feature in cur_item:
            transformed_data = self._pad_data(feature)
            t_item.append(transformed_data)

        t_item = np.array(t_item, dtype='float32')

        # Apply any specified transforms - Currently only supports wavelet transform
        # Note: Too slow, so avoid using inside dataset class
        for transform in self.transforms:
            if transform == 'wavelet':
                t_item = self.scattering(t_item)

        return {'data': t_item, 'label': self.Y[item]}

    def __len__(self):
        return len(self.X)

    def get_indices_split(self, train_percent):
        """
        Function to return train_indices and test_indices. This can be used to break this dataset
        into train and test sets. Indices splits are calculated in proportion to the class
        distribution.

        Args:
            train_percent (float): Percentage of samples to include in the training set

        Returns (list): Training indices and testing indices
        """

        total_samples = self.Y.shape[0]
        num_train = int(train_percent * total_samples)
        num_test = total_samples - num_train
        train_indices = []
        test_indices = []

        # Get indices for all classes
        for k, v in label_dict.items():
            indices = np.where(self.Y == v)[0]
            # Shuffle the indices
            np.random.shuffle(indices)
            # Find what percentage of the total samples are k
            percent_of_total = len(indices)/total_samples
            # Include the same percentage in test set
            num_in_test = int(percent_of_total * num_test)
            test_indices.extend(indices[:num_in_test])
            train_indices.extend(indices[num_in_test:])

        return train_indices, test_indices

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
    def _downsample_data(st, desired_rate=20.0):
        # Calculate decimating factor from desired rate and current sampling rate
        cur_rate = st[0].stats.sampling_rate
        assert cur_rate >= desired_rate, "Error: Current sampling rate is lower than desired rate"
        factor = int(cur_rate / desired_rate)
        st.decimate(factor=factor)

        return st

    @staticmethod
    def _resample_data(st, desired_rate=20.0):
        # Set no filter to False to perform automatic anti aliasing
        st.resample(sampling_rate=desired_rate, no_filter=False)

        return st

    def _avg_data(self, arr):
        """
        Average out n consecutive data points to get fewer samples while still retaining info
        Args:
            arr (numpy array): Numpy array containing the seismic data

        Returns (numpy array): Averaged out numpy array of length = excerpt len

        """

        num_of_samples = arr.shape[0]
        factor = int(num_of_samples / self.excerpt_len)

        assert num_of_samples % self.excerpt_len == 0, "Error: Number of samples not perfectly " \
                                                       "divisible"
        arr = np.mean(arr.reshape(-1, factor), axis=1)

        return arr

    def _load_data(self, file_name, training_folder, folder_type="trimmed_data", avg=False):
        """
        Function to load data from text file
        Args:
            file_name (str): Path to text file from which contains info on data. Assumption:
                           Lines in file  are of the form: Time_stamp network station component label
            training_folder (str): Folder name which contains the actual training data
            folder_type (str): Specify what kind of data to load (processed_data, raw_data, audio,
                               plots). Default = trimmed_data
            avg (bool): If set to true, data is averaged to len = excerpt len

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

                    # Re-sample the data
                    st1 = self._resample_data(st1)
                    st2 = self._resample_data(st2)
                    st3 = self._resample_data(st3)

                    if avg and label_dict[file[1]] == 0:
                        st1[0].data = self._avg_data(st1[0].data)
                        st2[0].data = self._avg_data(st2[0].data)
                        st3[0].data = self._avg_data(st3[0].data)

                    X.append([st1[0].data, st2[0].data, st3[0].data])
                    X_names.append(file[0])
                    Y.append(label_dict[file[1]])
                else:
                    # Warn users if some file is not found
                    warnings.warn("File not found: {}".format(file[0]))

        return X, Y, X_names

    def _load_data_from_folder(self, training_folder, folder_type, data_type):
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
            data_type (str): Can be 'earthquake' or 'tremor'

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
                                if file[:2] != '._':   # To avoid meta files in tremor dataset
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

                                # Re-sample the data
                                st1 = self._resample_data(st1)
                                st2 = self._resample_data(st2)
                                st3 = self._resample_data(st3)

                                X.append([st1[0].data, st2[0].data, st3[0].data])
                                X_names.append(file)
                                Y.append(folder_labels[data_type][folder_type])

        return X, Y, X_names

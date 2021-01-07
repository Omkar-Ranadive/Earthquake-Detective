from torch.utils.data import Dataset
import numpy as np
from utils import generate_file_name_from_labels
from constants import META_PATH, DATA_PATH, label_dict, folder_labels
from obspy import read
import os
import warnings
from kymatio.numpy import Scattering1D
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class QuakeDataSet(Dataset):
    def __init__(self, ld_files, ld_folders, ld_unlabeled, excerpt_len, transforms=[],
                 mode='train'):
        """
        Args:
            ld_files (list): A list of dictionaries where the dictionary contains parameters for
                            _load_data func
            ld_folders (list): List of dictionaries where each dictionary contains parameters
                            for _load_data_from_folder func
            ld_unlabeled (list) List of dictionaries where each dict contains params for
                            _load_unlabeled func
            excerpt_len (int): Length of each trace in final training data. Each signal will be
                                padded/trimmed to this len.
            transforms (list): A list of strings listing transforms to apply.
                               Supported transforms:
                               'wavelet' - Applies wavelet scattering transform
            mode (str): mode can be 'train' or any other str. For train, random offset sampling
                        is performed.
        """
        self.X, self.Y, self.X_names, self.X_users, self.X_ids, self.X_imgs = [], [], [], [], [], []
        self.excerpt_len = excerpt_len
        self.mode = mode
        self.transforms = transforms
        self.avg = False

        if ld_files is not None:
            for ld_file in ld_files:
                x, y, x_names, x_ids, x_users, x_imgs = self._load_data(**ld_file)
                self.X.extend(x)
                self.Y.extend(y)
                self.X_names.extend(x_names)
                self.X_ids.extend(x_ids)
                self.X_users.extend(x_users)
                if not x_imgs:
                    self.X_imgs.extend(len(y)*[-1])
                else:
                    self.X_imgs.extend(x_imgs)

        if ld_folders is not None:
            for ld_folder in ld_folders:
                x, y, x_names, x_imgs = self._load_data_from_folder(**ld_folder)
                self.X.extend(x)
                self.Y.extend(y)
                self.X_names.extend(x_names)
                # In case of folders, we don't keep track of subject ids / users
                # As folders are different data sources (not from Zooniverse)
                self.X_ids.extend(len(y)*[-1])
                self.X_users.extend(len(y)*[-1])
                if not x_imgs:
                    self.X_imgs.extend(len(y)*[-1])
                else:
                    self.X_imgs.extend(x_imgs)

        if ld_unlabeled is not None:
            for func_params in ld_unlabeled:
                x, y, x_names, x_imgs = self._load_unlabeled(**func_params)
                self.X.extend(x)
                self.Y.extend(y)
                self.X_names.extend(x_names)
                self.X_ids.extend(len(y) * [-1])
                self.X_users.extend(len(y) * [-1])
                if not x_imgs:
                    self.X_imgs.extend(len(y) * [-1])
                else:
                    self.X_imgs.extend(x_imgs)

        # Convert to numpy
        self.Y = np.array(self.Y, dtype='int64')
        self.X_ids = np.array(self.X_ids, dtype='int64')
        self.X_users = np.array(self.X_users, dtype='int64')

        # Print data distribution
        self.get_distribution()

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
            if self.avg:
                feature = self._avg_data(feature)
            transformed_data = self._pad_data(feature)
            t_item.append(transformed_data)

        t_item = np.array(t_item, dtype='float32')

        # Apply any specified transforms - Currently only supports wavelet transform
        # Note: Too slow, so avoid using inside dataset class
        for transform in self.transforms:
            if transform == 'wavelet':
                t_item = self.scattering(t_item)

        return {'data': t_item, 'label': self.Y[item], 'sub_id': self.X_ids[item],
                'user': self.X_users[item], 'img': np.array(self.X_imgs[item], dtype='float32'),
                'index': item}

    def __len__(self):
        return len(self.X)

    def get_indices_split(self, train_percent, seed=False, users=[]):
        """
        Function to return train_indices and test_indices. This can be used to break this dataset
        into train and test sets. Indices splits are calculated in proportion to the class
        distribution.

        Args:
            train_percent (float): Percentage of samples to include in the training set
            seed (bool): If true, same indices are produced everything for reproducibility
            users (list): If only data from specific users is required, it can be specified by
            giving a list of user ids

        Returns (list): Training indices and testing indices
        """

        # Filter the data by list of users if specified
        if users:
            user_indices = [-1]  # By default, all data from folders represented by
            # -1 will be considered
            user_indices.extend(users)

            # Get the samples classified by users in user_indices
            indices_all = []
            for user_index in user_indices:
                u_indices = np.where(self.X_users == user_index)[0]
                indices_all.extend(u_indices)
            indices_all = np.array(indices_all)
            total_samples = indices_all.shape[0]
        else:
            total_samples = self.Y.shape[0]
            indices_all = np.array([])

        num_train = int(train_percent * total_samples)
        num_test = total_samples - num_train
        train_indices = []
        test_indices = []
        if seed:
            np.random.seed(0)
        # Get indices for all classes
        for k, v in label_dict.items():
            indices = np.where(self.Y == v)[0]
            # Select user specific indices, if user names are given
            if indices_all.size != 0:
                indices = np.intersect1d(indices, indices_all)
            # Shuffle the indices
            np.random.shuffle(indices)
            # Find what percentage of the total samples are k
            percent_of_total = len(indices)/total_samples
            # Include the same percentage in test set
            num_in_test = int(percent_of_total * num_test)
            test_indices.extend(indices[:num_in_test])
            train_indices.extend(indices[num_in_test:])

        return train_indices, test_indices

    def get_distribution(self):
        # Print info on class distribution
        print("Total training examples: {}".format(len(self.Y)))
        print("-------------------------------------------")
        print("Class distribution: ")
        for k, v in label_dict.items():
            print("Number of {} examples: {}".format(k, np.count_nonzero(self.Y == v)))
        print("-------------------------------------------")

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

    @staticmethod
    def _process_data(st):
        st.detrend('demean')
        st.filter('bandpass', freqmin=2, freqmax=8, corners=4, zerophase=True)
        return st

    def modify_excerpt_len(self, new_len):
        self.excerpt_len = new_len

    def modify_flags(self, avg_flag):
        self.avg = avg_flag

    def _avg_data(self, arr):
        """
        Average out n consecutive data points to get fewer samples while still retaining info
        Args:
            arr (numpy array): Numpy array containing the seismic data

        Returns (numpy array): Averaged out numpy array of length = excerpt len

        """

        num_of_samples = arr.shape[0]
        factor = int(num_of_samples / self.excerpt_len)
        # assert num_of_samples % self.excerpt_len == 0, "Error: Number of samples not perfectly " \
        #                                                "divisible"

        if factor > 1 and num_of_samples % self.excerpt_len == 0:
            arr = np.mean(arr.reshape(-1, factor), axis=1)

        return arr

    @staticmethod
    def _gen_plot(tr, save_path):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(tr.times("matplotlib"), tr.data, "b-")
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(plt.gcf())  # Close the figure to release memory

    @staticmethod
    def _process_image(img, crop):
        # Convert to gray scale
        new_img = img.convert(mode="1", dither=Image.NONE)

        # Crop the time axis and boundaries for images from EQ detective
        if crop:
            width, height = new_img.size
            new_img = new_img.crop((80, 0, width, height - 275))

        # Resize the image
        new_img = new_img.resize((320, 200))
        new_img = np.asarray(new_img)

        return new_img

    def _load_data(self, file_name, training_folder, folder_type="trimmed_data", avg=False,
                   load_img=False):
        """
        Function to load data from text file
        Args:
            file_name (str): Path to text file from which contains info on data. Assumption:
                           Lines in file  are of the form: Time_stamp network station component label
            training_folder (str): Folder name which contains the actual training data
            folder_type (str): Specify what kind of data to load (processed_data, raw_data, audio,
                               plots). Default = trimmed_data
            avg (bool): If set to true, data is averaged to len = excerpt len
            load_img (bool): Also loads plots if set to true

        Returns (np arrays): Five arrays X and Y where X = training data, Y = training labels and
                            X_names = file name associated with the data in X, X_ids = subject
                            ids, X_users = user ids
        """
        fl_map = generate_file_name_from_labels(file_name)
        X, Y, X_names, X_ids, X_users, X_imgs = [], [], [], [], [], []
        train_path = DATA_PATH / training_folder
        training_folder = Path(training_folder)
        print("Loading data from folder {}".format(training_folder))
        for folder, files in fl_map.items():
            folder_path = train_path / folder / folder_type
            img_path = train_path / folder / 'plots'
            file_name = training_folder / folder / folder_type
            for file in files:
                # File is a list of following form = [file_name, label, sub_id, user]
                # Also, load the BHE and BHN components along with the Z component
                # NOTE: Its assumed that the labelled data only contains BHZ components. We are
                # assuming the labels of other components to be the same
                f_bhe = file[0].replace('BHZ', 'BHE')
                f_bhn = file[0].replace('BHZ', 'BHN')
                file_path_z = str(folder_path / (file[0] + '.sac'))
                file_path_e = str(folder_path / (f_bhe + '.sac'))
                file_path_n = str(folder_path / (f_bhn + '.sac'))
                img_path_z = str(img_path / (file[0] + '.png'))
                img_path_e = str(img_path / (f_bhe + '.png'))
                img_path_n = str(img_path / (f_bhn + '.png'))

                if os.path.exists(file_path_z) and os.path.exists(file_path_e) and os.path.exists(
                        file_path_n):
                    st1 = read(file_path_z)
                    st2 = read(file_path_e)
                    st3 = read(file_path_n)

                    if load_img:
                        # We are assuming that the images exist in case of EQ data
                        # If not, check the data_utils gen_plots function and generate them
                        # manually
                        img_z = Image.open(img_path_z)
                        img_e = Image.open(img_path_e)
                        img_n = Image.open(img_path_n)

                        # Process the images
                        img_z = self._process_image(img_z, crop=True)
                        img_e = self._process_image(img_e, crop=True)
                        img_n = self._process_image(img_n, crop=True)

                        X_imgs.append([img_z, img_e, img_n])

                    # Re-sample the data
                    st1 = self._resample_data(st1)
                    st2 = self._resample_data(st2)
                    st3 = self._resample_data(st3)

                    if avg and label_dict[file[1]] == 0:
                        st1[0].data = self._avg_data(st1[0].data)
                        st2[0].data = self._avg_data(st2[0].data)
                        st3[0].data = self._avg_data(st3[0].data)

                    X.append([st1[0].data, st2[0].data, st3[0].data])
                    X_names.append(str(file_name / (file[0] + '.sac')))
                    X_ids.append(int(file[2]))
                    X_users.append(int(file[3]))
                    Y.append(label_dict[file[1]])

                else:
                    # Warn users if some file is not found
                    # warnings.warn("File not found: {}".format(file[0]))
                    pass

        print("Number of samples loaded: {}".format(len(Y)))

        return X, Y, X_names, X_ids, X_users, X_imgs

    def _load_data_from_folder(self, training_folder, folder_type, data_type, load_img=False,
                               process_data=False):
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
            load_img (bool): If set to true, plots are also loaded. Plots will be created and
            then loaded if they don't exist.
            process_data (bool): If true, data is de-trended and bandpassed

       Returns (np arrays): Five arrays X and Y where X = training data, Y = training labels and
                            X_names = file name associated with the data in X, X_ids = subject
                            ids, X_users = user ids
        """

        print("Loading data from folder {}".format(training_folder))
        X, Y, X_names, X_imgs = [], [], [], []
        train_path = DATA_PATH / training_folder
        training_folder = Path(training_folder)
        for folder in os.listdir(train_path):
            folder_path = train_path / folder
            file_name = training_folder / folder
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

                                if process_data:
                                    st1 = self._process_data(st1)
                                    st2 = self._process_data(st2)
                                    st3 = self._process_data(st3)

                                if load_img:
                                    img_path_z = file_path_z[:-3] + '.png'
                                    img_path_e = file_path_e[:-3] + '.png'
                                    img_path_n = file_path_n[:-3] + '.png'

                                    # If plot doesn't exist, create them first
                                    if not os.path.exists(img_path_z):
                                        self._gen_plot(st1[0], img_path_z)
                                        self._gen_plot(st2[0], img_path_e)
                                        self._gen_plot(st3[0], img_path_n)

                                    img_z = Image.open(img_path_z)
                                    img_e = Image.open(img_path_e)
                                    img_n = Image.open(img_path_n)

                                    # Process the images
                                    img_z = self._process_image(img_z, crop=False)
                                    img_e = self._process_image(img_e, crop=False)
                                    img_n = self._process_image(img_n, crop=False)

                                    X_imgs.append([img_z, img_e, img_n])

                                X.append([st1[0].data, st2[0].data, st3[0].data])

                                X_names.append(str(file_name / inner_folder / (file + 'BHZ' +
                                                                             '.SAC')))
                                Y.append(folder_labels[data_type][folder_type])

        print("Number of samples loaded: {}".format(len(Y)))
        return X, Y, X_names, X_imgs

    def _load_unlabeled(self, folder_path, load_img=False, process_data=False, resample=False):
        """
        Function to load unlabeled data.
        Assume that the folder_path is the top level folder and then processed_data will be
        accessed to get the .sac data and plots folder will be accessed to get the images
        Args:
            folder_path (str): Path to top level folder
            load_img (bool): Load the images along with the signal data
            process_data (bool): Process data if true
            resample (bool) If true, resample the data

        Returns:

        """

        print("Loading data from folder {}".format(folder_path))
        X, Y, X_names, X_imgs = [], [], [], []
        folder = Path(folder_path)

        sac_path = folder / 'processed_data'
        img_path = folder / 'plots'

        for file in os.listdir(sac_path):
            # To avoid repeat of the same data, only choose file with BHZ in it
            if 'BHZ' in file:
                f_bhe = file.replace('BHZ', 'BHE')
                f_bhn = file.replace('BHZ', 'BHN')
                file_path_z = str(sac_path / file)
                file_path_e = str(sac_path / f_bhe)
                file_path_n = str(sac_path / f_bhn)
                img_path_z = str(img_path / (file[:-3] + 'png'))
                img_path_e = str(img_path / (f_bhe[:-3] + 'png'))
                img_path_n = str(img_path / (f_bhn[:-3] + 'png'))

                if os.path.exists(file_path_z) and os.path.exists(file_path_e) and os.path.exists(
                        file_path_n):
                    st1 = read(file_path_z)
                    st2 = read(file_path_e)
                    st3 = read(file_path_n)

                    if load_img:
                        # We are assuming that the images exist in case of EQ data
                        # If not, check the data_utils gen_plots function and generate them
                        # manually
                        img_z = Image.open(img_path_z)
                        img_e = Image.open(img_path_e)
                        img_n = Image.open(img_path_n)

                        # Process the images
                        img_z = self._process_image(img_z, crop=True)
                        img_e = self._process_image(img_e, crop=True)
                        img_n = self._process_image(img_n, crop=True)

                        X_imgs.append([img_z, img_e, img_n])

                    # Re-sample the data
                    if resample:
                        st1 = self._resample_data(st1)
                        st2 = self._resample_data(st2)
                        st3 = self._resample_data(st3)

                    X.append([st1[0].data, st2[0].data, st3[0].data])
                    X_names.append(file_path_z)
                    Y.append(-1)  # We don't care about the label, assign -1

                else:
                    # Warn users if some file is not found
                    warnings.warn("File not found: {}".format(file))
                    pass

        print("Number of samples loaded: {}".format(len(Y)))

        return X, Y, X_names, X_imgs


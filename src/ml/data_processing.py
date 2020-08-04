"""
Deprecated file: Just keeping it, separate loading into numpy arrays is needed in future
"""


from src.utils import generate_file_name_from_labels
from constants import DATA_PATH, label_dict, folder_labels
from obspy import read
import os
import warnings
import numpy as np


def load_data(file_name, training_folder, folder_type="trimmed_data"):
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
            file_path_z = str(folder_path / (file[0]+'.sac'))
            file_path_e = str(folder_path / (f_bhe+'.sac'))
            file_path_n = str(folder_path / (f_bhn+'.sac'))

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

    return np.array(X), np.array(Y, dtype='int64'), X_names


def load_data_from_folder(training_folder, folder_type):
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
                            files.append(file[:-7])   # Hence remove last 7 chars

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

    return np.array(X), np.array(Y, dtype='int64'), X_names


if __name__ == '__main__':
    X, Y, X_names = load_data_from_folder(training_folder='Training_Set_Prem',
                                           folder_type='positive')


    # X, Y, X_names = load_data('../../data/V_golden.txt', training_folder='Training_Set_Vivian')

    print(X.shape)
    for i in range(X.shape[0]):
        print(X[i][0].shape[-1], X[i][1].shape, X[i][2].shape)
    # X = np.expand_dims(X, axis=2)
    # print(X.shape)



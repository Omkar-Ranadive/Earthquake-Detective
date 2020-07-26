from src.utils import generate_file_name_from_labels
from constants import DATA_PATH, label_dict
from obspy import read
import os
import warnings
import numpy as np


def load_data(file_name, training_folder, folder_type="trimmed_data"):
    """

    Args:
        file_name (str): Path to text file from which contains info on data. Assumption:
                       Lines in file  are of the form: Time_stamp network station component label
        training_folder (str): Folder name which contains the actual training data
        folder_type (str): Specify what kind of data to load (processed_data, raw_data, audio,
                           plots). Default = trimmed_data

    Returns (list): Two lists X and Y where X = training data, Y = training labels

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

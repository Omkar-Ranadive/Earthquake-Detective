from src.utils import generate_file_name_from_labels
from constants import DATA_PATH
from obspy import read
import os
import warnings


def load_data(file_name, training_folder, folder_type="processed_data"):
    """

    Args:
        file_name (str): Path to text file from which contains info on data. Assumption:
                       Lines in file  are of the form: Time_stamp network station component label
        training_folder (str): Folder name which contains the actual training data
        folder_type (str): Specify what kind of data to load (processed_data, raw_data, audio,
                           plots). Default = processed_data

    Returns (list): Two lists X and Y where X = training data, Y = training labels

    """
    fl_map = generate_file_name_from_labels(file_name)
    X, Y = [], []
    train_path = DATA_PATH / training_folder

    for folder, files in fl_map.items():
        folder_path = train_path / folder / folder_type
        for file in files:
            # File is a list of following form = [file_name, label]
            file_path = str(folder_path / (file[0]+'.sac'))
            if os.path.exists(file_path):
                st = read(file_path)
                X.append(st[0])
                Y.append(file[1])
            else:
                # Warn users if some file is not found
                warnings.warn("File not found: {}".format(file[0]))

    return X, Y


X, Y = load_data('../../data/V_golden.txt', training_folder='Training_Set_Vivian')
for sample in X:
    print(sample.data.shape)
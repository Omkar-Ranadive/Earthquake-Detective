from collections import defaultdict
import pickle
from constants import DATA_PATH
import pandas as pd
import json


def clean_event_id(event_id):
    """
    Clean the event id by replacing few characters with underscore. Makes it easier to save.
    Args:
        event_id (str): Ideally should be time stamp YYYY-MM-DDTHH:MM:SS.000

    Returns (str): Cleaned event id

    """
    # Replace '.' and '-' in event_id before saving
    char_to_replace = ['.', '-', ':']
    event_id_new = event_id
    for char in char_to_replace:
        event_id_new = event_id_new.replace(char, "_")

    return event_id_new


def generate_file_name_from_labels(file_name):
    """
    Assumption: Labels are of the form: Time_stamp network station component label
    Args:
        file_name (str): Path to the file from which names are to be generated

    Returns (dict): List of file names which are compatible with the rest of the project
                    Format: {Folder name (event_id): [[file_name1, label], [file_name2, label] ..]}

    """
    file_label_dict = defaultdict(list)  # Key:Value pair of file_name: label
    with open(file_name) as f:
        for line in f.readlines():
            location = ""   # Empty location is assumed, change accordingly if required
            info = line.split()

            if len(info) == 5:
                event_id, network, station, channel, label = info
            else:
                continue
                # # Label = Unclear event. Space makes it length 6
                # label = "_".join(info[-2: ])
                # event_id, network, station, channel = info[:-2]

            # Clean the event id
            event_id = clean_event_id(event_id)

            file_name = "_".join((network, station, location, channel, event_id))

            file_label_dict[event_id].append([file_name, label])

    return file_label_dict


def extract_info_zooniverse(subjects, classfications, user_names=['suzanv']):
    """
    Extract the time, region, station, channel and classification from .csv files downloaded from
    Zooniverse. The extracted data is saved in a text file.

    Args:
        subjects (str): Path to .csv file containing subjects (seismic file) info
        classifications (str): Path to .csv file containing classifications done by Citizen
        Scientists.
        user_names (list): List of user names whose info is to be extracted

    """

    df_sub = pd.read_csv(DATA_PATH / subjects)
    df_class = pd.read_csv(DATA_PATH / classfications)
    subject_info = {}

    # Create a mapping of subject_id -> time_stamp region station channel
    for index, row in df_sub.iterrows():
        sub_id = row['subject_id']
        meta = json.loads(row['metadata'])
        file_name = meta['!image_name'].split('_')
        '''
        file_name[0] = Region 
        file_name[1] = Station code 
        file_name[2] = location 
        file_name[3] = channel 
        file_name[4] = year 
        file_name[5] = month 
        file_name[6] = day 
        file_name[7] = hours 
        file_name[8] = minutes 
        file_name[9] = seconds 
        file_name[10] = timestamp offset 
        '''
        # Ensure that the information is extracted only if the structure is correct
        if len(file_name) >= 11:
            time_stamp = "-".join(file_name[4:7]) + ":".join(file_name[7:10]) + '.' +\
                         file_name[10][:3]

            subject_info[sub_id] = [time_stamp, file_name[0], file_name[1], file_name[3]]

    # Filter the classifications based on the user_names list
    df_filt_class = df_class[df_class.user_name.isin(user_names)]

    with open(DATA_PATH / 'classification_data.txt', 'a') as f:
        for index, row in df_filt_class.iterrows():
            meta = json.loads(row['annotations'])
            sub_id = row['subject_ids']

            if sub_id in subject_info:
                label = meta[0]['value']
                info = " ".join((subject_info[sub_id])) + " " + label
                f.write(info + '\n')


def convert_to_seconds(val, t):
    """
    Convert a given unit of time to seconds
    Args:
        val (int): Value in number of years/days/minutes
        t (str): Specifies the type of value. It can be one of the following
                'y': Val denotes years
                'd': Val denotes days
                'h': Val denotes hours
                'm': Val denotes minutes
                's': Val denotes seconds

    Returns (int): Time in seconds
    """
    result = -1
    if t == 'y':
        result = val*365*24*3600
    elif t == 'd':
        result = val*24*3600
    elif t == 'h':
        result = val*3600
    elif t == 'm':
        result = val*60
    elif t == 's':
        result = val

    return result


def save_file(file, filename):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
    """
    with open(DATA_PATH / filename, 'wb') as f:
        pickle.dump(file, f)


def load_file(filename):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file

    Returns (Python obj): Returns the loaded pickle file

    """
    with open(DATA_PATH / filename, 'rb') as f:
        file = pickle.load(f)

    return file


if __name__ == '__main__':
    extract_info_zooniverse(classfications='earthquake-detective-classifications.csv',
                            subjects='earthquake-detective-subjects.csv',
                            user_names=['suzanv'])
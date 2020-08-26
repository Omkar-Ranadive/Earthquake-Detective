from collections import defaultdict
import pickle
from constants import DATA_PATH

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

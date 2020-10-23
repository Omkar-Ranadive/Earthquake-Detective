"""
Utility functions for data downloaded from Zooniverse
"""

import pandas as pd
import json
import functools
import operator
from constants import DATA_PATH, META_PATH
from collections import defaultdict, Counter
from datetime import datetime
from src.ml.retirement.r_utils import map_users_to_index


def get_subject_info(df_sub):
    """
    Args:
        df_sub (Pandas dataframe): Pandas dataframe containing
        earthquake-detective-subjects.csv file

    Returns (dictionary): Mapping of sub_id -> Time_Stamp Region_Code Station_Code Channel Label

    """
    subject_info = {}
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
        # If this is true then its a practice problem
        if '!Classification' in meta:
            subject_info[sub_id] = [meta['!image_name'][:-4], meta['!Classification']]
        # Ensure that the information is extracted only if the structure is correct
        elif len(file_name) >= 11:
            time_stamp = "-".join(file_name[4:7]) + ":".join(file_name[7:10]) + '.' + \
                         file_name[10][:3]

            subject_info[sub_id] = [time_stamp, file_name[0], file_name[1], file_name[3]]

    return subject_info


def choose_golden_samples(subjects, classifications, sub_stats, golden_users=['Vivitang',
                                                                              'suzanv']):
    """
    Args:
      subjects (str): Path to .csv file containing subjects (seismic file) info
      classifications (str): Path to .csv file containing classifications done by Citizen
      Scientists.
      sub_stats (str): Path to .txt file containing subject statistics (meta data on data samples)
      golden_users (list): List of users who will be used to score other users
    """

    df_sub = pd.read_csv(DATA_PATH / subjects)
    subject_info = get_subject_info(df_sub)
    df_class = pd.read_csv(DATA_PATH / classifications)
    # Filter out rows where users haven't logged in
    # Filter class to only contain golden users info
    n_samples = 200  # Number of samples to consider for the golden set
    df_class = df_class[df_class.user_name.isin(golden_users)]
    golden_ids = []

    with open(META_PATH / sub_stats, 'r') as f:
        counter = 0
        for line in f.readlines():
            info = line.split(" ")
            # Don't consider practice samples, hence len(line) > 4
            if len(info) > 4 and counter < n_samples:
                golden_ids.append(info[0])
                counter += 1

    print("Total golden samples: ", len(golden_ids))
    # From the set of golden ids, find if they are already classified by the golden users
    df_done = df_class[df_class.subject_ids.isin(golden_ids)]
    print("Golden samples already classified by golden users: ", len(df_done))
    remaining_ids = set(map(int, golden_ids)) - set(df_done['subject_ids'])
    print("Golden samples left to be classified by golden users: ", len(remaining_ids))

    # For the golden samples which are already classified, create a text file with sub_info labels
    with open(DATA_PATH / 'Golden' / 'golden_classified.txt', 'w') as f:
        for index, row in df_done.iterrows():
            meta = json.loads(row['annotations'])
            sub_id = row['subject_ids']

            if sub_id in subject_info:
                label = meta[0]['value']
                info = str(sub_id) + " " + " ".join(subject_info[sub_id]) + " " + label
                f.write(info + '\n')

    # For the rest, create a file with without the labels
    with open(DATA_PATH / 'Golden' / 'golden_to_be_classified.txt', 'w') as f:
        for sub_id in remaining_ids:
            if sub_id in subject_info:
                info = str(sub_id) + " " + " ".join(subject_info[sub_id])
                f.write(info + '\n')


def generate_stats(subjects, classifications):
    """
    Generate various statistics from the Zooniverse data. Data is saved in meta folder.
    Args:
      subjects (str): Path to .csv file containing subjects (seismic file) info
      classifications (str): Path to .csv file containing classifications done by Citizen
        Scientists.
    """

    df_sub = pd.read_csv(DATA_PATH / subjects)
    subject_info = get_subject_info(df_sub)
    df_class = pd.read_csv(DATA_PATH / classifications)
    # Filter out rows where users haven't logged in
    df_class = df_class[~df_class['user_name'].str.contains('not-logged-in')]
    df_common = df_class[df_class.duplicated(['subject_ids'], keep=False)]
    # Arrange according to sub_id count
    subs_count = Counter(df_common['subject_ids'])
    users_count = len(set(df_class['user_name']))
    cur_time = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    users_class_count = Counter(df_class['user_name'])
    #
    # Start saving the content to file
    with open(META_PATH / 'general_stats_{}.txt'.format(cur_time), 'w') as f:

        f.write('Total number of user classifications: {}\n'.format(len(df_class)))
        f.write("Total number of users: {}\n".format(users_count))
        # Save list of users and their classification count separately
        with open(META_PATH / 'stats_users_{}.txt'.format(cur_time), 'w') as f2:
            f2.write("Users\t Classification Count\tUnique Classification Count\n")
            counter = 0
            for user, count in sorted(users_class_count.items(), key=lambda x: x[1], reverse=True):
                df_user = df_class[df_class.user_name.isin([user])]
                unique_count = len(set(df_user['subject_ids']))
                f2.write("{} {} {}\n".format(user, count, unique_count))
                if unique_count > 100:
                    counter += 1

        f.write("Number of users with 100+ unique classifications {}\n".format(counter))

        # Subject_IDs with common classifications
        f.write("Total number of unique data samples: {}\n".format(len(set(df_sub['subject_id']))))

        with open(META_PATH / 'stats_subs{}.txt'.format(cur_time), 'w') as f2:
            f2.write("Subject ID\tCount\tInfo\n")
            counter = 0
            for sub_id, count in sorted(subs_count.items(), key=lambda x: x[1], reverse=True):
                if sub_id in subject_info:
                    f2.write("{} {} {}\n".format(sub_id, count, " ".join(subject_info[sub_id])))
                    if count > 100:
                        counter += 1

        f.write("Subjects which were classified 100+ times by the different users {}\n".format(
            counter))


def extract_info_zooniverse(subjects, classifications, user_names=['suzanv'], regions=[]):
    """
    Extract the time, region, station, channel and classification from .csv files downloaded from
    Zooniverse. The extracted data is saved in a text file.
    Each line of text file consists of the following:
    Time_Stamp Region_Code Station_Code Channel Label
    Example: 2018-01-10T02:51:31.000 AK SII BHZ Noise

    Args:
        subjects (str): Path to .csv file containing subjects (seismic file) info
        classifications (str): Path to .csv file containing classifications done by Citizen
        Scientists.
        user_names (list): List of user names whose info is to be extracted
        regions (list): If not empty, only info from the specified regions will be saved to file
    """

    df_sub = pd.read_csv(DATA_PATH / subjects)
    df_class = pd.read_csv(DATA_PATH / classifications)

    # Create a mapping of subject_id -> time_stamp region station channel
    subject_info = get_subject_info(df_sub)

    # Filter the classifications based on the user_names list
    df_filt_class = df_class[df_class.user_name.isin(user_names)]
    # Drop duplicate subject_ids
    df_filt_class = df_filt_class.drop_duplicates(subset=['subject_ids'], keep='first')

    f_name = 'classification_data_{}.txt'.format("_".join(user_names))
    # Save the content to text file
    with open(DATA_PATH / f_name, 'w') as f:
        for index, row in df_filt_class.iterrows():
            meta = json.loads(row['annotations'])
            sub_id = row['subject_ids']

            # The len() > 2 constraint ensures we don't save practice problems
            if sub_id in subject_info and len(subject_info[sub_id]) > 2:
                if regions:
                    # Only save if the subject id is part of the regions list
                    if subject_info[sub_id][1] in regions:
                        label = meta[0]['value']
                        info = str(sub_id) + " " + row['user_name'] + " " + " ".join((subject_info[
                                         sub_id])) + " " + label
                        f.write(info + '\n')
                else:
                    label = meta[0]['value']
                    info = str(sub_id) + " " + row['user_name'] + " " + " ".join((subject_info[
                                         sub_id])) + " " + label
                    f.write(info + '\n')


def extract_info_zooniverse_anon(subjects, classifications, user_names=[], regions=[]):
    """
    Extract the time, region, station, channel and classification from .csv files downloaded from
    Zooniverse. The extracted data is saved in a text file.
    Each line of text file consists of the following:
    Time_Stamp Region_Code Station_Code Channel Label
    Example: 2018-01-10T02:51:31.000 AK SII BHZ Noise

    Args:
        subjects (str): Path to .csv file containing subjects (seismic file) info
        classifications (str): Path to .csv file containing classifications done by Citizen
        Scientists.
        user_names (list): List of user names whose info is to be extracted
        regions (list): If not empty, only info from the specified regions will be saved to file
    """

    df_sub = pd.read_csv(DATA_PATH / subjects)
    df_class = pd.read_csv(DATA_PATH / classifications)
    df_class = df_class[~df_class['user_name'].str.contains('not-logged-in')]
    user_to_index = map_users_to_index(user_stats='stats_users_12_09_2020-20_10_24.txt')
    # Create a mapping of subject_id -> time_stamp region station channel
    subject_info = get_subject_info(df_sub)

    # Filter the classifications based on the user_names list
    if user_names:
        df_class = df_class[df_class.user_name.isin(user_names)]
        # Drop duplicate subject_ids
        df_class = df_class.drop_duplicates(subset=['subject_ids'], keep='first')

    f_name = 'classification_data_{}.txt'.format("all_users")
    # Save the content to text file
    with open(DATA_PATH / f_name, 'w') as f:
        for index, row in df_class.iterrows():
            meta = json.loads(row['annotations'])
            sub_id = row['subject_ids']

            # The len() > 2 constraint ensures we don't save practice problems
            if sub_id in subject_info and len(subject_info[sub_id]) > 2:
                if regions:
                    # Only save if the subject id is part of the regions list
                    if subject_info[sub_id][1] in regions:
                        label = meta[0]['value']
                        info = str(sub_id) + " " + str(user_to_index[row['user_name']]) + " " + \
                               " " \
                                                                                          "".join((
                            subject_info[
                                         sub_id])) + " " + label
                        f.write(info + '\n')
                else:
                    label = meta[0]['value']
                    info = str(sub_id) + " " + str(user_to_index[row['user_name']]) + " " + " " \
                                                                                          "".join((
                        subject_info[
                                         sub_id])) + " " + label
                    f.write(info + '\n')


def compare_classifications(subjects, classifications, user_names):
    """
    Compare the classifications done by different users
    Args:
    subjects (str): Path to .csv file containing subjects (seismic file) info
    classifications (str): Path to .csv file containing classifications done by Citizen
    Scientists.
    user_names (list): List of user names whose info is to be extracted
    """

    assert len(user_names) >= 2, "At least 2 users are needed for comparison"

    df_sub = pd.read_csv(DATA_PATH / subjects)
    df_class = pd.read_csv(DATA_PATH / classifications)

    # Filter the classifications file based on the user_names list
    df_filt_class = df_class[df_class.user_name.isin(user_names)]

    # Select the common rows
    df_common = df_filt_class[df_filt_class.duplicated(['subject_ids'], keep=False)]
    subject_info = defaultdict(dict)

    for index, row in df_common.iterrows():
        meta = json.loads(row['annotations'])
        sub_id = row['subject_ids']
        label = meta[0]['value']
        # Add info to subject info
        if label in subject_info[sub_id]:
            subject_info[sub_id][label].append(row['user_name'])
        else:
            subject_info[sub_id][label] = [row['user_name']]

    # Count the similarity between the classifications of user
    diff = 0
    total = 0
    same = 0
    total_users = len(user_names)
    for sub_id, classes in subject_info.items():
        users = functools.reduce(operator.iconcat, list(classes.values()), [])
        if len(set(users)) == total_users:
            total += 1
            if len(classes.keys()) > 1:
                diff += 1
                print(sub_id, classes)
            else:
                same += 1

    print("Total: {}, Diff {}, Same {}".format(total, diff, same))


if __name__ == '__main__':
    # extract_info_zooniverse(classifications='earthquake-detective-classifications.csv',
    #                        subjects='earthquake-detective-subjects.csv',
    #                        user_names=['suzanv'])

    extract_info_zooniverse_anon(classifications='earthquake-detective-classifications.csv',
                            subjects='earthquake-detective-subjects.csv')

    # compare_classifications(subjects='earthquake-detective-subjects.csv',
    #                         classifications='earthquake-detective-classifications.csv',
    #                         user_names=['Vivitang', 'suzanv'])


    # generate_stats(subjects='earthquake-detective-subjects.csv',
    #                          classifications='earthquake-detective-classifications.csv')

    # choose_golden_samples(subjects='earthquake-detective-subjects.csv',
    #                          classifications='earthquake-detective-classifications.csv',
    #                         sub_stats='stats_subs12_09_2020-20_10_24.txt')
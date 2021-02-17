"""
Functions to analyze the zooniverse data
"""

import pandas as pd
import json
from constants import DATA_PATH, META_PATH, label_dict, index_to_label
from collections import defaultdict, Counter
from src.ml.retirement.r_utils import map_users_to_index, map_index_to_users
from utils import load_file
import os
import numpy as np
from datetime import datetime


def extract_sub_ids(file, naive=False, label_of_interest=0):
    """
    Extracts sub_ids from dataset_x.txt files of the following format:
    SubjectID,Network,Station,Latitude,Longitude,PercentageEarthquakes,PercentageTremor,
    PercentageNoise,PercentageUnclearEvent

    Args:
        file (path): Path to file containing the dataset info
        naive (bool): If naive = True, apply naive selection based on threshold (no reliability
        score)
        label_of_interest (int): Only used if naive flag is set to true

    Returns (list): List containing extracted subject ids
    """

    sub_ids = []
    threshold = 70.0
    offset_dict = {0: 5, 1: 7, 2: 6, 3: 7}  # Map labels to indices according to headers
    label_index = offset_dict[label_of_interest]

    with open(file, 'r') as f:
        for index, line in enumerate(f.readlines()):
            # Don't consider the headers
            if index > 2:
                info = line.split(",")
                if not naive:
                    sub_ids.append(int(info[0]))
                elif naive and float(info[label_index]) > threshold:
                    sub_ids.append(int(info[0]))

    return sub_ids


def selection_criteria_1(users, label_of_interest):
    """
    Formula for Retirement/Selection score:
    x = sum_i=1_to_n (r_i) — sum_j=1_to_m (r_j).
    Where first summation contains reliability scores of users who have labeled it as the same
    as the label of interest, second summation contains reliability scores of users who have
    labeled it differently

    Args:
        users (list): List of users where each element is a tuple of the form (uid, ulabel,
        f1 score)
        label_of_interest (int): Label under consideration (left hand summation of formula)

    Returns (int): 1 = select the subject id, 0 = don't select
    """
    left_sum, right_sum = 0, 0
    threshold = 2.0

    for user in users:
        uid, ulabel, f1_score = user
        if ulabel == label_of_interest:
            left_sum += f1_score
        else:
            right_sum += f1_score

    if left_sum - right_sum >= threshold:
        return 1
    else:
        return 0


def selection_criteria_2(users, label_of_interest):
    """
    Select the subject id if the following criteria is met:
    If num_votes >= 4 and <= 5, agreement = 1.0
    if num_votes >= 6 and <= 7, agreement >= 0.8
    if num_votes >= 8 and <= 10, agreement >= 0.7


    Args:
        users (list): List of users where each element is a tuple of the form (uid, ulabel,
        f1 score)
        label_of_interest (int): Label under consideration (left hand summation of formula)

    Returns (int): 1 = select the subject id, 0 = don't select
    """
    total_votes = 0
    num_votes = 0
    for user in users:
        uid, ulabel, f1_score = user

        if ulabel == label_of_interest:
            num_votes += 1

        total_votes += 1

        agreement = num_votes/total_votes

        if 4 <= num_votes <= 5 and agreement >= 1.0:
            return 1
        elif 6 <= num_votes <= 7 and agreement >= 0.8:
            return 1
        elif 8 <= num_votes <= 10 and agreement >= 0.7:
            return 1

        #TODO  What if votes > 10 and agreement < threshold?

    return 0


def select_sub_ids(sub_ids, classifications, label_of_interest, criteria):

    sub_stats = gen_subids_stats(sub_ids, classifications)
    selected_ids = []
    for sub, users in sub_stats.items():
        out = criteria(users, label_of_interest=label_of_interest)

        if out == 1:
            selected_ids.append(int(sub))

    print("Selected {} ids out of {}".format(len(selected_ids), len(sub_stats)))

    return selected_ids


def gen_subids_stats(sub_ids, classifications):
    """
    Pass a list of subject ids and generate statistics w.r.t those ids
    Args:
    classifications (str): Path to .csv file containing classifications done by Citizen
    Scientists.
    sub_ids (list): List of subject ids to analyze

    Returns (dict): Dictionary of subject id -> List of all users who have labeled that sub id
    in the form (user_index, label, f1_score)
    """

    rel_scores = load_file(path=META_PATH, filename='rel_scores_v3')
    # index_to_users = map_index_to_users(user_stats='stats_users_12_09_2020-20_10_24.txt')
    mean_scores = {0: 0.78, 1: 0.74, 2: 0.4, 3: 0.58}

    users_to_index = map_users_to_index(user_stats='stats_users_12_09_2020-20_10_24.txt')

    df_class = pd.read_csv(DATA_PATH / classifications)

    # Remove users who haven't logged in
    df_class = df_class[~df_class['user_name'].str.contains('not-logged-in')]

    # Filter by subject ids
    df_class = df_class[df_class.subject_ids.isin(sub_ids)]
    # print(Counter(df_class['subject_ids']))
    #
    sub_stats = defaultdict(list)
    for index, row in df_class.iterrows():
        meta = json.loads(row['annotations'])
        label = meta[0]['value']
        label_in = label_dict[label]
        sub_id = row['subject_ids']
        user = row['user_name']
        uin = users_to_index[user]
        # print(label_in, uin)
        #
        # print(rel_scores[uin])
        f_score = rel_scores[uin][label_in][-1]
        # f_score = mean_scores[label_in] if np.isnan(f_score) else f_score
        f_score = 0.5 if np.isnan(f_score) else f_score

        sub_stats[sub_id].append((uin, label_in, f_score))

    return sub_stats


def compare_criterias(criterias, label_of_interest):
    """
     Selecting samples based on reliability thresholds

     Args:
         criterias (list): List of criterias (functions) to compare

     Returns:
         Saves comparision in a file
    """

    folder = DATA_PATH / 'EQ_Vivian_Analysis'
    all_ids = []
    save_folder = 'Results_{}'.format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))

    for file in os.listdir(folder):
        if not os.path.isdir(folder / file):
            print("-"*20)
            print("Processing {}".format(file))
            all_ids = extract_sub_ids(folder / file)
            selected_ids_list = []
            for index, criteria in enumerate(criterias):
                print("Running criteria: {}".format(index+1))
                selected_ids = select_sub_ids(all_ids,
                                              classifications='earthquake-detective-classifications.csv',
                                              label_of_interest=label_of_interest, criteria=criteria)

                selected_ids_list.append(selected_ids)

            # For comparison, also select them naively
            naive_ids = extract_sub_ids(folder / file, naive=True,
                                        label_of_interest=label_of_interest)
            # print(selected_ids)
            # print(naive_ids)
            new_file_name = file[:-3] + '_result_' + index_to_label[label_of_interest] + '.txt'

            if not os.path.exists(folder / save_folder):
                os.mkdir(folder / save_folder)

            with open(folder / save_folder / new_file_name, 'w') as f:
                for index, selected_ids in enumerate(selected_ids_list):
                    f.write("*"*15 + '\n')
                    print("*"*15)
                    f.write("Criteria {} \n".format(index+1))
                    f.write("Criteria {} : Selected {} ids out of {}\n".format(index+1, len(
                         selected_ids), len(all_ids)))

                    print("Naive: Selected {} ids out of {}".format(len(naive_ids), len(all_ids)))
                    f.write("Naive: Selected {} ids out of {}\n".format(len(naive_ids), len(all_ids)))

                    common_ids = set(selected_ids).intersection(set(naive_ids))
                    diff_ids = set(selected_ids).difference(set(naive_ids))

                    print("{} ids were selected in common with criteria {}".format(len(
                        common_ids), index+1))
                    f.write("{} ids were selected in common with critera {}\n".format(len(
                        common_ids), index+1))

                    # Save different ids to file
                    f.write("New selected ids are as follows: \n")
                    for nid in diff_ids:
                        f.write(str(nid) + '\n')

                    # Save common ids to file
                    f.write("The common ids are as follows: \n")
                    for cid in common_ids:
                        f.write(str(cid) + '\n')


if __name__ == '__main__':

    criterias = [selection_criteria_1, selection_criteria_2]
    compare_criterias(criterias=criterias, label_of_interest=2)
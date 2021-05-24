import sys
sys.path.append('../../src/')
from constants import DATA_PATH, META_PATH, n_classes, label_dict, index_to_label, metric_to_index
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from utils import save_file, load_file
import matplotlib.pyplot as plt


def calculate_reliability_mat(golden_samples, user_stats, classifications):
    """
    Calculates the reliability matrix for n_users. A class wise reliability score + an overall
    score is calculated for each user.
    Args:
        golden_samples (str): Path to golden_samples text file.
                Assumption: Each line in the text file is of the following form:
                Sub_ID UserID Time_Stamp Region_Code Sta_Code Channel Label
                Example: 28944230 0 2016-12-17T10:51:10.000 AK TNA BHZ Noise
        user_stats (str): Path to user stats file generated by
                zooniverse_utils.py/generate_stats func
        classifications (str): Path to .csv file downloaded from Zooniverse
    """
    df_class = pd.read_csv(DATA_PATH / classifications)
    n_users = -1   # If set to -1, all users will be considered
    user_names = []
    rel_dict = defaultdict(list)  # Mapping of user_names -> reliability in terms of accuracy
    # Reliability matrix has the following structure:
    golden_info = {}  # Mapping of subject id -> golden labels

    # Load the user names from the user stats text file
    with open(META_PATH / user_stats, 'r') as f:
        for index, line in enumerate(f.readlines()):
            # Skip the headers
            if index > 0:
                info = line.split()
                user_names.append(info[0])
    # Load the golden set info
    with open(DATA_PATH / golden_samples, 'r') as f:
        for line in f.readlines():
            info = line.split()
            golden_info[info[0]] = info[-1]  # Subject id -> labels

    # Trim the df_class set beforehand to optimize performance
    df_class = df_class[df_class.user_name.isin(user_names[:n_users])]
    subject_ids = list(map(int, golden_info.keys()))
    df_class = df_class[df_class.subject_ids.isin(subject_ids)]

    user_names = user_names[:n_users] if n_users != -1 else user_names
    n_users = len(user_names) if n_users == -1 else n_users
    rel_mat = np.zeros((n_users, (n_classes + 1), (n_classes + 1)))

    for index, user in enumerate(user_names):
        df_user = df_class[df_class['user_name'] == user]
        print("{} Processing user {}, num entries {}".format(index, user, len(df_user)))
        for sub_id, gold_label in golden_info.items():
            # Find what the user has labeled for the current sub_id
            row = df_user[df_user['subject_ids'] == int(sub_id)]
            if not row.empty:
                meta = json.loads(row['annotations'].values[0])
                label = meta[0]['value']
                l_index = label_dict[label]
                if l_index == label_dict[gold_label]:
                    rel_mat[index, l_index, l_index] += 1
                else:
                    rel_mat[index, label_dict[gold_label], l_index] += 1

        # Now, from the matrix, calculate the score for each label and overall score
        # Decide the beta to calculate F_beta score
        beta = 1.0  # beta < 1.0 -> more weight to precision / beta > 1.0 -> more weight to recall
        for c_id in range(n_classes+1):
            '''
            Rel Dict will contain the following info: 
            Index 0 to n_classes = [class wise precision (TP/(TP + FP)), 
                                      class wise recall (TP/(TP+FN)), 
                                      class wise accuracy = TP + TN / (all)
                                      f_beta score] 
            
            Precision - Proportion of positive samples which were actually positive 
            Recall - Proportion of correctly labeled positive samples out of total +ve samples 
         
            '''

            # Calculate the true +ve, -ve and false +ve, -ve
            tp = rel_mat[index, c_id, c_id]
            fn = np.sum(np.delete(rel_mat[index, c_id, :], c_id, axis=0))
            fp = np.sum(np.delete(rel_mat[index, :, c_id], c_id, axis=0))
            mask = np.ones((n_classes+1, n_classes+1), dtype=bool)
            mask[:, c_id] = False
            mask[c_id, :] = False
            tn = np.sum(rel_mat[index, :, :][mask])

            # Calculate precision, recall and acc
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            acc = (tp + tn)/(tp + tn + fp + fn)
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            rel_dict[index].append([precision, recall, acc, f_beta])

    # Set reliability of all gold users / clean data to 1
    golden_users = ['Vivitang', 'suzanv']
    # Find their index in user_names
    golden_indices = [-1]
    for user in golden_users:
        golden_indices.append(user_names.index(user))

    for i in golden_indices:
        if i not in rel_dict:
            for c_id in range(n_classes+1):
                rel_dict[i].append([1.0, 1.0, 1.0, 1.0])

    for index, row in enumerate(rel_mat):
        cur_user = user_names[index]
        print("User: {} \n {}".format(cur_user, row))
        print("Scores: ")
        for k, v in sorted(index_to_label.items()):
            print("Class: {} Precision: {}  Recall {} Accuracy {} F_Score {}"
                  .format(v, rel_dict[index][k][0], rel_dict[index][k][1], rel_dict[
                   index][k][2], rel_dict[index][k][3]))
        print("*"*50)

    save_file(path=META_PATH, file=rel_dict, filename="rel_scores_v3")


def map_users_to_index(user_stats):
    """
    Maps user_name strings to indices. Required for Pytorch processing. Can be used for user
    anonymization.
    Args:
        user_stats (str): Path to user stats file generated by
                zooniverse_utils.py/generate_stats func
    Returns (dict):  Mapping of user_name -> index

    """
    user_to_index = {}
    # Load the user names from the user stats text file
    with open(META_PATH / user_stats, 'r') as f:
        for index, line in enumerate(f.readlines()):
            # Skip the headers
            if index > 0:
                info = line.split()
                user_to_index[info[0]] = index-1

    return user_to_index


def map_index_to_users(user_stats):
    """
    Maps indices to users
    Args:
        user_stats (str): Path to user stats file generated by
                zooniverse_utils.py/generate_stats func
    Returns (dict):  Mapping of index -> users

    """
    index_to_user = {}
    # Load the user names from the user stats text file
    with open(META_PATH / user_stats, 'r') as f:
        for index, line in enumerate(f.readlines()):
            # Skip the headers
            if index > 0:
                info = line.split()
                index_to_user[index-1] = info[0]

    return index_to_user


def rel_stats(path, metric='f_beta', top=1.0):
    """

    Args:
        path (str): Path to the reliability scores dict
        metric (str): Specifies which metric to check for. Default = 'f_beta'.
            Other options:  'precision', 'recall', 'accuracy'
        top (float): Which percentage of users to consider. Default = 1.0; all users

    """
    rel_dict = load_file(path)
    rel_values = np.array(list(rel_dict.values()))
    index = metric_to_index[metric]

    # Get class-wise metric
    scores = []

    for i, label in sorted(index_to_label.items()):
        # TODO: Just getting rid of Nans for now, later when there aren't too many, handle it
        #  differently

        score = rel_values[:, i, index]
        total_nan = len(np.where(np.isnan(score))[0])
        print("For {}, Total values: {}, total nan values: {}".format(label, len(score),
                                                                      total_nan))

        score = score[~np.isnan(score)]  # Get rid of nan values
        scores.append((label, -np.sort(-score)))  # Sort and store (in reverse order)

    # Calculate results for the top n percent of users

    for label, score in scores:
        users = int(len(score)*top)  # Select top n%
        mean_score = np.mean(score[:users])

        print("Label: {}  Mean {} score for top {} ({}%) of users: {}".format(label, metric,
                                                                              users, top*100,
                                                                              mean_score))

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0, 0].hist(eq_scores_acc)
    # ax[0, 0].set_title("Earthquake")
    # ax[0, 1].hist(noise_scores_acc)
    # ax[0, 1].set_title("Noise")
    # ax[1, 0].hist(tremor_scores_acc)
    # ax[1, 0].set_title("Tremor")
    # ax[1, 1].hist(noa_scores_acc)
    # ax[1, 1].set_title("None of the above")
    # plt.show()


if __name__ == '__main__':
    # calculate_reliability_mat(golden_samples='Golden/golden_all.txt',
    #                           user_stats='stats_users_12_09_2020-20_10_24.txt',
    #                           classifications='earthquake-detective-classifications.csv')

    rel_stats(path=META_PATH / "rel_scores_v3", top=1.0, metric='f_beta')

    # a = map_users_to_index(user_stats='stats_users_12_09_2020-20_10_24.txt')
    # counter = 0
    # for k, v in sorted(a.items(), key=lambda  x: x[1]):
    #     print(k, v)
    #     if counter > 20:
    #         break
    #     counter += 1


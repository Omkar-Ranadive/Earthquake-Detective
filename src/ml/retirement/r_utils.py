from constants import DATA_PATH, META_PATH, n_classes, label_dict
import pandas as pd
import numpy as np
import json
from collections import defaultdict


def calculate_reliability_mat(golden_samples, user_stats, classifications):
    df_class = pd.read_csv(DATA_PATH / classifications)
    n_users = 20
    user_names = []
    rel_mat = np.zeros((n_users, 2*(n_classes+1)))
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

    for index, user in enumerate(user_names[:n_users]):
        df_user = df_class[df_class['user_name'] == user]
        print("Processing user {}, num entries {}".format(user, len(df_user)))
        for sub_id, gold_label in golden_info.items():
            # Find what the user has labeled for the current sub_id
            row = df_user[df_user['subject_ids'] == int(sub_id)]
            if not row.empty:
                meta = json.loads(row['annotations'].values[0])
                label = meta[0]['value']
                l_index = label_dict[label]
                if l_index == label_dict[gold_label]:
                    rel_mat[index, l_index] += 1
                else:
                    rel_mat[index, l_index + n_classes + 1] += 1

        # Now, from the matrix, calculate the score for each label and overall score
        for i in range(n_classes+1):
            # Each index i corresponds to some class like Earthquake, Tremor etc
            # We essentially calculate correct/total_samples for that specific class i
            class_total = rel_mat[index, i] + rel_mat[index, i + n_classes + 1]
            score = rel_mat[index, i] / class_total if class_total > 0 else -1
            rel_dict[user].append(score)

        # Finally, calculate the overall score correct_classifications_across_all/total_samples
        overall_total = np.sum(rel_mat[index:, ])
        if overall_total > 0:
            total_correct = np.sum(rel_mat[index, :n_classes+1]) / overall_total
        else:
            total_correct = -1

        rel_dict[user].append(total_correct)

    print(rel_mat)
    print(rel_dict)


if __name__ == '__main__':
    calculate_reliability_mat(golden_samples='Golden/golden_classified.txt',
                              user_stats='stats_users_12_09_2020-20_10_24.txt',
                              classifications='earthquake-detective-classifications.csv')

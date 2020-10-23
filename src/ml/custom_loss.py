import torch
from constants import metric_to_index
import torch.nn.functional as F
import numpy as np
cross_ent = torch.nn.CrossEntropyLoss()


def rel_cross_entropy(model, predicted, actual, users, rel_dict, metric='f_beta'):
    f_avg = 0.7  # TODO: Handle average calculations in a better way later
    # Calculate cross entropy loss
    log_prob = -1.0 * F.log_softmax(predicted, 1)
    loss = log_prob.gather(1, actual.unsqueeze(1))
    mi = metric_to_index[metric]

    # Calculate sum of model parameter weights
    param_sum = 0
    for param_name, param_value in model.named_parameters():
        if param_name.endswith('weight'):
            param_sum += param_value.abs().sum()

    # Add regularization based on reliability score
    for index, uid in enumerate(users):
        # Get class specific reliability for that user
        uid = uid.item()
        cid = actual[index].item()
        score = rel_dict[uid][cid][mi]
        # If score is nan, consider some average value
        if np.isnan(score):
            score = f_avg

        penalty = 1 - score
        # Add penalty to model weights
        loss[index] += penalty * param_sum

    return loss.mean()

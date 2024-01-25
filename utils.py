'''
utility functions
'''
import numpy as np
import torch
import torch.nn as nn
import os
import pdb
import torch.nn.functional as F

def normalize_duration(input, mask):
    input = torch.exp(input)*mask
    output = F.normalize(input, p=1, dim=-1)
    return output

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def get_unique(data):
    unique_elements = np.unique(data)
    if isinstance(data, np.ndarray):
        return unique_elements.tolist()
    else:
        return list(unique_elements)

def eval_file(gt_content, recog_content, obs_percentage, classes):
    # github.com/yabufarha/anticipating-activities
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))

    ground_truth_unique = list(set(ground_truth))
    recognized_unique = list(set(recognized))
    hamming_distance = modified_hamming_distance(ground_truth_unique, recognized_unique)

    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[classes[ground_truth[i]]] += 1
        else:
            n_F[classes[ground_truth[i]]] += 1
        if (i == 0):
            if (ground_truth[i] == recognized[i]):
                next_action_prediction = 1
            else:
                next_action_prediction = 0

    unique_recog = get_unique(recognized)
    unique_gt = get_unique(ground_truth)

    # single_action_recog = recognized[0]
    # single_action_gt = unique_gt

    true_positives = len(set(unique_recog) & set(unique_gt))
    precision = true_positives / len(unique_recog)
    recall = true_positives / len(unique_gt)

    return n_T, n_F, precision, recall, next_action_prediction, hamming_distance

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''

    loss = cal_loss(pred, gold.long(), trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    #gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Calculate cross entropy loss, apply label smoothing if needed'''

    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx)
    return loss

def weighted_binary_cross_entropy(pred, target):
    num_zeros = torch.sum(target == 0)
    num_ones = torch.sum(target == 1)
    total_samples = target.numel()

    weight_zeros = total_samples / (2 * num_zeros)
    weight_ones = total_samples / (2 * num_ones)

    loss = weight_zeros * (target * torch.log(pred + 1e-8)) + weight_ones * ((1 - target) * torch.log(1 - pred + 1e-8))
    loss = -torch.mean(loss)
    
    return loss

def compute_importance_loss_weighting(start_idx_list):
    '''
    Compute the weight for the importance loss component for each
    action to be predicted in the future.
    '''
    return torch.ones((len(start_idx_list)))


def readCSV(filename, single_element=False, gt_node_list=False):
    '''
    Read CSV file into a list
    '''
    data = []
    for line in open(filename):
        row = line.split(',')
        row[-1] = row[-1].strip()
        if single_element:
            data.append(row[1])
        else:
            data.append(row)

    if gt_node_list:
        action2nodes_mapping = {}
        for d in data:
            index, action_name, gt_nodes = d[0], d[1], d[2:]
            action2nodes_mapping[action_name] = gt_nodes
        return action2nodes_mapping

    return data

def modified_hamming_distance(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    
    # Create a 2D table to store the minimum cost
    dp = [[0 for _ in range(len_list2 + 1)] for _ in range(len_list1 + 1)]
    
    # Initialize the first row and first column
    for i in range(len_list1 + 1):
        dp[i][0] = i  # Cost of deleting elements from list1
    for j in range(len_list2 + 1):
        dp[0][j] = j  # Cost of adding elements to list1
    
    # Fill the DP table
    for i in range(1, len_list1 + 1):
        for j in range(1, len_list2 + 1):
            # If the current elements match, no cost is incurred
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Calculate the cost of various operations (add, delete, substitute)
                add_cost = dp[i][j - 1] + 1
                delete_cost = dp[i - 1][j] + 1
                substitute_cost = dp[i - 1][j - 1] + 1
                
                # Choose the minimum cost among the operations
                dp[i][j] = min(add_cost, delete_cost, substitute_cost)
    
    # The value in the bottom-right cell is the minimum cost
    return dp[len_list1][len_list2]

def kl_div_loss_with_ignore(distribution1, distribution2, target, 
                                ignore_index=-100, reduction='mean', 
                                log_target=False):
    input_prob = F.log_softmax(distribution1, dim=-1)
    target_prob = F.softmax(distribution2, dim=-1)

    mask = (target != ignore_index)

    # Compute the KL divergence loss
    kl_loss = F.kl_div(input_prob[mask], target_prob[mask], reduction='none')
    
    # Apply reduction if specified
    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()
    else:
        return kl_loss
    
def jensen_shannon_divergence_with_ignore(distribution1, distribution2, target, ignore_index=-100, reduction='mean'):
    input_prob = F.softmax(distribution1, dim=-1)
    target_prob = F.softmax(distribution2, dim=-1)

    # Compute the average distribution
    average_prob = 0.5 * (input_prob + target_prob)

    mask = (target != ignore_index)

    # Compute the KL divergences
    kl1 = F.kl_div(input_prob[mask], average_prob[mask], reduction='none')
    kl2 = F.kl_div(target_prob[mask], average_prob[mask], reduction='none')

    # Compute the Jensen-Shannon divergence
    jsd_loss = 0.5 * (kl1 + kl2)

    # Apply reduction if specified
    if reduction == 'mean':
        return jsd_loss.mean()
    elif reduction == 'sum':
        return jsd_loss.sum()
    else:
        return jsd_loss
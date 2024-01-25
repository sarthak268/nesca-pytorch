import torch
import csv

def readCSV(filename):
    data = []
    for line in open(filename):
        row = line.split(',')
        row[-1] = row[-1].strip()
        data.append(row)
    return data


def readDetections(filename):
    with open(filename) as f:
        content = f.readlines()
    
    detection_labels_all = []
    detection_conf_all = []

    for dets in content:
        detection = dets.strip().split(',')[:-1]
        detection_labels = [dets[:-6] for dets in detection]
        detection_conf = [float(dets[-6:][1:][:-1]) for dets in detection]
        detection_labels_all.append(detection_labels)
        detection_conf_all.append(detection_conf)
    
    return detection_labels_all, detection_conf_all


def readDetectionsStanford(filename, node_name_list):
    with open(filename) as f:
        content = f.readlines()

    detects_dict = {}
    
    for dets in content:
        name = dets.split(',')[0]
        gt_nodes = dets.split(',')[1:-1]
        detects_dict[name] = []

        for gt_node in gt_nodes:
            if gt_node[:-6] in node_name_list and gt_node[:-6] not in detects_dict[name]:
                detects_dict[name].append(gt_node[:-6])
    
    return detects_dict 


def extractNodenames(node_list):
    new_list = []
    for nodename in node_list:
        new_list.append(nodename[1])
    return new_list


def label2NodeIdx(label):
    '''
    From the label of the action returns the corresponding graph node for
    the affordance and object. ---------need to change this -could be 3 also
    '''

    label2nodeidx_dict = {1: [2, 3], 2: [4, 6]}
    # return label2nodeidx_dict[label]

    target_nodes_idx = torch.zeros((label.shape[0], 2)).to(label.device)
    for i in range(label.shape[0]):
        target_nodes_idx[i] = label2nodeidx_dict[label]

    return target_nodes_idx


def computeTargetNodes(label, n_total_nodes):
    '''
    Input: 
        label: Bx1 for label for the action
        n_total_nodes: total number of nodes in the graph
    Output: 
        target_nodes: Bxnum_nodes with ones at locations of the affordance 
                        and object and rest zeros
    '''

    target_nodes_idx = label2NodeIdx(label)

    target_nodes = torch.zeros((label.shape[0], n_total_nodes)).to(label.device)
    target_nodes.scatter_(1, target_nodes_idx, 1)

    return target_nodes


def makeOneHot(gt, num_classes):
    gt_onehot = torch.zeros(gt.size(0), num_classes).to(gt.device)
    gt_onehot.scatter_(1, gt.unsqueeze(1), 1)   
    return gt_onehot


def getGTImportanceForBatch(nodewise_imp_gt, all_candidates_idx):
    '''
    Given importance values for each node and the candidate nodes for which we 
    predicted the importance, returns the importance for those nodes.
    '''
    # print ('utils', nodewise_imp_gt.shape, all_candidates_idx.shape)
    return nodewise_imp_gt[all_candidates_idx.long()]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def readList(path):
    with open(path) as file:
        lines = [line.rstrip() for line in file]    
    return lines


def getGradientNorm(model):
    gradient_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            gradient_norm += torch.norm(param.grad.data)**2
    gradient_norm = gradient_norm.sqrt()
    return gradient_norm
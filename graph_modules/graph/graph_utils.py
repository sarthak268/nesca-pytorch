import torch

def select_k_indices(tensor, k, threshold):
    # find the indices where the values are greater than the threshold
    mask = tensor > threshold
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    
    # if there are less than k indices, select all of them
    if len(indices) <= k:
        return indices
            
    # select the k indices with the highest values
    _, topk_indices = torch.topk(tensor[indices].squeeze(-1), k)
    return indices[topk_indices]

def select_top_k(tensor, k):
    # Get the indices of the top k values
    _, indices = torch.topk(tensor.squeeze(-1), k)
    
    # Return the indices
    return indices
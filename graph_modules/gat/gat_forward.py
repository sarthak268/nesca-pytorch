import torch


def get_node_representations(args, graph, gat, device=None, conditioning_input=None):

    kg_nodes = torch.arange(graph.n_total_nodes).unsqueeze(-1).to(torch.device(device))
    adj_mat = graph.global_adj_mat.to(device)
    
    nodes_embedding = gat(kg_nodes.float(), 
                        adj_mat.unsqueeze(-1), 
                        conditioning_input)
    
    return nodes_embedding
    
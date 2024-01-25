'''
Code adopted from https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/graphs/gatv2

Paper:
HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS?
Published as a conference paper at ICLR 2022
'''


import torch
import torch.nn as nn

from graph_modules.gat.gatv2layer import GraphAttentionV2Layer


class GATv2(nn.Module):
    """
    ## Graph Attention Network v2 (GATv2)
    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, n_hidden: int, n_heads: int, dropout: float,
                 share_weights: bool = True):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                is_concat=True, dropout=dropout, share_weights=share_weights)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """

        x = self.layer1(x, adj_mat)
        return x


class ModifiedGATv2(nn.Module):
    """
    ## Modified Graph Attention Network v2 (GATv2)
    This graph attention network has two [graph attention layers](index.html).

    The main modification in this architecture is the presence of an additional
    input in the form of the image
    """

    def __init__(self, opt, in_features: int, n_hidden: int, n_heads: int, dropout: float,
                 share_weights: bool = True):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.opt = opt
        self.n_hidden = n_hidden

        self.embedding = nn.Embedding(num_embeddings=opt.vocab_size, embedding_dim=in_features)

        if self.opt.use_gsnn:
            if self.opt.condition_propagation:
                conditioning_input_dim = self.opt.condition_propagation_dim
                self.projection_layer = nn.Linear(in_features + conditioning_input_dim, in_features * 2)
                in_features *= 2

        else:
            conditioning_input_dim = self.opt.condition_propagation_dim
            self.projection_layer = nn.Linear(in_features + conditioning_input_dim, in_features * 2)
            in_features *= 2

        self.layer = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                is_concat=True, dropout=dropout, share_weights=share_weights)
        
    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, conditioning_input: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """

        node_embedding = self.embedding(x.squeeze(-1).long())

        if self.opt.use_gsnn:

            if self.opt.condition_propagation:
                repeated_embedding = conditioning_input.unsqueeze(0).repeat(node_embedding.shape[0], 1)
                new_node_embedding = torch.cat((node_embedding, repeated_embedding), -1)

                node_embedding = self.projection_layer(new_node_embedding) 
            
            return self.layer(node_embedding, adj_mat)

        else:

            output = torch.zeros((conditioning_input.shape[0], self.opt.vocab_size, self.n_hidden)).to(x.device) 

            for i in range(conditioning_input.shape[0]):
                current_video_input = conditioning_input[i].unsqueeze(0).repeat(node_embedding.shape[0], 1)

                new_node_embedding = torch.cat((node_embedding, current_video_input), -1)
                node_embedding_out = self.projection_layer(new_node_embedding) 

                output[i, :, :] = self.layer(node_embedding_out, adj_mat)

            return output
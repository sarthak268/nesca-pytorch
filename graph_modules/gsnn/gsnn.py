import torch
import torch.nn as nn

from graph_modules.gat.gatv2 import GATv2, ModifiedGATv2
from graph_modules.gsnn.gsnn_networks import ImportanceNet, ContextNet


class GSNN(nn.Module):

    def __init__(self, opt):

        super(GSNN, self).__init__()

        self.opt = opt
        self.n_total_nodes = opt.vocab_size
        self.device = torch.device(self.opt.device) if torch.cuda.is_available() else torch.device("cpu")
        
        self.createBaseNets()
        self.createOutputNets()

        if opt.imp_threshold_learnable:
            self.learnable_imp_threshold = nn.Parameter(torch.tensor(opt.imp_threshold))

    def createBaseNets(self):
        if self.opt.use_modified_gat:
            self.active_node_encoder = ModifiedGATv2(self.opt, in_features=self.opt.state_dim*2, n_hidden=self.opt.state_dim, 
                                        n_heads=self.opt.state_dim, dropout=self.opt.encoder_dropout, 
                                        share_weights=self.opt.encoder_share_weights)
        else:
            self.active_node_encoder = GATv2(in_features=1, n_hidden=self.opt.state_dim, n_heads=self.opt.state_dim, 
                                        dropout=self.opt.encoder_dropout, share_weights=self.opt.encoder_share_weights)
        self.active_node_encoder = self.active_node_encoder.to(self.device)

    def createOutputNets(self):
        self.context_out_net = ContextNet(self.opt, self.n_total_nodes)
        self.importance_out_net = ImportanceNet(self.opt, self.n_total_nodes)

    def forward(self, full_graph, initial_detections, conditioning_input=None):

        active_idx, active_node_lookup = full_graph.getInitialGraph(initial_detections)
        accumulated_importances, importance_prediction_idx = None, None

        for step in range(self.opt.num_steps):

            # Obtain all expansion candidates by looking at the neighbours of the active nodes
            # expansion_candidates is a combination of the previously active nodes and the current expansion candidates
            expansion_candidates, candidate_mask, candidate_idx_types = full_graph.getExpansionCandidates(active_idx, active_node_lookup, 
                                                                    return_nodetypes=self.opt.use_nodetypes)
            relative_adj_mat = full_graph.getRelativeAdjMat(expansion_candidates)

            active_nodes_embedding = self.active_node_encoder(expansion_candidates.float(), 
                                                    relative_adj_mat.unsqueeze(-1), 
                                                    conditioning_input)

            nodewise_importance = self.importance_out_net(x=active_nodes_embedding, node_input=expansion_candidates.float(),
                                                            nodetype_input=candidate_idx_types, mask=candidate_mask)
            
            if accumulated_importances == None:
                accumulated_importances = nodewise_importance[candidate_mask == 1]
                importance_prediction_idx = expansion_candidates[candidate_mask == 1]
            else:
                accumulated_importances = torch.cat((accumulated_importances, nodewise_importance[candidate_mask == 1]), 0)
                importance_prediction_idx = torch.cat((importance_prediction_idx, expansion_candidates[candidate_mask == 1]), 0)

            return_nodetypes = True if step == self.opt.num_steps - 1 else False
            
            if self.opt.imp_threshold_learnable:
                active_idx, active_node_lookup, active_idx_types = full_graph.updateGraphFromImportanceSelection(active_idx,\
                         active_node_lookup, expansion_candidates, nodewise_importance, return_nodetypes=return_nodetypes, 
                         learnable_imp_threshold=self.learnable_imp_threshold)

            else:
                active_idx, active_node_lookup, active_idx_types = full_graph.updateGraphFromImportanceSelection(active_idx,\
                         active_node_lookup, expansion_candidates, nodewise_importance, return_nodetypes=return_nodetypes)

        relative_adj_mat = full_graph.getRelativeAdjMat(active_idx)

        active_nodes_embedding = self.active_node_encoder(active_idx.float(), 
                                        relative_adj_mat.unsqueeze(-1), 
                                        conditioning_input)

        if self.opt.num_steps == 0:
            active_idx_types = torch.zeros((active_idx.shape[0], 2)).to(active_idx.device)
            active_idx_types[:, 1] = 1.

        active_idx_representations = self.context_out_net(x=active_nodes_embedding, node_input=active_idx.float(),
                                                            nodetype_input=active_idx_types)
        
        return accumulated_importances, importance_prediction_idx, active_idx_representations, active_idx, active_node_lookup
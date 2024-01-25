import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import readCSV


class ContextNet(nn.Module):
    def __init__(self, opt, n_total_nodes):
        super(ContextNet, self).__init__()

        self.opt = opt
        in_dim = self.opt.state_dim
        self.n_total_nodes = n_total_nodes

        self.sigmoid = nn.Sigmoid()

        if self.opt.context_use_node_input:
            self.embedding_node_bias = nn.Embedding(self.n_total_nodes, self.opt.node_bias_size)
            in_dim += self.opt.node_bias_size

        if self.opt.use_nodetypes:
            in_dim += 2

        if self.opt.context_architecture == 'gated':
            self.linear_gate = nn.Linear(in_dim, self.opt.context_dim)

        if self.opt.context_architecture == 'tanhout' :
            self.tanh = self.Tanh()

        self.layer_input_linear_list = []
        h_dim = self.opt.context_out_net_h_size
        if self.opt.context_out_net_h_size > 0:
            for i in range(self.opt.context_out_net_num_layer): 
                layer_input_linear = nn.Linear(in_dim, h_dim)
                self.layer_input_linear_list.append(layer_input_linear)
                in_dim = h_dim
            self.layer_input_linear_list = nn.ModuleList(self.layer_input_linear_list)

        if self.opt.context_transfer_function == 'tanh': 
            self.layer_input_act = nn.Tanh()        
        elif self.opt.context_transfer_function == 'sigmoid': 
            self.layer_input_act = nn.Sigmoid()
        elif self.opt.context_transfer_function == 'relu' :
            self.layer_input_act = nn.ReLU()
        else:
            raise Exception('Option ' + self.opt.context_transfer_function + ' not valid')

        self.output_linear = nn.Linear(in_dim, self.opt.context_dim)

        self.total_nodes_w_embedding = self.n_total_nodes


    def forward(self, x, node_input=None, nodetype_input=None):
        joined_input = x

        if self.opt.context_use_node_input:
            node_bias_size = self.opt.node_bias_size
            node_bias = torch.zeros((x.shape[0], node_bias_size)).to(x.device) 
            node_input = node_input.long()
            
            for i in range(len(node_input)):
                node_bias[i, :] = self.embedding_node_bias(node_input[i])

            joined_input = torch.cat((joined_input, node_bias), axis=-1)
        
        if self.opt.use_nodetypes:
            joined_input = torch.cat((joined_input, nodetype_input), axis=-1)

        if self.opt.context_architecture == 'gated':
            gate = self.sigmoid(self.linear_gate(joined_input))

        layer_input = joined_input 
        for layer_input_linear in self.layer_input_linear_list:
            layer_input = self.layer_input_act(layer_input_linear(layer_input))
        
        output = self.output_linear(layer_input)

        if self.opt.context_architecture == 'gated':
            final_output = output * gate
        elif self.opt.context_architecture == 'linout': 
            final_output = output
        elif self.opt.context_architecture == 'sigout' :
            final_output = self.sigmoid(output)
        elif self.opt.context_architecture == 'tanhout':
            final_output = 0.5 * (self.tanh(output) + 1)
        else:
            raise Exception('Option ' + self.opt.context_architecture + ' not valid')

        return final_output


class ImportanceNet(nn.Module):
    def __init__(self, opt, n_total_nodes):
        super(ImportanceNet, self).__init__()

        self.opt = opt
        in_dim = self.opt.state_dim
        self.n_total_nodes = n_total_nodes

        self.sigmoid = nn.Sigmoid()

        if self.opt.importance_use_node_input:
            self.embedding_node_bias = nn.Embedding(self.n_total_nodes, self.opt.node_bias_size)
            in_dim += self.opt.node_bias_size

        if opt.use_nodetypes:
            in_dim += 2

        if self.opt.importance_architecture == 'gated' or self.opt.importance_architecture == 'gatedsig':
            gate_linear = nn.Linear(in_dim, 1)

        if self.opt.importance_architecture == 'tanhout' :
            self.tanh = self.Tanh()

        if self.opt.imp_masking == 'active_cond':
            proj_dim = in_dim
            self.active_projection = nn.Linear(in_dim, in_dim)
            in_dim += proj_dim

        self.layer_input_linear_list = []
        h_dim = self.opt.importance_out_net_h_size
        if self.opt.importance_out_net_h_size > 0:
            for i in range(self.opt.importance_out_net_num_layer): 
                layer_input_linear = nn.Linear(in_dim, h_dim)
                self.layer_input_linear_list.append(layer_input_linear)
                in_dim = h_dim
            self.layer_input_linear_list = nn.ModuleList(self.layer_input_linear_list)

        if self.opt.importance_transfer_function == 'tanh': 
            self.layer_input_act = nn.Tanh()        
        elif self.opt.importance_transfer_function == 'sigmoid': 
            self.layer_input_act = nn.Sigmoid()
        elif self.opt.importance_transfer_function == 'relu' :
            self.layer_input_act = nn.ReLU()
        else:
            raise Exception('Option ' + self.opt.importance_transfer_function + ' not valid')

        self.output_linear = nn.Linear(in_dim, 1)

        self.total_nodes_w_embedding = self.n_total_nodes


    def forward(self, x, node_input=None, nodetype_input=None, mask=None):
        joined_input = x

        if self.opt.importance_use_node_input:
            node_bias_size = self.opt.node_bias_size
            node_bias = torch.zeros((x.shape[0], node_bias_size)).to(x.device) 
            node_input = node_input.long()
            
            for i in range(len(node_input)):
                node_bias[i, :] = self.embedding_node_bias(node_input[i])

            joined_input = torch.cat((joined_input, node_bias), axis=-1)
        
        if self.opt.use_nodetypes:
            joined_input = torch.cat((joined_input, nodetype_input), axis=-1)

        if self.opt.imp_masking == 'active_cond':
            candidate_idx = joined_input[mask == 1]
            already_active = self.active_projection(joined_input[mask == 0])
            projected_active = F.avg_pool1d(already_active.T, kernel_size=already_active.shape[0]).T
            projected_active = projected_active.repeat(candidate_idx.shape[0], 1)
            concat_candidate = torch.cat((candidate_idx, projected_active), -1)
            joined_input = concat_candidate

        if self.opt.importance_architecture == 'gated' or self.opt.importance_architecture == 'gatedsig':
            gate = self.sigmoid(self.linear_gate(joined_input))

        layer_input = joined_input
        for layer_input_linear in self.layer_input_linear_list:
            layer_input = self.layer_input_act(layer_input_linear(layer_input))
    
        output = self.output_linear(layer_input)

        if self.opt.importance_architecture == 'gated':
            final_output = output * gate
        elif self.opt.importance_architecture == 'gatedsig':
            final_output = self.sigmoid(output * gate)
        elif self.opt.importance_architecture == 'linout': 
            final_output = output
        elif self.opt.importance_architecture == 'sigout' :
            final_output = self.sigmoid(output)
        elif self.opt.importance_architecture == 'tanhout':
            final_output = 0.5 * (self.tanh(output) + 1)
        else:
            raise Exception('Option ' + self.opt.importance_architecture + ' not valid')

        if self.opt.imp_masking == 'naive':
            final_output = final_output * mask.unsqueeze(-1)

        elif self.opt.imp_masking == 'active_cond':
            final_output_1 = torch.zeros((x.shape[0], 1)).to(final_output.device)
            final_output_1[:final_output.shape[0], :] = final_output
            final_output = final_output_1 

        return final_output
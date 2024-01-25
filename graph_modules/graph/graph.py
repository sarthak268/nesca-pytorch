import torch
import copy
import math
import pickle
import numpy as np
import random

from graph_modules.graph.graph_utils import select_k_indices, select_top_k

from opts import parser
opt = parser.parse_args()


def saveObject(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

class Node():
    def __init__(self, index, name, nodetype='object'):
        
        # Index in graph
        self.index = index

        # Name of node
        self.name = name

        # Set up edge lists
        self.outgoing_edges = []
        self.incoming_edges = []

        self.nodetype = nodetype

class Edge():
    def __init__(self, start_node, end_node, index):
        self.start_node = start_node
        self.end_node = end_node
        self.index = index

class Graph():
    def __init__(self):

        # Default is empty graph
        self.n_total_nodes = 0
        self.n_total_edges = 0  
        self.nodes = []                # Table of nodes in the graph
        self.edges = []

        self.nodetype_mapping_list = []
        self.nodetype_list = []


    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    def printNodes(self):
        for node in self.nodes:
            print (node.index, node.name, node.nodetype)

    
    def writeNodes(self, filename):
        for node in self.nodes:
            # print (node.index, node.name, node.nodetype)
            with open(filename, 'a') as out:
                out.write(str(node.index) + ',' + node.name + ',' + node.nodetype + '\n')

    def printEdges(self):
        for edge in self.edges:
            print (edge.start_node.name, edge.end_node.name)


    def getNodeMask(self, nodetype):
        mask = torch.zeros((len(self.all_nodes)))
        indices = []
        for i, node in enumerate(self.nodetype_list):
            if node == nodetype:
                mask[i] = 1
                indices.append(i)
        return mask, indices


    def addNode(self, name, nodetype):
        '''
        Add empty node to graph
        '''

        newnode = Node(self.n_total_nodes, name, nodetype)

        # Update counts and lists 
        self.n_total_nodes += 1
        self.nodes.append(newnode)
        
        # assuming that the node added is a object node
        tensor = tensor = torch.zeros((2))
        if nodetype == 'object':
            tensor[1] = 1.
        else:
            tensor[0] = 1.
        self.nodetype_mapping_list.append(tensor)
        self.nodetype_list.append(nodetype)


    def checkNodeNameExists(self, node_name, node_type):
        '''
        Check if node with the name node_name exists
        '''
        for node in self.nodes:
            if node.name == node_name and node.nodetype == node_type:
                return True
        return False


    def getNode(self, node_name):
        '''
        Get the index for a node given its name
        '''
        for node in self.nodes:
            if node.name == node_name:
                return node.index
        return None


    def addEdge(self, start_node_idx, end_node_idx):
        '''
        Add new edge to graph with index
        '''

        # Create edge
        startnode = self.nodes[start_node_idx]
        endnode = self.nodes[end_node_idx]
        edge = Edge(startnode, endnode, self.n_total_edges + 1)
        startnode.outgoing_edges.append(edge)
        endnode.incoming_edges.append(edge)
        self.edges.append(edge)
        self.n_total_edges += 1


    def checkEdgeExists(self, start_node_idx, end_node_idx):
        '''
        Check if edge already exists with index
        '''

        edge_exists = False
        edge_idx = -1
        
        # Get start node
        startnode = self.nodes[start_node_idx]
        outgoing = startnode.outgoing_edges

        for i in range(len(outgoing)):
            edge = outgoing[i]
            if edge.start_node.index == start_node_idx and edge.end_node.index == end_node_idx:      
                edge_exists = True
                edge_idx = edge.index
                break

        return edge_exists, edge_idx


    def checkEdgeNameExists(self, start_node_name, end_node_name):
        '''
        Check if edge already exists with name
        '''

        edge_exists = False
        edge_idx = -1
        
        # Get start node
        start_node_idx = self.getNode(start_node_name)
        startnode = self.nodes[start_node_idx]
        outgoing = startnode.outgoing_edges

        for i in range(len(outgoing)):
            edge = outgoing[i]
            if edge.start_node.name == start_node_name and edge.end_node.name == end_node_name:      
                edge_exists = True
                edge_idx = edge.index
                break

        return edge_exists, edge_idx


    def getFullGraph(self):
        '''
        Get the edges of the entire graph
        '''

        edges = []
        for i, edge in enumerate(self.edges):
            e = [edge.start_node.index, edge.end_node.index]
            edges.append(e)

        return edges

    def removeEdge(self, start_node, end_node):
        '''
        Removes given edge from graph
        '''
        self.n_total_edges -= 1.
        for edge_idx, edge in enumerate(self.edges):
            if start_node.index == edge.start_node.index and end_node.index == edge.end_node.index:
                self.edges.pop(edge_idx)
                break
        
        for edge_idx, edge in enumerate(start_node.outgoing_edges):
            if edge.end_node.index == end_node.index:
                start_node.outgoing_edges.pop(edge_idx)
                break

        for edge_idx, edge in enumerate(end_node.incoming_edges):
            if edge.start_node.index == start_node.index:
                end_node.incoming_edges.pop(edge_idx)
                break   

    def removeNode(self, node_name):
        self.n_total_nodes -= 1
        for i, node in enumerate(self.nodes):
            if node.name == node_name:
                self.nodes.pop(i)
                for edge in self.edges:
                    if edge.start_node.name == node_name or \
                        edge.end_node.name == node_name:
                        self.removeEdge(edge.start_node, edge.end_node)
                break     

    def getNodeIdx(self, node_name):
        '''
        Returns the index of the node with name as node_name
        '''

        for idx, node in enumerate(self.nodes):
            if node.name == node_name:
                return idx
        
        return None

    def getGlobalAdjacencyMat(self):
        '''
        Compute global adjacency once so its easier to compute relative adjacency matrix
        each iteration
        '''
        self.global_adj_mat = torch.eye(self.n_total_nodes)
    
        for node_idx in range(self.n_total_nodes):
            outgoing_edges = self.nodes[node_idx].outgoing_edges
            outgoing_nodes = [edge.end_node.index for edge in outgoing_edges]
    
            self.global_adj_mat[node_idx, outgoing_nodes] = 1.
        
        return self.global_adj_mat

    def getInitialNodesFromDetections(self, initial_detections):

        above_threshold = torch.gt(initial_detections, opt.init_conf).float()
        active_node_lookup = torch.zeros((self.n_total_nodes))

        if above_threshold.sum().item() < opt.min_num_init:
            _, detect_indices = torch.topk(initial_detections, opt.min_num_init)

        else:
            detect_indices = torch.nonzero(initial_detections > opt.init_conf)
        active_node_lookup[detect_indices] = 1.

        return detect_indices, active_node_lookup

    def getInitialGraph(self, initial_detections):
        '''
        Get the initial set of detected nodes and the lookup which is basically a
        matrix that depicts which nodes are already visited
        '''

        detect_indices, active_node_lookup = self.getInitialNodesFromDetections(initial_detections)

        # Assuming that the detected indices are same as indices of the graph
        active_idx = copy.deepcopy(detect_indices)

        return active_idx, active_node_lookup
                    
    def getExpandedGraph(self, active_idx, active_node_lookup, to_expand):

        if len(active_idx.shape) == 1:
            active_idx = active_idx.unsqueeze(-1) 

        active_idx = torch.cat((active_idx, to_expand), 0)
        # if (to_expand.shape[0] == 0):
        #     print (active_idx.squeeze(-1))

        # Making sure that the idx are not added again
        # active_idx = torch.unique(active_idx, dim=0)
        
        active_node_lookup[to_expand.squeeze(-1).long()] = 1.

        return active_idx, active_node_lookup

    def getRelativeAdjMat(self, node_arr):
        '''
        Computes the relative adjacency matrix for the set of nodes provided using KG
        '''

        node_arr = node_arr.squeeze(-1).long()
        relative_adj_mat = self.global_adj_mat[node_arr.cpu(), :][:, node_arr.cpu()].to(node_arr.device)
        
        return relative_adj_mat

    def getExpansionCandidates(self, active_idx, active_node_lookup, return_nodetypes=False):
        '''
        Computes a list of all candidates that could be expanded -- neighbours of expanded nodes

        active_idx: All active node indices in a list
        active_node_lookup: An array that has 1 at locations which are active and 0 otherwise
        '''
        expansion_candidates = copy.copy(active_idx)

        if len(active_idx.shape) > 1:
            active_idx = active_idx.squeeze(-1)
        candidate_mask = torch.zeros_like(active_idx)

        for idx in active_idx:
            for outgoing_edge in self.nodes[int(idx.item())].outgoing_edges:
                candidate_idx = int(outgoing_edge.end_node.index)

                if opt.affordance_only_propagation:
                    condition = (active_node_lookup[candidate_idx] != 1) and (self.nodetype_list[candidate_idx] == 'affordance')
                else:
                    condition = (active_node_lookup[candidate_idx] != 1)
        
                if condition:
                    ones = torch.ones((1, 1)).to(active_idx.device)
                    if len(expansion_candidates.shape) == 1: expansion_candidates = expansion_candidates.unsqueeze(-1)
                    expansion_candidates = torch.cat((expansion_candidates, ones * candidate_idx), 0)
                    candidate_mask = torch.cat((candidate_mask, ones.squeeze(-1)), 0)

        candidate_idx_types = None
        if return_nodetypes:
            candidate_idx_types = []
            for i in range(len(expansion_candidates)):
                candidate_idx_types.append(self.nodetype_mapping_list[int(expansion_candidates[i].item())])
            candidate_idx_types = torch.stack(candidate_idx_types).to(active_idx)
        
        return expansion_candidates, candidate_mask, candidate_idx_types

    def updateGraphFromImportanceSelection(self, active_idx, active_node_lookup, expansion_candidates, importance, 
                                            return_nodetypes=False, learnable_imp_threshold=None):

        if opt.imp_threshold_learnable:
            imp_threshold = learnable_imp_threshold
        else:
            imp_threshold = opt.imp_threshold
            
        if opt.propagation_method == 'threshold_based':
            selected_indices = select_k_indices(importance, k=3, threshold=imp_threshold)
        
        elif opt.propagation_method == 'top_k':
            selected_indices = select_top_k(importance, k=opt.top_k_prop)

        to_expand_idx = expansion_candidates[selected_indices]

        active_idx, active_node_lookup = self.getExpandedGraph(active_idx, active_node_lookup, to_expand_idx)

        active_idx_types = None
        if return_nodetypes:
            active_idx_types = []
            for i in range(len(active_idx)):
                active_idx_types.append(self.nodetype_mapping_list[int(active_idx[i].item())])
            active_idx_types = torch.stack(active_idx_types).to(active_idx)

        return active_idx, active_node_lookup, active_idx_types

    def getNodewiseImportanceGT(self, target_nodes):
        '''
        Given a set of target nodes, computes discounted GT importance values for 
        relevant nodes for computing importance loss

        target_nodes: array of size = number of nodes, with ones
        at target node locations
        '''

        nodewise_imp_gt = copy.deepcopy(target_nodes)
        
        for step in range(opt.num_steps):
            value = opt.gamma ** step
            target_nodes_idx = torch.nonzero(target_nodes == value).squeeze()

            for edge in self.edges:
                if edge.start_node.index in target_nodes_idx:
                    node_value = nodewise_imp_gt[edge.end_node.index] + (opt.gamma ** (step + 1))
                    nodewise_imp_gt[edge.end_node.index] = node_value if node_value < 1. else 1.

        return nodewise_imp_gt


    def saveDiscountedAdjMat(self):

        self.discounted_adj_mats = []

        for node in self.nodes:

            immediate_neighbours = self.nonzero(self.global_adj_mat[node.index])
            neighbours = torch.zeros((self.n_total_nodes))

            for i in range(opt.num_steps):
                
                neighbours[immediate_neighbours] = opt.gamma ** i

                all_next_neighbours = None
                for neigh in immediate_neighbours:
                    next_neighbours = torch.nonzero(self.global_adj_mat[neigh])

                    if all_next_neighbours is None:
                        all_next_neighbours = next_neighbours
                    else:
                        all_next_neighbours = torch.cat((all_next_neighbours, next_neighbours))

                all_next_neighbours = torch.unique(all_next_neighbours)

                all_next_neighbours = torch.tensor([value for value in enumerate(all_next_neighbours) \
                                                        if neighbours[value] == 0])
                immediate_neighbours = all_next_neighbours

            self.discounted_adj_mats.append(neighbours)
        
        self.discounted_adj_mats = torch.stack(self.discounted_adj_mats)


    def getNodewiseImportanceGTAdjMat(self, target_nodes):
        '''
        Given a set of target nodes, computes discounted GT importance values for 
        relevant nodes for computing importance loss using saved adjacency matrices

        target_nodes: array of size = number of nodes, with ones
        at target node locations
        '''

        nodewise_imp_gt = copy.deepcopy(target_nodes)
        
        for step in range(opt.num_steps):
            value = opt.gamma ** step
            target_nodes_idx = torch.nonzero(target_nodes == value).squeeze()

            for edge in self.edges:
                if edge.start_node.index in target_nodes_idx:
                    node_value = nodewise_imp_gt[edge.end_node.index] + (opt.gamma ** (step + 1))
                    nodewise_imp_gt[edge.end_node.index] = node_value if node_value < 1 else 1

        return nodewise_imp_gt

        
            

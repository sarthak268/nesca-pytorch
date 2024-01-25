import torch
import torch.nn.functional as F

from graph_modules.graph_utils import getGTImportanceForBatch
from graph_modules.gsnn.gsnn import GSNN
from utils import weighted_binary_cross_entropy

device = torch.device('cuda')


def get_context_vectors(args, gsnn_net, graph, detections, target_nodes, encoder_out=None, conditioning_input=None, mode='train'):
    '''
    Given a set of detections compute the context vectors after the propagation 
    through KG. Also, can optionally condition this propagation on the image context.
    '''

    graph_size = args.context_dim * args.vocab_size
    graph_data = torch.zeros((args.batch_size, graph_size)).to(torch.device(device))
    # TODO we could try with three dimensional graph data also

    importance_loss = torch.tensor(0).float()
    active_idx_representations_list = []
        
    # For each batch, do forward pass
    for i in range(detections.shape[0]):

        initial_conf = detections[i]
        
        # Hack -- this basically ensures what scalar value we add to the second dim of annotation
        # All annotations are of the form of (num, 0)
        annotations_plus = torch.zeros((args.vocab_size, 1)).to(torch.device(device))
            
        # Forward through GSNN network
        # curr_image = None
        # if args.image_conditioned_propagation or args.image_conditioned_importance:
            # curr_image = image_data[i]
            # curr_image = encoder_out[i]
        
        if args.condition_propagation:
            accumulated_importances, importance_prediction_idx, active_idx_representations, active_idx, active_node_lookup = \
                                gsnn_net(graph, initial_conf, conditioning_input=conditioning_input[i])

        else:            
            accumulated_importances, importance_prediction_idx, active_idx_representations, active_idx, active_node_lookup = \
                                gsnn_net(graph, initial_conf)

        active_idx_representations_list.append(active_idx_representations)

        # Need to compute the importance loss only when training the KG transformer not during test / distillation
        if mode == 'train' and accumulated_importances is not None:

            # Loop over all future actions
            for j in range(len(target_nodes[i])):

                nodewise_imp_gt = graph.getNodewiseImportanceGT(target_nodes[i][j, :])
                
                if args.importance_over_predicted:
                    nodewise_imp_gt_batch = getGTImportanceForBatch(nodewise_imp_gt, importance_prediction_idx)
                    predicted_importance = accumulated_importances

                else:
                    nodewise_imp_gt_batch = nodewise_imp_gt
                    predicted_importance = torch.zeros((args.vocab_size)).to(accumulated_importances.device)
                    predicted_importance[importance_prediction_idx.squeeze(-1).long()] = accumulated_importances.squeeze(-1)
                    
                # Importance net
                ##############################################################################################
                importance_loss_sample = torch.tensor(0).float()
                if predicted_importance.shape[0] > 0:
                    
                    if args.weighted_importance_loss:
                        importance_loss_sample = weighted_binary_cross_entropy(predicted_importance.squeeze(-1), nodewise_imp_gt_batch.squeeze(-1))
                    else:
                        importance_loss_sample = F.binary_cross_entropy(predicted_importance.squeeze(-1), nodewise_imp_gt_batch.squeeze(-1))

                if torch.isnan(importance_loss_sample):
                    import pdb
                    pdb.set_trace()

                importance_loss = importance_loss + importance_loss_sample

    importance_loss *= args.importance_loss_weight

    return importance_loss, active_idx_representations_list
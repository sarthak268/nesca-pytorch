import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", default="futr", help='model type')
parser.add_argument("--mode", default="train_eval", help="select action: [\"train\", \
                    \"predict\", \"train_eval\"]")
parser.add_argument("--dataset", type=str, default='breakfast')
parser.add_argument('--predict', "-p", action='store_true', help="predict for whole videos mode")
parser.add_argument('--wandb', type=str, default='project name', help="wandb runs name")
parser.add_argument("--single_sample_eval", action='store_true', help='evaluating on single sample')

#Dataset
parser.add_argument("--mapping_file", default="./datasets/breakfast/mapping.txt")
parser.add_argument("--features_path", default="./datasets/breakfast/features/")
parser.add_argument("--gt_path", default="./datasets/breakfast/groundTruth/")
parser.add_argument("--split", default="1", help='split number')
parser.add_argument("--file_path", default="./datasets/breakfast/splits")
parser.add_argument("--model_save_path", default="./save_dir/models/transformer")
parser.add_argument("--results_save_path", default="./save_dir/results/transformer")
parser.add_argument("--task", type=str, help="Next Action Anticipation/long-term anticipation", default='long')

#Training options
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=5e-3) 
parser.add_argument("-warmup", '--n_warmup_steps', type=int, default=500)
parser.add_argument("--cpu", action='store_true', help='run in cpu')
parser.add_argument("--sample_rate", type=int, default=6)
parser.add_argument("--obs_perc", default=30)
parser.add_argument("--n_query", type=int, default=8)

#FUTR specific parameters
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--n_encoder_layer", type=int, default=2)
parser.add_argument("--n_decoder_layer", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--input_dim", type=int, default=2048)

#Model parameters
parser.add_argument("--seg", action='store_true', help='action segmentation')
parser.add_argument("--anticipate", action='store_true', help='future anticipation')
parser.add_argument("--pos_emb", action='store_true', help='positional embedding')
parser.add_argument("--max_pos_len", type=int, default=2000, help='position embedding number for linear interpolation')

#Test on GT or decoded input
parser.add_argument("--input_type", default="i3d_transcript", help="select input type: [\"decoded\", \"gt\"]")
parser.add_argument("--runs", default=0, help="save runs")

################################
# Graph general
parser.add_argument('--device', type=str, default='cuda', help="Device used to train: gpu / cpu")
parser.add_argument('--first_time', action='store_true', help="Running to save data for the first time")
parser.add_argument('--affordance_only_propagation', action='store_true', help="Perform propagation of only affordances")

parser.add_argument('--kg_attn', default=True, help="Use knowledge guided attention")
parser.add_argument('--use_gsnn', default=True, help="Use GSNN based propagation")

# Graph specific
parser.add_argument('--importance_loss_weight', type=float, default=0.01, help="Weight of importance loss")
parser.add_argument('--frame_save_interval', type=int, default=50, help="Interval to save each frame of the video")
parser.add_argument('--vocab_size', type=int, default=75, help="Size of concept vocabulary: 74 for 50salads")
parser.add_argument('--init_conf', type=float, default=0.35, help="Initialization confidence")
parser.add_argument('--min_num_init', type=int, default=1, help="Minimum number of initialization nodes")
parser.add_argument('--gradient_clipping', type=float, default=50, help="Use gradient clipping, don't use if -1")
parser.add_argument('--num_inter_steps', type=int, default=1, help="Number of intermediate steps")

# Propagation
parser.add_argument('--encoder_dropout', type=float, default=0.2, help="Dropout for node encoder")
parser.add_argument('--encoder_share_weights', type=bool, default=False, help="Share weights in node encoder")
parser.add_argument('--state_dim', type=int, default=10, help="State dimension for propagation net")
parser.add_argument('--condition_propagation', action='store_true', help="Condition the propagation on observed video context")
parser.add_argument('--condition_propagation_dim', type=int, default=50, help="Dimensionality of the conditioning input for propagation")
parser.add_argument('--use_nodetypes', type=bool, default=True, help="Use nodetypes for propagation")
parser.add_argument('--use_modified_gat', type=bool, default=True, help="Use modified GATv2 for encoding")
parser.add_argument('--node_bias_size', type=int, default=2, help="Dimensionality for node bias")
parser.add_argument('--num_steps', type=int, default=1, help="Number of propagation steps")

# Context net
parser.add_argument('--context_architecture', type=str, default='gated', help="Architecture for content net (gated)")
parser.add_argument('--context_transfer_function', type=str, default='tanh', help="Activation for context net (tanh)")
parser.add_argument('--context_use_node_input', type=bool, default=True, help="Use node bias for context net")
parser.add_argument('--context_use_ann', type=bool, default=True, help="Use node annotation for context net")
parser.add_argument('--context_out_net_h_size', type=int, default=10, help="")
parser.add_argument('--context_out_net_num_layer', type=int, default=0, help="")
parser.add_argument('--context_dim', type=int, default=5, help="Dimensionality for context encoding")

# Importance net
parser.add_argument('--gamma', type=float, default=0.5, help="Discount rate for importance")
parser.add_argument('--importance_architecture', type=str, default='sigout', help="Architecture for importance net (sigout)")
parser.add_argument('--importance_transfer_function', type=str, default='tanh', help="Activation for importance net (tanh)")
parser.add_argument('--importance_use_node_input', type=bool, default=True, help="Use node bias for importance net")
parser.add_argument('--importance_use_ann', type=bool, default=False, help="Use node annotation for importance net")
parser.add_argument('--importance_out_net_h_size', type=int, default=-1, help="")
parser.add_argument('--importance_out_net_num_layer', type=int, default=0, help="")
parser.add_argument('--importance_out_net_use_mask', type=bool, default=False, help="")
parser.add_argument('--image_conditioned_importance', type=bool, default=False, help="")
parser.add_argument('--imp_threshold', type=float, default=0.5, help="Threshold for expansion based on importance")
parser.add_argument('--imp_threshold_learnable', action='store_true', help="Use learnable threshold for expansion based on importance")
parser.add_argument('--imp_masking', type=str, default='active_cond', help="Importance prediction masking approach: naive / active_cond")
parser.add_argument('--importance_over_predicted', action='store_true', help="Compute importance loss over only predicted values")
parser.add_argument('--append_glove_emb_imp', action='store_true', help="Append glove embeddings to the node bias of importance net")
parser.add_argument('--propagation_method', type=str, default='threshold_based', help="Method for selecting indices for propagation: top_k / threshold_based")
parser.add_argument('--top_k_prop', type=int, default=3, help="If top_k selected, then indices to propagate at every step")
parser.add_argument('--weighted_importance_loss', action='store_true', help="Use a sample based weighting for importance loss computation")

# TO REMOVE -- both
parser.add_argument('--importance_accumulation', action='store_true', help="Increase importance for objects previously identified as active")
parser.add_argument('--importance_accumulation_weight', type=float, default=0.1, help="Weight for the previously active node's importance increment")

# Rectification matrix
parser.add_argument('--rectification_method', type=str, default='diagonal', help="Method to compute the rectification matrix: diagonal / full / weighting")
parser.add_argument('--rectification_residual', action='store_true', help="Instead of learning the rectification matrix learn its residual")

# Feature extraction
parser.add_argument('--feature_extraction_mode', type=str, default='rgb', help="Type of feature extraction: RGB / flow / both")
parser.add_argument('--last_k_frames', type=int, default=21, help="Use last n frames for computing video features")

# Finetune
parser.add_argument('--finetune', action='store_true', help="Finetune action anticipation using demonstations")
parser.add_argument('--demo_predict', action='store_true', help="Evaluate action anticipation on collected demonstations")
parser.add_argument('--demo_data_path', type=str, default='./demo_trajectories/', help="Path to collected demonstration data")
parser.add_argument('--prob_demos', type=float, default=0.7, help="Probability of choosing a vector from collected demonstration")
parser.add_argument('--scene_objects', type=str, default=['bowl', 'plate', 'tomato', 'lettuce', 
                                                        'cucumber', 'knife', 'salt', 'pepper', 'vinegar', 
                                                        'oil', 'spoon', 'spatula', 'glass', 'cheese'], 
                                                        help="Objects placed inside the scene")

# Real world HRC
parser.add_argument('--entropy_threshold', type=float, default=1., help="Threshold for entropy for certainty")
parser.add_argument('--min_pred_length', type=int, default=100, help="Min number frames, required for history collection")
parser.add_argument('--max_pred_length', type=int, default=1000, help="Max number frames, after this sliding window starts")

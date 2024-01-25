import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from torch.backends import cudnn
from opts import parser
from scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR

from utils import read_mapping_dict
from data.basedataset import BaseDataset
from model.futr import FUTR
from train import train
from predict import predict
import math

device = torch.device('cuda')

# Seed fix
# seed = 13452
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# cudnn.benchmark, cudnn.deterministic = False, True


def get_lr_lambda(warmup_epochs, total_epochs, min_lr=0.0):
    """Create a learning rate lambda function for Linear Warmup and Cosine Annealing schedule"""
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda

def main():
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    print('runs : ', args.runs)
    print('model type : ', args.model)
    print('input type : ', args.input_type)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    dataset = args.dataset
    task = args.task
    split = args.split

    if dataset == 'breakfast':
        data_path = './datasets/breakfast'
    elif dataset == '50salads' :
        data_path = './datasets/50salads'

    mapping_file = os.path.join(data_path, 'mapping.txt')
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train.split'+args.split+'.bundle' )
    video_file_test_path = os.path.join(data_path, 'splits', 'test.split'+args.split+'.bundle' )

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features')
    gt_path = os.path.join(data_path, 'groundTruth')

    n_class = len(actions_dict) + 1
    pad_idx = n_class + 1

    finetune = True if args.finetune else False
    trainset = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, finetune=finetune)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                shuffle=True, num_workers=args.workers,
                                                collate_fn=trainset.my_collate)

    # Model specification
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/transformer', split, args.input_type, \
                                    'runs'+str(args.runs))
    results_save_path = os.path.join('./save_dir'+'/'+args.dataset+'/'+args.task+'/results/transformer', 'split'+split,
                                    args.input_type )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    
    criterion = nn.MSELoss(reduction = 'none')

    if args.predict:
        if args.demo_predict:
            import glob
            video_test_list = glob.glob(args.demo_data_path + 'videos/*avi')
            video_test_list = [vid.split('/')[-1] for vid in video_test_list]

        obs_perc = [0.05, 0.1]
        
        results_save_path = results_save_path +'/runs'+ str(args.runs) +'.txt'
        if args.dataset == 'breakfast':
            model_path = './ckpt/bf_split'+args.split+'.ckpt'
        elif args.dataset == '50salads':
            model_path = './ckpt/50s_split'+args.split+'.ckpt'
        print("Predict with ", model_path)

        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for obs_p in obs_perc :
            predict(model, video_test_list, args, obs_p, n_class, actions_dict, device)

    else :
        if args.finetune:
            if args.dataset == 'breakfast':
                model_path = './ckpt/bf_split'+args.split+'.ckpt'
            elif args.dataset == '50salads':
                model_path = './ckpt/50s_split'+args.split+'.ckpt'
            print("Loading model from ", model_path)

            model.load_state_dict(torch.load(model_path))
            model.to(device)

        # Training
        train(args, model, train_loader, optimizer, scheduler, criterion,
                     model_save_path, pad_idx, device )


if __name__ == '__main__':
    main()

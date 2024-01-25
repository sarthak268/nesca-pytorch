import torch
import numpy as np
import random
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from numpy.random import randint
from utils import *
import cv2
import json
import glob


class BaseDataset(Dataset):
    def __init__(self, vid_list, actions_dict, features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None, finetune=False,
                 ):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1
        self.finetune = finetune
        
        if finetune:
            demo_vid_list = glob.glob(args.demo_data_path + '/videos/*.avi')
            self.finetune_vid_list = [vid.split('/')[-1] for vid in demo_vid_list]

        if self.mode == 'train' or self.mode == 'val':
            for vid in vid_list:
                self.vid_list.append([vid, .2])
                self.vid_list.append([vid, .3])
                self.vid_list.append([vid, .5])
        elif self.mode == 'test' :
            for vid in vid_list:
                self.vid_list.append([vid, obs_perc])
        
        if self.finetune:
            self.original = len(self.vid_list) + 1

            for vid in self.finetune_vid_list:
                self.vid_list.append([vid, .2])
                self.vid_list.append([vid, .3])
                self.vid_list.append([vid, .5])

            self.demos = len(self.vid_list) + 1 - self.original

        self.action2gt_dict = readCSV('./datasets/action_name2gt_{}.csv'.format(args.dataset), gt_node_list=True)
        self.all_actions = list(self.action2gt_dict.keys())

        self.node_list = readCSV('./datasets/nodelist_kitchen.csv', single_element=True)
        
        if args.first_time:
            file_names = []
            saved_frames = []

            for vid_file, obs_perc in self.vid_list:

                vid_file_split = vid_file.split('/')[-1]
                vid_name = vid_file_split
                gt_file = os.path.join(self.gt_path, vid_file_split)
                file_ptr = open(gt_file, 'r')
                all_content = file_ptr.read().split('\n')[:-1]
                vid_len = len(all_content)
                observed_len = int(float(obs_perc)*vid_len)
                start_frame = 0

                if self.args.dataset == 'breakfast':
                    data_path = '/data/sarthak/breakfast/BreakfastII_15fps_qvga_sync/'
                    vid_dir = vid_file.split('_')[0]
                    vid_type = vid_file.split('_')[1]
                    file_name_list = vid_file.split('.')[0].split('_')[2:]
                    file_name = file_name_list[0] + '_' + file_name_list[1] + '.avi'
                    video_path = os.path.join(data_path, vid_dir, vid_type, file_name)

                elif self.args.dataset == '50salads':
                    data_path = '/data/sarthak/50salads/rgb/'
                    file_name = vid_file.split('.')[0] + '.avi'
                    video_path = os.path.join(data_path, file_name)
                    
                video = cv2.VideoCapture(video_path)

                #### Save all frame of the video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                interval = args.frame_save_interval

                frame_save_dir = './saved_frames/{}/'.format(args.dataset)

                video_file_name = os.path.join(frame_save_dir, file_name.split('.')[0])

                if not os.path.exists(video_file_name):
                    print (video_file_name)
                
                    os.mkdir(video_file_name)

                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame_count += 1

                        # Select frames based on the specified interval
                        if frame_count % interval == 0:
                            # Perform your desired operations on the frame

                            frame_path = os.path.join(video_file_name, '{}.jpg'.format(frame_count))
                            cv2.imwrite(frame_path, frame)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    cap.release()
                    cv2.destroyAllWindows()

                #### Save only one random frame             
                # found = False
                # while not found:
                #     frame_num = random.randint(start_frame, observed_len)
                #     video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                #     ret, frame = video.read()

                #     if frame is not None:
                #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #         saved_frames.append(frame)
                #         file_names.append(vid_file_split)
                #         found = True

        else:

            with open('./datasets/detected_objects_{}.json'.format(args.dataset), 'r') as file:
                self.detected_objects_dict = json.load(file)


    def __getitem__(self, idx):
        if self.finetune:
            if random.uniform(0, 1) < self.args.prob_demos:
                vid_file, obs_perc = random.choice(self.vid_list[:self.original])
            else:
                vid_file, obs_perc = random.choice(self.vid_list[self.original:])
        else:
            vid_file, obs_perc = self.vid_list[idx]

        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc)
        return item


    def _make_input(self, vid_file, obs_perc):

        vid_file = vid_file.split('/')[-1]
        demo_vid = self.args.finetune and (vid_file in self.finetune_vid_list)

        if demo_vid: 
            # Loading the feature and GT labels stored from save_trajectory_data.py
            demo_gt_path = self.args.demo_data_path + 'labels/'
            demo_features_path = self.args.demo_data_path + 'features/'
            if vid_file in self.finetune_vid_list:
                gt_file = os.path.join(demo_gt_path, vid_file.split('.')[0]+'.txt') 
                feature_file = os.path.join(demo_features_path, vid_file.split('.')[0]+'.npy')

            objects_detected = self.args.scene_objects
        
        else:
            gt_file = os.path.join(self.gt_path, vid_file) # From: ./datasets/breakfast/groundTruth/P37_webcam01_P37_cereals.txt
            feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+'.npy')

            objects_detected = self.detected_objects_dict[vid_file.split('.')[0]]
        
        objects_detected_torch = torch.zeros((self.args.vocab_size))
        objects_detected_idx = []
        for objects in objects_detected:
            if objects in self.node_list:
                objects_detected_idx.append(self.node_list.index(objects))
        objects_detected_torch[objects_detected_idx] = 1.
        
        file_ptr = open(gt_file, 'r')
        all_content = file_ptr.read().split('\n')[:-1]
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)
        # print (observed_len, pred_len, vid_len) --> 248 620 1241

        features = np.load(feature_file)
        if demo_vid:
            features = np.squeeze(features, axis=1)
        else:
            features = features.transpose()
            
        start_frame = 0

        # feature slicing
        features = features[start_frame : start_frame + observed_len] #[S, C]
        
        features = features[::self.sample_rate]

        past_content = all_content[start_frame : start_frame + observed_len] #[S]
        past_content = past_content[::self.sample_rate]
        # past_content = past_content[::self.sample_rate]
        past_label = self.seq2idx(past_content)

        if np.shape(features)[0] != len(past_content) :
            features = features[:len(past_content),]

        future_content = \
        all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]
        future_content = future_content[::self.sample_rate]
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future #target
        
        # Computing the ground truth nodes for importance loss computation
        # future_action_idx contains the indices of the actions (wrt action_name2gt CSV file)
        # future_action_start_idx is the index where the new action starts 
        future_action_idx = []
        future_action_names = []
        future_action_start_idx = []

        for i, cont in enumerate(future_content):
            if cont == 'action_start' or cont == 'action_end':
                # for 50salads
                pass
            elif cont == 'SIL':
                # for breakfast
                pass
            else:
                action_index = self.all_actions.index(cont)
                
                if action_index not in future_action_idx:
                    future_action_idx.append(action_index)
                    future_action_names.append(cont)

                    if future_content[i] != future_content[i-1]:
                        future_action_start_idx.append(i)

        kg_nodes_gt = torch.zeros((len(future_action_idx), self.args.vocab_size))
        for i in range(len(future_action_idx)):
            corresponding_gt_nodes = self.action2gt_dict[future_action_names[i]]
            corresponding_gt_nodes_idx = [self.node_list.index(act) for act in corresponding_gt_nodes]
            kg_nodes_gt[i, corresponding_gt_nodes_idx] = 1.
        
        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))

        item = {'features':torch.Tensor(features),
                'past_label':torch.Tensor(past_label),
                'trans_future_dur':torch.Tensor(trans_future_dur),
                'trans_future_target' : torch.Tensor(trans_future_target),
                'detected_objects' : objects_detected_torch,
                'gt_nodes': kg_nodes_gt,
                'importance_weights': compute_importance_loss_weighting(future_action_start_idx),
                }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_detected_objects = torch.stack([item['detected_objects'] for item in batch])
        b_gt_nodes = [item['gt_nodes'] for item in batch]
        b_imp_weights = [item['importance_weights'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target, 
                    b_detected_objects, b_gt_nodes, b_imp_weights]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)









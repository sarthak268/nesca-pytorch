import torch
import torch.nn as nn
import numpy
import pdb
import os
import cv2
import copy
import json
import numpy as np
from utils import normalize_duration, eval_file, readCSV

from opts import parser
args = parser.parse_args()

def save_frames(vid_list, obs_p):

    file_names = []
    saved_frames = []

    for vid_file in vid_list:

        if args.dataset == 'breakfast':
            data_path = '/data/sarthak/breakfast/BreakfastII_15fps_qvga_sync/'
            vid_dir = vid_file.split('_')[0]
            vid_type = vid_file.split('_')[1]
            file_name_list = vid_file.split('.')[0].split('_')[2:]
            if vid_type == 'stereo01':
                file_name = file_name_list[0] + '_' + file_name_list[1] + '_ch0.avi'
                video_path = os.path.join(data_path, vid_dir, vid_type[:-2], file_name)
            else:
                file_name = file_name_list[0] + '_' + file_name_list[1] + '.avi'
                video_path = os.path.join(data_path, vid_dir, vid_type, file_name)
            dir_name = vid_file.split('.')[0]
            print (video_path)

        elif args.dataset == '50salads':
            data_path = '/data/sarthak/50salads/rgb/'
            file_name = vid_file.split('.')[0] + '.avi'
            video_path = os.path.join(data_path, file_name)
            dir_name = file_name.split('.')[0]
            print (dir_name)

        #### Save all frame of the video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        interval = args.frame_save_interval

        frame_save_dir = './datasets/saved_frames/{}/'.format(args.dataset)

        video_file_name = os.path.join(frame_save_dir, dir_name)

        if not os.path.exists(video_file_name):
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


def predict(model, vid_list, args, obs_p, n_class, actions_dict, device):
    
    model.eval()

    with torch.no_grad():
        
        if args.demo_predict:
            gt_path = args.demo_data_path + 'labels/'
            features_path = args.demo_data_path + 'features/'
            
        else:
            data_path = './datasets'
            if args.dataset == 'breakfast':
                data_path = os.path.join(data_path, 'breakfast')
            elif args.dataset == '50salads':
                data_path = os.path.join(data_path, '50salads')
            gt_path = os.path.join(data_path, 'groundTruth')
            features_path = os.path.join(data_path, 'features')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        if args.first_time:
            save_frames(vid_list, obs_p)
            print ('Done saving !')

        precision_avg, recall_avg = [0, 0, 0, 0], [0, 0, 0, 0]
        next_action_pred_avg, hamming_avg = np.zeros((len(eval_p))),  np.zeros((len(eval_p)))

        for vid in vid_list:
            file_name = vid.split('/')[-1].split('.')[0]

            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_read.close()

            if args.demo_predict:
                detected_object_names = args.scene_objects

            else:
                # Obtain the detection from the test video to begin propagation
                with open('./datasets/detected_objects_{}.json'.format(args.dataset), 'r') as file:
                    detected_objects_dict = json.load(file)
                detected_object_names = detected_objects_dict[file_name]

            # Get list of all nodes for the KG
            node_list = readCSV('./datasets/nodelist_kitchen.csv', single_element=True)

            detected_object_names_idx = []
            for obj in detected_object_names:
                if obj in node_list:
                    detected_object_names_idx.append(node_list.index(obj))
            
            detections = torch.zeros((args.vocab_size))
            detections[detected_object_names_idx] = 1.
            detections = detections.unsqueeze(0)

            # load features
            features_file = os.path.join(features_path, file_name+'.npy')
            features = np.load(features_file)
            if args.demo_predict: 
                features = np.squeeze(features, axis=1)
            else:
                features = features.transpose()

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len)
            future_len = int(pred_p*vid_len)

            past_seq = gt_seq[:past_len]
            features = features[:past_len]
            inputs = features[::sample_rate, :]
            inputs = torch.Tensor(inputs).to(device)
            
            detected_objs = detections if args.kg_attn == True else None
            target_nodes = None # we don't need GT KG nodes for evaluation

            # input shape: 1, num of frames, 2048
            outputs, _, _ = model(inputs.unsqueeze(0), detected_objs, target_nodes, mode='test')

            output_action = outputs['action']
            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # fine the forst none class
            none_mask = None
            for i in range(output_label.size(1)) :
                if output_label[0,i] == NONE :
                    none_idx = i
                    break
                else :
                    none = None
            if none_idx is not None :
                none_mask = torch.ones(output_label.shape).type(torch.bool)
                none_mask[0, none_idx:] = False

            output_dur = normalize_duration(output_dur, none_mask.to(device))

            pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

            pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
            predicted = torch.ones(future_len)
            action = output_label.squeeze()

            for i in range(len(action)) :
                predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                pred_len[i+1] = pred_len[i] + pred_len[i+1]
                if i == len(action) - 1 :
                    predicted[int(pred_len[i]):] = action[i]

            prediction = past_seq
            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                T_action, F_action, precision, recall, next_action_pred, hamming = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                T_actions[i] += T_action
                F_actions[i] += F_action
                precision_avg[i] += precision
                recall_avg[i] += recall
                next_action_pred_avg[i] += next_action_pred
                hamming_avg[i] += hamming

        results = []
        values = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i,j] != 0:
                    acc += float(T_actions[i,j]/total_actions[i,j])
                    n+=1

            result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(float(acc)/n) + ' Prec: %.4f'%(precision_avg[i]/len(vid_list)) \
                                        + ' Rec: %.4f'%(recall_avg[i]/len(vid_list)) + ' Next Action: %.4f'%(next_action_pred_avg[i]/len(vid_list)) \
                                        + ' Hamming Distance: %.4f'%(hamming_avg[i]/len(vid_list))
            results.append(result)
            values.append(round(float(acc)*100/n, 4))
            print(result)
        print('--------------------------------')

        # file_path = './results/gat_split5.txt'
        # with open(file_path, 'a') as file:
        #     # Convert the list to a string representation
        #     list_string = ', '.join(str(item) for item in values)
            
        #     # Append the string representation of the list to the file
        #     file.write(list_string)
        #     file.write('\n')

        return


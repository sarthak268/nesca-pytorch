from concurrent import futures
import grpc
import lp_pb2
import lp_pb2_grpc
import cv2
import numpy as np
import os
import random
import copy

import torch
import torch.nn as nn

from opts import parser
from utils import read_mapping_dict
from model.futr import FUTR
from utils import normalize_duration, readCSV

from features import I3DFeaturesExtractor

seed = 13452
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# The gRPC server's implementation of the ImageProcessor service.
class ImageProcessorServicer(lp_pb2_grpc.ImageProcessorServicer):
    def __init__(self):

        self.args = parser.parse_args()

        self.min_pred_length = self.args.min_pred_length    # tune this parameter ~ 15 seconds
        self.max_pred_length = self.args.max_pred_length   # tune this parameter

        self.action2gt_dict = readCSV('./datasets/action_name2gt_{}_objects.csv'.format(self.args.dataset), gt_node_list=True)

        if self.args.cpu:
            self.device = torch.device('cpu')
            print('using cpu')
        else:
            self.device = torch.device('cuda')
            print('using gpu')

        dataset = self.args.dataset
        if dataset == 'breakfast':
            data_path = './datasets/breakfast'
        elif dataset == '50salads' :
            data_path = './datasets/50salads'

        mapping_file = os.path.join(data_path, 'mapping.txt')
        actions_dict = read_mapping_dict(mapping_file)
        self.actions_dict_reverse = self.reverse_dict(actions_dict)

        self.n_class = len(actions_dict) + 1
        self.actions_dict = actions_dict
        self.pred_p = 0.1
        pad_idx = self.n_class + 1

        self.model = FUTR(self.n_class, self.args.hidden_dim, device=self.device, args=self.args, src_pad_idx=pad_idx,
                            n_query=self.args.n_query, n_head=self.args.n_head,
                            num_encoder_layers=self.args.n_encoder_layer, num_decoder_layers=self.args.n_decoder_layer).to(self.device)
        self.model = nn.DataParallel(self.model).to(self.device)
        if self.args.dataset == 'breakfast':
            model_path = './ckpt/bf_split'+self.args.split+'.ckpt'
        elif self.args.dataset == '50salads':
            model_path = './ckpt/50s_split'+self.args.split+'.ckpt'
        print("Using model from ", model_path)

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.extractor = I3DFeaturesExtractor()
        self.image_history = []
        self.feature_stack = None

        print("Ready")

    def check_uncertainity(self, action_likelihood) -> str:
        '''
        Computes the uncertainity of prediction to output an action only 
        if we are confident about our predictions
        '''
        probabilities = torch.nn.functional.softmax(action_likelihood)
        entropy = - (probabilities * torch.log(probabilities)).sum()
        certain = entropy > self.args.entropy_threshold

        return certain


    def compute_current_temporal_feature(self, rgb_stack) -> torch.Tensor:
        '''
        Computes the I3D features given the stack of all RGB observations 
        '''
        print ('Computing features....')
        return self.extractor.run_on_server(rgb_stack)
    
    
    def reverse_dict(self, d):
        new_d = {}
        for k in d.keys():
            v = d[k]
            new_d[v] = k
        return new_d 
       

    def get_action_name_list(self, action_idx_list) -> list:
        action_name_list = []
        for action_idx in action_idx_list:
            action_name = None if action_idx == 19 else self.actions_dict_reverse[action_idx] 
            action_name_list.append(action_name)
        return action_name_list


    def compute_action(self, label, durations, output):
        '''
        Compute the relevant actions to be performed from a list of actions 
        and their durations

        TODO -- check preconditions, time taken for robot to perform
        '''
        objects = self.args.scene_objects

        for l in label:
            # print ('durations --- ', durations[idx])
            # if durations[idx] > self.min_threshold_action_duration:

            if l is not None:
                required_objects = self.action2gt_dict[l]
                objects_present = True
                for obj in required_objects:
                    if obj not in objects:
                        objects_present = False
                        break
                
                if objects_present:
                    if self.check_uncertainity(output):
                        return l


    def get_action(self, features) -> str:
        '''
        Computes the anticipated actions from visuo-temporal features 
        '''
        
        with torch.no_grad():
            sample_rate = self.args.sample_rate
            NONE = self.n_class-1
            
            actions_dict_with_NONE = copy.deepcopy(self.actions_dict)
            actions_dict_with_NONE['NONE'] = NONE

            detected_object_names = self.args.scene_objects

            # Get list of all nodes for the KG
            node_list = readCSV('./datasets/nodelist_kitchen.csv', single_element=True)

            detected_object_names_idx = []
            for obj in detected_object_names:
                if obj in node_list:
                    detected_object_names_idx.append(node_list.index(obj))
            
            detections = torch.zeros((self.args.vocab_size))
            detections[detected_object_names_idx] = 1.
            detections = detections.unsqueeze(0)

            inputs = features[::sample_rate, :]
            inputs = torch.Tensor(inputs).to(self.device)
            
            detected_objs = detections if self.args.kg_attn == True else None
            target_nodes = None # we don't need GT KG nodes for evaluation

            # input shape: 1, num of frames, 2048
            outputs, _, _ = self.model(inputs.unsqueeze(0), detected_objs, target_nodes, mode='test')

            output_action = outputs['action']
            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # find the first none class
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

            output_dur = normalize_duration(output_dur, none_mask.to(self.device))

            return output_action, output_label, output_dur
    
    def ProcessImage(self, request, context):

        # Convert the bytes back to an image.
        image = np.frombuffer(request.data, dtype=np.uint8)

        # Decode JPEG data.
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # IMREAD_COLOR ensures it is read as a color image

        if image is None:
            raise Exception("Could not decode image data")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_history.append(image)

        if len(self.image_history) > self.max_pred_length:
            self.image_history.pop(0)
        
        if len(self.image_history) < self.min_pred_length:
            print ('Collecting frames for history ...... {} / {}'.format(len(self.image_history), self.min_pred_length))
            action = None
        else:
            # get action from history of images
            features = self.compute_current_temporal_feature(self.image_history)
            action_out, action_label, action_dur = self.get_action(features)
            action_name_list = self.get_action_name_list(list(action_label[0].cpu().numpy()))
            action = self.compute_action(action_name_list, action_dur[0], action_out[0])
            # action = 'add_salt'
        
        # Replace result with the output action string
        return lp_pb2.ProcessResponse(result=action)


def serve():  # Create an instance of your model
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_processor_servicer = ImageProcessorServicer()  # Pass the model to the service
    lp_pb2_grpc.add_ImageProcessorServicer_to_server(image_processor_servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        # Handle any cleanup here, if necessary
        print('Server shutdown.')

if __name__ == '__main__':
    serve()

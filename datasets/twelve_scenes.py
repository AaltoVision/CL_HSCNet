from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data
import pickle

from .utils import *


class TwelveScenes(data.Dataset):
    def __init__(self, root, dataset='12S', scene='apt2/bed', split='train', 
                    model='hscnet', aug='True', Buffer= False, dense_pred_flag= False, exp='exp_name'):
        self.Buffer = Buffer
        # self.buffer_size = buffer_size
        self.intrinsics_color = np.array([[572.0, 0.0,     320.0],
                       [0.0,     572.0, 240.0],
                       [0.0,     0.0,  1.0]])
                       
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)

        
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'12Scenes')
        self.scene = scene
        if self.dataset == '12S':
            self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        else: 
            self.scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
            self.transl = [[0,-20,0],[0,-20,0],[20,0,0],[20,0,0],[25,0,0],
                    [20,0,0],[-20,0,0],[-25,5,0],[-20,0,0],[-20,-5,0],[0,20,0],
                    [0,20,0]]
            if self.dataset == 'i12S':
                self.ids = [0,1,2,3,4,5,6,7,8,9,10,11]
            else:
                self.ids = [7,8,9,10,11,12,13,14,15,16,17,18]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')))

        self.split, scene = split.split('_')

        self.obj_suffixes = ['.color.jpg', '.pose.txt', '.depth.png', 
                '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        
        if self.dataset == '12S' or self.split == 'test':
            with open(os.path.join(self.root, self.scene, 
                    '{}{}'.format(self.split, '.txt')), 'r') as f:
                self.frames = f.readlines()
        else:
            self.frames = []
            
            with open(os.path.join(self.root, scene, 
                    '{}{}'.format(self.split, '.txt')), 'r') as f:
                    frames = f.readlines()
                    self.frames = [scene + ' ' + frame for frame in frames ]

        if self.Buffer:
            if self.dataset == 'i12S':
                split_buffer = 'train_buffer_{}'.format(exp)
                with open(os.path.join(self.root, '{}{}'.format(split_buffer, '.txt')), 'r') as f:
                    self.frames_buffer = f.readlines()
                self.dense_pred_prefix = 'dense_pred_{}'.format(exp)
                self.dense_pred_path = os.path.join(self.root, self.dense_pred_prefix)
                self.dense_pred_flag = dense_pred_flag

            if self.dataset == 'i19S':
                split_buffer = 'train_buffer_{}'.format(exp)
                root_19S = '/data/dataset/7_12_scenes/data/19Scenes'
                with open(os.path.join(root_19S, '{}{}'.format(split_buffer, '.txt')), 'r') as f:
                    self.frames_buffer = f.readlines()
                self.dense_pred_prefix = 'dense_pred_{}'.format(exp)
                self.dense_pred_path = os.path.join(root_19S, self.dense_pred_prefix)
                self.dense_pred_flag = dense_pred_flag

                

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        if self.dataset != '12S' and self.split == 'train':
            scene, frame_id = frame.split(' ')
            centers = self.scene_data[scene][2]
        else: 
            scene = self.scene
            if self.split == 'train':
                centers = self.centers
        
        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 'data', 
                    obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
        
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        pose = np.loadtxt(objs['pose'])
        if self.dataset != '12S' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(self.scene_data[scene][0])

        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
        
        lbl = cv2.imread(objs['label'],-1)

        ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000

        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
    
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
                mask, lbl, self.aug)
        
        if self.model == 'hscnet':
            coord = coord - ctr_coord
               
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
       
        if  self.dataset=='12S':
            lbl_1 = (lbl - 1) // 25
        else:
            lbl_1 = (lbl - 1) // 25 + 25*self.scene_data[scene][1]
        lbl_2 = ((lbl - 1) % 25) 
        
        if  self.dataset=='12S':
            N1=25
        if  self.dataset=='i12S':
            N1=300
        if  self.dataset=='i19S':
            N1=475
        
        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, 
                    coord, mask, lbl_1, lbl_2, N1)

        if self.Buffer:
            data_buffer = self.buffer(index)
        else:
            data_buffer = ()
        data_ori = (img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame)

        return data_ori, data_buffer 

    def buffer(self, index):
        max_length = len(self.frames_buffer)
        if index >= max_length:
            index = index % max_length
        frame = self.frames_buffer[index].rstrip('\n')
        if self.dataset != '12S' and self.split == 'train':
            if self.dataset == 'i12S':
                scene, frame_id = frame.split(' ')
                centers = self.scene_data[scene][2]
            if self.dataset == 'i19S':
                scene_name = frame.split(' ')[0]
                if scene_name in self.scenes:
                    scene, frame_id = frame.split(' ')
                    centers = self.scene_data[scene][2]
                else:
                    scene, seq_id, frame_id = frame.split(' ') 
                    img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame, dense_pred = self.load_from_7S(frame, index)

                    return (img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame, dense_pred)


        else: 
            scene = self.scene
            if self.split == 'train':
                centers = self.centers
            
        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 'data', 
                    obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data 
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        pose = np.loadtxt(objs['pose'])
        if self.dataset != '12S' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(self.scene_data[scene][0])

        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
            
        lbl = cv2.imread(objs['label'],-1)

        ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000

        depth = cv2.imread(objs['depth'],-1)
            
        pose[0:3,3] = pose[0:3,3] * 1000
        
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
                mask, lbl, self.aug)
            
        if self.model == 'hscnet':
            coord = coord - ctr_coord
                
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
        
        if  self.dataset=='12S':
            lbl_1 = (lbl - 1) // 25
        else:
            lbl_1 = (lbl - 1) // 25 + 25*self.scene_data[scene][1]
        lbl_2 = ((lbl - 1) % 25) 
            
        if  self.dataset=='12S':
                N1=25
        if  self.dataset=='i12S':
                N1=300
        if  self.dataset=='i19S':
                N1=475

        if self.dense_pred_flag:
            # load the dense preds from earlier cycles
            pkl_file = open('{}/{}_{}.pkl'.format(self.dense_pred_path, 'dense_pred', index), 'rb')
            preds = pickle.load(pkl_file)
            #preds = np.load('{}/{}_{}.npz'.format(self.dense_pred_path, self.dense_pred_prefix, index))
            dense_pred_lbl_2 = preds['lbl_2']
            dense_pred_lbl_1 = preds['lbl_1']
            coord_pred = preds['coord_pred']
            dense_pred_lbl_2 = torch.from_numpy(dense_pred_lbl_2)
            dense_pred_lbl_1 = torch.from_numpy(dense_pred_lbl_1)
            coord_pred = torch.from_numpy(coord_pred)
            dense_pred = (dense_pred_lbl_1, dense_pred_lbl_2,coord_pred )

        else:
            dense_pred = ()

        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img,
                coord, mask, lbl_1, lbl_2, N1)

        return (img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame, dense_pred)  



    def load_from_7S(self,frame, index):
        scene_name, seq_id, frame_id = frame.split(' ') 
        intrinsics_color = np.array([[525.0, 0.0,     320.0],
            [0.0,     525.0, 240.0],
            [0.0,     0.0,  1.0]])

        intrinsics_depth = np.array([[585.0, 0.0,     320.0],
                    [0.0,     585.0, 240.0],
                    [0.0,     0.0,  1.0]])
        
        intrinsics_depth_inv = np.linalg.inv(intrinsics_depth)
        intrinsics_color_inv = np.linalg.inv(intrinsics_color)
        root = '/data/dataset/7_12_scenes/data/7Scenes'
        calibration_extrinsics = np.loadtxt(os.path.join(root, 
                        'sensorTrans.txt'))


        scenes = ['chess','fire','heads','office','pumpkin',
                            'redkitchen','stairs']
        transl = [[0,0,0],[10,0,0],[-10,0,0],[0,10,0],[0,-10,0],
                            [0,0,10],[0,0,-10]]
        ids = [0,1,2,3,4,5,6]
        scene_data = {}
        for scene, t, d in zip(scenes, transl, ids):
            scene_data[scene] = (t, d, np.load(os.path.join(root,scene, 'centers.npy')),np.loadtxt(os.path.join(root, scene,'translation.txt')))

        obj_suffixes = ['.color.png','.pose.txt', '.depth.png','.label.png']
        obj_keys = ['color','pose', 'depth','label']


        centers = scene_data[scene_name][2] 
        scene_ctr = scene_data[scene_name][3] 

        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in obj_suffixes]
        obj_files_full = [os.path.join(root, scene_name, 
                    seq_id, obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(obj_keys, obj_files_full):
            objs[key] = data
    
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose = np.loadtxt(objs['pose'])

        pose[0:3,3] = pose[0:3,3] - scene_ctr
        if self.dataset != '7S' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(scene_data[scene_name][0])      
        
        lbl = cv2.imread(objs['label'],-1)
        ctr_coord = centers[np.reshape(lbl,(-1))-1,:]

        ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000
        
        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
    
        depth[depth==65535] = 0
        depth = depth * 1.0
        depth = get_depth(depth, calibration_extrinsics, 
            intrinsics_color, intrinsics_depth_inv)
        coord, mask = get_coord(depth, pose, intrinsics_color_inv)
        # comment the data augmentation 
        # img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
        #         mask, lbl, self.aug)
        
        coord = coord - ctr_coord
    
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
    
        lbl_1 = (lbl - 1)//25 + 25*scene_data[scene_name][1]
        lbl_2 = ((lbl - 1) % 25)      

        N1=475

        if self.dense_pred_flag:
            # load the dense preds from earlier cycles
            pkl_file = open('{}/{}_{}.pkl'.format(self.dense_pred_path, 'dense_pred', index), 'rb')
            preds = pickle.load(pkl_file)
            #preds = np.load('{}/{}_{}.npz'.format(self.dense_pred_path, self.dense_pred_prefix, index))
            dense_pred_lbl_2 = preds['lbl_2']
            dense_pred_lbl_1 = preds['lbl_1']
            coord_pred = preds['coord_pred']
            dense_pred_lbl_2 = torch.from_numpy(dense_pred_lbl_2)
            dense_pred_lbl_1 = torch.from_numpy(dense_pred_lbl_1)
            coord_pred = torch.from_numpy(coord_pred)
            dense_pred = (dense_pred_lbl_1, dense_pred_lbl_2,coord_pred )

        else:
            dense_pred = ()

        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img,
                coord, mask, lbl_1, lbl_2, N1)

        return (img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame, dense_pred)
                        


import os
import random
import numpy as np
import torch
import pickle
import pdb
from collections import ChainMap
# random.seed(521)
# deletelist = ['heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

def get_img2subscenes(data_path, dataset):
    #### load them in a more efficient way
    if dataset == '7Scenes':
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    if dataset == '12Scenes':
        scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
    if dataset == '19Scenes':
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs', 'apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
    for i in range(len(scenes)):
        scenes[i] = scenes[i].replace('/', '_')

            
    for i  in range(len(scenes)):
        with open(data_path +'/{}/img2subscene/{}.pkl'.format(dataset, scenes[i]),'rb') as data:
            scenes[i] = pickle.load(data)

    img2subscenes = dict(ChainMap(*scenes))

    return img2subscenes

class createBuffer():
    def __init__(self, buffer_size=256, data_path='', exp='exp_name', dataset = 'i7S'):

        if dataset == 'i7S':
            self.dataset = '7Scenes'
        if dataset == 'i12S':
            self.dataset = '12Scenes'
        if dataset == 'i19S':
            self.dataset = '19Scenes'


        self.buffer_scenes = []
        self.buffer_size = buffer_size
        # with open('{}/{}/train.txt'.format(data_path, self.dataset), 'r') as f:
        #     self.frames = f.readlines()
        self.buffer_list = []
        self.buffer_fn = '{}/{}/train_buffer_{}.txt'.format(data_path,self.dataset, exp)
        print(self.buffer_fn)
        self.N = 0
        self.buff_id = 100000000
        self.buffer_class = dict()
        self.img2sub_bin = []
        self.largest = 'awszfdeasqssq'
        self.dense_pred_path = '{}/{}/dense_pred_{}'.format(data_path,self.dataset, exp)
        if not os.path.exists(self.dense_pred_path):
            os.mkdir(self.dense_pred_path)
        self.img2subscenes = get_img2subscenes(data_path, self.dataset)


    def add_buffer_dense(self, frame, preds=None):

        frame = frame[0]
        self.N += 1
        if len(self.buffer_list)  < self.buffer_size:
            self.buffer_list.append(frame)
            buff_id = len(self.buffer_list) - 1
        else:
            s = int(random.random() * self.N)
            if s < self.buffer_size:
                self.buffer_list[s] = frame
                buff_id = s
            else:
                self.buffer_list[0] = frame
                buff_id = 0
        # save dense file
        if preds is not None:
            coord_pred = preds[0].squeeze().data.cpu().numpy()
            lbl_1_onehot = preds[1].squeeze().data.cpu().numpy()
            lbl_2_onehot = preds[2].squeeze().data.cpu().numpy()
            pkl_save = {'coord_pred':coord_pred, 'lbl_1': lbl_1_onehot, 'lbl_2': lbl_2_onehot}
            pkl_file = open('{}/dense_pred_{}.pkl'.format(self.dense_pred_path, buff_id), 'wb')
            pickle.dump(pkl_save, pkl_file)

        # dump to buffer file
        buffer_training = open(self.buffer_fn, 'w+')
        for item in self.buffer_list:
            buffer_training.write('{}\n'.format(item))

    def add_imb_buffer(self, fname, preds=None, nc=None):
        # frame: scene_name seq_num image_name
        frame = fname[0]
        scene = frame.split(' ')[0]

        # if 1st sample then init scene dict to contain indices of occupied positions
        if nc == 0:
            self.buffer_class[scene] = []

        # if buffer available store
        if len(self.buffer_list) < self.buffer_size:
            self.buffer_list.append(frame)
            buff_id = len(self.buffer_list)-1
            self.buffer_class[scene].append(buff_id)
        else:
            # if current scene is not the largest
            if scene != self.largest:
                # get the indices alloted to largest class and sample
                largest_inst = self.buffer_class[self.largest]
                buff_id = random.sample(largest_inst,1)[0]

                self.buffer_list[buff_id] = frame
                self.buffer_class[scene].append(buff_id) # add buff_id to scene dict
                self.buffer_class[self.largest].remove(buff_id) # remove buff_id from largest_scene dict
            else:
                mc = len(self.buffer_class[scene])
                uid = random.uniform(0,1)
                if uid <= mc/nc:
                    self_inst = self.buffer_class[scene]
                    buff_id = random.sample(self_inst,1)[0]

                    self.buffer_list[buff_id] = frame
                    # no need to add or remove as it is self-substituting

                else:
                    return 0

        
        # TODO online processing
        # find largest class
        scene_num = []
        scene_name = []
        for sc in self.buffer_class:
            #print(self.buffer_class[sc])
            scene_num.append(len(self.buffer_class[sc]))
            scene_name.append(sc)

        self.largest = scene_name[np.argsort(-np.array(scene_num))[0]]
        '''
        
        self.N += 1
        if len(self.buffer_list) < self.buffer_size:
            self.buffer_list.append(frame)
            buff_id = len(self.buffer_list)-1
            self.buffer_class[scene].append(buff_id)
        else:
            s = int(random.random() * self.N)
            if s < self.buffer_size:
                if scene != self.largest:
                    largest_inst = self.buffer_class[self.largest]
                    buff_id = random.sample(largest_inst,1)[0]
                    self.buffer_list[buff_id] = frame
                    self.buffer_class[scene].append(buff_id)
                    self.buffer_class[self.largest].remove(buff_id)
                else:
                     # no need to add or remove as it is self-substituting
                    self_inst = self.buffer_class[scene]
                    buff_id = random.sample(self_inst,1)[0]
                    self.buffer_list[buff_id] = frame

            else:
                self.buffer_list[0] = frame
                buff_id = 0
        
  
        scene_coverscore = []
        scene_name = []
        for sc in self.buffer_class:
            score_lists = []
            for id in self.buffer_class[sc]:
                frame_name = self.buffer_list[id]
                score_list= list(self.img2subscenes[frame_name])
                score_lists += score_list
            score_lists = set(score_lists)
            coverscore = len(score_lists) / 625
            scene_coverscore.append(coverscore)
            scene_name.append(sc)
        # print(scene_coverscore)
        self.largest = scene_name[np.argsort(-np.array(scene_coverscore))[0]]
        # print(self.largest)
        '''

        # save dense file
        if preds is not None:
            coord_pred = preds[0].squeeze().data.cpu().numpy()
            lbl_1_onehot = preds[1].squeeze().data.cpu().numpy()
            lbl_2_onehot = preds[2].squeeze().data.cpu().numpy()
            pkl_save = {'coord_pred': coord_pred, 'lbl_1': lbl_1_onehot, 'lbl_2': lbl_2_onehot}
            pkl_file = open('{}/dense_pred_{}.pkl'.format(self.dense_pred_path, buff_id), 'wb')
            pickle.dump(pkl_save, pkl_file)

        # dump to buffer file
        buffer_training = open(self.buffer_fn, 'w+')
        for item in self.buffer_list:
            buffer_training.write('{}\n'.format(item))

    def add_bal_buff(self, fname, preds=None, nc=None):
        # frame: scene_name seq_num image_name
        frame = fname[0]
        scene = frame.split(' ')[0]
        valid_subsc = np.array(list(self.img2subscenes[frame]))
        subsc_mask = -1*np.ones(625)
        subsc_mask[valid_subsc-1] = 1

        # if 1st sample then init scene dict to contain indices of occupied positions
        if nc == 0:
            self.buffer_class[scene] = []

        # if buffer available store
        if len(self.buffer_list) < self.buffer_size:
            self.buffer_list.append(frame)
            buff_id = len(self.buffer_list)-1
            self.buffer_class[scene].append(buff_id)
            self.img2sub_bin.append(subsc_mask)
        else:
            # if current scene is not the largest
            if scene != self.largest:
                # get the indices alloted to largest class and sample
                largest_inst = self.buffer_class[self.largest]
                buff_id = random.sample(largest_inst,1)[0]

                '''
                self.buffer_list[self.buff_id] = frame
                self.buffer_class[scene].append(self.buff_id) # add buff_id to scene dict
                self.img2sub_bin[self.buff_id] = subsc_mask
                self.buffer_class[self.largest].remove(self.buff_id) # remove buff_id from largest_scene dict
                buff_id = self.buff_id
                '''
                self.buffer_list[buff_id] = frame
                self.buffer_class[scene].append(buff_id)  # add buff_id to scene dict
                self.img2sub_bin[buff_id] = subsc_mask
                self.buffer_class[self.largest].remove(buff_id)  # remove buff_id from largest_scene dict
            else:
                # mc = len(self.buffer_class[scene])
                # uid = random.uniform(0, 1)
                # if uid <= mc / nc:

                # check if keep/drop (flag=1/0)
                flag = self.compute_subsc_difference(scene, subsc_mask)
                # make the replacement non deterministic for flag==0 items
                mc = len(self.buffer_class[scene])
                uid = random.uniform(0, 1)
                if uid <= mc / nc:
                    flag = 1

                if flag == 1:
                    self_inst = self.buffer_class[scene]
                    buff_id = random.sample(self_inst,1)[0]

                    #self.buffer_list[self.buff_id] = frame
                    #self.img2sub_bin[self.buff_id] = subsc_mask
                    #buff_id = self.buff_id
                    self.buffer_list[buff_id] = frame
                    self.img2sub_bin[buff_id] = subsc_mask
                    # no need to add or remove as it is self-substituting
                else:
                    return 0

        # TODO online processing
        # find largest class
        scene_num = []
        scene_name = []
        for sc in self.buffer_class:
            #print(self.buffer_class[sc])
            scene_num.append(len(self.buffer_class[sc]))
            scene_name.append(sc)

        self.largest = scene_name[np.argsort(-np.array(scene_num))[0]]

        # compute overlap score of the current largest scene
        if len(self.buffer_list) == self.buffer_size:
            # collect the binary img 2 subscene vectors
            # require : self.buffer_class to get scene to buffer list mapping
            #         : img2sub_bin to get binary vectors from buffer list mapping
            self_inst = self.buffer_class[self.largest]
            bin_vecs = np.array([self.img2sub_bin[k] for k in self_inst])
            S = bin_vecs@bin_vecs.T
            S = S.sum(1)
            self.buff_id = self_inst[np.argsort(-S)[0]]
      

        # save dense file
        if preds is not None:
            coord_pred = preds[0].squeeze().data.cpu().numpy()
            lbl_1_onehot = preds[1].squeeze().data.cpu().numpy()
            lbl_2_onehot = preds[2].squeeze().data.cpu().numpy()
            pkl_save = {'coord_pred': coord_pred, 'lbl_1': lbl_1_onehot, 'lbl_2': lbl_2_onehot}
            pkl_file = open('{}/dense_pred_{}.pkl'.format(self.dense_pred_path, buff_id), 'wb')
            pickle.dump(pkl_save, pkl_file)
        # dump to buffer file
        buffer_training = open(self.buffer_fn, 'w+')
        for item in self.buffer_list:
            buffer_training.write('{}\n'.format(item))


    def compute_subsc_difference(self,scene, subsc_mask):
        '''
        compute  the difference between imcoming image and current sub scene
        scene: name of the current scene
        sub_mask: array 1 * 625
        '''
        buff_subsc_lists = []
        for id in self.buffer_class[scene]:
            frame_name = self.buffer_list[id]
            buff_subsc_list= list(self.img2subscenes[frame_name])
            buff_subsc_lists += buff_subsc_list
        buff_subsc_lists = set(buff_subsc_lists)
        buff_subsc_valid = np.array(list(buff_subsc_lists))
        buff_subsc_mask = np.array([0] * 625)
        try:buff_subsc_mask[buff_subsc_valid-1] = 1
        except:
            import pdb
            pdb.set_trace()
        diff = subsc_mask - buff_subsc_mask
        if 1 in diff:
            return True
        else:
            return False








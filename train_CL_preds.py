from __future__ import division

import sys
import os
import random
import argparse
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

from models import get_model
from datasets import get_dataset
from loss import *
from utils import *
from buffer import createBuffer
import time

def train(args):
    # prepare datasets
    if args.dataset == 'i19S':
        datasetSs = get_dataset('7S')
        datasetTs = get_dataset('12S')
    else:
        if args.dataset in ['7S', 'i7S']:
            dataset_get = get_dataset('7S')
        if args.dataset in ['12S', 'i12S']:
            dataset_get = get_dataset('12S')


    # loss
    reg_loss = EuclideanLoss()
    if args.model == 'hscnet':
        cls_loss = CELoss()
        if args.dataset in ['i7S', 'i12S', 'i19S']:
            w1, w2, w3 = 1, 1, 100000
        else:
            w1, w2, w3 = 1, 1, 10

    # prepare model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset)
    model.init_weights()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, eps=1e-8,
                                 betas=(0.9, 0.999))

    # resume from existing or start a new session
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format\
                  (args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch{})".format(args.resume,
                  checkpoint['epoch']))
            save_path = Path(args.resume)
            args.save_path = save_path.parent
            #start_epoch = checkpoint['epoch'] + 1
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.exit()
    else:
        if args.dataset in ['i7S', 'i12S', 'i19S']:
            model_id = "{}-{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                        args.exp_name, args.dataset, args.model, args.init_lr, args.n_iter,
                        args.batch_size, int(args.aug), args.train_id)
        else:
            model_id = "{}-{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                        args.exp_name, args.dataset, args.scene.replace('/','.'),
                        args.model, args.init_lr, args.n_iter, args.batch_size,
                        int(args.aug), args.train_id)
        save_path = Path(model_id)
        args.save_path = 'checkpoints'/save_path
        args.save_path.mkdir(parents=True, exist_ok=True)
        start_epoch = 1

    # Continual learning over scenes
    buffer = createBuffer(data_path=args.data_path, exp=args.exp_name, buffer_size=args.buffer_size, dataset= args.dataset)
    if args.dataset == 'i7S':
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    if args.dataset == 'i12S':
        scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
    if args.dataset == 'i19S':
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs', 'apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']

    
    for i,scene in enumerate(scenes):
        # if not first scene

        if args.dataset in ['i7S', 'i12S']:
            if i > 0:
                dataset = dataset_get(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=args.aug, Buffer=True, dense_pred_flag=args.dense_pred, exp=args.exp_name)
            else:
                dataset = dataset_get(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=args.aug, Buffer=False, exp=args.exp_name)

            trainloader = data.DataLoader(dataset, batch_size=args.batch_size,
                                        num_workers=4, shuffle=True) 


            buffer_dataset = dataset_get(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=False, Buffer=False, dense_pred_flag=args.dense_pred, exp=args.exp_name)
            buffer_trainloader = data.DataLoader(buffer_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

        if args.dataset == 'i19S':
            if i == 0:
                dataset = datasetSs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=args.aug, Buffer=False, exp=args.exp_name)
                buffer_dataset = datasetSs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=False, Buffer=False, dense_pred_flag=args.dense_pred, exp=args.exp_name)
            if i >0 and i < 7:
                dataset = datasetSs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=args.aug, Buffer=True, dense_pred_flag=args.dense_pred, exp=args.exp_name)
                buffer_dataset = datasetSs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=False, Buffer=False, dense_pred_flag=args.dense_pred, exp=args.exp_name)
            if i >= 7:          
                dataset = datasetTs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=args.aug, Buffer=True, dense_pred_flag=args.dense_pred, exp=args.exp_name)
                buffer_dataset = datasetTs(args.data_path, args.dataset, args.scene, split='train_{}'.format(scene),
                                    model=args.model, aug=False, Buffer=False, dense_pred_flag=args.dense_pred, exp=args.exp_name)

            trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

            buffer_trainloader = data.DataLoader(buffer_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
      
        # start training
        
        args.n_epoch = int(np.ceil(args.n_iter * args.batch_size / len(dataset)))
        
        #for epoch in range(start_epoch, start_epoch + args.n_epoch+1):
        for epoch in range(1, args.n_epoch+1):
            lr = args.init_lr

            model.train()
            train_loss_list = []
            coord_loss_list = []
            if args.model == 'hscnet':
                lbl_1_loss_list = []
                lbl_2_loss_list = []

            for _, (data_ori, data_buffer) in enumerate(tqdm(trainloader)):
                img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, _ = data_ori

                if mask.sum() == 0:
                    continue
                optimizer.zero_grad()

                img = img.to(device)
                coord = coord.to(device)
                mask = mask.to(device)
                train_loss, coord_loss, lbl_1_loss, lbl_2_loss = loss(img, coord, mask, lbl_1, lbl_2, lbl_1_oh,
                                lbl_2_oh, model, reg_loss, cls_loss, device, w1, w2, w3)


                
                # compute loss for buffer if not first scene
                if i > 0 :
                    # sample a random minibatch from buffer dataloader
                    img_buff, coord_buff, mask_buff, lbl_1_buff, lbl_2_buff, lbl_1_oh_buff, lbl_2_oh_buff, _, dense_pred = data_buffer

                    if mask_buff.sum() == 0:
                        continue
                    img_buff = img_buff.to(device)
                    coord_buff = coord_buff.to(device)
                    mask_buff = mask_buff.to(device)

                    buff_loss = loss_buff_DK(img_buff, coord_buff, mask_buff, lbl_1_buff, lbl_2_buff, lbl_1_oh_buff,
                                lbl_2_oh_buff, model, reg_loss, cls_loss, device, w1, w2, w3, dense_pred=dense_pred)

                    train_loss+= 1 * buff_loss


                coord_loss_list.append(coord_loss.item())
                if args.model == 'hscnet':
                    lbl_1_loss_list.append(lbl_1_loss.item())
                    lbl_2_loss_list.append(lbl_2_loss.item())
                train_loss_list.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

            with open(args.save_path/args.log_summary, 'a') as logfile:
                if args.model == 'hscnet':
                    logtt = 'task {}:Epoch {}/{} - lr: {} - reg_loss: {} - cls_loss_1: {}' \
                    ' - cls_loss_2: {} - train_loss: {} '.format(scene,
                     epoch, args.n_epoch, lr, np.mean(coord_loss_list),
                     np.mean(lbl_1_loss_list), np.mean(lbl_2_loss_list),
                     np.mean(train_loss_list))
                else:
                    logtt = 'Epoch {}/{} - lr: {} - reg_loss: {} - train_loss: {}' \
                            '\n'.format(
                             epoch, args.n_epoch, lr, np.mean(coord_loss_list),
                             np.mean(train_loss_list))
                print(logtt)
                logfile.write(logtt)

            if epoch % int(np.floor(args.n_epoch / 1.)) == 0:
                save_state(args.save_path, epoch, model, optimizer)

        #start_epoch = epoch

        # add buffer data
        with torch.no_grad():
            for i, (data_ori, data_buffer) in enumerate(tqdm(buffer_trainloader)):
                img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, frame = data_ori

                if mask.sum() == 0:
                    continue
                optimizer.zero_grad()

                img = img.to(device)
                coord = coord.to(device)
                mask = mask.to(device)

                if args.dense_pred:
                    # predictions
                    lbl_1 = lbl_1.to(device)
                    lbl_2 = lbl_2.to(device)
                    lbl_1_oh = lbl_1_oh.to(device)
                    lbl_2_oh = lbl_2_oh.to(device)
                    coord_pred, lbl_2_pred, lbl_1_pred = model(img, lbl_1_oh,
                                                               lbl_2_oh)
                    preds = (coord_pred, lbl_1_pred, lbl_2_pred)
                    if args.sampling == 'CoverageS':
                        buffer.add_bal_buff(frame, preds, i)
                    if args.sampling == 'Imgbal':
                        buffer.add_imb_buffer(frame, preds, i)
                    if args.sampling == 'Random':
                        buffer.add_buffer_dense(frame, preds)
                else:
                    if args.sampling == 'CoverageS':
                        buffer.add_bal_buff(frame, nc=i)
                    if args.sampling == 'Imgbal':
                        buffer.add_imb_buffer(frame, nc=i)
                    if args.sampling == 'Random':
                        buffer.add_buffer_dense(frame)


    save_state(args.save_path, epoch, model, optimizer)

def loss(img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, model, reg_loss, cls_loss, device, w1, w2, w3, dense_pred=None):
    
    lbl_1 = lbl_1.to(device)
    lbl_2 = lbl_2.to(device)
    lbl_1_oh = lbl_1_oh.to(device)
    lbl_2_oh = lbl_2_oh.to(device)
    coord_pred, lbl_2_pred, lbl_1_pred = model(img,lbl_1_oh,
                                                lbl_2_oh)

    lbl_1_loss = cls_loss(lbl_1_pred, lbl_1 , mask )
    lbl_2_loss = cls_loss(lbl_2_pred, lbl_2 , mask )
    coord_loss = reg_loss(coord_pred, coord , mask )

    train_loss = w3*coord_loss + w1*lbl_1_loss + w2*lbl_2_loss
        

    return train_loss, coord_loss, lbl_1_loss, lbl_2_loss

def loss_buff(img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, model, reg_loss, cls_loss, device, w1, w2, w3, dense_pred=None):
    lbl_1 = lbl_1.to(device)
    lbl_2 = lbl_2.to(device)
    lbl_1_oh = lbl_1_oh.to(device)
    lbl_2_oh = lbl_2_oh.to(device)
    coord_pred, lbl_2_pred, lbl_1_pred = model(img,lbl_1_oh,
                                                lbl_2_oh)

    lbl_1_loss = cls_loss(lbl_1_pred, lbl_1, mask)
    lbl_2_loss = cls_loss(lbl_2_pred, lbl_2, mask)
    coord_loss = reg_loss(coord_pred, coord, mask)

    train_loss = w3 * coord_loss + w1 * lbl_1_loss + w2 * lbl_2_loss

    if dense_pred:
        dense_pred_lbl_1 = dense_pred[0].to(device)
        dense_pred_lbl_2 = dense_pred[1].to(device)
        dense_pred_coord = dense_pred[2].to(device)

        L2_loss = nn.MSELoss()

        buff_lbl_1_loss = L2_loss(lbl_1_pred, dense_pred_lbl_1)
        buff_lbl_2_loss = L2_loss(lbl_2_pred, dense_pred_lbl_2)
        buff_coord_loss = L2_loss(coord_pred, dense_pred_coord)

        train_loss += 0.5 * (w1 * buff_lbl_1_loss + w2 * buff_lbl_2_loss + w3 * buff_coord_loss)

    #return buff_lbl_1_loss, buff_lbl_2_loss
    return train_loss


def loss_buff_DK(img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh, model, reg_loss, cls_loss, device, w1, w2, w3, dense_pred=None):
    ## teacher loss as a upper bound ##
    lbl_1 = lbl_1.to(device)
    lbl_2 = lbl_2.to(device)
    lbl_1_oh = lbl_1_oh.to(device)
    lbl_2_oh = lbl_2_oh.to(device)
    coord_pred, lbl_2_pred, lbl_1_pred = model(img,lbl_1_oh,
                                                lbl_2_oh)

    lbl_1_loss = cls_loss(lbl_1_pred, lbl_1, mask)
    lbl_2_loss = cls_loss(lbl_2_pred, lbl_2, mask)
    coord_loss = reg_loss(coord_pred, coord, mask)
    # student VS gt loss

    if dense_pred:
        train_loss = 0.5 * (w3 * coord_loss + w1 * lbl_1_loss + w2 * lbl_2_loss)
        dense_pred_lbl_1 = dense_pred[0].to(device)
        dense_pred_lbl_2 = dense_pred[1].to(device)
        dense_pred_coord = dense_pred[2].to(device)

        L2_loss = nn.MSELoss()

        buff_lbl_1_loss = L2_loss(lbl_1_pred, dense_pred_lbl_1)
        buff_lbl_2_loss = L2_loss(lbl_2_pred, dense_pred_lbl_2)
        buff_coord_loss = L2_loss(coord_pred, dense_pred_coord)
        
        # teacher loss
        buff_teacher_loss = reg_loss(dense_pred_coord, coord, mask)
        buff_student_loss = reg_loss(coord_pred, coord, mask)
        if buff_student_loss > buff_teacher_loss:
            train_loss += 0.5 * (w1 * buff_lbl_1_loss + w2 * buff_lbl_2_loss + w3 * buff_coord_loss)
        else:
            train_loss += 0.5 * (w1 * buff_lbl_1_loss + w2 * buff_lbl_2_loss)
    else:
        train_loss = (w3 * coord_loss + w1 * lbl_1_loss + w2 * lbl_2_loss)
    return train_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hscnet")
    parser.add_argument('--model', nargs='?', type=str, default='hscnet',
                        choices=('hscnet', 'scrnet'),
                        help='Model to use [\'hscnet, scrnet\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='7S',
                        choices=('7S', '12S', 'i7S', 'i12S', 'i19S',
                        'Cambridge'), help='Dataset to use')
    parser.add_argument('--scene', nargs='?', type=str, default='heads',
                        help='Scene')
    parser.add_argument('--n_iter', nargs='?', type=int, default=30000,
                        help='# of iterations (to reproduce the results from ' \
                        'the paper, 300K for 7S and 12S, 600K for ' \
                        'Cambridge, 900K for the combined scenes)')
    parser.add_argument('--init_lr', nargs='?', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--aug', nargs='?', type=str2bool, default=True,
                        help='w/ or w/o data augmentation')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to saved model to resume from')
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to dataset')
    parser.add_argument('--log-summary', default='progress_log_summary.txt',
                        metavar='PATH',
                        help='txt where to save per-epoch stats')
    parser.add_argument('--train_id', nargs='?', type=str, default='',
                        help='An identifier string'),
    parser.add_argument('--dense_pred', nargs='?', type=str2bool, default=False,
                        help='store dense predictions in buffer')
    parser.add_argument('--exp_name', nargs='?', type=str, default='exp',
                        help='store dense predictions in buffer')
    parser.add_argument('--buffer_size', nargs='?', type=int, default=1024,
                        help='the length of buffer size')
    parser.add_argument('--sampling', nargs='?', type=str, default='Random',
                        help='choose from Random, Imgbal, CoverageS')
    
    args = parser.parse_args()

    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin',
                              'redkitchen','stairs']:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == '12S':
        if args.scene not in ['apt1/kitchen', 'apt1/living', 'apt2/bed',
                              'apt2/kitchen', 'apt2/living', 'apt2/luke',
                              'office1/gates362', 'office1/gates381',
                              'office1/lounge', 'office1/manolis',
                              'office2/5a', 'office2/5b']:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in ['GreatCourt', 'KingsCollege', 'OldHospital',
                              'ShopFacade', 'StMarysChurch']:
            print('Selected scene is not valid.')
            sys.exit()

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.dense_pred:
        print('Dense predictions will be stored in buffer !')
    train(args)
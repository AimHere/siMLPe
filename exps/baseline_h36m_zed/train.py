import argparse
import os, sys
import json
import math
import numpy as np
import copy

from config import config
from model import siMLPe as Model
from datasets.h36m_zed import H36MZedDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
#from datasets.h36m_eval import H36MEval
from datasets.h36m_zed_eval import H36MZedEval

from datasets.h36m_zed import exp_distance_torch, quat_distance_torch

from test import test

from time import localtime, strftime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zed_utilities import ForwardKinematics_Torch

torch.autograd.set_detect_anomaly(True)

BONE_COUNT = 18

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--rotations', action='store_true', help='=train on rotations')
parser.add_argument('--quaternions', action = 'store_true', help = '=train on quaternions')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_zed_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_zed_motion_input, h36m_zed_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,n,c = h36m_zed_motion_input.shape
        h36m_zed_motion_input__ = h36m_zed_motion_input.clone()
        h36m_zed_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], h36m_zed_motion_input__.cuda())
    else:
        h36m_zed_motion_input_ = h36m_zed_motion_input.clone()

    if (args.quaternions):
        OUTPUT_BONE_COMPONENTS = 4
        config.data_component_size = 4
    else:
        OUTPUT_BONE_COMPONENTS = 3
        config.data_component_size = 3

    nan_count = torch.where(torch.isnan(h36m_zed_motion_input_))
    if (len(nan_count[0]) > 0):
        print("Nan count in model input is bigger than 0")

    eps = h36m_zed_motion_input.clone().normal_(std=1e-8).cuda()
    motion_pred_ = model(h36m_zed_motion_input_.cuda() + eps)
    nan_count = torch.where(torch.isnan(motion_pred_))

    if (len(nan_count[0]) > 0):
        print("Nan count in model output is bigger than 0")

    motion_pred__ = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], motion_pred_)
    # print("iDCT = ", idct_m[:, :config.motion.h36m_zed_input_length, :])
    # print("MPred_ = ",motion_pred_)
          
    mpnp = motion_pred__.cpu().detach().numpy()
    idctnp = idct_m[:, :config.motion.h36m_zed_input_length, :].detach().cpu().numpy()

    np.savez("BadStuff.npz", idct = idctnp, motion_pred = mpnp, input = h36m_zed_motion_input_.cpu().detach().numpy())
    
    if config.deriv_output:
        offset = h36m_zed_motion_input[:, -1:].cuda()
        motion_pred = motion_pred__[:, :config.motion.h36m_zed_target_length] + offset
    else:
        motion_pred = motion_pred__[:, :config.motion.h36m_zed_target_length]

    b,n,c = h36m_zed_motion_target.shape
    #motion_pred = motion_pred.reshape(b,n,BONE_COUNT,3).reshape(-1,3)
    #h36m_zed_motion_target = h36m_zed_motion_target.cuda().reshape(b,n,BONE_COUNT,3).reshape(-1,3)

    motion_pred = motion_pred.reshape(b,n,BONE_COUNT,OUTPUT_BONE_COMPONENTS)
    h36m_zed_motion_target = h36m_zed_motion_target.cuda().reshape(b,n,BONE_COUNT,OUTPUT_BONE_COMPONENTS)
    
    if (config.loss_rotation_metric):
        print("Rotation Metric used")
        # print("Motion pred: ",motion_pred)
        # print("Motion target: ", h36m_zed_motion_target)

        mpr = motion_pred.reshape([-1, BONE_COUNT, OUTPUT_BONE_COMPONENTS])
        hzmtr = h36m_zed_motion_target.reshape([-1, BONE_COUNT, OUTPUT_BONE_COMPONENTS])
        # Exponential to stop the loss value touching zero, where it breaks everything
        
        eps = mpr.clone().normal_(std = 1e-8)
        edist = exp_distance_torch(mpr, hzmtr + eps)

        print(mpr.shape, hzmtr.shape, edist.shape)
        
        loss = torch.mean(edist)
        print("Mean Loss is ", loss)
        if torch.isnan(loss):
            print("Invalid loss value, halting")
            print(loss)
            exit(0)

        minloss = torch.min(edist)
        maxloss = torch.max(edist)
        print("Loss min: %f, max: %f"%(minloss, maxloss))

        # if (minloss == 0.00000):
        #     print("Zero loss - saving!")
            
        #     np.savez("Zerovals.npz",
        #              losses = edist.cpu().detach().numpy(),
        #              pred = motion_pred.cpu().detach().numpy(),
        #              gt = h36m_zed_motion_target.cpu().detach().numpy(),
        #              )

    elif config.loss_quaternion_distance:
        print("Quaternion Metric used")

        not_three = [i for i in range(18) if i != 3]
        
        
        mpr = motion_pred.reshape([-1, BONE_COUNT, OUTPUT_BONE_COMPONENTS])[:, not_three, :]
        hzmtr = h36m_zed_motion_target.reshape([-1, BONE_COUNT, OUTPUT_BONE_COMPONENTS])[:, not_three, :]
        eps = hzmtr.clone().normal_(std = 1e-8)        
        edist = quat_distance_torch(mpr, hzmtr + eps)
        loss = torch.mean(edist)

        if (torch.isnan(loss)):
            print("Invalid loss value, halting")
            exit(0)
            
        minloss = torch.min(edist)
        maxloss = torch.max(edist)
        print("Loss %f,  min: %f, max: %f"%(loss, minloss, maxloss))        

        # if (minloss == 0.0000):
        #     print("(Q) Zero loss - saving!")
        #     print("Shapes: ", mpr.shape, hzmtr.shape)            
        #     np.savez("Zerovals.npz",
        #              losses = edist.cpu().detach().numpy(),
        #              pred = motion_pred.cpu().detach().numpy(),
        #              gt = h36m_zed_motion_target.cpu().detach().numpy(),
        #              )
        
    elif (config.loss_convert_to_xyz):
        print("Quaternion-to-xyz metric used")

        mpr = motion_pred.reshape([-1, BONE_COUNT, OUTPUT_BONE_COMPONENTS])


        

    elif (config.loss_6D):
        pass

    # elif (config.convert_rotations_to_mpjpe):
    #     pass
    #     # Convert to xyz then take the mpjpe

    else:
        
        motion_pred = motion_pred.reshape(-1, OUTPUT_BONE_COMPONENTS)
        h36m_zed_motion_target = h36m_zed_motion_target.reshape(-1, OUTPUT_BONE_COMPONENTS)

        
        loss = torch.mean(torch.norm(motion_pred - h36m_zed_motion_target, 2, 1))            
        if config.use_relative_loss:
            motion_pred = motion_pred.reshape(b,n,BONE_COUNT,OUTPUT_BONE_COMPONENTS)
            dmotion_pred = gen_velocity(motion_pred)
            motion_gt = h36m_zed_motion_target.reshape(b,n,BONE_COUNT,OUTPUT_BONE_COMPONENTS)
            dmotion_gt = gen_velocity(motion_gt)

            dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,OUTPUT_BONE_COMPONENTS), 2, 1))
            loss = loss + dloss
        else:
            loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, config.cos_lr_max, config.cos_lr_min, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

def mainfunc():

    if (args.quaternions):
        ckpt_name = './model-bone-iter-'
        config.use_quaternions = True
        config.loss_quaternion_distance = True
        config.motion.dim = 72 # 4 * 18
        config.motion_mlp.hidden_dim = config.motion.dim
        config.motion_fc_in.in_features = config.motion.dim
        config.motion_fc_in.in_features = config.motion.dim        
        config.motion_fc_in.out_features = config.motion.dim
        config.motion_fc_out.in_features = config.motion.dim
        config.motion_fc_out.out_features = config.motion.dim
        config.data_component_size = 4
        
    if (args.rotations):
        ckpt_name = './model-bone-iter-'
        config.use_rotations = True
        config.loss_rotation_metric = True
    else:
        ckpt_name = './model-iter-'
        config.use_rotations = False
    
    model = Model(config)
    model.train()
    model.cuda()
    
    config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_train
    dataset = H36MZedDataset(config, 'train', config.data_aug, rotations = args.rotations, quaternions = args.quaternions)

    shuffle = True
    sampler = None
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    eval_config = copy.deepcopy(config)
    eval_config.motion.h36m_zed_target_length = eval_config.motion.h36m_zed_target_length_eval
    eval_dataset = H36MZedEval(eval_config, 'test', rotations = args.rotations, quaternions = args.quaternions)


    shuffle = False
    sampler = None
    eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                                 num_workers=1, drop_last=False,
                                 sampler=sampler, shuffle=shuffle, pin_memory=True)


    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.cos_lr_max,
                                 weight_decay=config.weight_decay)

    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file, 'train')
    link_file(config.log_file, config.link_log_file)

    print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

    if config.model_pth is not None :
        state_dict = torch.load(config.model_pth)
        model.load_state_dict(state_dict, strict=True)
        print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

        ##### ------ training ------- #####
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.

    snapshot_subdir = strftime('snapshot_%Y%m%d%H%M%S', localtime())

    while (nb_iter + 1) < config.cos_lr_total_iters:
    
        for (h36m_zed_motion_input, h36m_zed_motion_target) in dataloader:
        
            loss, optimizer, current_lr = train_step(h36m_zed_motion_input, h36m_zed_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
            avg_loss += loss
            avg_lr += current_lr
        
            if (nb_iter + 1) % config.print_every ==  0 :
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every
            
                print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                avg_loss = 0
                avg_lr = 0
                
            if (nb_iter + 1) % config.save_every ==  0 :
                try:
                    odir = os.path.join(config.snapshot_dir, snapshot_subdir)
                    os.mkdir(odir)
                except(FileExistsError):
                    print("Failed to create output dir %s at iteration %d"%(odir, nb_iter + 1))
                output_file = os.path.join(config.snapshot_dir, snapshot_subdir, ckpt_name + str(nb_iter + 1) + '.pth')
                torch.save(model.state_dict(), output_file)

                model.eval()
                print("Evaluating with component and size %d"%eval_config.data_component_size)
                acc_tmp = test(eval_config, model, eval_dataloader)
                print("Acc tmp: ", acc_tmp)

                acc_log.write(''.join(str(nb_iter + 1) + '\n'))
                line = ''
                for ii in acc_tmp:
                    line += str(ii) + ' '
                line += '\n'
                acc_log.write(''.join(line))
                model.train()
    
            if (nb_iter + 1) == config.cos_lr_total_iters :
                break
            nb_iter += 1
    
    writer.close()

if __name__ == '__main__':
    mainfunc()

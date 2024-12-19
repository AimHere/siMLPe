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

from datasets.h36m_zed import exp_distance_torch, quat_distance_torch, quat_normalize_torch, quat_distance_dotprod

from test import test

from time import localtime, strftime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zed_utilities import body_34_parts, body_34_tree, body_34_tpose, Position, MotionUtilities_Torch

torch.autograd.set_detect_anomaly(True)

# config.motion_fc_in.temporal_fc - boolean
# config.motion.h36m_zed_input_length_dct - (history size)
# config.motion.dim -  (component size) * (bones + orientation_keypoints)

# config.motion.dim_ - (component size) * (bones + orientation_keypoints) ?

# joint_subset

motionutils = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

def normalnorm(a, b):
    v = (torch.linalg.norm(a - b, axis = 3) - 1)
    return v * v
    

def prepare_config(config, use_quaternions, full_joints, orientation_keypoints, history_size):
    
    if (use_quaternions):
        component_size = 4
        config.use_quaternions = True
        config.loss_convert_to_xyz = True
    else:
        component_size = 3
        
    if (full_joints):
        used_joints = [i for i in range(34)]

    elif(orientation_keypoints):
        used_joints = [i for i in range(3*34)]        
    else:
        used_joints = [ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]

    bone_count = len(used_joints)
    train_bones = bone_count
    
    config.joint_subset = used_joints
        
    config.motion.h36m_zed_input_length_dct = history_size
    config.dim_ = component_size * train_bones
    config.motion.dim = component_size * train_bones
    
    config.motion_mlp.hidden_dim = config.motion.dim
    config.motion_fc_in.in_features = config.motion.dim
    config.motion_fc_in.in_features = config.motion.dim        
    config.motion_fc_in.out_features = config.motion.dim
    config.motion_fc_out.in_features = config.motion.dim
    config.motion_fc_out.out_features = config.motion.dim

    config.data_component_size = component_size
    
    config.use_rotations = False
    
    return {'component_size' : component_size,
            'frame_history' : history_size,
            'bone_count' : bone_count,
            'total_bones' : train_bones,
            'used_bones' : used_joints}

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

train_step_count = 0

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



def train_step(config_values, h36m_zed_motion_input, h36m_zed_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr):

    global train_step_count # Why is this necessary?
    if (config.deriv_input):
        b, n, c = h36m_zed_motion_input.shape
        h36m_zed_motion_input__ = h36m_zed_motion_input.clone()
        h36m_zed_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], h36m_zed_motion_input__.cuda())

    else:
        h36m_zed_motion_input_ = h36m_zed_motion_input.clone()

    bone_count = config_values['bone_count']
    train_bones = config_values['total_bones']
    output_bone_components = config_values['component_size']

    nan_count = torch.where(torch.isnan(h36m_zed_motion_input_))
    if (len(nan_count[0]) > 0):
        print("Warning: Nan count in model input > 0")

    eps = h36m_zed_motion_input.clone().normal_(std=1e-8).cuda()

    motion_pred_ = model(h36m_zed_motion_input_.cuda() + eps)
    nan_count = torch.where(torch.isnan(motion_pred_))
    if (len(nan_count[0]) > 0):
        print("Warning: Nan count in model input > 0")

    motion_pred__ = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], motion_pred_)

    mpnp = motion_pred__.cpu().detach().numpy()

    idctnp = idct_m[:, :config.motion.h36m_zed_input_length, :].detach().cpu().numpy()

    if config.deriv_output:
        offset = h36m_zed_motion_input[:, -1:].cuda()
        motion_pred = motion_pred__[:, :config.motion.h36m_zed_target_length] + offset
    else:
        motion_pred = motion_pred__[:, :config.motion.h36m_zed_target_length]

    b, n, c = h36m_zed_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, train_bones, output_bone_components)
    h36m_zed_motion_target = h36m_zed_motion_target.cuda().reshape(b, n, train_bones, output_bone_components)

    if (config.loss_rotation_metric):
        print("Rotation Metric used")

        mpr = motion_pred.reshape([-1, bone_count, output_bone_components])
        hzmtr = h36m_zed_motion_target.reshape([-1, bone_count, output_bone_components])

        # Perturb to stop the loss value touching zero
        eps = mpr.clone().normal_(std = 1e-8)
        edist = exp_distance_torch(mpr, hzmtr + eps)

        loss = torch.mean(edist)


        if torch.isnan(loss):
            print("Invalid loss, halting")
            print(loss)
            exit(0)

        minloss = torch.min(edist)
        maxloss = torch.max(edist)

        print("Loss min: %f, max: %f, mean: %f"%(minloss, maxloss, loss))

    elif (config.loss_quaternion_distance):
        not_three = [i for i in range(bone_count) if i != 3]

        mpr = motion_pred.reshape([-1, bone_count, output_bone_components])[:, not_three, :]
        hzmtr = h36m_zed_motion_target.reshape([-1, bone_count, output_bone_components])[:, not_three, :]

        eps = hzmtr.clone().normal_(std = 1e-8)

        hzmtr_normed = quat_normalize_torch(hzmtr + eps)

        edist = quat_distance_torch(mpr, hzmtr_normed)

        main_loss = torch.mean(edist)

        if (torch.isnan(main_loss)):
            print("Invalid loss value, halting")
            exit(0)

        qnlroot = torch.sum(motion_pred * motion_pred, dim = 3) - 1
        quat_norm_loss = torch.mean(qnlroot * qnlroot)

        loss = main_loss + args.quat_norm_weight * quat_norm_loss

        minloss = torch.min(edist)
        maxloss = torch.max(edist)

        print("Quaternion Metric: Loss %f, main loss: %f, quat_norm_loss: %f,  min: %f, max: %f"%(loss, main_loss, quat_norm_loss, minloss, maxloss))        

    elif (config.loss_quaternion_dotprod):

        not_three = [i for i in range(bone_count) if i != 3]

        mpr = motion_pred.reshape([-1, bone_count, output_bone_components])[:, not_three, :]
        hzmtr = h36m_zed_motion_target.reshape([-1, bone_count, output_bone_components])[:, not_three, :]

        eps = hzmtr.clone().normal(std = 1e-8)

        mpr_normed = quat_normalize_torch(mpr + eps)
        main_loss = torch.mean(edist)

        if (torch.isnan(main_loss)):
            print("Invalid loss value, halting")
            exit(0)

        qnlroot = torch.sum(motion_pred * motion_pred, dim = 3) - 1
        quat_norm_loss = torch.mean(qnlroot * qnlroot)

        loss = main_loss + args.quat_norm_weight * quat_norm_loss

        minloss = torch.min(edist)
        maxloss = torch.max(edist)
        print("Quaternion DotProd Metric: Loss %f, main loss: %f, quat_norm_loss: %f,  min: %f, max: %f"%(loss, main_loss, quat_norm_loss, minloss, maxloss))        

    elif (config.loss_convert_to_xyz):
        rotation_substrate = torch.zeros([motion_pred.shape[0], motion_pred.shape[1], full_bone_count, motion_pred.shape[3]]).cuda()

        rotation_substrate[:, :, :, 3] = 1
        rotation_substrate[:, :, joints_used, :] = motion_pred

        rotation_gt_substrate = torch.zeros([motion_pred.shape[0], motion_pred.shape[1], full_bone_count, motion_pred.shape[3]]).cuda()
        rotation_gt_substrate[:, :, :, 3] = 1
        rotation_gt_substrate[:, :, joints_used, :] = h36m_zed_motion_target

        ## What the fuck? Why aren't these in the loss?
        pred_xyz = motionutils.forwardkinematics(rotation_substrate)
        gt_xyz = motionutils.forwardkinematics(rotation_gt_substrate)

        qnlroot = torch.sum(motion_pred * motion_pred, dim = 3) - 1
        quat_norm_loss = torch.mean(qnlroot * qnlroot)


        main_loss = torch.mean(torch.norm(pred_xyz - gt_xyz, 2, 1))        
        #main_loss = torch.mean(torch.norm(rotation_substrate - rotation_gt_substrate, 2, 3))

        loss = main_loss + args.quat_norm_weight * quat_norm_loss
                               
        print("Main Loss: %f, Norm Loss: %f, total: %f"%(main_loss, quat_norm_loss, loss))
        
    else: # Default loss - mpjpe
        if (config.use_orikip_normalization):
            #motion_pred = motion_pred.reshape(b,n,BONE_COUNT,OUTPUT_BONE_COMPONENTS)            
            mpc = motion_pred.clone()
            aa = mpc[:, :, :34, :]
            au = mpc[:, :, 34:68, :]
            ar = mpc[:, :, 68:, :]
            
            orientation_normalize_loss = torch.mean(normalnorm(aa, au) + normalnorm(aa, ar))
        else: 
            orientation_normalize_loss = 0


        motion_pred = motion_pred.reshape(-1, output_bone_components)
        h36m_zed_motion_target = h36m_zed_motion_target.reshape(-1, output_bone_components)

        main_loss = torch.mean(torch.norm(motion_pred - h36m_zed_motion_target, 2, 1))

        if config.use_relative_loss:
            motion_pred = motion_pred.reshape([b, n, train_bones, output_bone_components])
            dmotion_pred = gen_velocity(motion_pred)
            motion_gt = h36m_zed_motion_target.reshape([b, n, train_bones, output_bone_components])
            dmotion_gt = gen_velocity(motion_gt)

            dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, output_bone_components), 2, 1))
            loss = main_loss + dloss + orientation_normalize_loss * config.orikip_normalization_weight
            print("Vanilla MPJ error loss: %f + %f + %f*%f = %f"%(main_loss.mean(), dloss, config.orikip_normalization_weight, orientation_normalize_loss, loss))
        else:
            loss = main_loss.mean() + orientation_normalize_loss * config.orikip_normalization_weight
            print("Vanilla MPJ error loss: %f + %f*%f = %f"%(main_loss.mean(), config.orikip_normalization_weight, orientation_normalize_loss, loss))

        train_step_count += 1
            
        if (train_step_count%5000 == 0):
            test_file = "training_step_%05d.npz"%train_step_count
            print("Saving to %s"%test_file)
            np.savez(test_file, pred = motion_pred.detach().cpu().numpy(), gt = h36m_zed_motion_target.cpu().numpy(), input = h36m_zed_motion_input_.cpu().numpy())
                
    writer.add_scalar('loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, config.cos_lr_max, config.cos_lr_min, optimizer)

    writer.add_scalar('LR/train', current_lr, nb_iter)
    return loss.item(), optimizer, current_lr

def mainfunc():
    config_values = prepare_config(config, args.quaternions, args.full_joints, args.ori_kps, 50)

    config.motion_fc_in.temporal_fc = args.temporal_only
    config.motion_fc_out.temporal_fc = args.temporal_only
    config.motion_mlp.norm_axis = args.layer_norm_axis
    config.motion_mlp.spatial_fc_only = args.spatial_fc
    config.motion_mlp.with_normalization = args.with_normalization
    config.motion_mlp.num_layers = args.num
    acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

    if (args.quaternions):
        ckpt_name = './model_bone_iter-'
    elif(args.rotations):
        ckpt_name = './model_kp_iter-'
    elif(args.ori_kps):
        ckpt_name = './model_orikp_iter-'
    else:
        ckpt_name = './model_kp_iter-'


    if (args.rotations):
        config.data_type = 'axis-ang'
    elif(args.quaternions):
        config.data_type = 'quat'
    elif(args.ori_kps):
        config.data_type = 'ori_xyz'
    else:
        config.data_type = 'xyz'

    model = Model(config)

    model.train()
    model.cuda()
    
    config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_train

    # if (config.use_orientation_keypoints):
    #     dataset = H36MZedOrientation
    # else:
    dataset = H36MZedDataset(config, 'train', config.data_type, config.data_aug)

    # shuffle = True
    # sample = None
    dataloader = DataLoader(dataset,
                            batch_size = config.batch_size,
                            num_workers = config.num_workers,
                            drop_last = True,
                            sampler = None,
                            shuffle = True,
                            pin_memory = True)

    eval_config = copy.deepcopy(config)
    eval_config.motion.h36m_zed_target_length = eval_config.motion.h36m_zed_target_length_eval

    eval_dataset = H36MZedDataset(eval_config, 'test', config.data_type, config.data_aug)
    #eval_dataset = H36MZedEval(eval_config, 'test', data_type = eval_config.data_type)
    
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size = 128,
                                 num_workers = 1,
                                 drop_last = False,
                                 sampler = None,
                                 shuffle = False,
                                 pin_memory = True)

    # eval_dataloader = DataLoader(eval_dataset,
    #                              batch_size = 128,
    #                              num_workers = 1,
    #                              drop_last = False,
    #                              sampler = None,
    #                              shuffle = False,
    #                              pin_memory = True)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = config.cos_lr_max,
                                 weight_decay = config.weight_decay)


    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file, 'train')
    link_file(config.log_file, config.link_log_file)
          
    print_and_log_info(logger, json.dumps(config, indent = 4, sort_keys = True))

    if (config.model_pth is not None):
        state_dict = torch.load(config.model_pth)
        model.load_state_dict(state_dict, strict = True)
        print_and_log_info(logger, "Loading model path from {} ".format(config_model_pth))

        ##### ------ training ------- #####
    
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.

    snapshot_subdir = strftime('snapshot_%Y%m%d%H%M%S', localtime())

    while (nb_iter + 1) < config.cos_lr_total_iters:

        for (h36m_zed_motion_input, h36m_zed_motion_target) in dataloader:
            loss, optimizer, current_lr = train_step(config_values, h36m_zed_motion_input, h36m_zed_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)

            avg_loss += loss
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0 :
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every

                print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                print_and_log_info(logger, f"\t lr: {avg_lr} \t Training Loss: {avg_loss}")

                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0 :
                try:
                    odir = os.path.join(config.snapshot_dir, snapshot_subdir)
                    os.mkdir(odir)
                except(FileExistsError):
                    print("Failed to create output dir %s at iteration %d"%(odir, nb_iter + 1))

                output_file = os.path.join(config.snapshot_dir, snapshot_subdir, ckpt_name + str(nb_iter + 1) + '.pth'
)
                torch.save(model.state_dict(), output_file)
                model.eval()

                print("Iter: %d, Evaluating with component and size %d"%(nb_iter,eval_config.data_component_size))
                acc_tmp = test(eval_config, model, eval_dataloader, joint_prefiltered = True)
                print("Test config is ", eval_config)
                print("Acc tmp: ", acc_tmp)

                acc_log.write(''.join(str(nb_iter + 1) + '\n'))

                line = ''
                for ii in acc_tmp:
                    line += str(ii) + ' '
                line += '\n'
                acc_log.write(''.join(line))
                model.train()

            if (nb_iter + 1) == config.cos_lr_total_iters:
                break
            nb_iter += 1

    writer.close()
                
                                 
## TODO: Write a model that hallucinates ori_kps without orientation keypoint input
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')

group = parser.add_mutually_exclusive_group()
group.add_argument('--rotations', action='store_true', help='=train on rotations')
group.add_argument('--quaternions', action = 'store_true', help = '=train on quaternions')
group.add_argument('--ori_kps', action = 'store_true', help = "Train on all joints, not just the main 18")

parser.add_argument('--quat_norm_weight', type = float, default = 0.0, help = '=Quaternion normalization weight')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--full_joints', action = 'store_true', help = "Train on all joints, not just the main 18")
parser.add_argument('--dumptrace', type = str, help = "Dump the model to a Torchscript trace function for use in C++")

args = parser.parse_args()

# cc = config
# cc.data_aug = True
# cc.data_type = 'xyz'
# cc.h36m_zed_target_length = cc.motion.h36m_zed_target_length_train
# cc.motion.h36m_zed_target_length = cc.motion.h36m_zed_target_length_train
# #cc.h36m_zed_target_length = cc.motion.h36m_zed_target_length_train

# datasetorigtrain = H36MZedDataset(config, 'train', cc.data_type, cc.data_aug)
# datasetorigtest = H36MZedDataset(config, 'test', cc.data_type, cc.data_aug)
# datasetevaltest = H36MZedEval(config, 'test', cc.data_type, cc.data_aug)

# print("Lengths: ", len(datasetorigtrain), len(datasetorigtest), len(datasetevaltest))

# v5orig = datasetorigtrain[5]
# v5origtest = datasetorigtest[5]
# v5eval = datasetevaltest[5]

# print("Shapes: Orig Train: ", v5orig[0].shape)
# print("Shapes: Orig Test: ", v5origtest[0].shape)
# print("Shapes: Eval Test: ", v5eval[0].shape)

# exit(0)


torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

if __name__ == '__main__':
    mainfunc()

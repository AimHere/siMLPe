import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config import config
from model import siMLPe as Model

from datasets.h36m_zed_eval import H36MZedEval
from datasets.h36m_zed import H36MZedDataset
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

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

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def regress_pred(model, pbar, num_samples, m_p3d_h36, data_component_size = 3, bones_used_count = 34):

    joint_used = [i for i in range(bones_used_count)]
    
    for (motion_input, motion_target) in pbar:

        motion_input = motion_input.cuda()
        b, n, c = motion_input.shape
        num_samples += b
        
        #motion_input = motion_input.reshape(b, n, -1, 3)
        motion_input = motion_input.reshape(b, n, -1)
        outputs = []

        step = config.motion.h36m_zed_target_length_train

        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1

        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()

                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            output = output.reshape([-1, bones_used_count * data_component_size])
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis = 1)

        motion_pred = torch.cat(outputs, axis = 1)[:, :25]

        motion_target = motion_target.detach()

        motion_gt = motion_target.clone().cuda()

        b, n, c = motion_target.shape


        if False:
            motion_pred = motion_pred.detach().cpu()
            print("B=%d, n=%d, mopred = "%(b, n), motion_pred.shape)
            pred_rot = motion_pred.clone().reshape(b, n, -1, data_component_size)
            motion_pred = motion_target.clone().reshape(b, n, -1, data_component_size)
            print("Pred Rot shape is ", pred_rot.shape)
            motion_pred[:, :, joint_used] = pred_rot        
            tmp = motion_gt.clone()
            #tmp[:, :, joint_used, data_component_size] = motion_pred[:, :, joint_used]        
            motion_pred = tmp ## Wut?

        motion_pred = motion_pred.reshape(b, n, -1, data_component_size)
        motion_gt = motion_gt.reshape(b, n, -1, data_component_size)        
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim = 3), dim = 2), dim = 0)

        torch.set_printoptions(precision = 2, sci_mode = False)

        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
        
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader, joint_prefiltered = None):
    print("Test config is: ", config)

    m_p3d_h36 = np.zeros([config.motion.h36m_zed_target_length])
    titles = np.array(range(config.motion.h36m_zed_target_length)) + 1

    num_samples = 0

    pbar = dataloader

    #joints_used = np.array(config.joint_subset).astype(np.int64)
    if (config.data_type == 'ori_xyz'):
        bones_used_count = 102
    else:
        bones_used_count = 34

    data_component_size = 3
    
    m_p3d_h36 = regress_pred(model, pbar, num_samples, m_p3d_h36,
                             bones_used_count = bones_used_count,
                             data_component_size = data_component_size)

    ret = {}
    for j in range(config.motion.h36m_zed_target_length):
        ret['#{:d}'.format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]

    return [round(float(ret[key][0]), 1) for key in results_keys]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_kps', action = 'store_true', help = "Train on all joints, not just the main 18")    
    parser.add_argument('model_pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    if (args.ori_kps):
        config.data_type = 'ori_xyz'
        full_bone_count = 34 * 3
        data_component_size = 3
    else:
        config.data_type = 'xyz'
        full_bone_count = 34
        data_component_size = 3

    config.motion.dim = data_component_size * len(joints_used)       
    config.dim_ = data_component_size  * len(joints_used)
    config.joint_subset = joints_used

    config.motion_mlp.hidden_dim = config.motion.dim
    config.motion_fc_in.in_features = config.motion.dim
    config.motion_fc_in.in_features = config.motion.dim        
    config.motion_fc_in.out_features = config.motion.dim
    config.motion_fc_out.in_features = config.motion.dim
    config.motion_fc_out.out_features = config.motion.dim


    joints_used = [i for i in range(full_bone_count)]

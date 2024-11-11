import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config import config
from model import siMLPe as Model
#from datasets.h36m_eval import H36MEval
from datasets.h36m_zed_eval import H36MZedEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

# COMPACT_BONE_COUNT = 18
# FULL_BONE_COUNT = 34

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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

def regress_pred(model, pbar, num_samples, joint_used, m_p3d_h36, data_component_size = 3):
    #joint_to_ignore = np.array([8, 9, 10, 15, 16, 17, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    #joint_equal = np.array([])

    bones_used_count = len(joint_used)

    for (motion_input, motion_target) in pbar:

        motion_input = motion_input.cuda()
        b, n, c, _ = motion_input.shape
        num_samples += b
        motion_input = motion_input.reshape(b, n, -1, data_component_size)

        #motion_input = motion_input[:, :, joint_used].reshape(b, n, -1)
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

        b, n, c, _ = motion_target.shape
        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()

        pred_rot = motion_pred.clone().reshape(b, n, -1, data_component_size)
        motion_pred = motion_target.clone().reshape(b, n, -1, data_component_size)

        motion_pred[:, :, joint_used] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used] = motion_pred[:, :, joint_used]

        motion_pred = tmp ## Wut?

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim = 3), dim = 2), dim = 0)

        print("MPJPE_P3D_H36: ", mpjpe_p3d_h36)
        
        mpjpe_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()

    mpjpe_p3d_h36 = mpjpe_p3d_h36 / num_samples
    return mpjpe_p3d_h36

        
########## FINISH ###############

def test(config, model, dataloader, full_bone_count = 34):

    m_p3d_h36 = np.zeros([config.motion.h36m_zed_target_length])
    titles = np.array(range(config.motion.h36m_zed_target_length)) + 1

    num_samples = 0
    
    pbar = dataloader
    joints_used = config.joint_subset
    
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joints_used, m_p3d_h36, data_component_size = config.data_component_size)
    
    ret = {}
    for j in range(config.motion.h36m_zed_target_length):
        ret['#{:d}'.format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]

    return [round(float(ret[key][0]), 1) for key in results_keys]

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fulljoints', action = 'store_true', default=None, help='=encoder path')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rotations', action='store_true', help='=train on rotations')
    group.add_argument('--quaternions', action = 'store_true', help = '=train on quaternions')
    group.add_argument('--ori_kps', action = 'store_true', help = "Train on all joints, not just the main 18")
    
    parser.add_argument('model_pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()


    config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_eval
    
    if (args.rotations):
        config.data_type = 'axis-ang'
        full_bone_count = 34
        data_component_size = 3
    elif(args.quaternions):
        config.data_type = 'quat'
        full_bone_count = 34
        data_component_size = 4
    elif(args.ori_kps):
        config.data_type = 'ori_xyz'
        full_bone_count = 34 * 3
        data_component_size =3
    else:
        config.data_type = 'xyz'
        full_bone_count = 34
        data_component_size = 3

    if (args.fulljoints):
        joints_used = [i for i in range(full_bone_count)]
    else:
        joints_used = [ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]

    config.motion_dim = 3 * len(joints_used)       
    config.dim_ = data_component_size  * config.motion_dim
    config.joint_subset = joints_used
    
    print("Test: %s data, %d, %d"%(config.data_type, config.motion.dim, config.dim_))        

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    dataset = H36MZedEval(config, 'test', data_type = config.data_type)
    
    # if (config.use_orientation_keypoints):
    #     dataset = H36MZedEval(config, 'test')
    # else:
    #     dataset = H36MZedOrientationEval(config, 'test')
        
    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    a, b = dataset[10]

    print("Test value is", (test(config, model, dataloader, full_bone_count = full_bone_count)))


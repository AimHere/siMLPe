import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch, expmap2rotmat_torch

from utils.visualize import Animation, Loader

import torch
from torch.utils.data import Dataset, DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

COMPACT_BONE_COUNT = 18
REALFULL_BONE_COUNT = 34
FULL_BONE_COUNT = 18


# Loads a single specified animation file
class AnimationSet(Dataset):
    
    def __init__(self, config, filename, zeros = False):
        super(AnimationSet, self).__init__()

        self.used_joint_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]) # Bones 32 and 33 are non-zero rotations, but constant        

        self.filename = filename

        self.zeros = zeros
        
        self.h36_motion_input_length  = config.motion.h36m_zed_input_length
        self.h36_motion_target_length  = config.motion.h36m_zed_target_length
        
        pose = self._preprocess(self.filename)

        self.h36m_seqs = []
        self.h36m_seqs.append(pose)

        num_frames = pose.shape[0]


    def __len__(self):
        return self.h36m_seqs[0].shape[0] - (self.h36_motion_input_length + self.h36_motion_target_length)

    def __getitem__(self, index):
        start_frame = index
        end_frame_inp = index + self.h36_motion_input_length
        end_frame_target = index + self.h36_motion_input_length + self.h36_motion_target_length
        input = 0.001 *  self.h36m_seqs[0][start_frame:end_frame_inp].float()
        target = 0.001 * self.h36m_seqs[0][end_frame_inp:end_frame_target].float()
        return input, target


    def _preprocess(self, filename):
        fbundle = np.load(filename, allow_pickle = True)
        
        xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))
        xyz_info = xyz_info[:, self.used_joint_indices, :]

        xyz_info = xyz_info.reshape([xyz_info.shape[0], -1])

        N = xyz_info.shape[0]
        
        sample_rate = 2
        sampled_index = np.arange(0, N, sample_rate)
        h36m_zed_motion_poses = xyz_info[sampled_index]

        T = h36m_zed_motion_poses.shape[0]
        h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, FULL_BONE_COUNT, 3)
        return h36m_zed_motion_poses
                
    def upplot(self, t):
        print(t.shape)
        newvals = np.zeros([t.shape[0], REALFULL_BONE_COUNT, 3])
        for i, b in enumerate(self.used_joint_indices):
            newvals[:, b, :] = t[:, i, :]

        return newvals

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



def fetch(config, model, dataset, frame):
    
    joint_used_xyz = np.arange(0, COMPACT_BONE_COUNT)
    motion_input, motion_target = dataset[frame]

    orig_input = motion_input.clone()
    
    motion_input = motion_input.cuda()
    motion_target = motion_target.cuda()
    
    motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])
    motion_target = motion_target.reshape([1, motion_target.shape[0], motion_target.shape[1], -1])

    b, n, c, _ = motion_input.shape
    
    motion_input = motion_input.reshape(b, n, FULL_BONE_COUNT, 3)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)

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

            # Should this only apply if config.deriv_input ?
            output = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], output)[:, :step, :]

            if config.deriv_output:
                output = output + motion_input[:, -1, :].repeat(1, step, 1)

        output = output.reshape(-1, COMPACT_BONE_COUNT * 3)
        output = output.reshape(b, step, -1)
        outputs.append(output)

        motion_input = torch.cat([motion_input[:, step:], output], axis = 1)

    motion_pred = torch.cat(outputs, axis = 1)[:, :25]

    b, n, c, _ = motion_target.shape

    motion_gt = motion_target.clone()
    pred_rot = motion_pred.clone().reshape(b, n, COMPACT_BONE_COUNT, 3)

    motion_pred = motion_pred.detach().cpu()
    motion_pred = motion_target.clone().reshape(b, n, FULL_BONE_COUNT, 3)
    motion_pred[:, :, joint_used_xyz] = pred_rot

    tmp = motion_gt.clone()
    tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]

    motion_pred = tmp
    #motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

    gt_out = np.concatenate([orig_input, motion_gt.cpu().squeeze(0)], axis = 0)    
    pred_out = np.concatenate([orig_input, motion_pred.cpu().squeeze(0)], axis = 0)

    return gt_out, pred_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default="./log/snapshot/model-iter-40000.pth", help='=encoder path')

    parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
    parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
    parser.add_argument("--scale", type = float, default = 1.0)
    parser.add_argument("--elev", type = float, help = "Elevation", default = 90)
    parser.add_argument("--azim", type = float, help = "Azimuth", default = 270)
    parser.add_argument("--roll", type = float, help = "Roll", default = 0)

    parser.add_argument("--zeros", action = 'store_true', help = "Zero-expmap")

    parser.add_argument("--fps", type = float, default = 50, help = "Override animation fps")
    
    parser.add_argument('file', type = str)
    parser.add_argument('start_frame', type = int)
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_eval

    dataset = AnimationSet(config, args.file, zeros = args.zeros)
    shuffle = False
    sampler = None
    train_sampler = None

    gt_, pred_ = fetch(config, model, dataset, args.start_frame)

    gt = dataset.upplot(gt_)
    pred = dataset.upplot(pred_)
    
    anim = Animation([gt, pred], dots = not args.nodots, skellines = args.lineplot, scale = args.scale, unused_bones = True, skeltype = 'zed', elev = args.elev, azim = args.azim, roll = args.roll, fps = args.fps)


import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

#from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

import torch
import torch.utils.data as data

# Dataset loader for h36m files in a zed-friendly format

def quat_to_expmap(rot_info):
    halfthetas = np.arccos(rot_info[:, :, 3])
    http = np.where(halfthetas == 0, 0, 2 * halfthetas/np.sin(halfthetas))
    https = np.stack([http, http, http], axis = 2)
    rots = https * rot_info[:, :, :3]
    return rots

def expmap_to_quat(expmaps):
    rads = np.linalg.norm(expmaps, axis = 2)
    rv = np.stack([rads, rads, rads], axis = 2)
    qv = np.where(rv == 0, 0, (expmaps[:, :, :3] / rv))
    cosses = np.cos (rads / 2)
    sins = np.sin(rads / 2)
    sinss = np.stack([sins, sins, sins], axis = 2)
    exps = np.concatenate([qv * sinss , np.expand_dims(cosses, 2)], axis = 2)
    return exps

def quat_inverse(quats):
    exps = np.concatenate([-quats[:, :, :3], quats[:, :, 3:]], axis = 2)    
    return exps

def quat_mult(qa, qb):
    a = qa[:, :, 0]
    b = qa[:, :, 1]
    c = qa[:, :, 2]
    d = qa[:, :, 3]
    e = qb[:, :, 0]
    f = qb[:, :, 1]
    g = qb[:, :, 2]
    h = qb[:, :, 3]

    ww = -a * e - b * f - g * c + d * h
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = np.concatenate([ii, jj, kk, ww], axis = 2)
    return qq


def exp_distance(ea, eb):
    qa = expmap_to_quat(ea)
    qb = expmap_to_quat(eb)

    qdiff = quat_mult(quat_inverse(ea), qb)

    # DO we need to calculate sines?
    halfthetas = np.arccos(qdiff[:, :, 3])
    
    return 2 * halfthetas


class H36MZedDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug = False, rotations = False):
        super(H36MZedDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        self._h36m_zed_anno_dir = config.h36m_zed_anno_dir

        self.used_joint_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]) # Bones 32 and 33 are non-zero rotations, but constant
        self.axis_ang = rotations
        
        self._h36m_zed_files = self._get_h36m_zed_files()

        self.h36m_zed_motion_input_length = config.motion.h36m_zed_input_length
        self.h36m_zed_motion_target_length = config.motion.h36m_zed_target_length
        self.motion_dim = config.motion.dim

        self.shift_step = config.shift_step

        self._collect_all()
        self._file_length = len(self.data_idx)
        
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_zed_files)
    
    def _get_h36m_zed_files(self):
        seq_names = []
        if (self._split_name == 'train'):
            seq_names += np.loadtxt(
                os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                             'h36m_zed_train.txt'),
                dtype = str).tolist()
        else:
            seq_names += np.loadtxt(
                os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                             'h36m_zed_test.txt'),
                dtype = str).tolist()

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._h36m_zed_anno_dir + "/" + dataset + "/*")
            for subject in subjects:
                file_list.append(subject)

        h36m_zed_files = []

        
        if (self.axis_ang):
            # TODO: Fix the NaN issues with this
            for path in file_list:
                fbundle = np.load(path, allow_pickle = True)

                quats = fbundle['quats'].astype(np.float32)[:, self.used_joint_indices, :]
                rots = quat_to_expmap(quats)
                h36m_zed_files.append(torch.tensor(rots))
        else:
            for path in file_list:
                fbundle = np.load(path, allow_pickle = True)
                xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))
                xyz_info = xyz_info[:, self.used_joint_indices, :]
                xyz_info = xyz_info.reshape([xyz_info.shape[0], -1])
                h36m_zed_files.append(xyz_info)


        
        
                
        return h36m_zed_files

    def _collect_all(self):
        self.h36m_zed_seqs = []
        self.data_idx = []
        idx = 0

        for h36m_zed_motion_poses in self._h36m_zed_files:
            N = len(h36m_zed_motion_poses)
            if (N < self.h36m_zed_motion_target_length + self.h36m_zed_motion_input_length):
                continue

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = h36m_zed_motion_poses[sampled_index]
    
            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses.reshape(T, -1)

            self.h36m_zed_seqs.append(h36m_zed_motion_poses)

            valid_frames = np.arange(0, T - self.h36m_zed_motion_input_length - self.h36m_zed_motion_target_length + 1, self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))

            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length)

        motion = self.h36m_zed_seqs[idx][frame_indexes]
        
        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                idx = [i for i in range(motion.size(0) -1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        h36m_zed_motion_input = motion[:self.h36m_zed_motion_input_length] / 1000
        h36m_zed_motion_target = motion[self.h36m_zed_motion_input_length:] / 1000
        h36m_zed_motion_input = h36m_zed_motion_input.float()
        h36m_zed_motion_target = h36m_zed_motion_target.float()
            
        
        return h36m_zed_motion_input, h36m_zed_motion_target
            

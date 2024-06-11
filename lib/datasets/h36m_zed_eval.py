import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import find_indices_256, find_indices_srnn #expmap2rotmat_torch, rotmat2xyz_torch

import torch
import torch.utils.data as data

class H36MZedEval(data.Dataset):
    def __init__(self, config, split_name, paired=True, rotations = False, quaternions = False):
        super(H36MZedEval, self).__init__()
        self._split_name = split_name
        self._h36m_zed_anno_dir = config.h36m_zed_anno_dir
        self._actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

        self.h36m_zed_motion_input_length =  config.motion.h36m_zed_input_length
        self.h36m_zed_motion_target_length =  config.motion.h36m_zed_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._h36m_zed_files = self._get_h36m_zed_files()
        self._file_length = len(self.data_idx)

        self.rotations = rotations
        self.quaternions = quaternions

        
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_zed_files)

    def _get_h36m_zed_files(self):

        # create list
        seq_names = []

        seq_names += open(
            os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''), "h36m_zed_test.txt"), 'r'
            ).readlines()

        self.h36m_zed_seqs = []
        self.data_idx = []
        idx = 0
        for subject in seq_names:
            subject = subject.strip()
            for act in self._actions:
                
                filename0 = '{0}/{1}/{1}_{2}_{3}.npz'.format(self._h36m_zed_anno_dir, subject, act, 1)
                filename1 = '{0}/{1}/{1}_{2}_{3}.npz'.format(self._h36m_zed_anno_dir, subject, act, 2)
                
                poses0 = self._preprocess(filename0)
                poses1 = self._preprocess(filename1)

                self.h36m_zed_seqs.append(poses0)
                self.h36m_zed_seqs.append(poses1)

                num_frames0 = poses0.shape[0]
                num_frames1 = poses1.shape[0]

                fs_sel1, fs_sel2 = find_indices_256(num_frames0, num_frames1,
                                   self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length,
                                   input_n=self.h36m_zed_motion_input_length)
                #fs_sel1, fs_sel2 = find_indices_srnn(num_frames0, num_frames1,
                #                   self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length,
                #                   input_n=self.h36m_zed_motion_input_length)
                valid_frames0 = fs_sel1[:, 0]
                tmp_data_idx_1 = [idx] * len(valid_frames0)
                tmp_data_idx_2 = list(valid_frames0)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                valid_frames1 = fs_sel2[:, 0]
                tmp_data_idx_1 = [idx + 1] * len(valid_frames1)
                tmp_data_idx_2 = list(valid_frames1)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                idx += 2


    def _preprocess(self, filename):
        bundle = np.load(filename, allow_pickle = True)
        pose_info = bundle['keypoints']
        N = pose_info.shape[0]
        
        sample_rate = 2
        sampled_index = np.arange(0, N, sample_rate)
        h36m_zed_motion_poses = pose_info[sampled_index]

        #T = h36m_zed_motion_poses.shape[0]
        #h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, 32, 3)
        return h36m_zed_motion_poses

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length)
        motion = self.h36m_zed_seqs[idx][frame_indexes]

        h36m_zed_motion_input = motion[:self.h36m_zed_motion_input_length] / 1000.
        h36m_zed_motion_target = motion[self.h36m_zed_motion_input_length:] / 1000.

        h36m_zed_motion_input = h36m_zed_motion_input.astype(np.float32)
        h36m_zed_motion_target = h36m_zed_motion_target.astype(np.float32)
        return h36m_zed_motion_input, h36m_zed_motion_target


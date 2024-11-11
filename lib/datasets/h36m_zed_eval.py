import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import find_indices_256, find_indices_srnn #expmap2rotmat_torch, rotmat2xyz_torch

from zed_utilities import MotionUtilities_Torch, body_34_parts, body_34_tree, body_34_tpose, Position


import torch
import torch.utils.data as data


class H36MZedEval(data.Dataset):
    def __init__(self, config, split_name, paired = True, data_type = 'xyz', sample_rate = 2):
        super(H36MZedEval, self).__init__()
        self._split_name = split_name
        self._h36m_zed_anno_dir = config.h36m_zed_anno_dir
        self.sample_rate = sample_rate

        self._actions =  ["walking", "eating", "smoking", "discussion", "directions",
                          "greeting", "phoning", "posing", "purchases", "sitting",
                          "sittingdown", "takingphoto", "waiting", "walkingdog",
                          "walkingtogether"]

        self.motionutils = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

        self.data_type = data_type

        self.h36m_zed_motion_input_length = config.motion.h36m_zed_input_length
        self.h36m_zed_motion_target_length = config.motion.h36m_zed_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._h36m_zed_files = self._get_h36m_zed_files()
        self._file_length = len(self.data_idx)
        
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_zed_files)


    
    def _get_h36m_zed_files(self):

        seq_names = []
        seq_names += open(os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                                       'h36m_zed_test.txt'),
                          'r').readlines()
        self.h36m_zed_seqs = []
        self.data_idx = []

        self.data_idx_counter = 0

        for subject in seq_names:
            subject = subject.strip()
            idx = 0
            for act in self._actions:
                
                filename0 = '{0}/{1}/{1}_{2}_{3}_zed34_test.npz'.format(self._h36m_zed_anno_dir, subject, act, 1)
                filename1 = '{0}/{1}/{1}_{2}_{3}_zed34_test.npz'.format(self._h36m_zed_anno_dir, subject, act, 2)
                
                poses0 = self._preprocess(filename0)
                poses1 = self._preprocess(filename1)

                self.h36m_zed_seqs.append(poses0)
                self.h36m_zed_seqs.append(poses1)
                
                num_frames0 = poses0.shape[0]
                num_frames1 = poses1.shape[0]

                fs_sel1, fs_sel2 = find_indices_256(num_frames0,
                                                    num_frames1,
                                                    self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length,
                                                    input_n = self.h36m_zed_motion_input_length )

#                print("Indexes: ", fs_sel1, fs_sel2)
                valid_frames0 = fs_sel1[:, 0]
                tmp_data_idx_1 = [idx] * len(valid_frames0)
                tmp_data_idx_2 = list(valid_frames0)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                idx += 1
                
                valid_frames1 = fs_sel2[:, 0]
                tmp_data_idx_1 = [idx] * len(valid_frames1)
                tmp_data_idx_2 = list(valid_frames1)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                idx += 1

    def _preprocess(self, filename, quats = False):
        bundle = np.load(filename, allow_pickle = True)
        #print("Preprocessing %s data from %s"%(self.data_type, filename))
        if (self.data_type == 'quat'):
            pose_info = bundle['quats']

        elif(self.data_type == 'axis_ang'):
            print("Axis angles not implemented yet")
            exit(0)
        elif(self.data_type == 'ori_xyz'):
            quat_info = torch.tensor(bundle['quats']).unsqueeze(0).float().cuda()
            kp_info = bundle['keypoints']
            orients = self.motionutils.orientation_kps(quat_info).squeeze(0).cpu().numpy()
            pose_info = np.concatenate([kp_info, orients], axis = 1)

        else:
            pose_info = bundle['keypoints']

        N = pose_info.shape[0]
                
        sampled_index = np.arange(0, N, self.sample_rate)
        h36m_zed_motion_poses = pose_info[sampled_index]
        return h36m_zed_motion_poses

    def __getitem__(self, index):

        idx, start_frame = self.data_idx[index]

        # print("Item: %d, idx: %d, start: %d"%(index, idx, start_frame))
        # print("Seq: ", self.h36m_zed_seqs[idx].shape)
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length)

        vv = self.h36m_zed_seqs[idx]
        if (vv.shape[0] < max(frame_indexes)):
            print("Problem with seq shape: ", vv.shape)
            print("And frame indices: ", frame_indexes)
            exit(0)
        

        motion = self.h36m_zed_seqs[idx][frame_indexes]
        #print("Motion shape is ", motion.shape)
        h36m_zed_motion_input = motion[:self.h36m_zed_motion_input_length] / 1000.
        h36m_zed_motion_target = motion[self.h36m_zed_motion_input_length:] / 1000.

        h36m_zed_motion_input = h36m_zed_motion_input.astype(np.float32)
        h36m_zed_motion_target = h36m_zed_motion_target.astype(np.float32)

        return h36m_zed_motion_input, h36m_zed_motion_target

    

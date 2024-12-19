# encooding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 304

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'siMLPe'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))


exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'lib'))

"""Data Dir and Weight Dir"""
# TODO

"""Dataset Config"""
C.h36m_zed_anno_dir = osp.join(C.root_dir, 'data/h36m_zed/')
C.motion = edict()

C.motion.h36m_zed_input_length = 50
C.motion.h36m_zed_input_length_dct = 50
C.motion.h36m_zed_target_length_train = 10 
C.motion.h36m_zed_target_length_eval = 25
#C.motion.dim = 66
C.motion.dim = 54
#C.motion.dim = 34 * 3 * 3


C.data_aug = True
# C.deriv_input = True
# C.deriv_output = True
C.deriv_input = False
C.deriv_output = False
C.use_relative_loss = True

C.use_rotations = False       # Whether to use rotations
C.use_rotation_loss = False   # Whether or not to measure the rotation loss or go with mpjpe
C.use_quaternions = False     # Train on quaternions
C.use_orikip_normalization = True # Put in loss factors to normalize orientation keypoint distances
C.use_orikip_orthonormalization = False # Loss factor to make orientation keypoints orthogonal

C.orikip_normalization_weight = 1.0 # Weight multiplier to balance normalizationx

C.data_component_size = 3

C.loss_quaternion_dotprod = False # Dot Product derived quaternion loss
C.loss_quaternion_distance = False # Distance based purely on quaternions
C.loss_convert_to_xyz = False # Convert rotations to keypoints and take the loss
C.loss_6D = False # Convert rotations to keypoints and take the loss
C.loss_rotation_metric = False # Convert rotations to keypoints and take the loss

C.use_orientation_keypoints = False # Orientation Keypoints

C.joint_subset = [ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24] # Bones 32 and 33 are non-zero rotations, but constant

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False
## Motion Network mlp
#dim_ = 66

dim_ = 54
#dim_ = 102
#dim_ = 306

#C.hidden_dim = 54

C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = C.motion.h36m_zed_input_length_dct
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'
## Motion Network FC In
C.motion_fc_in = edict()
C.motion_fc_in.in_features = C.motion.dim
C.motion_fc_in.out_features = dim_
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False
## Motion Network FC Out
C.motion_fc_out = edict()
C.motion_fc_out.in_features = dim_
C.motion_fc_out.out_features = C.motion.dim
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config"""
C.batch_size = 256
C.num_workers = 8

C.cos_lr_max=1e-5
C.cos_lr_min=5e-8

# C.cos_lr_max=3e-4
# C.cos_lr_min=1e-5

C.cos_lr_total_iters=100000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
# C.print_every = 5
# C.save_every = 50
C.print_every = 100
C.save_every = 5000


if __name__ == '__main__':
    print(config.decoder.motion_mlp)

import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config import config
from model import siMLPe as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch, expmap2rotmat_torch

from utils.visualize import Animation, Loader

import torch
from torch.utils.data import Dataset, DataLoader

from zed_utilities import quat_to_expmap_torch, MotionUtilities_Torch, body_34_parts, body_34_tree, body_34_tpose, Position, Quaternion, expmap_to_quat

import time

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

# COMPACT_BONE_COUNT = 18
# REALFULL_BONE_COUNT = 34
# FULL_BONE_COUNT = 18

def prepare_config(config, use_quaternions, full_joints, orientation_keypoints, history_size):

    if (use_quaternions):
        component_size = 4
        config.use_quaternions = True
        config.loss_convert_to_xyz = True
        bone_count = len(used_joints)    
        train_bones = bone_count
    else:
        component_size = 3
        
    if(orientation_keypoints):
        used_joints = [i for i in range(3*34)]        
        bone_count = len(used_joints)    
        train_bones = bone_count
    elif (full_joints):
        used_joints = [i for i in range(34)]
        bone_count = len(used_joints)    
        train_bones = bone_count
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


def bone_counts(all_bones = False, ori_kps = False):
    
    if (all_bones):
        used_bones = 34
        full_bones = 34
    else:
        used_bones = 18
        full_bones = 34

    if (ori_kps):
        used_bones *= 3
        full_bones *= 3
        
    return used_bones, full_bones



# Loads a single specified animation file
class AnimationSet(Dataset):

    def __init__(self, config, filename, zeros = False, rotations = False, quaternions = False, all_bones = False, ori_kps = False, scale = 0.001):
        super(AnimationSet, self).__init__()

        if (all_bones or ori_kps):
            self.used_joint_indices = [i for i in range(34)]
            # self.used_bone_count = 34
            # self.full_bone_count = 34
            
        else:
            self.used_joint_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]) # Bones 32 and 33 are non-zero rotations, but constant
            # self.used_bone_count = 18
            # self.full_bone_count = 34

        self.used_bone_count, self.full_bone_count = bone_counts(all_bones, ori_kps)
        self.ori_keypoints = ori_kps            
            
        self.filename = filename
        self.rotations = rotations
        self.quaternions = quaternions

        self.scale = scale
            
        if (self.quaternions):
            self.component_size = 4
        else:
            self.component_size = 3            
        
        self.zeros = zeros
        
        self.h36_motion_input_length  = config.motion.h36m_zed_input_length
        self.h36_motion_target_length  = config.motion.h36m_zed_target_length
        
        pose = self._preprocess(self.filename).float().cuda()

        self.h36m_seqs = []
        
        # if (ori_kps):
        
        #     quats = self._preprocess(self.filename, quaternions = True).float().unsqueeze(0).cuda()

        #     motionutils = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
        #     addendum = motionutils.orientation_kps(quats)
        #     fullpose = torch.concat([pose, addendum.squeeze(0)], axis = 1)
            
        #     self.h36m_seqs.append(fullpose)

        self.h36m_seqs.append(pose)

        num_frames = pose.shape[0]


    def __len__(self):
        return self.h36m_seqs[0].shape[0] - (self.h36_motion_input_length + self.h36_motion_target_length)

    def __getitem__(self, index):
        start_frame = index
        end_frame_inp = index + self.h36_motion_input_length
        end_frame_target = index + self.h36_motion_input_length + self.h36_motion_target_length
        input = self.h36m_seqs[0][start_frame:end_frame_inp].float()
        target = self.h36m_seqs[0][end_frame_inp:end_frame_target].float()

        return self.scale * input, self.scale * target

    def _preprocess(self, filename):
        fbundle = np.load(filename, allow_pickle = True)
        if (self.rotations):
            quat = torch.tensor(fbundle['quats'].astype(np.float32))
            quat = quat[:, self.used_joint_indices, :]
            rots = quat_to_expmap_torch(quat)
            rots = np.reshape(rots, [rots.shape[0], -1])

            N = rots.shape[0]

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = rots[sampled_index]

            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, self.used_bone_count, self.component_size)
            return h36m_zed_motion_poses
        elif (self.quaternions):
            quat = torch.tensor(fbundle['quats'].astype(np.float32))
            quat = quat[:, self.used_joint_indices, :]

            N = quat.shape[0]
            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = quat[sampled_index]
            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, self.used_bone_count, self.component_size)
        elif(self.ori_keypoints):
            quat = torch.tensor(fbundle['quats'].astype(np.float32))
            quat = quat[:, self.used_joint_indices, :]

            kps = torch.tensor(fbundle['keypoints'].astype(np.float32))
            kps = kps[:, self.used_joint_indices, :]
            
            sample_rate = 2
            sampled_index = np.arange(0, quat.shape[0], sample_rate)

            quats = quat[sampled_index, :, :]
            kps = kps[sampled_index, :, :]

            motionutils = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
            #addendum = motionutils.orientation_kps(quats.unsqueeze(0).cuda())
            addendum = motionutils.orientation_kps_withkeypoints(quats.unsqueeze(0).cuda(), torch.unsqueeze(kps, 0).cuda())
            
            xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))

            print("Shapes: ", addendum.shape, xyz_info.shape, self.used_joint_indices)            
            xyz_info = xyz_info[sampled_index, :, :][:, self.used_joint_indices, :]
            
            fullpose = torch.concat([xyz_info.cuda(), addendum.squeeze(0)], axis = 1)
            
            fullpose = fullpose.reshape([fullpose.shape[0], -1])

            return fullpose
        else:
            xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))
            xyz_info = xyz_info[:, self.used_joint_indices, :]

            xyz_info = xyz_info.reshape([xyz_info.shape[0], -1])

            N = xyz_info.shape[0]
        
            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            #h36m_zed_motion_poses = 0.001 * xyz_info[sampled_index]
            h36m_zed_motion_poses = xyz_info[sampled_index]            

            T = h36m_zed_motion_poses.shape[0]

            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, self.used_bone_count, self.component_size)
        return h36m_zed_motion_poses
               
    def upplot(self, t):
        
        newvals = np.zeros([t.shape[0], self.full_bone_count, self.component_size])

        if (self.quaternions):
            newvals[:, :, 3] = 1.0
            
        for i, b in enumerate(self.used_joint_indices):
            newvals[:, b, :] = t[:, i, :]
        return newvals

    def fk(self, anim, quats = False):
        fk = ForwardKinematics(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
        if (quats):
            uquats = anim
        else:
            uquats = expmap_to_quat(anim)
            
        big_array = np.zeros([anim.shape[0], anim.shape[1], 3])

        for i in range(uquats.shape[0]):
            rots = [Quaternion(u) for u in uquats[i]]
            xyz = fk.propagate(rots, Position([0, 0, 0]))
            big_array[i, :] = np.array([k.np() for k in xyz])

        return big_array

    
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


def get_trace(config, model, dataset, frame, used_bone_count, full_bone_count):
    joint_used_xyz = np.arange(0, used_bone_count)
    motion_input, motion_target = dataset[frame]

    
    motion_input = motion_input.cuda()
    motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])
    
    b, n, c, _ = motion_input.shape
    
    motion_input = motion_input.reshape(b, n, full_bone_count, -1)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
    

    with torch.no_grad():
        trace = torch.jit.trace(model, motion_input)
    return trace

def fetch(config, model, dataset, frame, used_bone_count, full_bone_count):

    joint_used_xyz = np.arange(0, used_bone_count)
    motion_input, motion_target = dataset[frame]

    orig_input = motion_input.clone()
    
    motion_input = motion_input.cuda()
    motion_target = motion_target.cuda()
    
    motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])
    motion_target = motion_target.reshape([1, motion_target.shape[0], motion_target.shape[1], -1])

    b, n, c, _ = motion_input.shape

    motion_input = motion_input.reshape(b, n, used_bone_count, -1)
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
                print("Using deriv input")
                motion_input_ = motion_input.clone()                
                motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], motion_input_.cuda())
            else:
                motion_input_ = motion_input.clone()
                
            start_time = time.time() * 1000
            print("Size in: ", motion_input_.shape)

            output = model(motion_input_)
            print("Output shape is ", output.shape, " and input is ",motion_input_.shape)

            end_time = time.time() * 1000

            print("Time to eval is %f"%(end_time - start_time))
            # Should this only apply if config.deriv_input ?

            output = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], output)[:, :step, :]
            print("Size out: ", output.shape)

            if config.deriv_output:
                print("Using deriv output")                
                output = output + motion_input[:, -1, :].repeat(1, step, 1)

            
            # if config.deriv_output:
            #     offset = motion_input[:, -1:, :].cuda()
            #     motion_pred = output[:, :config.motion.h36m_zed_target_length] + offset
            # else:
            #     motion_pred = output[:, :config.motion.h36m_zed_target_length]

        output = output.reshape(-1, used_bone_count * config.data_component_size)
        output = output.reshape(b, step, -1)
        print("Output shape is ", output.shape)

        outputs.append(output)

        motion_input = torch.cat([motion_input[:, step:], output], axis = 1)

    motion_pred = torch.cat(outputs, axis = 1)[:, :25]

    print("Motion pred is ", motion_pred)
    b, n, c, _ = motion_target.shape
    motion_gt = motion_target.clone()

    motion_pred2 = motion_target.clone().reshape(b, n, used_bone_count, config.data_component_size)

    print("Motion Pred shape: ", motion_pred.shape)
    print("Motion Pred2 shape: ", motion_pred2.shape)

    motion_pred2[:, :, joint_used_xyz] = motion_pred.reshape([motion_pred.shape[0],motion_pred.shape[1], -1, config.data_component_size])
    
    mgtreshape = motion_gt.reshape([motion_gt.shape[1], -1, config.data_component_size])
    mpredreshape = motion_pred2.reshape([motion_pred.shape[1], -1, config.data_component_size])

    origreshape = orig_input.reshape([orig_input.shape[0], -1, config.data_component_size])

    print("Torch diff: %f"%(torch.norm(mgtreshape - mpredreshape)))
    gt_out = np.concatenate([origreshape.cpu(), mgtreshape.cpu()], axis = 0)    
    pred_out = np.concatenate([origreshape.cpu(), mpredreshape.cpu()], axis = 0)

    return gt_out, pred_out


    # motion_pred = torch.cat(outputs, axis = 1)[:, :25]

    # b, n, c, _ = motion_target.shape
    # print("b=%d, n=%d, c=%d, _=%d"%(b, n, c, _))
    # motion_gt = motion_target.clone()
    
    # pred_rot = motion_pred.clone().reshape(b, n, used_bone_count, config.data_component_size)

    # motion_pred = motion_pred.detach().cpu()
    # motion_pred = motion_target.clone().reshape(b, n, used_bone_count, config.data_component_size)
    # motion_pred[:, :, joint_used_xyz] = pred_rot

    # tmp = motion_gt.clone()
    # motion_pred = tmp


    # mgtreshape = motion_gt.reshape([motion_gt.shape[1], -1, 3])
    # mpredreshape = motion_pred.reshape([motion_pred.shape[1], -1, 3])

    # origreshape = orig_input.reshape([orig_input.shape[0], -1, 3])

    # print("Torch diff: %f"%(torch.norm(mgtreshape - mpredreshape)))
    
    # gt_out = np.concatenate([origreshape.cpu(), mgtreshape.cpu()], axis = 0)    
    # pred_out = np.concatenate([origreshape.cpu(), mpredreshape.cpu()], axis = 0)

    # return gt_out, pred_out

def initialize(modelpth, input_file, start_frame, quats = False, rots = False, zeros = False,
               layer_norm_axis = False,
               with_normalization = False,
               dumptrace = None,
               all_bones = False,
               ori_kps = False
               ):


    cfgvals = prepare_config(config, quats, all_bones, ori_kps, 50)
    
    config.motion_mlp.with_normailzation = with_normalization
    config.motion.norm_axis = layer_norm_axis

    
    if (quats):
        config.use_quaternions = True
        #config.loss_quaternion_distance = True
        config.motion.dim = 72 # 4 * 18
        config.motion_mlp.hidden_dim = config.motion.dim
        config.motion_fc_in.in_features = config.motion.dim
        config.motion_fc_in.in_features = config.motion.dim        
        config.motion_fc_in.out_features = config.motion.dim
        config.motion_fc_out.in_features = config.motion.dim
        config.motion_fc_out.out_features = config.motion.dim
        config.data_component_size = 4
        
    if (ori_kps):
        config.motion.dim = 306
        config.dim_ = 306

    # import json
    # print(json.dumps(config, indent = 4))
    # print("Orikips is ", ori_kps)
    # exit(0)

    print("Config is ", config)
    
    model = Model(config)

    state_dict = torch.load(modelpth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_eval

    dataset = AnimationSet(config, input_file, zeros = zeros, rotations = rots, quaternions = quats, all_bones = all_bones, ori_kps = ori_kps)
    shuffle = False
    sampler = None
    train_sampler = None

    used_bone_count, full_bone_count = bone_counts(all_bones, ori_kps)

    if (dumptrace):
        traced_module = get_trace(config, model, dataset, start_frame, used_bone_count, full_bone_count)
        traced_module.save(dumptrace)


    gt_, pred_ = fetch(config, model, dataset, start_frame, used_bone_count, full_bone_count)
    
    print("gt_ shape: ", gt_.shape)
    print("pred_ shape: ", pred_.shape)
    print("Config Deriv: ", config.deriv_input, config.deriv_output)
    if (rots):
        gt = dataset.fk(dataset.upplot(gt_))
        pred = dataset.fk(dataset.upplot(pred_))

    elif(quats):
        gt = dataset.fk(dataset.upplot(gt_), quats = True)
        pred = dataset.fk(dataset.upplot(pred_), quats = True)
    elif(ori_kps):
        gt = gt_
        pred = pred_
    else:

        gt = dataset.upplot(gt_)
        pred = dataset.upplot(pred_)
        
    return gt, pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_pth', type=str, default="./log/snapshot/model-iter-40000.pth", help='=encoder path')

    parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
    parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
    parser.add_argument("--scale", type = float, default = 1.0)
    parser.add_argument("--elev", type = float, help = "Elevation", default = 0)
    parser.add_argument("--azim", type = float, help = "Azimuth", default = 0)
    parser.add_argument("--roll", type = float, help = "Roll", default = 0)

    parser.add_argument("--zeros", action = 'store_true', help = "Zero-expmap")

    parser.add_argument("--fps", type = float, default = 50, help = "Override animation fps")

    parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
    parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')

    parser.add_argument("--save", type = str, help = "File to save animation to")
    parser.add_argument("--save_npz", type = str, help = "File to save input and output files to")

    parser.add_argument("--fullbones", action = 'store_true', help = "Use all skeleton bones")
    
    parser.add_argument('--rotations', action = 'store_true', help = 'Rotation-based data')
    parser.add_argument('--quaternions', action = 'store_true', help = 'Rotation-based data')    
    parser.add_argument('--dumptrace', type = str, help = "Dump the model to a Torchscript trace function for use in C++")
    parser.add_argument('--orient_kps', action = 'store_true', help = 'Orientation Keypoints')        
    parser.add_argument('file', type = str)
    parser.add_argument('start_frame', type = int)
    args = parser.parse_args()

    gt, pred = initialize(args.model_pth, args.file, args.start_frame, quats = args.quaternions, rots = args.rotations, layer_norm_axis = args.layer_norm_axis, with_normalization = args.with_normalization, dumptrace = args.dumptrace, all_bones = args.fullbones, ori_kps = args.orient_kps)


    if (args.save_npz):
        np.savez(args.save_npz, gt = gt, pred = pred, input_file = np.array(args.file), startframe = np.array(args.start_frame))

    print(pred.shape)
    pred[:, 1, :] = 0.0
        
    anim = Animation([gt, pred], dots = not args.nodots, skellines = args.lineplot, scale = args.scale, unused_bones = True, skeltype = 'zed', elev = args.elev, azim = args.azim, roll = args.roll, fps = args.fps, save = args.save)

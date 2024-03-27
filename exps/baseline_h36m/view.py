import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch, expmap2rotmat_torch

from utils.visualize import Animation, Loader

import torch
from torch.utils.data import Dataset, DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']


# Loads a single specified animation file
class AnimationSet(Dataset):
    
    def __init__(self, config, filename):
        super(AnimationSet, self).__init__()
        
        self.filename = filename

        self.h36_motion_input_length  = config.motion.h36m_input_length
        self.h36_motion_target_length  = config.motion.h36m_target_length
        
        pose = self._preprocess(self.filename)
        self.h36m_seqs = []
        self.h36m_seqs.append(pose)
        print(self.h36m_seqs[0].shape)
        num_frames = pose.shape[0]


    def __len__(self):
        return self.h36m_seqs[0].shape[0] - (self.h36_motion_input_length + self.h36_motion_target_length)

    def __getitem__(self, index):
        start_frame = index
        end_frame_inp = index + self.h36_motion_input_length
        end_frame_target = index + self.h36_motion_input_length + self.h36_motion_target_length
        input = self.h36m_seqs[0][start_frame:end_frame_inp].float()
        target = self.h36m_seqs[0][end_frame_inp:end_frame_target].float()
        return input, target
        # frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        # motion = self.h36m_seqs[idx][frame_indexes]
        # if self.data_aug:
        #     if torch.rand(1)[0] > .5:
        #         idx = [i for i in range(motion.size(0)-1, -1, -1)]
        #         idx = torch.LongTensor(idx)
        #         motion = motion[idx]

        # h36m_motion_input = motion[:self.h36m_motion_input_length] / 1000 # meter
        # h36m_motion_target = motion[self.h36m_motion_input_length:] / 1000 # meter



        
        # h36m_motion_input = h36m_motion_input.float()
        # h36m_motion_target = h36m_motion_target.float()
        # return h36m_motion_input, h36m_motion_target


    def _preprocess(self, filename):
        info = open(filename, 'r').readlines()
        pose_info = []
        for line in info:
            line = line.strip().split(',')
            if len(line) > 0:
                pose_info.append(np.array([float(x) for x in line]))
        pose_info = np.array(pose_info)
        pose_info = pose_info.reshape(-1, 33, 3)
        pose_info[:, :2] = 0
        N = pose_info.shape[0]
        pose_info = pose_info.reshape(-1, 3)
        pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(N, 33, 3, 3)[:, 1:]
        pose_info = rotmat2xyz_torch(pose_info)

        sample_rate = 2
        sampled_index = np.arange(0, N, sample_rate)
        h36m_motion_poses = pose_info[sampled_index]

        T = h36m_motion_poses.shape[0]
        h36m_motion_poses = h36m_motion_poses.reshape(T, 32, 3)
        return h36m_motion_poses


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

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b, n,c,_ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                print("model in: ", motion_input_.shape)
                output = model(motion_input_)
                print("model out: ", output.shape)                
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36



def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]


def fetch(config, model, dataset, frame):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    inp, gt = dataset[frame]
    motion_input = inp.cuda()
    motion_gt = gt.cuda()
    
    b = 1
    n, c, _ = motion_input.shape

    motion_input = motion_input.reshape(b, n, 32, 3)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
    motion_gt = motion_gt[:, joint_used_xyz, :]
    motion_gt = motion_gt.reshape(b, motion_gt.shape[0], -1)    

    step = config.motion.h36m_target_length_eval    

    if (step == 25):
        num_step = 1
    else:
        num_step = 25 // step + 1
    for idx in range(num_step):
        
        with torch.no_grad():
            if config.deriv_input:

                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
            else:
                motion_input_ = motion_input_.clone()

            output = model(motion_input_)
        
            output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]

            if config.deriv_output:
                output = output + motion_input[:, -1, :].repeat(1, step, 1)

            
            output = output.reshape(-1, 22 * 3)
            output = output.reshape(b, step, -1)
            motion_input = torch.cat([motion_input[: step:], output], axis = 1)

    print("Shapes after looping:",motion_input.shape, motion_gt.shape, output.shape)
    new_gt = torch.cat([motion_input, motion_gt], axis = 1)
    new_gt = new_gt.reshape([new_gt.shape[0], new_gt.shape[1], -1, 3]).squeeze(0)
    
    pred = torch.cat([motion_input, output], axis = 1)
    pred = pred.reshape([pred.shape[0], pred.shape[1], -1, 3]).squeeze(0)

    print("Shapes before dumping:", gt.shape, new_gt.shape, pred.shape)
    
    return new_gt.cpu().numpy(), pred.cpu().numpy(), inp

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default="./log/snapshot/model-iter-40000.pth", help='=encoder path')

    parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
    parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
    parser.add_argument("--scale", type = float, default = 1000.0)
    parser.add_argument('file', type = str)
    parser.add_argument('start_frame', type = int)
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    
    #dataset = H36MEval(config, 'test')
    dataset = AnimationSet(config, args.file)
    shuffle = False
    sampler = None
    train_sampler = None
    # dataloader = DataLoader(dataset, batch_size=128,
    #                         num_workers=1, drop_last=False,
    #                         sampler=sampler, shuffle=shuffle, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print("Num entries: %d"%len(dataset))

    gt, pred, inp = fetch(config, model, dataset, args.start_frame)

    np.savez("test_output.npz", gt = gt, pred = pred, inp = inp)
    
    anim = Animation(gt, dots = not args.nodots, skellines = args.lineplot, scale = args.scale, unused_bones = False)


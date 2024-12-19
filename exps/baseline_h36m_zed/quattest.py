
import torch
import numpy as np
import argparse
from zed_utilities import MotionUtilities_Torch, body_34_parts, body_34_tree, body_34_tpose

from zed_utilities import batch_rotate_vector, batch_quat_multiply


def quatdiff(a, b):
    binv = torch.tensor([[[-1, -1, -1, 1]]]).float().cuda() * b
    op =  batch_quat_multiply(a, binv)

    return op

def quatalign(a):
    print(a.shape)
    return torch.where(a[:, :, :, 3:] > 0, a, -a)



    
if (__name__ == '__main__'): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--midpoints", action = "store_true")
	
    parser.add_argument('file', type = str)
    parser.add_argument('testframe', type = int)
    
    args = parser.parse_args()
    
    testframe = args.testframe
    
    mu = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

    if(args.file[-4:] == '.npz'):
        mofile = np.load(args.file, allow_pickle = True)
    
        kp_data = torch.tensor(mofile['keypoints']).unsqueeze(0).float().cuda()
        quat_data = torch.tensor(mofile['quats']).unsqueeze(0).float().cuda()
    else:
        rawkp = np.loadtxt('%s_keypoints.csv'%args.file, delimiter = ',')
        rawquat = np.loadtxt('%s_quaternions.csv'%args.file, delimiter = ',')        

        kp_data = torch.tensor(rawkp).reshape([1, rawkp.shape[0],34, 3]).float().cuda()
        quat_data = torch.tensor(rawquat).reshape([1, rawquat.shape[0],34, 4]).float().cuda()


    zero_amt = kp_data[:, :, 0:1, :]
    kp_data = kp_data - zero_amt
    print("Zero thing: ", zero_amt)
    print("Recoup: ", kp_data + zero_amt)
    # globrots = mu.globalrotations(quat_data)
    # locrots = mu.localrotations(globrots)
    
    # torch.set_printoptions(precision = 3,sci_mode = False)
    
    # print("Global: ", globrots)
    # print("Orig: ", quat_data)
    # print("Localized: ", locrots)
    
    # print("Diff: ", (locrots - quat_data))
    
    globrots = mu.globalrotations(quat_data)
    if (args.midpoints):
        orients = mu.orientation_kps(quat_data, printframe = 0)        
    else:
        orients = mu.orientation_kps_withkeypoints(quat_data, kp_data, printframe = 0)
        
    print(orients.shape, quat_data.shape)
    full_mocap = torch.concat([kp_data, orients], dim = 2) + zero_amt
    
    torch.set_printoptions(precision = 3,sci_mode = False)
    print("Full Orikips: ", full_mocap[:, testframe, :, :])

    
    rebuild_rots = mu.rebuild_quaternions(full_mocap, use_midpoints = args.midpoints).cuda()
    alt_rebuild = mu.rebuild_quaternions(full_mocap, use_midpoints = args.midpoints, altfn = True).cuda()
    print("Orig: ", quat_data[:, testframe, :, :])
    print("Rebuild: ", rebuild_rots[:, testframe, :, :])
    print("Alt Rebuild: ", alt_rebuild[:, testframe, :, :])    
    print("Global: ", globrots[:, testframe, :, :])


    # print("Diff: ", quat_data[:, testframe, :, :] - rebuild_rots[:, testframe, :, :].cuda())
    
    # print("Norm: %f"%(torch.norm(quat_data[:, testframe, :, :] - rebuild_rots[:, testframe, :, :].cuda())))
    # print("Norms: ", np.linalg.norm(rebuild_rots[:, testframe, :, :], axis = 2))
    
    print("Diff is ", (quatalign(rebuild_rots) - quatalign(quat_data))[:, testframe, :, :])

    ad = rebuild_rots[:, testframe, :, :].unsqueeze(2)
    bd = quat_data[:, testframe, :, :].unsqueeze(2)
    qd = quatdiff(ad, bd).squeeze(0).squeeze(1)
    print("Quatdiff is ", qd)
    print("QD shape is ",qd.shape)

    print("Diffnorm is ", torch.norm(quatalign(rebuild_rots) - quatalign(quat_data)))

    print("Quat shape is ", quat_data.shape)
    # testvec1 = batch_rotate_vector(quat_data[:, :, 1:2, :], torch.tensor([0, 1, 0]).cuda().float())
    # testvec2 = batch_rotate_vector(quat_data[:, :, 1:2, :], torch.tensor([0, 0, 1]).cuda().float())

    
    # print("BRV shapes: ", testvec1.shape)
    # print("Batch Rotate Vec 1 test: ", testvec1[:, :, :, :])
    # print("Batch Rotate Vec 2 test: ", testvec2[:, :, :, :])
    
else:

    infile = '../../data/h36m_zed/S1/S1_directions_1_zed34_test.npz'
    torch.set_printoptions(precision = 4, sci_mode = False)
    bundle = np.load(infile)
    tquats = torch.tensor(bundle['quats']).float().cuda().unsqueeze(0)
    tkps = torch.tensor(bundle['keypoints']).float().cuda().unsqueeze(0)

    mu = MotionUtilities_Torch(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)

    midpoints = mu.midpoints(tquats)
    orikips = mu.orientation_kps(tquats)
    
    globalrots = mu.globalrotations(tquats)
    fullori = torch.concatenate([tkps, orikips], axis = 2)
    rebuild, reglobrots = mu.rebuild_quaternions(fullori, globrot = True)
    












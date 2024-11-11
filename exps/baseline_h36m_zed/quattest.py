
import torch
import numpy as np
import argparse
from zed_utilities import MotionUtilities_Torch, body_34_parts, body_34_tree, body_34_tpose

#from zed_utilities import quat2rotmat_torch, batch_rotate_vector


def quatalign(a):
    print(a.shape)
    return torch.where(a[:, :, :, 3:] > 0, a, -a)



    
if (__name__ == '__main__'): 
    parser = argparse.ArgumentParser()
	
    parser.add_argument('file', type = str)
    parser.add_argument('testframe', type = int)
    
    args = parser.parse_args()
    
    testframe = args.testframe
    
    mu = MotionUtilities_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
    
    mofile = np.load(args.file, allow_pickle = True)
    
    kp_data = torch.tensor(mofile['keypoints']).unsqueeze(0).float().cuda()
    quat_data = torch.tensor(mofile['quats']).unsqueeze(0).float().cuda()
    
    # globrots = mu.globalrotations(quat_data)
    # locrots = mu.localrotations(globrots)
    
    # torch.set_printoptions(precision = 3,sci_mode = False)
    
    # print("Global: ", globrots)
    # print("Orig: ", quat_data)
    # print("Localized: ", locrots)
    
    # print("Diff: ", (locrots - quat_data))
    
    globrots = mu.globalrotations(quat_data)
    orients = mu.orientation_kps(quat_data)
    
    print(orients.shape, quat_data.shape)
    full_mocap = torch.concat([kp_data, orients], dim = 2)
    
    torch.set_printoptions(precision = 3,sci_mode = False)
    rebuild_rots = mu.rebuild_quaternions(full_mocap).cuda()
    
    print("Orig: ", quat_data[:, testframe, :, :])
    print("Rebuild: ", rebuild_rots[:, testframe, :, :])
    print("Global: ", globrots[:, testframe, :, :])
    # print("Diff: ", quat_data[:, testframe, :, :] - rebuild_rots[:, testframe, :, :].cuda())
    
    # print("Norm: %f"%(torch.norm(quat_data[:, testframe, :, :] - rebuild_rots[:, testframe, :, :].cuda())))
    # print("Norms: ", np.linalg.norm(rebuild_rots[:, testframe, :, :], axis = 2))
    
    print("Diff is ", (quatalign(rebuild_rots) - quatalign(quat_data))[:, testframe, :, :])
    print("Diffnorm is ", torch.norm(quatalign(rebuild_rots) - quatalign(quat_data)))
    
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
    

















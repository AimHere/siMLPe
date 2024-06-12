
import numpy as np
import argparse

from zed_utilities import quat_to_expmap_torch, ForwardKinematics, body_34_parts, body_34_tree, body_34_tpose, Position, Quaternion, expmap_to_quat, PointsToRotations

# Loads a single specified animation file
class AnimationSet:
    used_joint_indices = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]) # Bones 32 and 33 are non-zero rotations, but constant            

    def __init__(self, filename, zeros = False):
        super(AnimationSet, self).__init__()

        self.filename = filename

        self.zeros = zeros
        
        ff = np.load(filename, allow_pickle = True)
        
        self.quats, self.quantized_quats, self.keypoints = [ff[i] for i in ['quats', 'quantized_quats', 'keypoints']]

        
    def upplot(self, t):
        print(t.shape)
        newvals = np.zeros([t.shape[0], REALFULL_BONE_COUNT, 3])
        for i, b in enumerate(self.used_joint_indices):
            newvals[:, b, :] = t[:, i, :]
        return newvals

    def fk(self, anim):
        fk = ForwardKinematics(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
        uquats = expmap_to_quat(anim)
        big_array = np.zeros_like(anim)

        for i in range(uquats.shape[0]):
            rots = [Quaternion(u) for u in uquats[i]]
            xyz = fk.propagate(rots, Position([0, 0, 0]))
            big_array[i, :] = np.array([k.np() for k in xyz])

        return big_array
                                                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--quaternions', action = 'store_true', help = 'Rotation-based data')    
    parser.add_argument('file', type = str)
    #parser.add_argument('start_frame', type = int)
    args = parser.parse_args()

    animdata = AnimationSet(args.file, zeros = False)

    p2r = PointsToRotations(animdata.keypoints, body_34_parts, body_34_tree)
    print(p2r.root_rot().shape)
    print(p2r.root_rot())
  

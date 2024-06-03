import numpy as np
import torch
import argparse
import math

from scipy.spatial.transform import Rotation


body_34_parts = [
    "PELVIS",
    "NAVALSPINE",
    "CHESTSPINE",
    "NECK",
    "LEFTCLAVICLE",
    "LEFTSHOULDER",
    "LEFTELBOW",
    "LEFTWRIST",
    "LEFTHAND",
    "LEFTHANDTIP",
    "LEFTTHUMB",
    "RIGHTCLAVICLE",
    "RIGHTSHOULDER",
    "RIGHTELBOW",
    "RIGHTWRIST",
    "RIGHTHAND",
    "RIGHTHANDTIP",
    "RIGHTTHUMB",
    "LEFTHIP",
    "LEFTKNEE",
    "LEFTANKLE",
    "LEFTFOOT",
    "RIGHTHIP",
    "RIGHTKNEE",
    "RIGHTANKLE",
    "RIGHTFOOT",
    "HEAD",
    "NOSE",
    "LEFTEYE",
    "LEFTEAR",
    "RIGHTEYE",
    "RIGHTEAR",
    "LEFTHEEL",
    "RIGHTHEEL"
]


body_38_parts = [
    "PELVIS",
    "SPINE_1",
    "SPINE_2",
    "SPINE_3",
    "NECK",
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_CLAVICLE",
    "RIGHT_CLAVICLE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_BIG_TOE",
    "RIGHT_BIG_TOE",
    "LEFT_SMALL_TOE",
    "RIGHT_SMALL_TOE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_HAND_THUMB_4",
    "RIGHT_HAND_THUMB_4",
    "LEFT_HAND_INDEX_1",
    "RIGHT_HAND_INDEX_1",
    "LEFT_HAND_MIDDLE_4",
    "RIGHT_HAND_MIDDLE_4",
    "LEFT_HAND_PINKY_1",
    "RIGHT_HAND_PINKY_1"]

body_34_tree = { 
    "PELVIS": ["NAVALSPINE", "LEFTHIP", "RIGHTHIP"],
    "NAVALSPINE" : ["CHESTSPINE"],
    "CHESTSPINE" : ["LEFTCLAVICLE", "RIGHTCLAVICLE", "NECK"],

    "LEFTCLAVICLE" : ["LEFTSHOULDER"],
    "LEFTSHOULDER" : ["LEFTELBOW"],
    "LEFTELBOW" : ["LEFTWRIST"],
    "LEFTWRIST" : ["LEFTHAND", "LEFTTHUMB"],
    "LEFTHAND" : ["LEFTHANDTIP"],
     
    "RIGHTCLAVICLE" : ["RIGHTSHOULDER"],
    "RIGHTSHOULDER" : ["RIGHTELBOW"],
    "RIGHTELBOW" : ["RIGHTWRIST"],
    "RIGHTWRIST" : ["RIGHTHAND", "RIGHTTHUMB"],
    "RIGHTHAND" : ["RIGHTHANDTIP"],
     
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTFOOT", "LEFTHEEL"],
    "LEFTHEEL" : ["LEFTFOOT"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTFOOT", "RIGHTHEEL"],
    "RIGHTHEEL" : ["RIGHTFOOT"],

    "NECK" : ["HEAD", "LEFTEYE", "RIGHTEYE"],
    "HEAD" : ["NOSE"],
    "LEFTEYE" : ["LEFTEAR"],
    "RIGHTEYE" : ["RIGHTEAR"],

    "LEFTHANDTIP" : [],
    "LEFTTHUMB" : [],
    
    "RIGHTHANDTIP" : [],
    "RIGHTTHUMB" : [],
    
    "NOSE" : [],
    "LEFTEAR" : [],
    "RIGHTEAR" : [],
    
    "LEFTFOOT" : [],
    "RIGHTFOOT" : []

    
    }

body_38_tree = {
    "PELVIS": ["SPINE1", "LEFTHIP", "RIGHTHIP"],
    
    "SPINE1": ["SPINE2"],
    "SPINE2": ["SPINE3"],
    "SPINE3": ["NECK", "LEFTCLAVICLE", "RIGHTCLAVICLE"],

    "NECK": ["NOSE"],
    "NOSE": ["LEFTEYE", "RIGHTEYE"],
    "LEFTEYE": ["LEFTEAR"],
    "RIGHTEYE": ["RIGHTEAR"],
    
    "LEFTCLAVICLE": ["LEFTSHOULDER"],
    "LEFTSHOULDER": ["LEFTELBOW"],
    "LEFTELBOW": ["LEFTWRIST"],
    "LEFTWRIST": ["LEFTHANDTHUMB4",
                   "LEFTHANDINDEX1",
                   "LEFTHANDMIDDLE4",
                   "LEFTHANDPINKY1"],

    "RIGHTCLAVICLE": ["RIGHTSHOULDER"],
    "RIGHTSHOULDER": ["RIGHTELBOW"],
    "RIGHTELBOW": ["RIGHTWRIST"],
    "RIGHTWRIST": ["RIGHTHANDTHUMB4",
                   "RIGHTHANDINDEX1",
                   "RIGHTHANDMIDDLE4",
                   "RIGHTHANDPINKY1"],
    
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTHEEL", "LEFTBIGTOE", "LEFTSMALLTOE"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTHEEL", "RIGHTBIGTOE", "RIGHTSMALLTOE"],

    
}


# Utility functions for handling Zed data

body_34_tpose = [[0,0,0],
                 [-0.000732270938924443,175.158289701814,0.0000404],
                 [0.104501423707752,350.306093180137,0.061023624087894],
                 [0.209734388920577,525.459141360196,0.122005361332611],
                 [-47.5920764358155,526.439401661188,0.945479045649915],
                 [-173.508163386225,526.509589382348,2.98848444604651],
                 [-413.490495136004,529.070006884027,4.30015451481757],
                 [-644.259234923473,531.557805055099,5.5156119864156],
                 [-690.412981869986,532.055366973068,5.75870846651414],
                 [-782.7204827236,533.050484095638,6.24488492674942],
                 [-737.171470366004,477.177940319642,6.03507728101326],
                 [48.0126939292815,526.382576868867,-0.700813930486266],
                 [173.928779031094,526.312389136708,-2.74381818341968],
                 [413.965281719423,525.913766951398,-5.58751914341834],
                 [644.770225962532,525.03114718918,-8.36370311424063],
                 [690.931216942206,524.854620798345,-8.91894225928869],
                 [783.253196286002,524.501573309859,-10.029413939978],
                 [736.897241876494,469.296359031723,-9.22217198040476],
                 [-97.2538992002065,0,-0.0216466824273884],
                 [-97.2534603765872,-398.665678321646,-0.0280368622539885],
                 [-97.236806268837,-753.034804644565,-0.0408179870540185],
                 [-97.2541801180481,-841.630928132314,106.266711069826],
                 [97.2538982249762,0,0.0216480069085377],
                 [97.2606469820357,-398.664829041876,0.0265545153545871],
                 [97.2769776133806,-753.033961966181,0.0252733646276821],
                 [97.2596464463626,-841.626635633415,106.335670500348],
                 [1.30731388292955,660.085767955841,62.6295143653575],
                 [1.37823874062583,704.938429698234,62.5519366785149],
                 [-25.8996591150869,736.342683178085,31.5073346749533],
                 [-76.4225396149832,715.693774717041,-52.9081045914678],
                 [27.8706194688158,736.259256448681,30.7476818991719],
                 [75.9266183557466,715.457396322932,-55.0604622597071],
                 [-97.2254670188798,-841.625806800333,-35.4809212169046],
                 [97.2881989464546,-841.626116231009,-35.4119624250308]]

def quat_to_expmap(rot_info):
    halfthetas = np.arccos(rot_info[:, :, 3])
    sinhalves = np.sin(halfthetas)
    http = np.where(sinhalves == 0, 0, 2 * halfthetas/sinhalves)
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
    a = qa[:, :, 0:1]
    b = qa[:, :, 1:2]
    c = qa[:, :, 2:3]
    d = qa[:, :, 3:4]
    e = qb[:, :, 0:1]
    f = qb[:, :, 1:2]
    g = qb[:, :, 2:3]
    h = qb[:, :, 3:4]

    ww = -a * e - b * f - g * c + d * h
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = np.concatenate([ii, jj, kk, ww], axis = 2)
    return qq

def quat_to_expmap_torch(rot_info):
    halfthetas = torch.acos(rot_info[:, :, 3])
    sinhalves = torch.sin(halfthetas)
    http = torch.where(sinhalves == 0, 0, 2 * halfthetas/sinhalves)
    https = torch.stack([http, http, http], axis = 2)
    rots = https * rot_info[:, :, :3]
    return rots

def expmap_to_quat_torch(exps):
    if (len(exps.shape) == 2):
        exps = torch.reshape(exps, [exps.shape[0], -1, 3])
    rads = torch.norm(exps, dim = 2)
    rv = torch.stack([rads, rads, rads], axis = 2)
    qv = torch.where(rv == 0, 0, (exps[:, :, :3] / rv))
    cosses = torch.cos (rads / 2)
    sins = torch.sin(rads / 2)
    sinss = torch.stack([sins, sins, sins], axis = 2)
    quats = torch.cat([qv * sinss , torch.unsqueeze(cosses, 2)], axis = 2)
    return quats

# Return the rotation distance between two quaternion arrays
def quat_distance(qa, qb): 
    qdiff = np.clip(quat_mult(quat_inverse(qa), qb), -1, 1)
    # Is it better to calculate sines and use np.arctan2?
    halfthetas = np.arccos(qdiff[:, :, 3])
    return 2 * halfthetas
    
# Return the rotation distance between two expmap arrays
def exp_distance(ea, eb):
    qa = expmap_to_quat(ea)
    qb = expmap_to_quat(eb)

    return quat_distance(qa, qb)


def quat_inverse_torch(quats):
    exps = torch.cat([-quats[:, :, :3], quats[:, :, 3:]], axis =2)
    return exps

def quat_mult_torch(qa, qb):
    a = qa[:, :, 0:1]
    b = qa[:, :, 1:2]
    c = qa[:, :, 2:3]
    d = qa[:, :, 3:4]
    e = qb[:, :, 0:1]
    f = qb[:, :, 1:2]
    g = qb[:, :, 2:3]
    h = qb[:, :, 3:4]

    ww = -a * e - b * f - g * c + d * h
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = torch.cat([ii, jj, kk, ww], axis = 2)
    return qq

def quat_distance_torch(qa, qb):
    qdiff = torch.clamp(quat_mult_torch(quat_inverse_torch(qa), qb), -1, 1)
    halfthetas = torch.acos(qdiff[:, :, 3])
    return 2 * halfthetas

def exp_distance_torch(ea, eb):
    qa = expmap_to_quat_torch(ea)
    qb = expmap_to_quat_torch(eb)

    return quat_distance_torch(qa, qb)


class Quantized_Quaternion:
    # Represent a quaternion with three 16-bit fixed-point ints
    def __init__(self, ints):
        self.fixed = ints

    def toQuaternion(self):
        floats = [f / 32767 for f in self.fixed]
        sqrs = [f * f for f in floats]
        # print("Sqrs is ", sqrs)
        # print("Sumsq is %f"%(1.0 - sum(sqrs)))
        floats.append(math.sqrt(1.0 - sum(sqrs)))
        return Quaternion(floats)

    def zero():
        return Quantized_Quaternion([0.0, 0.0, 0.0])

    def __str__(self):
        return Quaternion.toQuaternion.__str__()

    def np(self):
        return np.array(self.fixed).astype(np.int16)
    
class Quaternion:

    def __init__(self, floats):
        self.rot = Rotation.from_quat(floats)

    def __mul__(self, q):
        rmul = (self.rot * q.rot).as_quat()
        return Quaternion(rmul)

    def zero():
        return Quaternion([0.0, 0.0, 0.0, 1.0])
        
    def toEuler(self, perm = 'xyz'):
        e = self.rot.as_euler(perm, degrees = True)
        return Euler([e[0], e[1], e[2]])

    def cstr(self, sep = ","):
        q = self.rot.as_quat()
        return (sep.join([str(i) for i in q]))
    
    def __str__(self):
        q = self.rot.as_quat()
        return (" ".join([str(i) for i in q]))

    def apply(self, x):
        return Position(self.rot.apply([x.x, x.y, x.z]))

    def np(self):
        return self.rot.as_quat()
    
    def toQuantQuat(self):
        q = self.rot.as_quat()
        if (q[3] < 0):
            qq = -q
        else:
            qq = q            
        ints = [round(32767 * f) for f in qq]
        return Quantized_Quaternion(ints)
    
class Euler:
    def __init__(self, floats, perm = 'xyz'):

        self.perm = perm
        self.e0 = floats[0]
        self.e1 = floats[1]
        self.e2 = floats[2]

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.e0, self.e1, self.e2]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.e0, self.e1, self.e2]]))

    def toQuat(self, xneg = False, zneg = False):

        rot = Rotation.from_euler(self.perm, [self.e0,
                                              self.e1,
                                              self.e2], degrees = True)
        q = rot.as_quat()

        
        if (xneg == True):
            xcoord = -q[0]
        else:
            xcoord = q[0]
        if (zneg == True):
            zcoord = -q[2]
        else:
            zcoord = q[2]

        return Quaternion([xcoord, q[1], zcoord, q[3]])


    def toQuantQuat(self, xneg = False, zneg = False):
        return self.toQuat(xneg = xneg, zneg = zneg).toQuantQuat()

class Position:
    def __init__(self, floats):
        [self.x, self.y, self.z] = floats

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.x, self.y, self.z]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.x, self.y, self.z]]))


    def scale(self, s):
        return Position([s * self.x , s * self.y, s * self.z])

    def __add__(self, a):
        return Position([self.x + a.x, self.y + a.y, self.z + a.z])

    def __sub__(self, a):
        return Position([self.x - a.x, self.y - a.y, self.z - a.z])

    def __mul__ (self, k):
        return Position([k * self.x, k * self.y, k * self.z])

    
    def np(self):
        return np.array([self.x, self.y, self.z])

    
class Transform:
    def __init__(self, pos, ori):
        self.pos = pos
        self.ori = ori

    def cstrquat(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.cstr(sep))
    
    def cstr(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.toEuler(args.convert_order).cstr(sep))
    
    def __str__(self):
        return "%s %s"%(self.pos,self.ori.toEuler(args.convert_order))

    def scale(self, x):
        return Transform(self.pos.scale(x), self.ori)

    def offset_pos(self, p):
        return Transform(self.pos + p, self.ori)


class ForwardKinematics:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpos = Position([0,0,0])):
        self.bonetree = bonetree
        self.bonelist = bonelist
        self.root = rootbone
        self.tpose = [Position(p) for p in tpose]
        
    def propagate(self, rotations, initial_position):

        keyvector = [Position([0, 0, 0]) for i in range(34)]
        
        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
                new_pos = initial_position
            else:
                n_rot = c_rot * rotations[pIdx]
                new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])

            keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return keyvector
    

class ForwardKinematics_Torch:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpose = torch.tensor([0, 0, 0])):
        self.bonelist = bonelist
        self.bonetree = bonetree
        self.root = rootbone
        self.tpose = torch.tensor(tpose)


    def propagate(self, rotations, initial_position):

        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                pass
            else:
                pass

            keyVector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)

            initial_rot = rotations[self.bonelist.index(self.root)]

                

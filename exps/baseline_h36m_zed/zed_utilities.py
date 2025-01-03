import numpy as np
import torch
import argparse
import math

import argparse

from scipy.spatial.transform import Rotation

import torch.nn.functional

import torch

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

    def __init__(self, floats, precision = None):
        self.rot = Rotation.from_quat(floats)
        self.precision = precision

    def set_precision(self, p):
        self.precision = p
        
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
        if (self.precision):
            pstr = '%%.%df'%self.precision
        else:
            pstr = "%f"
            
        q = self.rot.as_quat()
        return (" ".join([pstr%i for i in q]))

    def apply(self, x):
        return Position(self.rot.apply([x.x, x.y, x.z]))

    def np(self):
        return self.rot.as_quat()

    def torch(self):
        return torch.tensor(self.rot.as_quat())
    
    def toQuantQuat(self):
        q = self.rot.as_quat()
        if (q[3] < 0):
            qq = -q
        else:
            qq = q            
        ints = [round(32767 * f) for f in qq]
        return Quantized_Quaternion(ints)

    def fwdvec(self):
        return self.apply(Position([1, 0, 0]))

    def upvec(self):
        return self.apply(Position([0, 1, 0]))

    def rightvec(self):
        return self.apply(Position([0, 0, 1]))
    
    
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
    def __init__(self, floats, precision = None):
        #print("Floats: ", floats)
        [self.x, self.y, self.z] = floats
        self.precision = precision
        
    def set_precision(self, p):
        self.precision = p
        
    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.x, self.y, self.z]]))
    
    def __str__(self):
        if (self.precision):
            pstr = "%%.%df"%self.precision
        else:
            pstr = "%f"

        return (" ".join([pstr%i for i in [self.x, self.y, self.z]]))

    def scale(self, s):
        return Position([s * self.x , s * self.y, s * self.z], precision = self.precision)

    def __add__(self, a):
        return Position([self.x + a.x, self.y + a.y, self.z + a.z], precision = self.precision)

    def __sub__(self, a):
        return Position([self.x - a.x, self.y - a.y, self.z - a.z], precision = self.precision)

    def __mul__ (self, k):
        if isinstance(k, Position):
            return k.x * self.x + k.y * self.y + k.z * self.z
        else:
            return Position([k * self.x, k * self.y, k * self.z], precision = self.precision)
    
    def __lmul__ (self, k):
        if isinstance(k, Position):
            return k.x * self.x + k.y * self.y + k.z * self.z
        else:
            return Position([k * self.x, k * self.y, k * self.z], precision = self.precision)
    
    def __rmul__ (self, k):
        if isinstance(k, Position):
            return k.x * self.x + k.y * self.y + k.z * self.z
        else:
            return Position([k * self.x, k * self.y, k * self.z], precision = self.precision)
    
    def np(self):
        return np.array([self.x, self.y, self.z])

    def torch(self):
        return torch.tensor([self.x, self.y, self.z])
    
    def norm(self):
        return math.sqrt(self * self)

    def cross(self, b):
        new_x = self.y * b.z - self.z * b.y
        new_y = - self.x * b.z + self.z * b.x
        new_z = self.x * b.y - self.y * b.x
        return Position([new_x, new_y, new_z])
    
    
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


def rotmat2quat(mm, transpose = True):
    # Takes a 3x3 numpy array and turns it into our bespoke quaternion
    if transpose:
        m = mm.T
    else:
        m = mm
    
    if (m[2,2] < 0):
        if (m[0,0] > m[1,1]):
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = np.array([t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]])
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = np.array([m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]])
    else:
        if (m[0,0] < -m[1,1]):
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = np.array([m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]])
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = np.array([m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t])

    q *= 0.5 / math.sqrt(t)
    return Quaternion(q)

    
def rotmat2quat_torch(m, transpose = True):
    # Takes a batch of 3x3 rotation matrices and turns them into a batch of quaternions
    # Shape is [batch, frame, 3, 3]
    
    if (transpose):
        m = torch.transpose(m, 2, 3)

    taa = 1 + m[:, :, 0, 0] - m[:, :, 1, 1] - m[:, :, 2, 2]
    tab = 1 - m[:, :, 0, 0] + m[:, :, 1, 1] - m[:, :, 2, 2]

    tba = 1 - m[:, :, 0, 0] - m[:, :, 1, 1] + m[:, :, 2, 2]
    tbb = 1 + m[:, :, 0, 0] + m[:, :, 1, 1] + m[:, :, 2, 2]

    qaa = torch.stack([taa, m[:,:,0,1] + m[:,:,1,0], m[:,:,2,0] + m[:,:,0,2], m[:,:,1,2] - m[:,:,2,1]], dim = 2)
    qab = torch.stack([m[:,:,0,1] + m[:,:,1,0], tab, m[:,:,1,2] + m[:,:,2,1], m[:,:,2,0] - m[:,:,0,2]], dim = 2)
    
    qba = torch.stack([m[:,:,2,0] + m[:,:,0,2], m[:,:,1,2] + m[:,:,2,1], tba, m[:,:,0,1] - m[:,:,1,0]], dim = 2)
    qbb = torch.stack([m[:,:,1,2] - m[:,:,2,1], m[:,:,2,0] - m[:,:,0,2], m[:,:,0,1] - m[:,:,1,0], tbb], dim = 2)

    va = torch.where(m[:,:,0,0] > m[:,:,1,1],  (qaa / torch.sqrt(taa).unsqueeze(2)).transpose(2, 1), (qab / torch.sqrt(tab).unsqueeze(2)).transpose(2, 1))
    vb = torch.where(m[:,:,0,0] < -m[:,:,1,1], (qba / torch.sqrt(tba).unsqueeze(2)).transpose(2, 1), (qbb / torch.sqrt(tbb).unsqueeze(2)).transpose(2, 1))

    
    
    quats = 0.5 * torch.where(m[:,:,2,2] > m[:,:,1,1], va, vb).transpose(2, 1)
    return quats



def rotmat2quat_torch_again(m, transpose = True):
    m00 = m[:, :, 0, 0]
    m11 = m[:, :, 1, 1]
    m22 = m[:, :, 2, 2]
    
    m01 = m[:, :, 0, 1]
    m10 = m[:, :, 1, 0]
    
    m02 = m[:, :, 0, 2]
    m20 = m[:, :, 2, 0]
    
    m12 = m[:, :, 1, 2]
    m21 = m[:, :, 2, 1]

    taa = 1 + m00 - m11 - m22
    tab = 1 - m00 + m11 - m22
    tba = 1 - m00 - m11 + m22
    tbb = 1 + m00 + m11 + m22

    min01 = m01 - m10
    add01 = m01 + m10    
    
    min20 = m20 - m02
    add02 = m20 + m02

    min12 = m12 - m21
    add12 = m12 + m21

    qaa = torch.stack([  taa, add01, add02, min12], 2)
    qab = torch.stack([add01,   tab, add12, min20], 2)
    qba = torch.stack([add02, add12,   tba, min01], 2)
    qbb = torch.stack([min12, min20, min01,   tbb], 2)    


    # print("taa: ", taa)
    # print("tab: ", tab)
    # print("tba: ", tba)
    # print("tbb: ", tbb)
    # print("qaa: ", qaa)
    # print("qab: ", qab)
    # print("qba: ", qba)
    # print("qbb: ", qbb)

    
    va = torch.where(m00 > m11,
                     (qaa/torch.sqrt(taa).unsqueeze(2)).transpose(2, 1),
                     (qab/torch.sqrt(tab).unsqueeze(2)).transpose(2, 1))
    
    vb = torch.where(m00 < -m11,
                     (qba/torch.sqrt(tba).unsqueeze(2)).transpose(2, 1),
                     (qbb/torch.sqrt(tbb).unsqueeze(2)).transpose(2, 1))

    # print("Va:" , va)
    # print("Vb:" , vb)    
    quats = 0.5 * torch.where(m22 < 0, va, vb).transpose(2, 1);

#    print(quats)
#    exit(0)
    
    return quats

def quat2rotmat_torch(q):
    vr = torch.tensor([1, 0, 0]).float()
    vu = torch.tensor([0, 1, 0]).float()
    vf = torch.tensor([0, 0, 1]).float()

    print(q)
    print(q.shape)
    
    vvr = batch_rotate_vector(q, vr)
    vvu = batch_rotate_vector(q, vu)
    vvf = batch_rotate_vector(q, vf)

    print(vr.shape, vvr.shape)
    
    return torch.stack([vvr, vvu, vvf], axis = 3).transpose(3, 4)

def old_rotate_vector(v, k, theta):
    v = np.asarray(v)
    k = np.asarray(k) / np.linalg.norm(k)  # Normalize k to a unit vector
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    q
    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1 - cos_theta)


    print("Precross k: ", k)
    print("Precross v: ", v)
    print("Precross Sin: ", sin_theta)
    
    print("Cross Product is ", np.cross(k, v))
    return term1 + term2 + term3, term1, term2, term3
    
def batch_rotate_vector(quats, vector):

    """ Rotates a vector by a batch of quaternions using Rodriguez rotation formula
    - 'quats': Double-batched quaternions in [batch, frame, bone, 4] shape
    - 'vector' : 3-vector Vector to be rotated by the given bone quaternion
    """
    #print("BRV Input shape : ", quats.shape, " and vector = ", vector)

    halftheta = torch.acos(quats[:, :, :, 3])
    sinhalves = torch.unsqueeze(torch.sin(halftheta), dim = 2)
    #print("BRV: sinhalves halftheta: ", sinhalves.shape, halftheta.shape)
    kvecs = torch.div(quats[:, :, :, :3], sinhalves)
    
    sines = torch.unsqueeze(torch.sin(2 * halftheta), dim = 2)

    costheta = torch.unsqueeze(torch.cos(2 * halftheta), dim = 2)

    t1 = costheta * vector

    #print("BRV: t1 sines kvecs  ", t1.shape, sines.shape, kvecs.shape)

    
    t2 = torch.cross(kvecs, vector.expand_as(kvecs), dim = 3) * sines

    dotproduct = torch.sum(kvecs * vector.expand_as(kvecs), dim = 3)
    t3 = kvecs * torch.unsqueeze(dotproduct, dim = 3) * (1 - costheta)
    #print("Dotprod shape is ", dotproduct.shape)
    #print("T1 t2 t3 shapes: ", t1.shape, t2.shape, t3.shape)

    outval = t1 + t2 + t3
    #print("Outval vs vector: ", outval.shape, vector.shape)

    # if it's a Nan here, it's because it's a 0-rotation quaternion, most likely
    return torch.where(torch.isnan(outval), vector, outval)


# Takes a batched set of tensors and another one and quaternion-multiply them
def batch_quat_multiply(qa, qb, cIdx = None):
    if (cIdx is None):
        a = qa[:, :, :, 0:1]
        b = qa[:, :, :, 1:2]
        c = qa[:, :, :, 2:3]
        d = qa[:, :, :, 3:4]
        
        e = qb[:, :, :, 0:1]
        f = qb[:, :, :, 1:2]
        g = qb[:, :, :, 2:3]
        h = qb[:, :, :, 3:4]
        
        ww = -a * e - b * f - g * c + d * h
        ii = a * h + b * g - c * f + d * e
        jj = b * h + c * e - a * g + d * f
        kk = c * h + a * f - b * e + d * g
        
        qq = torch.cat([ii, jj, kk, ww], axis = 3)
        return qq

    else:
        a = qa[:, :, cIdx:cIdx + 1, 0:1]
        b = qa[:, :, cIdx:cIdx + 1, 1:2]
        c = qa[:, :, cIdx:cIdx + 1, 2:3]
        d = qa[:, :, cIdx:cIdx + 1, 3:4]
        
        e = qb[:, :, :, 0:1]
        f = qb[:, :, :, 1:2]
        g = qb[:, :, :, 2:3]
        h = qb[:, :, :, 3:4]
        
        ww = -a * e - b * f - g * c + d * h
        ii = a * h + b * g - c * f + d * e
        jj = b * h + c * e - a * g + d * f
        kk = c * h + a * f - b * e + d * g

        print("II shape is: ", ii.shape)
        print("TC shape is ",torch.cat([ii, jj, kk, ww], axis = 3).shape)
        qq = qa.clone()
        qq[:, :, cIdx:cIdx + 1, :] = torch.cat([ii, jj, kk, ww], axis = 3)
        return qq
        
    
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
                # print("Old: %d, Nrot:"%cIdx, n_rot)
                # print("Old: %d, NewPos: "%cIdx, new_pos)


            keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return keyvector


def normalize(v):
    norm = np.linalg.norm(v, axis = 1)
    nms = np.stack([norm, norm, norm]).T
    return v / nms
    
                
class PointsToRotations:
    
    def __init__(self, keypoints, body_names, body_tree, root_bone = 'PELVIS'):
        self.keypoints = keypoints
        self.tree = body_tree
        self.names = body_names
        self.root = root_bone
        self.root_idx = self.names.index(self.root)

        
    def root_rot(self):

        root_pos = self.keypoints[:, self.root_idx, :]
        lhip_idx = [self.names.index(i) for i in self.tree[self.root] if 'left' in i.lower()][0]
        rhip_idx = [self.names.index(i) for i in self.tree[self.root] if 'right' in i.lower()][0]
        spine_idx = [self.names.index(i) for i in self.tree[self.root] if 'spine' in i.lower()][0]
        
        lhval = self.keypoints[:, lhip_idx, :]
        rhval = self.keypoints[:, rhip_idx, :]        
        sval = self.keypoints[:, spine_idx, :]
        

        leftvec = normalize(lhval - root_pos)
        upvec = normalize(sval - root_pos)
        
        fwdvec = np.cross(upvec, leftvec)

        return np.stack([leftvec, upvec, fwdvec], axis = 2)

    def calculate_rot_mats(self):

        rrot = self.root_rot

        def _recurse(bone, rotation, c_idx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                #n_rot = c_rot
                new_pos = initial_position
            else:
                #n_rot = c_rot * rotations[pIdx]
                #new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])

                keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)

            _recurse(self.root_idx, None, None)
        

# test_file = '../../data/h36m_zed/S7/S7_posing_2_zed34_test.npz'

# tfdata = np.load(test_file, allow_pickle = True)
class MotionUtilities_Torch:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpose = torch.tensor([0, 0, 0])):
        self.bonelist = bonelist
        self.bonetree = bonetree
        self.root = rootbone
        self.tpose = torch.tensor(tpose).cuda()

    def forwardkinematics(self, rotations, initial_position = None):

        if (initial_position):
            ipos = initial_position
        else:
            ipos = torch.zeros([3])

        key_tensor = torch.zeros([rotations.shape[0], rotations.shape[1], rotations.shape[2], 3]).cuda()
        
        def _recurse(parentbone, cur_rot, pIdx):

            cIdx = self.bonelist.index(parentbone)

            if pIdx < 0:
                new_rot = cur_rot.clone()
                new_pos = ipos.clone()
            else:
                new_rot = batch_quat_multiply(cur_rot, rotations[:, :, pIdx:pIdx + 1, :])

                brv = batch_rotate_vector(new_rot, self.tpose[cIdx] - self.tpose[pIdx])
                new_pos = key_tensor[:, :, pIdx:pIdx + 1, :] + brv

            key_tensor[:, :, cIdx:cIdx + 1, :] = new_pos

            for child in self.bonetree[parentbone]:
                _recurse(child, new_rot, cIdx)

        iidx = self.bonelist.index(self.root)
        initial_rot = rotations[:, :, iidx:iidx + 1, :]
        _recurse(self.root, initial_rot, -1)
        return key_tensor

    def globalrotations(self, rotations, initial_position = None):
        # if (initial_position):
        #     ipos = initial_position
        # else:
        #     ipos = torch.zeros([3])
            
        glob_rot_tensor = torch.zeros([rotations.shape[0], rotations.shape[1], rotations.shape[2], 4]).cuda()
        glob_rot_tensor[:, :, :, 3] = 1.0
        def _recurse(parentbone, cur_rot, pIdx):

            cIdx = self.bonelist.index(parentbone)
            if pIdx < 0:
                new_rot = cur_rot.clone()
            else:
                new_rot = batch_quat_multiply(cur_rot, rotations[:, :, cIdx:cIdx + 1, :])
                #print("GlobRot shape: ", new_rot.shape)
                #brv = batch_rotate_vector(new_rot, self.tpose[cIdx] - self.tpose[pIdx])
                #oprint("BRV Out shape: ", brv.shape)
            glob_rot_tensor[:, :, cIdx:cIdx+1, :] = new_rot
            
            for child in self.bonetree[parentbone]:
                _recurse(child, new_rot, cIdx)


        
        iidx = self.bonelist.index(self.root)
        initial_rot = rotations[:, :, iidx:iidx + 1, :]
        # initial_rot = torch.zeros([rotations.shape[0], rotations.shape[1], 1, 4]).cuda()
        # initial_rot[:, :, :, 3] = 1
        _recurse(self.root, initial_rot, -1)

        return glob_rot_tensor

    def midpoints(self, rotations, initial_position = None):

        if (initial_position is not None):
            ipos = initial_position
        else:
            ipos = torch.zeros([3])

        key_tensor = torch.zeros([rotations.shape[0], rotations.shape[1], rotations.shape[2], 3]).cuda()
        mid_tensor = torch.zeros([rotations.shape[0], rotations.shape[1], rotations.shape[2], 3]).cuda()

        def _recurse(parentbone, cur_rot, pIdx):

            cIdx = self.bonelist.index(parentbone)

            if pIdx < 0:
                new_rot = cur_rot.clone()
                new_pos = ipos.clone()
            else:
                new_rot = batch_quat_multiply(cur_rot, rotations[:, :, pIdx:pIdx + 1, :])
                brv = batch_rotate_vector(new_rot, self.tpose[cIdx] - self.tpose[pIdx])
                new_pos = key_tensor[:, :, pIdx:pIdx + 1, :] + brv

            key_tensor[:, :, cIdx:cIdx + 1, :] = new_pos 
            if (pIdx >= 0):
                mid_tensor[:, :, cIdx:cIdx + 1, :] = 0.5 * (key_tensor[:, :, cIdx:cIdx + 1, :] + key_tensor[:, :, pIdx:pIdx + 1, :])
            
            for child in self.bonetree[parentbone]:
                _recurse(child, new_rot, cIdx)

        iidx = self.bonelist.index(self.root)
        initial_rot = rotations[:, :, iidx:iidx + 1, :]
        _recurse(self.root, initial_rot, -1)

        return mid_tensor
        #return torch.concatenate([key_tensor, mid_tensor], axis = 2)

    def localrotations(self, globrots):
        # Take a set of global rotations and return the local rotations based on the body tree

        localrots = torch.zeros_like(globrots).cuda()
        localrots[:, :, :, 3] = 1.0

        def qinv(q):
            return q * torch.tensor([-1, -1, -1, 1]).cuda()
        
        def _recurse(parentbone, pIdx):
            cIdx = self.bonelist.index(parentbone)
            
            if (pIdx < 0):
                localrots[:, :, cIdx, :] = globrots[:, :, cIdx, :]

            else:
                localrots[:, :, cIdx:cIdx + 1, :] = batch_quat_multiply(qinv(globrots[:, :, pIdx:pIdx + 1, :]) , globrots[:, :, cIdx:cIdx + 1, :])                 

            for child in self.bonetree[parentbone]:
                _recurse(child, cIdx)

        initial_rot = globrots[:, :, :1, :]
        _recurse(self.root, -1)
        
        return localrots

        
    def rebuild_quaternions(self, keypoints, use_midpoints = True, globrot = False, altfn = False):
        """ Rebuild the rotation set from the 102-bone orientation keypoints """

        # Bones 0-33 are the original keypoints
        # Bones 34-67 are the [0, 1, 0] vector rotated by the rotation vector and added to tbe bone midpoint
        # Bones 68-102 are the [0, 0, 1] vector rotated by the rotation vector and added to the bone midpoint

        globrots = torch.zeros([keypoints.shape[0], keypoints.shape[1], 34, 4]).cuda()
        globrots[:, :, :, 3] = 1.0

        # First recreate the global rotations from the orientation keypoints
        # The Root rotation
        rIdx = self.bonelist.index(self.root)
        midpoint = keypoints[:, :, rIdx, :]
        vecup = torch.nn.functional.normalize(keypoints[:, :, rIdx + 34, :] - midpoint, dim = 2)
        vecfwd = torch.nn.functional.normalize(keypoints[:, :, rIdx + 68, :] - midpoint, dim = 2)
        vecright = torch.nn.functional.normalize(torch.cross(vecup, vecfwd, dim = 2), dim = 2)
        
        rotmats = torch.stack([vecright, vecup, vecfwd], axis = 3)

        if (altfn):
            globrots[:, :, rIdx, :] = rotmat2quat_torch_again(rotmats)
        else:
            globrots[:, :, rIdx, :] = rotmat2quat_torch(rotmats)

        for pName in self.bonetree:
            for cName in self.bonetree[pName]:
                    pIdx = self.bonelist.index(pName)
                    cIdx = self.bonelist.index(cName)

                    if use_midpoints:
                        midpoint = 0.5 * (keypoints[:, :, pIdx, :] + keypoints[:, :, cIdx, :])
                    else:
                        midpoint = keypoints[:, :, cIdx, :]
                    vecup = torch.nn.functional.normalize(keypoints[:, :, cIdx + 34, :] - midpoint, dim = 2)
                    vecfwd = torch.nn.functional.normalize(keypoints[:, :, cIdx + 68, :] - midpoint, dim = 2)
                    vecright = torch.nn.functional.normalize(torch.cross(vecup, vecfwd, dim = 2), dim = 2)

                    rotmats = torch.stack([vecright, vecup, vecfwd], axis = 3)
                    if (altfn):
                        globrots[:, :, cIdx, :] = rotmat2quat_torch_again(rotmats)                        
                    else:
                        globrots[:, :, cIdx, :] = rotmat2quat_torch(rotmats)
                    #print("BT: %d->%d - Mid %s, Up %s, Fwd: %s, Glob: %s"%(pIdx, cIdx, str(vecup), str(vecfwd), std(globrots[:, :, cIdx, :]))
        # Now we've reconstructed the global rotations, we have to turn them into local rotations
        localrots = self.localrotations(globrots)

        # Lets return the reconstructed global rotations
        if (globrot):
            return localrots, globrots
        return localrots
        
    def orientation_kps_midpoints(self, rotations, initial_position = None, printframe = None):

        grots = self.globalrotations(rotations, initial_position)
        mpoints = self.midpoints(rotations, initial_position)
        mpvecs = torch.concatenate([mpoints, mpoints], axis = 2)


        for i in range(34):
            
            # Right Vectors
            mpvecs[:, :, i :i + 1, :] = mpoints[:, :, i:i+1, :] + batch_rotate_vector(grots[:, :, i:i+1, :], torch.tensor([0.0, 1.0, 0.0]).cuda())

            # Up Vectors
            mpvecs[:, :, i + 34:i+35, :] = mpoints[:, :, i:i+1, :] + batch_rotate_vector(grots[:,:,i:i+1,:], torch.tensor([0.0, 0.0, 1.0]).cuda())

        return mpvecs

    def orientation_kps_withkeypoints(self, rotations, keypoints, initial_position = None, printframe = None):
        grots = self.globalrotations(rotations, initial_position)
        mpvecs = torch.concatenate([keypoints, keypoints], axis = 2)
        for i in range(34):
            
            # Right Vectors
            mpvecs[:, :, i :i + 1, :] = keypoints[:, :, i:i+1, :] + batch_rotate_vector(grots[:, :, i:i+1, :], torch.tensor([0.0, 1.0, 0.0]).cuda())
            # Up Vectors
            mpvecs[:, :, i + 34:i+35, :] = keypoints[:, :, i:i+1, :] + batch_rotate_vector(grots[:,:,i:i+1,:], torch.tensor([0.0, 0.0, 1.0]).cuda())
        return mpvecs
        
    def orientation_kps(self, rotations, initial_position = None, printframe = None):
        return self.orientation_kps_midpoints(rotations, initial_position = None, printframe = None)


fk = ForwardKinematics(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

#fktorch = ForwardKinematics_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

if (False):
    test_quats, test_kps, test_quant_quats = [tfdata[i] for i in ['quats', 'keypoints', 'quantized_quats']]

    rots = []
    for tq in test_quats:
        rots.append([Quaternion(u) for u in tq])

    #quant_torch = torch.unsqueeze(torch.tensor(test_quats), dim = 0).type(torch.Tensor)
    btpose_torch = torch.tensor(body_34_tpose)

    test_quat1_np = np.array([-0.1419, -0.0820, 0.400, 0.9018])
    test_quat2_np = np.array([ 0.27907279,  0.33488734, -0.44651646,  0.7814038 ])

    test_axis1 = test_quat1_np[:3] / np.linalg.norm(test_quat1_np[:3])
    test_theta1 = 2 * math.acos(test_quat1_np[3])

    test_axis2 = test_quat2_np[:3] / np.linalg.norm(test_quat2_np[:3])
    test_theta2 = 2 * math.acos(test_quat2_np[3])

    test_v_np = np.array([1.0, 2.0, 4.0000001])
    test_v_torch = torch.unsqueeze(torch.tensor(test_v_np), dim = 0)

    test_quat1_torch = torch.tensor(test_quat1_np)
    test_quat2_torch = torch.tensor(test_quat2_np)

    test_quat_torch = torch.reshape(torch.stack([test_quat1_torch, test_quat2_torch]), [1, 2, 1, 4])

    quats_torch = torch.unsqueeze(torch.tensor(test_quats), dim = 0).float()
#from zed_utilities import ForwardKinematics, ForwardKinematics_Torch, old_rotate_vector, batch_rotate_vector, test_v_np, test_v_torch, test_quat_torch, test_axis1, test_theta1, test_axis2, test_theta2, test_quat1_np, test_quat2_np, fktorch, quats_torch, Position, test_kps, body_34_parts, body_34_tree, fk

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_npz", type = str)
    parser.add_argument("in_npz", type = str)
    parser.add_argument("frame", type = int)
    parser.add_argument("out_csv", type = str)

    args = parser.parse_args()

    if (args.out_csv[-4:].lower() != ".csv"):
        print("Error: Not writing to a file that doesn't end in '.csv'")
        exit(0)
                        
    inf_data = np.load(args.in_npz, allow_pickle = True)
    npquat, np_kps = [inf_data[i] for i in ['quats', 'keypoints']]
    
    #fkt = ForwardKinematics_Torch(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)
    fkt = MotionUtilities_Torch(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)
    fkn = ForwardKinematics(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)

    oframe = args.frame
    
    quats_torch = torch.unsqueeze(torch.tensor(npquat), dim = 0).float()[:, oframe:oframe + 1, :, :].cuda()
    kp_torch = fkt.forwardkinematics(quats_torch).cuda()

    framelist = []
    # for frame in range(npquat.shape[0]):
    #     frot = [Quaternion(i) for i in npquat[frame]]
    #     framelist.append([j.np() for j in fk.propagate(frot, Position([0, 0, 0]))])


    frot = [Quaternion(i) for i in npquat[oframe]]
    framelist.append([j.np() for j in fk.propagate(frot, Position([0, 0, 0]))])
    recalc_kps = np.array(framelist)

    # Now npquat == quaternions [frame, bone, coord]
    # recalc_kps == keypoints using the old forward kinematics, calculated [frame, bone, coord]
    # np_kps = keypoints from the data file [frame, bone, coord]
    # kp_torch = keypoints calculated via torch batches - [batch, frame, bone, coord]

    header = ['Bone',
              'Qx', 'Qy', 'Qz', 'Qw',
              'Datafile_x', 'Datafile_y', 'Datafile_z',
              'Old_calc_x', 'Old_calc_y', 'Old_calc_z', 
              'Torch_calc_x', 'Torch_calc_y', 'Torch_calc_z']
    
    with open(args.out_csv, 'w') as ofp:

        ofp.write(",".join(header))
        ofp.write("\n")
        

        for i, p in enumerate(body_34_parts):
            line = []            
            line.append(p) # String
            line.append(str(float(npquat[oframe, i, 0])))
            line.append(str(float(npquat[oframe, i, 1])))
            line.append(str(float(npquat[oframe, i, 2])))
            line.append(str(float(npquat[oframe, i, 3])))           

            line.append(str(float(recalc_kps[0, i, 0])))
            line.append(str(float(recalc_kps[0, i, 1])))        
            line.append(str(float(recalc_kps[0, i, 2])))

            line.append(str(float(np_kps[oframe, i, 0])))
            line.append(str(float(np_kps[oframe, i, 1])))        
            line.append(str(float(np_kps[oframe, i, 2])))

            line.append(str(float(kp_torch[0, 0, i, 0])))
            line.append(str(float(kp_torch[0, 0, i, 1])))        
            line.append(str(float(kp_torch[0, 0, i, 2])))        

            ofp.write(",".join(line))
            ofp.write("\n")



import struct
#import math
import argparse


import json
import torch

from utils import Quaternion, Vector, traverse_tree
import ang2joint

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import numpy as np

file_magic = "DGSAV"
frame_magic = "DGSV"


METADATASIZE=16
ZEDBODIESHEADERSIZE=16

BONE_34_UNPACKS = [ 2,3,5,6,12,13,18,19,22,23,26 ]
BONE_38_UNPACKS = [ 1,2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23 ]




body_38_tree = [
    ["PELVIS", "SPINE_1"],
    ["SPINE_1", "SPINE_2"],
    ["SPINE_2", "SPINE_3"],
    ["SPINE_3", "NECK"],

    ["NECK", "NOSE"],
    ["NOSE", "LEFT_EYE"],
    ["LEFT_EYE", "LEFT_EAR"],
    ["NOSE", "RIGHT_EYE"],
    ["RIGHT_EYE", "RIGHT_EAR"],

    ["SPINE_3", "LEFT_CLAVICLE"],
    ["LEFT_CLAVICLE", "LEFT_SHOULDER"],
    ["LEFT_SHOULDER", "LEFT_ELBOW"],
    ["LEFT_ELBOW", "LEFT_WRIST"],
    ["LEFT_WRIST", "LEFT_HAND_THUMB_4"],
    ["LEFT_WRIST", "LEFT_HAND_INDEX_1"],
    ["LEFT_WRIST", "LEFT_HAND_MIDDLE_4"],
    ["LEFT_WRIST", "LEFT_HAND_PINKY_1"],
    
    ["SPINE_3", "RIGHT_CLAVICLE"],
    ["RIGHT_CLAVICLE", "RIGHT_SHOULDER"],
    ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
    ["RIGHT_ELBOW", "RIGHT_WRIST"],
    ["RIGHT_WRIST", "RIGHT_HAND_THUMB_4"],
    ["RIGHT_WRIST", "RIGHT_HAND_INDEX_1"],
    ["RIGHT_WRIST", "RIGHT_HAND_MIDDLE_4"],
    ["RIGHT_WRIST", "RIGHT_HAND_PINKY_1"],
    
    ["PELVIS", "LEFT_HIP"],
    ["LEFT_HIP", "LEFT_KNEE"],
    ["LEFT_KNEE", "LEFT_ANKLE"],
    ["LEFT_ANKLE", "LEFT_HEEL"],
    ["LEFT_ANKLE", "LEFT_BIG_TOE"],
    ["LEFT_ANKLE", "LEFT_SMALL_TOE"],
    
    ["PELVIS", "RIGHT_HIP"],
    ["RIGHT_HIP", "RIGHT_KNEE"],
    ["RIGHT_KNEE", "RIGHT_ANKLE"],
    ["RIGHT_ANKLE", "RIGHT_HEEL"],
    ["RIGHT_ANKLE", "RIGHT_BIG_TOE"],
    ["RIGHT_ANKLE", "RIGHT_SMALL_TOE"]
]

body_38_newtree = {
    "PELVIS": ["SPINE_1", "LEFT_HIP", "RIGHT_HIP"],
    
    "SPINE_1": ["SPINE_2"],
    "SPINE_2": ["SPINE_3"],
    "SPINE_3": ["NECK", "LEFT_CLAVICLE", "RIGHT_CLAVICLE"],

    "NECK": ["NOSE"],
    "NOSE": ["LEFT_EYE", "RIGHT_EYE"],
    "LEFT_EYE": ["LEFT_EAR"],
    "RIGHT_EYE": ["RIGHT_EAR"],
    
    "LEFT_CLAVICLE": ["LEFT_SHOULDER"],
    "LEFT_SHOULDER": ["LEFT_ELBOW"],
    "LEFT_ELBOW": ["LEFT_WRIST"],
    "LEFT_WRIST": ["LEFT_HAND_THUMB_4",
                   "LEFT_HAND_INDEX_1",
                   "LEFT_HAND_MIDDLE_4",
                   "LEFT_HAND_PINKY_1"],

    "RIGHT_CLAVICLE": ["RIGHT_SHOULDER"],
    "RIGHT_SHOULDER": ["RIGHT_ELBOW"],
    "RIGHT_ELBOW": ["RIGHT_WRIST"],
    "RIGHT_WRIST": ["RIGHT_HAND_THUMB_4",
                   "RIGHT_HAND_INDEX_1",
                   "RIGHT_HAND_MIDDLE_4",
                   "RIGHT_HAND_PINKY_1"],
    
    "LEFT_HIP" : ["LEFT_KNEE"],
    "LEFT_KNEE" : ["LEFT_ANKLE"],
    "LEFT_ANKLE" : ["LEFT_HEEL", "LEFT_BIG_TOE", "LEFT_SMALL_TOE"],
    
    "RIGHT_HIP" : ["RIGHT_KNEE"],
    "RIGHT_KNEE" : ["RIGHT_ANKLE"],
    "RIGHT_ANKLE" : ["RIGHT_HEEL", "RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"],
}

body_34_tree = [
    ["PELVIS", "NAVAL_SPINE"],
    ["NAVAL_SPINE", "CHEST_SPINE"],
    ["CHEST_SPINE", "LEFT_CLAVICLE"],
    ["LEFT_CLAVICLE", "LEFT_SHOULDER"],
    ["LEFT_SHOULDER", "LEFT_ELBOW"],
    ["LEFT_ELBOW", "LEFT_WRIST"],
    ["LEFT_WRIST", "LEFT_HAND"],
    ["LEFT_HAND", "LEFT_HANDTIP"],
    ["LEFT_WRIST", "LEFT_THUMB"],
    ["CHEST_SPINE", "RIGHT_CLAVICLE"],
    ["RIGHT_CLAVICLE", "RIGHT_SHOULDER"],
    ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
    ["RIGHT_ELBOW", "RIGHT_WRIST"],
    ["RIGHT_WRIST", "RIGHT_HAND"],
    ["RIGHT_HAND", "RIGHT_HANDTIP"],
    ["RIGHT_WRIST", "RIGHT_THUMB"],
    ["PELVIS", "LEFT_HIP"],
    ["LEFT_HIP", "LEFT_KNEE"],
    ["LEFT_KNEE", "LEFT_ANKLE"],
    ["LEFT_ANKLE", "LEFT_FOOT"],
    ["PELVIS", "RIGHT_HIP"],
    ["RIGHT_HIP", "RIGHT_KNEE"],
    ["RIGHT_KNEE", "RIGHT_ANKLE"],
    ["RIGHT_ANKLE", "RIGHT_FOOT"],
    ["CHEST_SPINE", "NECK"],
    ["NECK", "HEAD"],
    ["HEAD", "NOSE"],
    ["NOSE", "LEFT_EYE"],
    ["LEFT_EYE", "LEFT_EAR"],
    ["NOSE", "RIGHT_EYE"],
    ["RIGHT_EYE", "RIGHT_EAR"],
    ["LEFT_ANKLE", "LEFT_HEEL"],
    ["RIGHT_ANKLE", "RIGHT_HEEL"],
    ["LEFT_HEEL", "LEFT_FOOT"],
    ["RIGHT_HEEL", "RIGHT_FOOT"]]

body_34_newtree = {
    "PELVIS": ["NAVAL_SPINE", "LEFT_HIP", "RIGHT_HIP"],
    "NAVAL_SPINE" : ["CHEST_SPINE"],
    "CHEST_SPINE" : ["LEFT_CLAVICLE", "RIGHT_CLAVICLE", "NECK"],

    "LEFT_CLAVICLE" : ["LEFT_SHOULDER"],
    "LEFT_SHOULDER" : ["LEFT_ELBOW"],
    "LEFT_ELBOW" : ["LEFT_WRIST"],
    "LEFT_WRIST" : ["LEFT_HAND", "LEFT_THUMB"],
    "LEFT_HAND" : ["LEFT_HANDTIP"],
     
    "RIGHT_CLAVICLE" : ["RIGHT_SHOULDER"],
    "RIGHT_SHOULDER" : ["RIGHT_ELBOW"],
    "RIGHT_ELBOW" : ["RIGHT_WRIST"],
    "RIGHT_WRIST" : ["RIGHT_HAND", "RIGHT_THUMB"],
    "RIGHT_HAND" : ["RIGHT_HANDTIP"],
     
    "LEFT_HIP" : ["LEFT_KNEE"],
    "LEFT_KNEE" : ["LEFT_ANKLE"],
    "LEFT_ANKLE" : ["LEFT_FOOT", "LEFT_HEEL"],
    "LEFT_HEEL" : ["LEFT_FOOT"],
    
    "RIGHT_HIP" : ["RIGHT_KNEE"],
    "RIGHT_KNEE" : ["RIGHT_ANKLE"],
    "RIGHT_ANKLE" : ["RIGHT_FOOT", "RIGHT_HEEL"],
    "RIGHT_HEEL" : ["RIGHT_FOOT"],

    "NECK" : ["HEAD", "LEFT_EYE", "RIGHT_EYE"],
    "HEAD" : ["NOSE"],
    "LEFT_EYE" : ["LEFT_EAR"],
    "RIGHT_EYE" : ["RIGHT_EAR"]    
    }

body_34_parents = {"PELVIS" : None,
                   "LAST" : None,
                   "NAVAL_SPINE" : "PELVIS",
                   "CHEST_SPINE" : "NAVAL_SPINE",
                   "LEFT_CLAVICLE" : "CHEST_SPINE",
                   "LEFT_SHOULDER" : "LEFT_CLAVICLE",
                   "LEFT_ELBOW" : "LEFT_SHOULDER",
                   "LEFT_WRIST" : "LEFT_ELBOW",
                   "LEFT_HAND" : "LEFT_WRIST",
                   "LEFT_HANDTIP" : "LEFT_HAND",
                   "LEFT_THUMB" : "LEFT_WRIST",
                   "RIGHT_CLAVICLE" : "CHEST_SPINE",
                   "RIGHT_SHOULDER" : "RIGHT_CLAVICLE",
                   "RIGHT_ELBOW" : "RIGHT_SHOULDER",
                   "RIGHT_WRIST" : "RIGHT_ELBOW",
                   "RIGHT_HAND" : "RIGHT_WRIST",
                   "RIGHT_HANDTIP" : "RIGHT_HAND",
                   "RIGHT_THUMB" : "RIGHT_WRIST",
                   "LEFT_HIP" : "PELVIS",
                   "LEFT_KNEE" : "LEFT_HIP",
                   "LEFT_ANKLE" : "LEFT_KNEE",
                   "LEFT_FOOT" : "LEFT_ANKLE",
                   "RIGHT_HIP" : "PELVIS",
                   "RIGHT_KNEE" : "RIGHT_HIP",
                   "RIGHT_ANKLE" : "RIGHT_KNEE",
                   "RIGHT_FOOT" : "RIGHT_ANKLE",
                   "NECK" : "CHEST_SPINE",
                   "HEAD" : "NECK",
                   "NOSE" : "HEAD",
                   "LEFT_EYE" : "NOSE",
                   "LEFT_EAR" : "LEFT_EYE",
                   "RIGHT_EYE" : "NOSE",
                   "RIGHT_EAR" : "RIGHT_EYE",
                   "LEFT_HEEL" : "LEFT_ANKLE",
                   "RIGHT_HEEL" : "RIGHT_ANKLE",
                   "LEFT_FOOT" : "LEFT_HEEL",
                   "RIGHT_FOOT" : "RIGHT_HEEL"
                }


body_38_parents = {"PELVIS" : None,
                   "LAST" : None,                   
                   "SPINE_1" : "PELVIS",
                   "SPINE_2" : "SPINE_1",
                   "SPINE_3" : "SPINE_2",
                   "NECK" : "SPINE_3",
                   
                   "NOSE" : "NECK",
                   "LEFT_EYE" : "NOSE",
                   "LEFT_EAR" : "LEFT_EYE",
                   "RIGHT_EYE" : "NOSE",
                   "RIGHT_EAR" : "RIGHT_EYE",
                   
                   "LEFT_CLAVICLE" : "SPINE_3",
                   "LEFT_SHOULDER" : "LEFT_CLAVICLE",
                   "LEFT_ELBOW" : "LEFT_SHOULDER",
                   "LEFT_WRIST" : "LEFT_ELBOW",
                   "LEFT_HAND_THUMB_4" : "LEFT_WRIST",
                   "LEFT_HAND_INDEX_1" : "LEFT_WRIST",
                   "LEFT_HAND_MIDDLE_4" : "LEFT_WRIST",
                   "LEFT_HAND_PINKY_1" : "LEFT_WRIST",
                   
                   "RIGHT_CLAVICLE" : "SPINE_3",
                   "RIGHT_SHOULDER" : "RIGHT_CLAVICLE",
                   "RIGHT_ELBOW" : "RIGHT_SHOULDER",
                   "RIGHT_WRIST" : "RIGHT_ELBOW",
                   "RIGHT_HAND_THUMB_4" : "RIGHT_WRIST",
                   "RIGHT_HAND_INDEX_1" : "RIGHT_WRIST",
                   "RIGHT_HAND_MIDDLE_4" : "RIGHT_WRIST",
                   "RIGHT_HAND_PINKY_1" : "RIGHT_WRIST",
                   
                   "LEFT_HIP" : "PELVIS",
                   "LEFT_KNEE" : "LEFT_HIP",
                   "LEFT_ANKLE" : "LEFT_KNEE",
                   "LEFT_HEEL" : "LEFT_ANKLE",
                   "LEFT_BIG_TOE" : "LEFT_ANKLE",
                   "LEFT_SMALL_TOE" : "LEFT_ANKLE",
                   
                   "RIGHT_HIP" : "PELVIS",
                   "RIGHT_KNEE" : "RIGHT_HIP",
                   "RIGHT_ANKLE" : "RIGHT_KNEE",
                   "RIGHT_HEEL" : "RIGHT_ANKLE",
                   "RIGHT_BIG_TOE" : "RIGHT_ANKLE",
                   "RIGHT_SMALL_TOE" : "RIGHT_ANKLE"
                   }




body_34_names = ["PELVIS",
                 "NAVAL_SPINE",
                 "CHEST_SPINE",
                 "NECK",
                 "LEFT_CLAVICLE",
                 "LEFT_SHOULDER",
                 "LEFT_ELBOW",
                 "LEFT_WRIST",
                 "LEFT_HAND",
                 "LEFT_HANDTIP",
                 "LEFT_THUMB",
                 "RIGHT_CLAVICLE",
                 "RIGHT_SHOULDER",
                 "RIGHT_ELBOW",
                 "RIGHT_WRIST",
                 "RIGHT_HAND",
                 "RIGHT_HANDTIP",
                 "RIGHT_THUMB",
                 "LEFT_HIP",
                 "LEFT_KNEE",
                 "LEFT_ANKLE",
                 "LEFT_FOOT",
                 "RIGHT_HIP",
                 "RIGHT_KNEE",
                 "RIGHT_ANKLE",
                 "RIGHT_FOOT",
                 "HEAD",
                 "NOSE",
                 "LEFT_EYE",
                 "LEFT_EAR",
                 "RIGHT_EYE",
                 "RIGHT_EAR",
                 "LEFT_HEEL",
                 "RIGHT_HEEL"]


#     enum class BODY_38_PARTS {

body_38_names = ["PELVIS",
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

class Metadata:
    def __init__(self, timeStamp, pId, uId, sigType, sigIdx):
        self.time = timeStamp
        self.pId = pId
        self.sigIdx = sigIdx
        self.sigType = sigType        
        self.uId = uId

    def __str__(self):
        return("Time: %d, pID: %d, uId: %d, sigIdx: %d, sigType: %d"%(self.time, self.pId, self.uId, self.sigIdx, self.sigType))

class ZedBodiesHeader:
    def __init__(self, num, timeStamp):
        self.numskels = num
        self.time = timeStamp

    def __str__(self):
        return("Time: %d, skels: %d"%(self.time, self.numskels))

        

class ZedBodyDataKPRot:
    def __init__(self, id, bonecounts, rootpos, rootori, kpdata, rotdata):        
        self.id = id
        self.bonecounts = bonecounts
        self.rootpos = rootpos
        self.rootori = rootori
        self.keypoints = kpdata
        self.rots = rotdata

    def read(fp, bonecounts):
        iseek = fp.tell()
        fullbonecount, compactbonecount = bonecounts
        id = struct.unpack("i", fp.read(4))[0]
        rootpos = Vector.read(fp)
        rootori = Quaternion.read_quant(fp)
        if (args.verbose):
            print("%6x: Read Body %d with rootpos: %s and ori: %s"%(iseek, id, rootpos, rootori))
        
        keypoints = [Vector.read(fp) for b in range(fullbonecount)]
        rotations = [Quaternion.read_quant(fp) for b in range(compactbonecount)]

        ## What the fuck is this?
        #padding = fp.read(7)

        return ZedBodyDataKPRot(id, bonecounts, rootpos, rootori, keypoints, rotations)

    def boneval(self, i):
        if (self.bonecounts[0] == 38) and (self.bonecounts[1] == len(BONE_38_UNPACKS)):
            try:
                bidx = BONE_38_UNPACKS.index(i)
                return self.rots[bidx]                
            except(ValueError):
                return Quaternion.identity()
                
        elif (self.bonecounts[0] == 34) and (self.bonecounts[1] == len(BONE_34_UNPACKS)):
            try:
                bidx = BONE_34_UNPACKS.index(i)
                return self.rots[bidx]                
            except(ValueError):
                return Quaternion.identity()
        else:
            print("Error: unhandled bonecount scheme: %d,%d"%(self.bonecounts[0], self.bonecounts[1]))

    def print_bonelength(self, chi, par):
        pIdx = body_34_names.index(par)
        cIdx = body_34_names.index(chi)        
        bonelength = (self.keypoints[pIdx] - self.keypoints[cIdx])

#        print("%s: %f, %s = %s->%s"%(chi, bonelength.mag(), bonelength, self.keypoints[pIdx], self.keypoints[cIdx]))

            
    # Remove the rotations applied and get back to the keypose
    # def bone_sizes(self):
    #     if (self.bonecounts[0] == 34):
    #         nvals = [body_34_names.index(i) for i in traverse_tree(body_34_newtree, "PELVIS", lambda x, y: x)]
    #     elif (self.bonecount[0] == 38):
    #         nvals = [body_38_names.index(i) for i in traverse_tree(body_38_newtree, "PELVIS", lambda x, y: x)]
    #     else:
    #         print("Error in bone_sizes: unhandled bonecount scheme: %d,%d"%(self.bonecounts[0], self.bonecounts[1]))
    #         exit(0)
            
        #traverse_tree(body_34_newtree, "PELVIS", self.print_bonelength)


    def untarget(self):
        
        new_positions = [Vector.zero()] * len(self.keypoints)
        def recurse(bone_name, rot, parentIdx):
            curIdx = self.bodynames.index(bone_name)
            if (parentIdx < 0):
                if (static):
                    new_positions[curIdx] = Vector.zero()
                else:
                    new_positions[curIdx] = self.keypoints[curIdx]
                newrot = rot
            else:
                newrot = rot * self.boneval(parentIdx)
                new_positions[curIdx] = new_positions[parentIdx] + newrot.inv() * (self.keypoints[curIdx] - self.keypoints[parentIdx])

            if (args.verbose and curIdx == 3):
                print("Bone: %d, parent = %d"%(curIdx, parentIdx))
                print("Rotation: %s, antiRot = %s, newRot=%s"%(rot, self.boneval(parentIdx), newrot))
                print("Root ori is %s"%self.rootori)
                print("Parent bone = %s, mult=%s"%(new_positions[parentIdx], newrot.inv()))
                print("Diff = %s, bone = %s"%(self.keypoints[curIdx] - self.keypoints[parentIdx], new_positions[curIdx]))
                      
            if (bone_name in self.bodytree):
                for child_name in self.bodytree[bone_name]:
                    recurse(child_name, newrot, curIdx)

        Origstr = ",".join([str(i) for i in self.keypoints])


        recurse("PELVIS", self.rootori, -1)

        if (args.static):
            self.rootpos = Vector(0, 0, 0)

        
        Rotstr = ",".join([str(self.rots[i]) for i in range(11)])
        Newstr = ",".join([str(i) for i in new_positions])
        # print("Orig = %s"%(Origstr))
        # print("Rots = %s"%(Rotstr))        
        # print("New = %s"%(Newstr))
        return new_positions   # 


            
        


        
    # Unpack the skeleton back to what should be the t-pose
    # def untarget(self):
    #     new_positions = [Vector.zero()] * len(self.keypoints)
    #     new_positions[0] = self.keypoints[0]

    #     # Rot and pivot just describe the last global rotation
    #     def untarget_inner(tree, root, xform):
    #         pIdx = body_34_names.index(root)
    #         try:
    #             children = tree[root]

    #             for child in children:

    #                 cIdx = body_34_names.index(child)

    #                 bone = rot * (self.keypoints[cIdx] - pivot) - new_positions[pIdx]

    #                 newrot = self.boneval(cIdx).inv()
    #                 newbone =  newrot * bone
    #                 print("Unrotating %s by %s to %s"%(bone, self.boneval(cIdx), newbone))
    #                 new_positions[cIdx] = newbone + new_positions[pIdx]
    #                 untarget_inner(tree, child, newrot * rot, new_positions[pIdx])
    #         except(KeyError):
    #             pass

    #     tf = Transform(Quaternion.identity(), Vector.zero(), 1.0)
    #     untarget_inner(body_34_newtree, "PELVIS", tf)

    #     return new_positions

    def retarget(self, rotations):
        pass
            
def readBodyDataPlaceholder(fp, bonecounts):
    fp.read(framesize)
    
                                 
bodyDataFn = {"Body_34_KeypointsPlus" : lambda fp, num : [ZedBodyDataKPRot.read(fp, [34, 11]) for i in range(num)],
              "Body_38_KeypointsPlus" : lambda fp, num, : [ZedBodyDataKPRot.read(fp, [38, 18]) for i in range(num)],
              "Body_34_Compact" :readBodyDataPlaceholder,              
              "Body_34_Full" : readBodyDataPlaceholder,              
              "Body_38_Compact" : readBodyDataPlaceholder,
              "Body_38_Full" : readBodyDataPlaceholder}


def readHeader(fp):
    magic = fp.read(5).decode('utf-8')
    if (magic != file_magic):
        print("Magic number wrong: %s"%magic)
        exit(0)

    jsonSize = struct.unpack('i', fp.read(4))[0]

    jsonheader = fp.read(jsonSize).decode('utf-8')
    return jsonheader


def readZedBodiesHeader(fp):
    numSkels = struct.unpack('b', fp.read(1))[0]
    padding = fp.read(7)    
    timeStamp = struct.unpack('q', fp.read(8))[0]
    return ZedBodiesHeader(numSkels, timeStamp)

def readMetadata(infile):
    timestamp = struct.unpack('q', fp.read(8))[0]
    packetIdx = struct.unpack('i', fp.read(4))[0]
    userIdx = struct.unpack('h', fp.read(2))[0]
    sigType = struct.unpack('b', fp.read(1))[0]
    sigIdx = struct.unpack('b', fp.read(1))[0]
    metadata = Metadata(timestamp, packetIdx, userIdx, sigType, sigIdx)
    return metadata

def readZedBodies(fp, bodyType, framesize):
    zbHeader = readZedBodiesHeader(fp)
    if (args.verbose):
        print("%06x: Reading %d bytes for frames: %s"%(fp.tell(), framesize, zbHeader))
    return bodyDataFn[bodyType](fp, zbHeader.numskels)

    
def readFrames(infile, bodyType):
    framedata = []
    metadata = []

    while(True):
        ival = fp.tell()        
        fmagicv = fp.read(4)
        try:
            fmagic = fmagicv.decode('utf-8')
            if (fmagic != frame_magic):
                if (args.verbose):
                    print("Can't find frame magic at %x, assuming EOF"%(fp.tell() - 4))
                return [metadata, framedata]
        except(UnicodeDecodeError):
            print("Malformed frame data at %x, quitting"%(fp.tell() - 4))
            exit(0)
        framesize = struct.unpack('i', fp.read(4))[0] # Framesize includes header data
        if (args.verbose):
            print("%x: Read frame with size %d"%(ival, framesize))
        mdata = readMetadata(infile) # Metadata is 16 bytes long
        frames = readZedBodies(infile, bodyType, framesize - METADATASIZE)
        metadata.append(mdata)
        framedata.append(frames)
    return [metadata, framedata]


class AnimatedScatterKP:
    def __init__(self, keypoints):
        if (len(persondata) == 0):
            print("No frames to show, sorry")
            exit(0)

        self.numpoints = len(keypoints[0])

        self.data = keypoints

        t = np.array([np.ones(self.numpoints) * i for i in range(len(self.data))]).flatten()
        x = np.array([frame[i].x for frame in self.data for i in range(self.numpoints)])
        y = np.array([frame[i].y for frame in self.data for i in range(self.numpoints)])
        z = np.array([frame[i].z for frame in self.data for i in range(self.numpoints)])

        self.df = pd.DataFrame({'time' : t,
                                'x' : x,
                                'y' : y,
                                'z' : z})
        
        self.fig, self.ax = plt.subplots(2, subplot_kw = dict(projection='3d'))
        # self.ax = self.fig.add_subplot(projection = '3d')
        # print(self.ax)
        self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)
        #self.graph2 = self.ax[1].scatter(self.df2.x, self.df2.y, self.df2.z)        
        #self.title = self.ax.set_title("frame=%d"%(self.numpoints))
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(persondata),
                                           interval = 1)
        #plt.savefig("ExtractedTPose.gif")
        plt.show()
        
    def update_plot(self, num):
        data = self.df[self.df['time'] == num]
        self.graph._offsets3d = (data.x, data.y, data.z)
        #self.title.set_text("frame=%d"%num)

        #data2 = self.df2[self.df2['time'] == num]
        #self.graph._offsets3d = (data2.x, data2.y, data2.z)
        #self.title.set_text("frame=%d"%num)

class AnimatedScatter:
    def __init__(self, persondata):
        if (len(persondata) == 0):
            print("No frames to show, sorry")
            exit(0)

        self.numpoints = persondata[0].bonecounts[0]

        self.data = persondata

        t = np.array([np.ones(self.numpoints) * i for i in range(len(self.data))]).flatten()
        x = np.array([frame.keypoints[i].x for frame in self.data for i in range(self.numpoints)])
        y = np.array([frame.keypoints[i].y for frame in self.data for i in range(self.numpoints)])
        z = np.array([frame.keypoints[i].z for frame in self.data for i in range(self.numpoints)])

        self.df = pd.DataFrame({'time' : t,
                                'x' : x,
                                'y' : y,
                                'z' : z})
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection = '3d')
        print(self.ax)
        self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)
        self.title = self.ax.set_title("frame=%d"%(self.numpoints))
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(persondata),
                                           interval = 33)

        #plt.savefig("my_plot.png", dpi=300)
        plt.show()
        

        
    def update_plot(self, num):
        data = self.df[self.df['time'] == num]
        self.graph._offsets3d = (data.x, data.y, data.z)
        self.title.set_text("frame=%d"%num)


class AnimatedScatterDual:

    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])
        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()
        x = np.array([frame[i].x for frame in keypoints for i in range(numpoints)])
        y = np.array([frame[i].y for frame in keypoints for i in range(numpoints)])
        z = np.array([frame[i].z for frame in keypoints for i in range(numpoints)])

        df = pd.DataFrame({'time' : t,
                           'x' : x,
                           'y' : y,
                           'z' : z})
        return df

    def drawlines(self, dfx, dfy, dfz):
        linex = []
        liney = []
        linez = []

        for b in self.parents:
            if (self.parents[b] >= 0):
                linex.append([dfx[b], dfx[self.parents[b]]])
                liney.append([dfy[b], dfy[self.parents[b]]])
                linez.append([dfz[b], dfz[self.parents[b]]])                    
                    
        for i in range(len(self.parents)-1):
            self.lines.append(self.ax.plot(linex[i], liney[i], linez[i]))

    
    def __init__(self, data, skellines = False, amass_parents= {}):
        if (len(data) == 0):
            print("No frames to show, sorry")
            exit(0)
        self.parents = amass_parents
        #self.numpoints1 = len(data1[0])
        #self.numpoints2 = len(data2[0])

        self.data= data
        #self.data2 = data2

        self.df = self.build_frame(data)
        #self.df2 = self.build_frame(data2)
        #self.fig, self.ax = plt.subplots(1, 2, subplot_kw = dict(projection='3d')) ##check documentation
        # self.ax = self.fig.add_subplot(projection = '3d')
        # print(self.ax)
        #self.graph1 = self.ax[0].scatter(self.df1.x, self.df1.y, self.df1.z)
        #self.graph2 = self.ax[1].scatter(self.df2.x, self.df2.y, self.df2.z)        
        #self.title = self.ax.set_title("frame=%d"%(self.numpoints))
        self.skellines = skellines
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        #self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)

        self.lines = []
        if (self.skellines):
            self.drawlines(self.df.x, self.df.y, self.df.z)
            # linex = []
            # liney = []
            # linez = []

            # for b in ZED_34_PARENTS:
            #     if (ZED_34_PARENTS[b] >= 0):
            #         linex.append([self.df1.x[b], self.df1.x[ZED_34_PARENTS[b]]])
            #         liney.append([self.df1.y[b], self.df1.y[ZED_34_PARENTS[b]]])
            #         linez.append([self.df1.z[b], self.df1.z[ZED_34_PARENTS[b]]])                    
                    
            # for i in range(33):
            #     self.lines.append(self.ax[0].plot(linex[i], liney[i], linez[i]))
        else:
        #     self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)            
            self.graph = self.ax.scatter(self.df.x, self.df.y, self.df.z)


        self.ax.set_xlim(-args.scale, args.scale)
        self.ax.set_ylim(-args.scale, args.scale)
        self.ax.set_zlim(-args.scale, args.scale)

        #self.ax[1].set_xlim(-1000, 1000)
        # self.ax[1].set_ylim(-1000, 1000)
        # self.ax[1].set_zlim(-1000, 1000)
       

        self.ax.view_init(elev=11, azim=88)
        #self.ax[1].view_init(elev=93, azim=-90)

        #self.fig.tight_layout()
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=len(self.data), interval = 16)

        plt.show()
        writergif = animation.PillowWriter(fps = 60)

        #writer = animation.FFMpegFileWriter(fps = 60)

        self.ani.save("hip_poses.gif", writer = writergif)
        #self.ani.save("salsa.mp4", writer = writer)
        
    def update_plot(self, num):
        data = self.df[self.df['time'] == num]        

        if (self.skellines):
            linex = []
            liney = []
            linez = []
            for b in self.parents:
                if (self.parents[b] >= 0):
                    qb = len(self.parents) * num + b
                    qp = len(self.parents) * num + self.parents[b]
                    linex.append([data.x[qb], data.x[qp]])
                    liney.append([data.y[qb], data.y[qp]])
                    linez.append([data.z[qb], data.z[qp]])                    
                    
            for i in range(len(self.parents)-1):
                self.lines[i][0].set_data_3d(linex[i], liney[i], linez[i])
                # self.lines[i][0].set_ydata(liney[i])
                # self.lines[i][0].set_zdata(linez[i])
        else:            
            self.graph._offsets3d = (data.x, data.y, data.z)


        #data2 = self.df2[self.df2['time'] == num]
        #self.graph2._offsets3d = (data2.x, data2.y, data2.z)
        #self.title.set_text("frame=%d"%num)
    
    
def keypoints(filename, reshape= False):
    kth = np.load(filename, allow_pickle=True)
    #amass = np.load(args.inputfile2, allow_pickle=True)
    if args.tpose:
         poses = np.zeros([1,156])
        
    else:
        poses = kth['poses']
        
    if args.convert:
        print("p3d0 is ", p3d0.shape)    
        p3d0_tmp = p3d0.repeat([poses.shape[0], 1, 1])
        print(p3d0_tmp.shape)
        # p_tmp = np.expand_dims(poses.reshape([poses.shape[0], -1, 3]), 0)
        p_tmp = poses.reshape([poses.shape[0], -1, 3])

        print(p_tmp.shape)
        print(len(parent))
        p_tmp2 = torch.from_numpy(p_tmp).float().cuda()

        print(p_tmp2.shape)
        #exit(0)
        p3d_t= ang2joint.ang2joint(p3d0_tmp, p_tmp2, parent).squeeze(0).reshape([poses.shape[0], -1])
        p3d = p3d_t.cpu().data.numpy()

        print(p3d.shape, p3d0_tmp.shape, p_tmp.shape, poses.shape, len(parent))
    else :
        p3d = poses
    print(p3d.shape)
    if reshape:
        p3d = p3d[:,12:66]    
    #exit(0)

    
    keypoints = []
    print("Total frames is ",p3d.shape[0])
    for frame in range(p3d.shape[0]):
        keypoints.append([])
        for b in range(0,p3d.shape[1], 3):
            keypoints[-1].append(Vector(p3d[frame, b],p3d[frame, b+1],p3d[frame, b+2]))
        keypoints[-1].append(Vector(0,0,0))
    #print(",".join([str(k) for k in keypoints[4]]))
    print(len(keypoints))
   
    return keypoints

skel = np.load('../body_models/smpl_skeleton.npz')
p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()
parents = skel['parents']
parent = {}
for i in range(len(parents)):
    parent[i] = parents[i]
print(parent)   

new_parent= {}
for b in parent:
    if b <= 3 or b >=22 :
        pass
    elif (parent[b] >=4 and parent[b]<22):
        new_parent[b-4] = parent[b]-4
    else :
        
        pass ##handle case when parent bone is not in 4:22 range 
new_parent[0] = 18
new_parent[1] = 18
new_parent[2] = 18
new_parent[18] = -1

print(new_parent)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action = 'store_true', help = "Wordier console spam")
parser.add_argument("--convert", action = 'store_true', help = "Convert from ang2joint to keypoints")
parser.add_argument("--tpose", action = 'store_true', help = "Just a tpose")
parser.add_argument("--scale", type = int, help = "Scaling factor", default = 1.0)
parser.add_argument("inputfile",type = str)
parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
parser.add_argument("--reshape", action = 'store_true') ## nboiler to 4:22 

args = parser.parse_args()

if args.reshape:
    amass_parents = new_parent
else:
    amass_parents = parent


kp = keypoints(args.inputfile, args.reshape)
#kp2 = keypoints(args.inputfile2)



ascat = AnimatedScatterDual(kp, skellines = args.lineplot, amass_parents= amass_parents)
    

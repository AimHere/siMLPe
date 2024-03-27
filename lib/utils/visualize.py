import numpy as np
import torch

from utils.misc import expmap2rotmat_torch,  rotmat2xyz_torch
import argparse
import csv

from utils.utils import Quaternion, Vector, traverse_tree

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
class Animation:

    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])

        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()

        x = keypoints[:, :, 0].reshape([-1])
        y = keypoints[:, :, 1].reshape([-1])
        z = keypoints[:, :, 2].reshape([-1])

        print(t.shape, x.shape, y.shape, z.shape)
        
        df = pd.DataFrame({'time' : t,
                           'x' : x,
                           'y' : y,
                           'z' : z})
        
        return df

    def __init__(self, origdata, dots = True, skellines = False):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        self.origdata = origdata
        self.df = self.build_frame(self.origdata)
        self.skellines = skellines

        self.lines = []

        self.dots = dots
        
        self.used_bones = [2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]

        data = self.df[self.df['time'] == 0]
        if (self.skellines):
            self.drawlines(data)

        if (self.dots):
            self.graph = self.ax.scatter(data.x, data.y, data.z)
        
        self.ax.set_xlim(-args.scale, args.scale)
        self.ax.set_ylim(-args.scale, args.scale)
        self.ax.set_zlim(-args.scale, args.scale)

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames = len(self.origdata), interval = 16)
        plt.show()




        
    def drawlines(self, data, update = False):
        linex = []
        liney = []
        linez = []
        
        #for f, t in enumerate(parents[:-1]):
        for f in self.used_bones:
            t = parents[f]
            if (t >= 0):
                print("%d versus %d,"%(f, t), data.x[f], data.x[t], data.y[f], data.y[t])                
                linex.append([data.x[f], data.x[t]])
                liney.append([data.y[f], data.y[t]])
                linez.append([data.z[f], data.z[t]])
                
        for i, lx in enumerate(linex):
            self.lines.append(self.ax.plot(linex[i], liney[i], linez[i]))
        

    def updatelines(self, data, num):
        linex = []
        liney = []
        linez = []

        for f in self.used_bones:
            t = parents[f]
            if (t >= 0):
                linex.append([data.x[num * 32 + f], data.x[num * 32 + t]])
                liney.append([data.y[num * 32 + f], data.y[num * 32 + t]])
                linez.append([data.z[num * 32 + f], data.z[num * 32 + t]])

        for i, lx in enumerate(linex):
            self.lines[i][0].set_data_3d(linex[i], liney[i], linez[i])



            
    def update_plot(self, num):
        data = self.df[self.df['time'] == num]        

        if (self.dots):
            self.graph._offsets3d = (data.x, data.y, data.z)
        
        if (self.skellines):
            self.updatelines(data, num)            


class Loader:
    def __init__(self, filename):
        with open(args.file, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, 3:]
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], 32, 3])
            

    def xyz(self):
        rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
        print(rm.shape)
        return rotmat2xyz_torch(rm)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type = int, help = "Scaling factor", default = 1000.0)
    parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
    parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
    parser.add_argument("--output", action = 'store_true', help = "Visualize model output too")
    parser.add_argument("--model_pth", type = str, help = "Draw a skel")
    parser.add_argument("file", type = str)
    
    args = parser.parse_args()
    
    l = Loader(args.file)

    anim = Animation(l.xyz(), dots = not args.nodots, skellines = args.lineplot)
                 

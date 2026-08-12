import torch
import torch.utils.data as data_utl

import numpy as np
import random
from utils import *
import os
#import lintel

import json

import cv2

class CL(data_utl.Dataset):

    def __init__(self, split_file, root):
        with open(split_file, 'r') as f:
            self.data = f.readlines()
        
        self.maindata = []
        self.feats = []
        self.all_feats = []
        for o in range(len(self.data)):

           totalfeat = torch.load(os.path.join(root, self.data[o][0:14]+'1.mp4.pt'), map_location=torch.device('cpu'))        
           
           totalfeat = totalfeat.unsqueeze(0)
           
           if totalfeat.size(1) == 1:
               continue
           
           
           self.all_feats.append(totalfeat)
           self.maindata.append(self.data[o])
           self.feats.append(self.data[o][0:14]+'1.mp4.pt')
           
        
        

        self.split_file = split_file
        self.root = root
        self.cls1 = 'N' 
        self.cls = 0
        
        
    def __getitem__(self, index):
        
        feat = self.feats[index] #+ '_capture' + str(j + 1) + '.avi'

        self.cls1 = self.maindata[index][15:]
        cls1 = float(self.cls1)
        
        if cls1 <3.8:
           self.cls = 0
        else:
             self.cls = 1
    
        dff = self.all_feats[index]

        return dff, self.cls, index, feat#, self.minnimum  #feat
        
    def __len__(self):
        return len(self.maindata)#.keys())



if __name__ == '__main__':
    train = ''
    val = ''
    test = ''
    root = ''
    dataset_tr = CL(train, root)
    


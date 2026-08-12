import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
#import lintel

import json

import torch

class JL(data_utl.Dataset):

    def __init__(self, split_file, root):
        with open(split_file, 'r') as f:
            self.data = f.readlines()
        
        self.maindata = []
        self.feats = []
        self.all_feats = []
        for o in range(len(self.data)):
            line = self.data[o].strip()
            if not line:
                continue
            parts = line.split()
            entry = parts[0]
            # Determine feature file names for each dataset type
            if entry.startswith('Suturing'):
                feature_names = [entry[0:13] + '_capture1.avi.pt', entry[0:13] + '_capture2.avi.pt']
            elif entry.startswith('Knot_Tying'):
                feature_names = [entry[0:15] + '_capture1.avi.pt', entry[0:15] + '_capture2.avi.pt']
            elif entry.startswith('Needle_Passing'):
                feature_names = [entry[0:19] + '_capture1.avi.pt', entry[0:19] + '_capture2.avi.pt']
            elif entry.startswith('SK_'):
                feature_names = [entry + '.mp4.pt'] if not entry.endswith('.mp4.pt') else [entry]
                #feature_names = [entry + '.avi.pt'] if not entry.endswith('.avi.pt') else [entry]
            else:
                feature_names = [entry + '.mp4.pt'] if not entry.endswith('.mp4.pt') else [entry]
                #feature_names = [entry + '.avi.pt'] if not entry.endswith('.avi.pt') else [entry]

            for feat_name in feature_names:
                feat_path = os.path.join(root, feat_name)
                if not os.path.exists(feat_path):
                    continue
                totalfeat = torch.load(feat_path)
                self.all_feats.append(totalfeat)
                self.maindata.append(line)
                self.feats.append(feat_name)
        
       
        
        self.split_file = split_file
        self.root = root
        self.cls1 = 'N' 
        self.cls = 0
        
    def __getitem__(self, index):
        
        feat = self.feats[index] 
     
        if feat[0:10] == 'Knot_Tying':
           self.cls1 = self.maindata[index][17]
        if feat[0:14] == 'Needle_Passing':
           self.cls1 = self.maindata[index][21]
        if feat[0:8] == 'Suturing':
           self.cls1 = self.maindata[index][15]
        
        cls1 = ord(self.cls1[0])
        if cls1 == 73:
           self.cls = 0
        elif cls1 == 69:
             self.cls = 1
        elif cls1== 78:
              self.cls = 2
        

        dff = self.all_feats[index]
        
        return dff, self.cls, index, feat
        
    def __len__(self):
        return len(self.maindata)#.keys())

    

if __name__ == '__main__':
    train = ''
    val = ''
    root = ''
    dataset_tr = JL(train, root, length=16, model='2d', mode='flow')
    

import os

import numpy as np
from models.metrics import AreaSource, bad_n_error
from utils import load_rgb_img
from torch.utils import data
import csv

class Kitti15Dataset(data.Dataset):
    def __init__(self,abspath,training=True, indices = [], transform=None, test_phase=False, keys=
        set(['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fgmap','resolution','index'])) -> None:
        """ Indices must be in 0-199"""
        self.keys = keys
        self.abspath = abspath
        self.indices = indices
        self.training = training # whether to use the training set split
        self.test_phase = test_phase # decides which transform to use
        self.transform = transform

        self.key_idxs = {}

        idx = 0
        for k in ['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fgmap','resolution','index']:
            if k in self.keys:
                self.key_idxs[k] = idx
                idx+=1
            



    def get_key_idxs(self):
        return self.key_idxs
        
    def __len__(self):
        return len(self.indices) or 200

    def __getitem__(self, i):
        directory = 'training'
        if not self.training:
            directory = 'testing'
        directory = os.path.join(self.abspath,directory)

        if self.indices:
            i = self.indices[i]

        index_string = "{:06d}".format(i)
        inputs = []
        if 'l0' in self.keys:
            left_t0_path = os.path.join(directory,'image_2',f'{index_string}_10.png')
            inputs.append(load_rgb_img(left_t0_path))
        if 'r0' in self.keys:
            right_t0_path = os.path.join(directory,'image_3',f'{index_string}_10.png')
            inputs.append(load_rgb_img(right_t0_path))
        if 'l1' in self.keys:
            left_t1_path = os.path.join(directory,'image_2',f'{index_string}_11.png')
            inputs.append(load_rgb_img(left_t1_path))
        if 'r1' in self.keys:
            right_t1_path = os.path.join(directory,'image_3',f'{index_string}_11.png')
            inputs.append(load_rgb_img(right_t1_path))
            


        labels = []
        if 'd0' in self.keys:
            disparity_t0_path = os.path.join(directory,'disp_occ_0',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t0_path)/256)
        if 'd0noc' in self.keys:
            disparity_t0_noc_path = os.path.join(directory,'disp_noc_0',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t0_noc_path)/256)


        if 'd1' in self.keys:
            disparity_t1_path = os.path.join(directory,'disp_occ_1',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t1_path)/256)
        if 'd1noc' in self.keys:
            disparity_t1_noc_path = os.path.join(directory,'disp_noc_1',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t1_noc_path)/256)


        outputs = [*inputs,*labels]
        if 'fgmap' in self.keys:
            fg_map = load_rgb_img(os.path.join(directory,'obj_map',f'{index_string}_10.png'))
            outputs.append(fg_map)

        if 'resolution' in self.keys:
            resolution = inputs[0].shape[1:]
            outputs.append(resolution)

        if 'index' in self.keys:
            outputs.append(index_string)

        if self.transform:
            return self.transform([*outputs],self.get_key_idxs(),self.test_phase)
        else:
            return [*outputs]


    def eval_to_csv(self, X,y,runtime, writer : csv.writer, write_headers=False):
        keys = self.get_key_idxs()

        gt_noc = y[keys['d0noc']].astype(float)
        gt_oc = y[keys['d0']].astype(float)
        fg_mask =y[keys['fgmap']]


        if write_headers:
            writer.writerow(['sample','nocc_fg_d1','nocc_all_d1','occ_fg_d1','occ_all_d1','runtime'])

        nocc_fg_d1 = bad_n_error(3,
            X,
            gt_noc,
            AreaSource.FOREGROUND,
            fg_mask=fg_mask)   

        nocc_all_d1 = bad_n_error(3,
            X,
            gt_noc) 

        occ_fg_d1 = bad_n_error(3,
            X,
            gt_oc,
            AreaSource.FOREGROUND,
            fg_mask=fg_mask)     

        occ_all_d1 = bad_n_error(3,
            X,
            gt_oc) 
        
        writer.writerow([y[keys['index']],
                nocc_fg_d1,
                nocc_all_d1,
                occ_fg_d1,
                occ_all_d1,
                runtime
                ])

import os

import numpy as np
from fetch_network import get_leastereo
from utils import load_rgb_img
from torch.utils import data

class Kitti15Dataset(data.Dataset):
    def __init__(self,abspath,training=True, indices = [], transform=None, target_transform=None, mask='1111') -> None:
        """ Indices must be in 0-199"""
        self.abspath = abspath
        self.indices = indices
        self.training = training
        self.transform = transform
        self.target_transform = target_transform
        self.mask = mask

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

        left_t0_path = os.path.join(directory,'image_2',f'{index_string}_10.png')
        left_t1_path = os.path.join(directory,'image_2',f'{index_string}_11.png')
        right_t0_path = os.path.join(directory,'image_3',f'{index_string}_10.png')
        right_t1_path = os.path.join(directory,'image_3',f'{index_string}_11.png')

        inputs = [ load_rgb_img(x) for i,x in enumerate([left_t0_path,right_t0_path,left_t1_path,right_t1_path]) if self.mask[i] == '1']

        resolution = inputs[0].size

        labels = []
        fg_map = None
        if self.training:
            if self.mask[0] == '1' or self.mask[1] == '1':
                disparity_t0_path = os.path.join(directory,'disp_occ_0',f'{index_string}_10.png')
                disparity_t0_noc_path = os.path.join(directory,'disp_noc_0',f'{index_string}_10.png')
                labels.append(load_rgb_img(disparity_t0_path))
                labels.append(load_rgb_img(disparity_t0_noc_path))

            if self.mask[2] == '1' or self.mask[3] == '1':
                disparity_t1_path = os.path.join(directory,'disp_occ_1',f'{index_string}_10.png')
                disparity_t1_noc_path = os.path.join(directory,'disp_noc_1',f'{index_string}_10.png')
                labels.append(load_rgb_img(disparity_t1_path))
                labels.append(load_rgb_img(disparity_t1_noc_path))
            
            fg_map = load_rgb_img(os.path.join(directory,'obj_map',f'{index_string}_10.png'))
        
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            labels = self.target_transform(labels)

        return {
            'inputs': inputs,
            'labels': labels,
            'fg_mask' : fg_map,
            'index' : index_string,
            'resolution': np.array([resolution[1],resolution[0]])
        }

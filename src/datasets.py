import os

import numpy as np
from models.metrics import AreaSource, bad_n_error
from utils import load_rgb_img
from torch.utils import data
import csv

MAX_FLOW_KITTI=512
MAX_DISP_KITTI=256

class Kitti15Dataset(data.Dataset):
    def __init__(self,abspath,training=True, indices = [], transform=None, test_phase=False, keys=
        set(['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fgmap','fl','resolution','index']),
        permute_keys=set(),
        replace_keys={},
        ) -> None:
        """ Indices must be in 0-199"""
        self.keys = keys
        self.abspath = abspath
        self.indices = indices
        self.training = training # whether to use the training set split
        self.test_phase = test_phase # decides which transform to use
        self.transform = transform
        self.permute_keys = permute_keys
        self.replace_keys = replace_keys
        self.key_idxs = {}

        idx = 0
        for k in ['l0','r0','l1','r1','d0','d0noc','d1','d1noc','fl','fgmap','resolution','index']:
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
        directory = os.path.join(self.abspath,"kitti2015",directory)

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
            labels.append(load_rgb_img(disparity_t0_path,dtype=np.float32)/256)
        if 'd0noc' in self.keys:
            disparity_t0_noc_path = os.path.join(directory,'disp_noc_0',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t0_noc_path,dtype=np.float32)/256)


        if 'd1' in self.keys:
            disparity_t1_path = os.path.join(directory,'disp_occ_1',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t1_path,dtype=np.float32)/256)
        if 'd1noc' in self.keys:
            disparity_t1_noc_path = os.path.join(directory,'disp_noc_1',f'{index_string}_10.png')
            labels.append(load_rgb_img(disparity_t1_noc_path,dtype=np.float32)/256)

        if 'fl' in self.keys:
            import cv2
            flow_left_path = os.path.join(directory,'flow_occ',f'{index_string}_10.png')
            flo_file = cv2.imread(flow_left_path, -1)
            flo_img = flo_file[:,:,3:0:-1].astype(np.float32)
            invalid = (flo_file[:,:,0] == 0)
            flo_img = flo_img - 32768
            flo_img = flo_img / 64 
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] =  0# MAX_FLOW_KITTI * 2
            
            labels.append(flo_img)

        outputs = [*inputs,*labels]
        if 'fgmap' in self.keys:
            fg_map = load_rgb_img(os.path.join(directory,'obj_map',f'{index_string}_10.png'))
            outputs.append(fg_map)

        if 'resolution' in self.keys:
            resolution = [inputs[0].shape[1],inputs[0].shape[0]]
            outputs.append(resolution)

        if 'index' in self.keys:
            outputs.append(index_string)

        # shuffle requested keys
        for pk in self.permute_keys:
            np.random.shuffle(outputs[self.key_idxs[pk]].flat)

        if self.transform:
            outputs = self.transform([*outputs],self.get_key_idxs(),self.test_phase)

        if self.replace_keys:
            new_outputs = [None]*len(outputs)
            for k in self.keys:
                if k in self.replace_keys:
                    new_outputs[self.key_idxs[k]] = outputs[self.key_idxs[self.replace_keys[k]]] 
                else:
                    new_outputs[self.key_idxs[k]] = outputs[self.key_idxs[k]]
            outputs = new_outputs

      
        return outputs

    def eval_to_csv(self,X,y,gt_label_to_idx_map,runtime, writer : csv.writer, write_headers=False):
        keys = self.get_key_idxs()
        headers = ['sample','runtime']
        datas = [y[keys['index']],f"{runtime:7.3f}"]
        disp_frames = []
        if "d0" in gt_label_to_idx_map:
            disp_frames += ["d0"]
        if "d1" in gt_label_to_idx_map:
            disp_frames += ["d1"]

        for d in disp_frames:
            gt_noc = y[keys[f"{d}noc"]]
            gt_oc = y[keys[d]]
            headers += [f'nocc_fg_{d}',f'nocc_all_{d}',f'occ_fg_{d}',f'occ_all_{d}']

            # we only have fg maps for first frame
            if d == "d0":
                fg_mask = y[keys['fgmap']] 
                nocc_fg_d1 = bad_n_error(3,
                    X[gt_label_to_idx_map[d]],
                    gt_noc,
                    AreaSource.FOREGROUND,
                    fg_mask=fg_mask)   
                occ_fg_d1 = bad_n_error(3,
                    X[gt_label_to_idx_map[d]],
                    gt_oc)    

            nocc_all_d1 = bad_n_error(3,
                X[gt_label_to_idx_map[d]],
                gt_noc,
                AreaSource.FOREGROUND,
                fg_mask=fg_mask) 

            occ_all_d1 = bad_n_error(3,
                X[gt_label_to_idx_map[d]],
                gt_oc) 

            datas += [  
                f"{nocc_fg_d1:10.3f}",
                f"{nocc_all_d1:11.3f}",
                f"{occ_fg_d1:9.3f}",
                f"{occ_all_d1:10.3f}",
            ]

        if "fl" in gt_label_to_idx_map:
            gt_oc = y[keys["fl"]].astype(float)
            headers += ["fl_all"]
            fl_oc = bad_n_error(3,
                X[gt_label_to_idx_map['fl']],
                gt_oc)
            datas += [f"{fl_oc:5.3f}\t"]

        if write_headers:
            writer.writerow(headers)

        writer.writerow(datas)

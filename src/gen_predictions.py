import json
from re import L
import argparse 
import os
from time import time
import numpy as np
import skimage
from skimage import io
import csv
import torch
from torch.autograd import Variable 
from fetch_network import get_leastereo
from datasets import Kitti15Dataset
from metrics import AreaSource, bad_n_error

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

SCRIPT_DIR =os.path.dirname(os.path.realpath(__file__))
REPRODS_PATH = os.path.join(SCRIPT_DIR,'reproductions')

def get_splits(path):
    with open(path,'r') as f:
        return json.load(f)

class LEASTereoRunner():
    def __init__(self,args) -> None:
        self.args = args
        
        dataset = args.dataset
        self.crop_width = 1248
        self.crop_height = 384 
        self.device = 'cuda'

        if dataset == 'kitti2012':
            self.resume = os.path.join(REPRODS_PATH,'LEAStereo',"run","Kitti12","best","best_1.16.pth") 
        elif dataset == 'kitti2015':
            self.resume = os.path.join(REPRODS_PATH,'LEAStereo','run','Kitti15','best','best.pth')
        elif dataset == 'sceneflow':
            self.resume = os.path.join(REPRODS_PATH,'LEAStereo',"run","sceneflow","best","checkpoint","best.pth")
            self.crop_width = 960
            self.crop_height = 576
        

    def get_transform(self):
        def transform(x):
            left = x[0]
            right = x[1]
            size = np.shape(left)
            height = size[0]
            width = size[1]
            temp_data = np.zeros([6, height, width], 'float32')
            left = np.asarray(left)
            right = np.asarray(right)

            r = left[:, :, 0]
            g = left[:, :, 1]
            b = left[:, :, 2]
            temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
            temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
            temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
            r = right[:, :, 0]
            g = right[:, :, 1]
            b = right[:, :, 2]	
            temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
            temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
            temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])

            _, h, w=np.shape(temp_data)

            crop_height = self.crop_height
            crop_width = self.crop_width

            if h <= crop_height and w <= crop_width: 
                # padding zero 
                temp = temp_data
                temp_data = np.zeros([6, crop_height, crop_width], 'float32')
                temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp    
            else:
                start_x = int((w - crop_width) / 2)
                start_y = int((h - crop_height) / 2)
                temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
            left = np.ones([1, 3,crop_height,crop_width],'float32')
            left[0, :, :, :] = temp_data[0: 3, :, :]
            right = np.ones([1, 3, crop_height, crop_width], 'float32')
            right[0, :, :, :] = temp_data[3: 6, :, :]

            return [torch.from_numpy(left).float(), torch.from_numpy(right).float()]
            
        return transform

    def get_target_transform(self):
        return lambda x : x 

    def get_model(self):
        return get_leastereo(resume_non_converted=self.resume)

    def get_mask(self):
        return "1100"


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

METHODS = {
    'LEAStereo': lambda args: LEASTereoRunner(args)
}
DATASETS = {
    'kitti2015': lambda *args: Kitti15Dataset(os.path.join(SCRIPT_DIR,'..','datasets','kitti2015'),*args)
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run given network on the given split and store outpus + running times')

    parser.add_argument('file', default='splits.json')
    parser.add_argument('--outdir','-o', default=os.path.join('..','predictions'))
    parser.add_argument('--dataset','-d', default='kitti2015')
    parser.add_argument('--datasetsplit','-ds', default='training')
    parser.add_argument('--splitname', '-s', default='validation')
    parser.add_argument('--method', '-m', default='LEAStereo')

    args = parser.parse_args()
    splits = get_splits(args.file)
    
    method = METHODS[args.method](args)

    indices = splits[args.dataset][args.method][args.datasetsplit][args.splitname]
    dataset = DATASETS[args.dataset](
        args.datasetsplit == "training", 
        indices,
        method.get_transform(),
        method.get_target_transform(),
        method.get_mask())

    model = method.get_model()
    model.eval()
    torch.backends.cudnn.benchmark = True

    out_path = os.path.join(args.outdir,args.method,args.dataset,args.datasetsplit,args.splitname)
    eval_file = os.path.join(out_path,'results.csv')

    if os.path.exists(eval_file):
        os.remove(eval_file)

    for idx,sample in enumerate(dataset):

        i = sample['inputs']

        start_time = time()
        with torch.no_grad():
            prediction = model(*([Variable(x,requires_grad=False).cuda() for x in i]))
        end_time = time()
        
        temp = prediction.cpu()
        temp = temp.detach()
        temp = temp.numpy()

        output_resolution = sample['resolution']
        height = temp.shape[-2]
        width = temp.shape[-1]

        if output_resolution[0] <= height and output_resolution[1]<= width:
            temp = temp[:, height - output_resolution[0]: height, width - output_resolution[1]: width]

        temp = temp[0, :, :]            
        target_dir = os.path.join(out_path,args.datasetsplit,args.splitname,str(sample['index'])+'.png')
        print(target_dir)
        if not os.path.exists(os.path.dirname(target_dir)):
            os.makedirs(os.path.dirname(target_dir))
        
        with open(eval_file, 'a') as f:
            csvreader = csv.writer(f)

            if args.dataset == 'kitti2015':
                if idx == 0:
                    csvreader.writerow(['sample','nocc_fg_d1','nocc_all_d1','occ_fg_d1','occ_all_d1','runtime'])
                

                gt_noc = np.asarray(sample['labels'][1]).astype(float) / 256
                gt_oc = np.asarray(sample['labels'][0]).astype(float) / 256
                fg_mask = np.asarray(sample['fg_mask'])

                nocc_fg_d1 = bad_n_error(3,
                    temp,
                    gt_noc,
                    AreaSource.FOREGROUND,
                    fg_mask=sample['fg_mask'])   

                nocc_all_d1 = bad_n_error(3,
                    temp,
                    gt_noc) 

                occ_fg_d1 = bad_n_error(3,
                    temp,
                    gt_oc,
                    AreaSource.FOREGROUND,
                    fg_mask=sample['fg_mask'])     

                occ_all_d1 = bad_n_error(3,
                    temp,
                    gt_oc) 
                
                csvreader.writerow([sample['index'],nocc_fg_d1,nocc_all_d1,occ_fg_d1,occ_all_d1,end_time-start_time])

        temp = (temp * 256).astype('uint16')
        skimage.io.imsave(target_dir, temp)

    # average each column apart from first one
    average_row = []
    with open(eval_file, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)

        rows = []
        for row in csvreader:
            rows.append(row[1:])

        rows = np.array(rows).astype(float)
        average_row = list(rows.mean(axis=0).astype(str))
    
    with open(eval_file, 'a') as f:
        csvwriter = csv.writer(f)

        average_row.insert(0,'AVG')
        csvwriter.writerow(average_row)
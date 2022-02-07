import json
import os
import numpy as np
from skimage.io import imsave 
import csv
import torch
from models.runners import LEASTereoRunner, STSEarlyFusionConcatRunner , STSEarlyFusionConcat2Runner, STSEarlyFusionConcat2BigRunner, LEAStereoOrigMockRunner,STSEarlyFusionTimeMatchRunner,RAFTRunner, STSLateFusion2Runner,STSLateFusionGTFlowRunner
from datasets import Kitti15Dataset
from args import PARSER
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)
from models.raft.utils.flow_viz import flow_to_image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_splits(path):
    with open(path,'r') as f:
        return json.load(f)

METHODS = {
    'LEAStereoOrig' : lambda args: LEAStereoOrigMockRunner(args, training= False),
    'LEAStereo': lambda args: LEASTereoRunner(args,training=False),
    'STSEarlyFusionConcat' : lambda args: STSEarlyFusionConcatRunner(args, training=False),
    'STSEarlyFusionConcat2' : lambda args: STSEarlyFusionConcat2Runner(args, training=False),
    'STSEarlyFusionConcat2Big' : lambda args: STSEarlyFusionConcat2BigRunner(args, training=False),
    'STSEarlyFusionTimeMatch' : lambda args: STSEarlyFusionTimeMatchRunner(args, training=False),
    'STSLateFusionGTFlow' : lambda args: STSLateFusionGTFlowRunner(args,training=False),
    'STSLateFusion2' : lambda args : STSLateFusion2Runner(args,training=False),
    'RAFT' : lambda args: RAFTRunner(args, training=False)
}

DATASETS = {
    'kitti2015': lambda *args: Kitti15Dataset(*args)
}

if __name__ == "__main__":


    args = PARSER.parse_args()
    splits = get_splits(args.file)
    method = METHODS[args.method](args)
    if args.resume_method:
        resume_cls = METHODS[args.resume_method](args).model_cls
    else:
        resume_cls = None 

    indices = splits[args.dataset][args.method][args.datasetsplit][args.splitname]
    dataset = DATASETS[args.dataset](
        args.datasets_dir,
        args.datasetsplit == "training", 
        indices,
        method.transform,
        True, # test phase
        method.get_keys(),
        args.permute_keys,
        args.replace_keys)
    model = method.get_model(args.resume,resume_cls)
    model.eval()

    torch.backends.cudnn.benchmark = True
    
    save_dir = args.method if not args.save_name else args.save_name
    out_path = os.path.join(args.outdir,save_dir,args.dataset,args.datasetsplit,args.splitname)
    eval_file = os.path.join(out_path,'results.csv')

    if os.path.exists(eval_file):
        os.remove(eval_file)

    key_idxs = dataset.get_key_idxs()
    gt_label_to_idx_map = method.gt_label_to_idx_map()
    inv_gt_label_to_idx_map = {y:x for x,y in gt_label_to_idx_map.items()}
    # run predictions
    for idx,sample in enumerate(dataset):

        output = method.get_output(model,sample, key_idxs)

        target_dir = os.path.join(out_path,str(sample[key_idxs['index']])+'.png')
        print(target_dir)
        if not os.path.exists(os.path.dirname(target_dir)):
            os.makedirs(os.path.dirname(target_dir))
        
        with open(eval_file, 'a') as f:
            csvwriter = csv.writer(f)
            write_headers = False if idx != 0 else True
            dataset.eval_to_csv(output["outputs"],sample,
                gt_label_to_idx_map,
                output["runtime"],csvwriter, write_headers=write_headers)

        # save images for reference
        for i,o in enumerate(output["outputs"]):
            
            corr_gt = inv_gt_label_to_idx_map.get(i,None)

            if(not corr_gt):
                continue 

            if(corr_gt in ['d0','d1']):
                o = (o * 256).astype('uint16')
            else:
                permuted = np.moveaxis(o,0,-1)
                o = flow_to_image(permuted).astype('uint8')


            imsave(target_dir[0:-4] + f'_{corr_gt}' + ".png", o)

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
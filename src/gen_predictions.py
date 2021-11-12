import json
import os
import numpy as np
from skimage.io import imsave 
import csv
import torch
from models.runners import LEASTereoRunner, STSEarlyFusionConcatRunner 
from datasets import Kitti15Dataset
from args import PARSER
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_splits(path):
    with open(path,'r') as f:
        return json.load(f)



METHODS = {
    'LEAStereo': lambda args: LEASTereoRunner(args,training=False),
    'STSEarlyFusionConcat' : lambda args: STSEarlyFusionConcatRunner(args, training=False)
}
DATASETS = {
    'kitti2015': lambda *args: Kitti15Dataset(os.path.join(SCRIPT_DIR,'..','datasets','kitti2015'),*args)
}

if __name__ == "__main__":


    args = PARSER.parse_args()
    splits = get_splits(args.file)
    method = METHODS[args.method](args)
    resume_method = METHODS[args.resume_method](args)
    indices = splits[args.dataset][args.method][args.datasetsplit][args.splitname]
    dataset = DATASETS[args.dataset](
        args.datasetsplit == "training", 
        indices,
        method.transform,
        True, # test phase
        method.get_keys())
    model = method.get_model(args.resume,resume_method.model_cls)
    model.eval()

    torch.backends.cudnn.benchmark = True

    out_path = os.path.join(args.outdir,args.method,args.dataset,args.datasetsplit,args.splitname)
    eval_file = os.path.join(out_path,'results.csv')

    if os.path.exists(eval_file):
        os.remove(eval_file)

    key_idxs = dataset.get_key_idxs()
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
            dataset.eval_to_csv(output["outputs"],sample,output["runtime"],csvwriter, write_headers=write_headers)


        # save images for reference
        output["outputs"] = (output["outputs"] * 256).astype('uint16')
        imsave(target_dir, output["outputs"])

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
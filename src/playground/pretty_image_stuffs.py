import numpy as np
import argparse 
from enum import Enum 
from PIL import Image 
import matplotlib.pyplot as plt 
import os 
from pathlib import Path
from turbo_cmap import TURBO

class Mode(Enum):
    DISP_CMAP = 'dispcmap'

    def __str__(self):
        return self.value 

PARSER = argparse.ArgumentParser()
PARSER.add_argument('mode', type=Mode,choices=list(Mode))
PARSER.add_argument('img')
PARSER.add_argument('--img2','-i2', default=None)
PARSER.add_argument('--dataset','-d',default="kitti",choices=['kitti'])
PARSER.add_argument('--out','-o',default=None)

# PARSER_TRAIN.add_argument('file')
# PARSER_TRAIN.add_argument('datasets_dir', help="directory containing root folders of each dataset")
# PARSER_TRAIN.add_argument('save', help="path to overwrite/save new weights to")
# PARSER_TRAIN.add_argument('valsplit',help="hold out split name")
# PARSER_TRAIN.add_argument('trainsplits',nargs="+", help="split names to be used for training")
# PARSER_TRAIN.add_argument('--resume','-r', help="path to weights dir to resume from (None)")
# PARSER_TRAIN.add_argument('--dataset','-d', default='kitti2015')
# PARSER_TRAIN.add_argument('--method', '-m', default='LEAStereo')
# PARSER_TRAIN.add_argument('--resume_method', '-rs', default="LEAStereo")
# PARSER_TRAIN.add_argument('--epochs', '-e', default=600, type=int)
# PARSER_TRAIN.add_argument('--batch', '-b', default=4, type=int)
# PARSER_TRAIN.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
# PARSER_TRAIN.add_argument('--finetuning_resume', '-fr', default=True, action="store_false" ,help="if true, optimizer and scheduler are not loaded from checkpoint (True)")
# PARSER_TRAIN.add_argument('--freeze-starting-with', '-f', default=None ,help="freeze submodules who start with this string (None)")
# PARSER_TRAIN.add_argument('--crop_width', '-cw', default=336 ,help="the size of the random crop applied to the input image during training")
# PARSER_TRAIN.add_argument('--crop_height', '-ch', default=168 ,help="the size of the random crop applied to the input image during training")
# PARSER_TRAIN.add_argument('--local_rank', '-loc_r',type=int,default=-1 ,help="the uniue id of the process (0 = master), decides which GPU is used")
# PARSER_TRAIN.add_argument('--permute_keys', '-sk',nargs="+",default=[], help="permute the given keys from the dataset (inputs), for calculating feature importance")
# PARSER_TRAIN.add_argument('--replace_keys', '-rk',type=json.loads)
# PARSER_TRAIN.add_argument('--seed', '-s',type=int,default=0)

if __name__ =="__main__":
    args = PARSER.parse_args()
    if(args.out == None):
        args.out = Path(args.img).stem + "_out"

    if args.mode == Mode.DISP_CMAP:
        if args.dataset == "kitti":
            img = np.asarray(Image.open(args.img)) / 256
            img /= np.max(img)
            img[img == 0] = 0
            print(np.max(img))
            colored = TURBO(img)
            plt.imsave(args.out + ".png",colored)
        else:
            pass
    else:
        print("unknown mode")
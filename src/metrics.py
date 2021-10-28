import numpy as np 
import os
import enum
from PIL import Image 
import sys 

class AreaSource(enum.Enum):
    FOREGROUND = 1,
    BACKGROUND = 2,
    BOTH = 3,

def bad_n_error(n, disp, gt, area : AreaSource = AreaSource.BOTH , fg_mask = None):

    if area == AreaSource.FOREGROUND:
        gt[fg_mask == 0] = 0
    elif area == AreaSource.BACKGROUND: 
        gt[fg_mask > 0 ] = 0

    valid_gt = np.greater(gt,0).astype(int)


    disp_over_valid = disp * valid_gt
    
    sum_gt = np.sum(valid_gt)

    all_err = np.greater( np.abs(disp_over_valid-gt),n).astype(int)
    res = np.sum(all_err)/sum_gt
        
    return res*100
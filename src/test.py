from models.metrics import ST_loss 
from datasets import Kitti15Dataset
from PIL import Image 
import numpy as np
import torch 
if __name__ == "__main__":
    ds = Kitti15Dataset("../datasets/kitti2015",training=True)

    keys = ds.get_key_idxs()

    average_loss= 0
    for i in range(len(ds)):
        i = ds[i]
        fl = i[keys["fl"]][np.newaxis,:]
        n = ST_loss(
            torch.Tensor(np.moveaxis(i[keys["l0"]],-1,0)[np.newaxis,:]),
            torch.Tensor(np.moveaxis(i[keys["r0"]],-1,0)[np.newaxis,:]),
            torch.Tensor(i[keys["l1"]]),
            torch.Tensor(np.moveaxis(i[keys["r1"]],-1,0)[np.newaxis,:]),
            torch.Tensor(i[keys["d0"]]),
            torch.Tensor(i[keys["d1"]][np.newaxis,np.newaxis,:]),
            torch.Tensor(fl),
            None,
            None)#1 - torch.Tensor(np.moveaxis(i[keys["fgmap"]],-1,1)[np.newaxis,np.newaxis,:]))
        average_loss += n
        print(n)
    print(average_loss / len(ds))

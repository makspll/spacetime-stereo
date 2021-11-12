import numpy as np 
import os
import enum
from PIL import Image 
import sys 
import matplotlib.pyplot as plt
from numpy.lib.arraypad import pad
import torch 
import torch.nn.functional as F 

class AreaSource(enum.Enum):
    FOREGROUND = 1,
    BACKGROUND = 2,
    BOTH = 3,

def bad_n_error(n, disp, gt, area : AreaSource = AreaSource.BOTH , fg_mask = None, max_disp=None):

    gt_cp = np.array(gt,copy=True) 

    if area == AreaSource.FOREGROUND:
        gt_cp[fg_mask == 0] = 0
    elif area == AreaSource.BACKGROUND: 
        gt_cp[fg_mask > 0 ] = 0

    if max_disp:
        gt_cp[gt_cp > max_disp] = 0


    valid_gt = np.greater(gt_cp,0).astype(int)


    disp_over_valid = disp * valid_gt

    sum_gt = np.sum(valid_gt)

    all_err = np.greater( np.abs(disp_over_valid-gt_cp),n).astype(int)
    res = np.sum(all_err)/sum_gt
        
    return res*100

def ST_loss(l0,r0,l1,r1,d0,d1,ofl,ofr,disc_map=None):
    """[summary]

    Args:
        l0 ([type]): N x C x H x W
        r0 ([type]): N x C x H x W
        l1 ([type]): N x C x H x W
        r1 ([type]): N x C x H x W
        d0 ([type]): N x C x H x W
        d1 ([type]): N x C x H x W
        ofl ([type]): N x H x W x 2 (x then y)
        ofr ([type]): N x H x W x 2 (x then y)
        disc_map ([type], optional): N x H x W
    """
    n,c,h,w = l0.size()
    hh,hw = (h)/2,(w)/2
    grid = torch.Tensor(np.moveaxis(np.meshgrid(np.linspace(-1,1,num=w),np.linspace(-1,1,num=h)),0,-1))
    n_ofl = ofl / torch.Tensor([hw,hh]) + grid
    n_d1 = d1 / (hw) 

    # blow up non-valid values, causes them to land outsid boundary
    # important: invalid values for flow and disparity must be set to 0
    n_ofl[n_ofl == 0] = 1e10 
    n_d1[n_d1 == 0] = 1e10 

    d1_warped_by_fl = F.grid_sample(n_d1,n_ofl, mode="nearest",align_corners=True,padding_mode='zeros')
    n_ofl[:,:,:,0] -= d1_warped_by_fl[:,0,:,:]

    L_t = F.grid_sample(r1,n_ofl, mode="nearest",align_corners=True,padding_mode='zeros' )
    
    if(disc_map is not None):
        l0 = l0 * disc_map
        L_t *= disc_map

    mask = L_t > 0

    return F.mse_loss(l0[mask],L_t[mask])

    # print(L_t.max())
    # new = torch.Tensor(np.zeros((1,c,h,w)))
    # print(ofl.size())
    # for i in range(h):
    #     for j in range(w):
    #         im = i
    #         jm = j
    #         x_ofl_i = im + (int)(ofl[0,im,jm,1]) 
    #         x_ofl_j = jm + (int)(ofl[0,im,jm,0])
    #         d_at_flow = (int)(-d1[0,0,x_ofl_i%h,x_ofl_j%w])

    #         # print(ofl[0,im,jm,1]/h,(int)(ofl[0,im,jm,0]/w),d_at_flow)
            
    #         new[0,:,i,j] = r1[0,:,x_ofl_i%h,+ (x_ofl_j + d_at_flow) % w]
    # return new 
    return L_t

if __name__ == "__main__":
    from PIL import Image 
    # with Image.open(sys.argv[1]) as i:

        # w,h = i.size
        # d = 10
        # u = 0
        # v = 0
        # l = torch.Tensor(np.expand_dims(np.moveaxis(np.asarray(i),-1,0),0))
        # d1 = torch.Tensor((np.ones((1,1,h,w)) * d))
        # ofl = torch.Tensor(np.concatenate((np.ones((1,h,w,1))*u,np.ones((1,h,w,1))*v),axis=-1))
        # grid = torch.Tensor(np.moveaxis(np.meshgrid(np.linspace(0,w-1,num=w),np.linspace(0,w-1,num=w)),0,-1))[np.newaxis,:]
        # for i in range(h):
        #     for j in range(w):
        #         ofl[:,i,j,0] = u
        #         ofl[:,i,j,1] =  v

        # print(d1,d1.size())
        # n = smooth_l1_ST_loss(l,None,None,None,None,d1,ofl,None,None)
        # i_new = Image.fromarray(np.uint8(np.moveaxis(n.numpy()[0],0,-1)))
        # i_new.show()
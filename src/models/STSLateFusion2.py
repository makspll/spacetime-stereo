import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Disparity
from .blocks.AutoEncoder import AutoEncoder
from .LEAStereo import MatchingNetwork,FeatureNetwork
import re
from .raft.raft import RAFT
import numpy as np
from .warps import invert_flow,warp_with_flow
from scipy.ndimage import median_filter

class STSLateFusion2(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()

        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork() 
        self.disp = Disparity.DisparitySelector(self.max_disp)
        self.flow = RAFT(iterations=24)
        self.flow.requires_grad_(False)
        self.refiner = AutoEncoder(4, 2,stages=3)

    def forward(self, l0_img, r0_img, l1_img, r1_img):
        # disparity at 0
        l0 = self.feature(l0_img)       
        r0 = self.feature(r0_img) 

        with torch.cuda.device_of(l0):
            cost0 = l0.new().resize_(l0.size()[0], l0.size()[1]*2, int(self.max_disp/3),  l0.size()[2],  l0.size()[3]).zero_() 
        for i in range(int(self.max_disp/3)):
            if i > 0 : 
                cost0[:,:l0.size()[1], i,:,i:] = l0[:,:,:,i:]
                cost0[:,l0.size()[1]:, i,:,i:] = r0[:,:,:,:-i]
            else:
                cost0[:,:l0.size()[1],i,:,i:] = l0
                cost0[:,l0.size()[1]:,i,:,i:] = r0

        cost0 = self.matching(cost0)     
        disp0 = self.disp(cost0)   

        # disparity at 1
        l1 = self.feature(l1_img)       
        r1 = self.feature(r1_img) 

        with torch.cuda.device_of(l1):
            cost1 = l1.new().resize_(l1.size()[0], l1.size()[1]*2, int(self.max_disp/3),  l1.size()[2],  l1.size()[3]).zero_() 
        for i in range(int(self.max_disp/3)):
            if i > 0 : 
                cost1[:,:l1.size()[1], i,:,i:] = l1[:,:,:,i:]
                cost1[:,l1.size()[1]:, i,:,i:] = r1[:,:,:,:-i]
            else:
                cost1[:,:l1.size()[1],i,:,i:] = l1
                cost1[:,l1.size()[1]:,i,:,i:] = r1

        cost1 = self.matching(cost1)     
        disp1 = self.disp(cost1)   

        # flow left 
        flow_left = self.flow(l0_img,l1_img)[-1].detach().cpu()
        inv_flow =  invert_flow(flow_left)
        warped_d1 = warp_with_flow(disp1[:].detach().cpu(),inv_flow)
        for i in range(warped_d1.shape[0]):
            warped_d1[i] = torch.Tensor(median_filter(warped_d1[i],size=3))
            # warped_d1[i] = torch.Tensor(median_filter(warped_d1[i],size=5))
            # warped_d1[i] = torch.Tensor(median_filter(warped_d1[i],size=7))
        warped_d1 = warped_d1.cuda()
        

        flow_left = flow_left.cuda()

        # autoencoder
        # combine all of the above, and 'refine' the disparities
        refined_disparities = self.refiner(torch.cat((disp0,warped_d1,flow_left),1))
        # skip connection + warping

        refined_disparities[:,0] += disp0[:,0]
        refined_disparities[:,1] += warped_d1[:,0]


        return [refined_disparities[:,0],refined_disparities[:,1]]

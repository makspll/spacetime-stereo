import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Disparity
from .blocks.AutoEncoder import AutoEncoder
from .LEAStereo import MatchingNetwork,FeatureNetwork
import re
from .raft.raft import RAFT


class STSLateFusion2(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()

        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork() 
        self.disp = Disparity.DisparitySelector(self.max_disp)
        self.flow = RAFT(iterations=24)
        self.flow.requires_grad_(False)
        self.refiner = AutoEncoder(6, 2,stages=3)

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
        flow_left = self.flow(l0_img,l1_img)[-1]

        # flow right
        flow_right = self.flow(r0_img,r1_img)[-1]

        # autoencoder
        # combine all of the above, and 'refine' the disparities
        # print(disp0.shape,disp1.shape)
        refined_disparities = self.refiner(torch.cat((disp0,disp1,flow_left,flow_right),1))
        # print(refined_disparities.shape)

        # skip connection
        refined_disparities[:,0] += disp0[:,0]
        refined_disparities[:,1] += disp1[:,0]

        return refined_disparities

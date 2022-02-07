import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Disparity
from .blocks.AutoEncoder import AutoEncoder
from .LEAStereo import MatchingNetwork,FeatureNetwork
import numpy as np

class STSLateFusionGTFlow(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()

        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork() 
        self.disp = Disparity.DisparitySelector(self.max_disp)
        self.refiner = AutoEncoder(2, 1,stages=3)
        
    def forward(self, l0_img, r0_img, d1_gt_pov_d0):
        d1_gt_pov_d0 = d1_gt_pov_d0[:,np.newaxis,:]

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

        refined_disparities = self.refiner(torch.cat((disp0,d1_gt_pov_d0),1))
        # skip connection
        refined_disparities[:,0] += disp0[:,0]

        return [refined_disparities]

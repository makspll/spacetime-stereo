from os import stat
from types import new_class
import torch
import torch.nn as nn
from .blocks import Disparity
from .LEAStereo import MatchingNetwork,FeatureNetwork, LEAStereo
import re 

class STSEarlyFusionConcat2Big(LEAStereo):
    def __init__(self, max_disp=192):
        super().__init__()


        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork(in_channels=128,out_channels=2,
            resolution_levels=[1,1,2,2,1,2,2,2,1,1,0,1,1,1,2,2,1,2,2,2,1,1,0,1],
            skip_connections=[0,4,0,0,8,0,0,0,0,0,0,0,0,16,0,0,20,0,0,0,0,0,0,0],)

        self.disp = Disparity.DisparitySelector(self.max_disp)
    def forward(self, l0, r0, l1, r1):
        l0_f = self.feature(l0)       
        r0_f = self.feature(r0) 

        l1_f = self.feature(l1)
        r1_f = self.feature(r1)

        b,f,h,w = l0_f.size()
        with torch.cuda.device_of(l0):
            cost = l0_f.new().resize_(b,   
                                        f*4, # features 
                                        int(self.max_disp/3),  
                                        h, # h
                                        w # w
                                        ).zero_() 
        for i in range(int(self.max_disp/3)):
            if i > 0 : 
                cost[:,:f, i,:,i:] = l0_f[:,:,:,i:]
                cost[:,f:f*2, i,:,i:] = r0_f[:,:,:,:-i]
                cost[:,f*2:f*3, i,:,i:] = l1_f[:,:,:,i:]
                cost[:,f*3:f*4, i,:,i:] = r1_f[:,:,:,:-i]

            else:
                cost[:,:f,i,:,i:] = l0_f
                cost[:,f:f*2,i,:,i:] = r0_f
                cost[:,f*2:f*3,i,:,i:] = l1_f
                cost[:,f*3:f*4,i,:,i:] = r1_f

        cost = self.matching(cost)
        disp = self.disp(cost)
        return disp

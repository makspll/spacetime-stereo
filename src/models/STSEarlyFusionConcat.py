import torch
import torch.nn as nn
from .blocks import Disparity
from .LEAStereo import MatchingNetwork,FeatureNetwork

class STSEarlyFusionConcat(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()


        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork() 
        self.disp = Disparity.DisparitySelector(self.max_disp)

    def forward(self, x, y):
        x = self.feature(x)       
        y = self.feature(y) 
        
        # print(x.size(), y.size())

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.max_disp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.max_disp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        # print(cost.size())

        cost = self.matching(cost)     
        disp = self.disp(cost)    
        return disp
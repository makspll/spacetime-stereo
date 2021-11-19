import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Feature,Matching,Disparity,Ops
from .blocks import cell_params_iterator
import re

class LEASTereoOrigMock():
    pass

class LEAStereo(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()


        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork() 
        self.disp = Disparity.DisparitySelector(self.max_disp)

    def forward(self, x, y):
        x = self.feature(x)       
        y = self.feature(y) 

        
        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.max_disp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.max_disp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        

        cost = self.matching(cost)     
        disp = self.disp(cost)    
        return disp
    
    

class MatchingNetwork(nn.Module):
    def __init__(self,
            in_channels=64,
            out_channels=1,
            resolution_levels=[1,1,2,2,1,2,2,2,1,1,0,1], 
            resolution_level_to_disparities={0: 8,1: 16, 2: 32, 3: 64},
            skip_connections=[0,4,0,0,8,0,0,0,0,0,0,0],
            dim=3,
            kernel_size=3,):     

        super().__init__()

        self.dim = dim 

        assert(resolution_levels[-1] == 1 and resolution_levels[0] == 1)
        self.cells = nn.ModuleList()
        in_disparities=in_channels
        same_padding = (kernel_size - 1) // 2

        self.stem0 = Ops.ConvBR(in_disparities, in_disparities//2, kernel_size, stride=1, padding=same_padding, dim=dim)
        self.stem1 = Ops.ConvBR(in_disparities//2, in_disparities//2, kernel_size, stride=1, padding=same_padding, dim=dim)
        self.skip_connections = skip_connections

        for c_params in cell_params_iterator(in_disparities//2,resolution_levels,resolution_level_to_disparities,dim=dim,kernel_size=kernel_size):
            self.cells.append(Matching.MatchingCell(**c_params))
        features_last_layer = resolution_level_to_disparities[resolution_levels[-1]]*4

        self.conv_out  = Ops.ConvBR(features_last_layer//2, out_channels, kernel_size, stride=1, padding=same_padding,  bn=False, relu=False, dim=dim)  
        self.last_6  = Ops.ConvBR(features_last_layer , features_last_layer//2, 1, stride=1, padding=0,dim=dim)  


        self.skips = nn.ModuleList()

        for s,t in enumerate(skip_connections):

            if t == 0:
                continue 

            source_disparities = resolution_level_to_disparities[resolution_levels[s]]
            target_disparities = resolution_level_to_disparities[resolution_levels[t]]
            self.skips.append(
                Ops.ConvBR((source_disparities+target_disparities)*4, (source_disparities+target_disparities)*2, kernel_size, stride=1, padding=same_padding, dim=dim)
                )

    def upsample_to_size(self,t,target_size,**kwargs):
        if self.dim == 4:
            out = t.new().resize_(
                t.shape[0],t.shape[1],t.shape[2],target_size[0],target_size[1],target_size[2]
            )
            for time in range(t.shape[1]):
                out[:,time] = F.upsample(t[:,time],target_size,**kwargs)
            t = out 
        else:
            t = F.upsample(t,target_size,**kwargs)
        return t 

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        out = (stem0, stem1)


        skip_buffer = [None]*len(self.skip_connections)
        skips_count = 0
        for i,c in enumerate(self.cells):
            out = c(out[0],out[1])
            if self.skip_connections[i] != 0:
                skip_buffer[self.skip_connections[i]] = (skips_count,out[1])
                skips_count += 1

            if skip_buffer[i] is not None:
                skip = self.skips[skip_buffer[i][0]](torch.cat((skip_buffer[i][1], out[1]),dim=1))
                out = (out[0],skip)
            


        last_output = out[-1]

          
        mat = self.last_6(last_output)
        mat = self.upsample_to_size(mat,x.shape[-3:],mode='trilinear',align_corners=True)
        mat = self.conv_out(mat)
        return mat  



class FeatureNetwork(nn.Module):
    def __init__(self, 
        resolution_levels=[1,0,1,0,0,0], 
        resolution_level_to_features={0: 8,1: 16, 2: 32, 3: 64} ):
        """

        Args:
            resolution_levels (list, optional): a list of layers, with each element corresponding to the scale down factor at that layer, last layer has to be 0. Defaults to [1,0,1,0,0,0].
            resolution_level_to_features (dict, optional): a list maping each scaling down factor to the number of features there. Defaults to {0: 8,1: 16, 2: 32, 3: 64}.
        """
        super().__init__()
        assert(resolution_levels[-1] == 0 and resolution_levels[0] == 1)

        initial_fm = 32

        # we operate over one third of the resolution at most
        self.stem0 = Ops.ConvBR(3, initial_fm//2, 3, stride=1, padding=1) # initial_fm/2 x H x W
        

        self.stem1 = Ops.ConvBR(initial_fm//2, initial_fm, 3, stride=3, padding=1) # initial_fm x H/3 x W/3
        self.stem2 = Ops.ConvBR(initial_fm, initial_fm, 3, stride=1, padding=1) # initial_fm x H/3 x W/3
        
        # from then each cell scales one level up, down, or not at all
        # the number of features is dependent on the level (half res = double features)
        # each cell, always preproceses input to have the same amount of channels, and so is flexible
        self.cells = nn.ModuleList()

        # we append a zero to resolution levels for the input to the network
        for c_params in cell_params_iterator(initial_fm,resolution_levels,resolution_level_to_features):
            self.cells.append(Feature.FeatureCell(**c_params))

        self.conv_out = Ops.ConvBR(initial_fm , initial_fm, kernel_size=1, stride=1, padding=0, bn=False, relu=False) 
        
    def forward(self, x):
            stem0 = self.stem0(x)

            stem1 = self.stem1(stem0)
            stem2 = self.stem2(stem1)
            out = (stem1, stem2)

            for c in self.cells:
                out = c(out[0], out[1])

            last_output = out[-1]

            return self.conv_out(last_output)

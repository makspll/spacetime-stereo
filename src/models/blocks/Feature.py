import torch
import torch.nn as nn 
from .Ops import ConvBR,Identity
from . import scale_dimension
import torch.nn.functional as F

class FeatureCell(nn.Module):
    def __init__(self,c_in_prev_prev,c_in_prev,c_out,scale):
        super().__init__()
        
        self.pre_preprocess = ConvBR(c_in_prev_prev, c_out, kernel_size=1, stride=1, padding=0)
        self.preprocess = ConvBR(c_in_prev, c_out, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.c_out = c_out

        conv_args = {
            'c_in':c_out,
            'c_out':c_out,
            'kernel_size':3,
            'stride':1,
            'padding':1
            }

        self.conv_prev_prev_to_zero = ConvBR(**conv_args)
        self.skip_prev_to_zero = Identity()
        self.conv_prev_to_one = ConvBR(**conv_args)
        self.conv_zero_to_one = ConvBR(**conv_args)
        self.conv_prev_prev_to_two = ConvBR(**conv_args)
        self.conv_one_to_two = ConvBR(**conv_args)
        
    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.scale != 1:
            feature_size_h = scale_dimension(s1.shape[2], self.scale)
            feature_size_w = scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.c_out) else s0
        s1 = self.preprocess(s1)

        # intermediate nodes
        i0 = sum([self.conv_prev_prev_to_zero(s0),self.skip_prev_to_zero(s1)])
        i1 = sum([self.conv_prev_to_one(s1),self.conv_zero_to_one(i0)])
        i2 = sum([self.conv_prev_prev_to_two(s0),self.conv_one_to_two(i1)])

        # skip connection, and intermediary nodes features get concatenated
        states = [s1,i0,i1,i2]

        return prev_input, torch.cat(states, dim=1) 

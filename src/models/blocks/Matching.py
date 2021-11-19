import torch
from torch.functional import align_tensors
import torch.nn as nn 
from .Ops import ConvBR
from . import scale_dimension
import torch.nn.functional as F

class MatchingCell(nn.Module):
    def __init__(self, c_in_prev_prev,c_in_prev,c_out,scale, dim=3, kernel_size=3):
        super().__init__()

        assert(dim >= 3 and dim <= 4)
        self.dim = dim 
        
        self.pre_preprocess = ConvBR(c_in_prev_prev, c_out, kernel_size=1, stride=1, padding=0,dim=dim)
        self.preprocess = ConvBR(c_in_prev, c_out, kernel_size=1, stride=1, padding=0,dim=dim)
        self.scale = scale 
        self.c_out = c_out

        conv_args = {
            'c_in':c_out,
            'c_out':c_out,
            'kernel_size':kernel_size,
            'stride':1,
            'padding':(kernel_size - 1) // 2,
            'dim':dim,
            }

        self.conv_prev_prev_to_zero = ConvBR(**conv_args)
        self.conv_prev_to_zero = ConvBR(**conv_args)
        self.conv_prev_to_one = ConvBR(**conv_args)
        self.conv_zero_to_one = ConvBR(**conv_args)
        self.conv_prev_to_two = ConvBR(**conv_args)
        self.conv_one_to_two = ConvBR(**conv_args)


    def interpolate_scale(self, t, scale,target_size=None, **kwargs):

        dims = [scale_dimension(x, scale) for x in t.shape[-3:]] if not target_size else target_size[-3:]
        if scale != 1 or not target_size == t.shape[-3:]:
            if self.dim == 3:
                t = F.interpolate(t,dims,**kwargs)
            else:
                out = t.new().resize_(t.shape[0],t.shape[1],t.shape[2],dims[0],dims[1],dims[2])
                for time in range(t.shape[1]):
                    out[:,time] = F.interpolate(t[:,time],dims,**kwargs)
                t = out 
        return t
    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input

        s1 = self.interpolate_scale(s1,self.scale, mode='trilinear', align_corners=True)
        s0 = self.interpolate_scale(s0,self.scale, target_size=s1.shape, mode='trilinear', align_corners=True)

        # if self.scale != 1:
        #     feature_size_d = scale_dimension(s1.shape[2], self.scale)
        #     feature_size_h = scale_dimension(s1.shape[3], self.scale)
        #     feature_size_w = scale_dimension(s1.shape[4], self.scale)
        #     s1 = F.interpolate(s1, [feature_size_d, feature_size_h, feature_size_w], mode='trilinear', align_corners=True)
        # if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
        #     s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3], s1.shape[4]),
        #                                     mode='trilinear', align_corners=True)
        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.c_out) else s0
        s1 = self.preprocess(s1)

        # intermediate nodes
        i0 = sum([self.conv_prev_prev_to_zero(s0),self.conv_prev_to_zero(s1)])
        a,b= self.conv_prev_to_one(s1),self.conv_zero_to_one(i0)
        i1 = sum([a,b])
        i2 = sum([self.conv_prev_to_two(s1),self.conv_one_to_two(i1)])

        # skip connection, and intermediary nodes features get concatenated
        states = [s1,i0,i1,i2]

        return prev_input, torch.cat(states, dim=1)
        

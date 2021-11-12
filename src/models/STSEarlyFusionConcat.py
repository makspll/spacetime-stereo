from os import stat
from types import new_class
import torch
import torch.nn as nn
from .blocks import Disparity
from .LEAStereo import MatchingNetwork,FeatureNetwork, LEAStereo

class STSEarlyFusionConcat(LEAStereo):
    def __init__(self, max_disp=192):
        super().__init__()


        self.max_disp = max_disp
        self.feature = FeatureNetwork()
        self.matching = MatchingNetwork(128) 
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
                cost[:,f*2:f*3,i,:,i:] = r1_f
                cost[:,f*3:f*4,i,:,i:] = r1_f

        cost = self.matching(cost)     
        disp = self.disp(cost)    
        return disp

    def convert_weights(self,state_dict, weights_source):
        state_dict = super().convert_weights(state_dict, weights_source)


        if weights_source is type(self):
            return state_dict
        elif weights_source is type(LEAStereo):
            # clear affected weights

            filters = set(["module.matching.cells.0.pre_preprocess.conv.weight",
                            "module.matching.cells.0.preprocess.conv.weight",
                            "module.matching.cells.1.pre_preprocess.conv.weight",
                            "module.matching.stem0.conv.weight",
                            "module.matching.stem0.bn.weight",
                            "module.matching.stem0.bn.bias",
                            "module.matching.stem0.bn.running_mean",
                            "module.matching.stem0.bn.running_var",
                            "module.matching.stem1.conv.weight",
                            "module.matching.stem1.bn.weight",
                            "module.matching.stem1.bn.bias",
                            "module.matching.stem1.bn.running_mean",
                            "module.matching.stem1.bn.running_var"])


            new_state_dict = {}

            for k in state_dict:
                if k in filters:
                    continue
                new_state_dict[k] = state_dict[k]

            return new_state_dict


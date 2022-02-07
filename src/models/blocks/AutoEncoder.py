import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary 
import math 

class AutoEncoder(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,stages=3,kernel=3) -> None:
        super().__init__()
        
        self.kernel = kernel

        init_channels = 2
        while init_channels < in_channels:
            init_channels *= 2

        stages = [2**i * init_channels for i,_ in enumerate(range(stages))] 
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i,s in enumerate(stages):
            self.add_encoder_stage(i,in_channels,s)
            in_channels = s

        rev_stages = list(reversed(stages))
        for i,s in enumerate(reversed(stages)):
            quit = False
            nxt_stage = None
            if i >= len(stages) - 1:
                nxt_stage = out_channels
                quit = True
            else:
                nxt_stage = rev_stages[i+1]
            self.add_decoder_stage(i,s,nxt_stage)
            if quit:
                break
        

    def add_encoder_stage(self,stage,in_c,out_c):
        self.encoder.add_module(f"conv{stage}", nn.Conv2d(in_c,out_c,self.kernel,padding=1,stride=2))

    def add_decoder_stage(self,stage,in_c,out_c):
        self.encoder.add_module(f"deconv{stage}", nn.ConvTranspose2d(in_c,out_c,self.kernel,padding=1,output_padding=1,stride=2))

    # expects 3D input ((B) x D x H x W)
    def forward(self,x):  
        x = self.encoder(x)
        x = self.decoder(x)
        return x
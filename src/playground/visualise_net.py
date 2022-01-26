import sys
import os
import torch.nn as nn
from torchsummary import summary
from fetch_network import get_leastereo

def visualise_net(net : nn.Module, in_size, device):
    print(summary(net,in_size,device=device))


def visualise_LEAStereo(**kwargs):
    device = 'cpu'
    model = get_leastereo(device=device,**kwargs)

    width = kwargs.get('crop_width',1248)
    height = kwargs.get('crop_height',384) 

    visualise_net(model.feature,[(3,height,width)], device)
if __name__ == "__main__":
    visualise_LEAStereo()
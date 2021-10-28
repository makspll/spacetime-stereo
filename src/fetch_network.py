import argparse
import sys
import os
import torch

def get_leastereo(resume_non_converted=None,device='cuda'):
    from models.LEAStereo import LEAStereo
  
    torch.backends.cudnn.benchmark = True

    cuda = device == 'cuda'
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    model = LEAStereo()

    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if resume_non_converted: #opt.resume:
        if os.path.isfile(resume_non_converted):
            print("=> loading checkpoint '{}'".format(resume_non_converted))
            checkpoint = torch.load(resume_non_converted)
            model.load_state_dict(model.module.convert_weights(checkpoint['state_dict']), strict=True)      
        else:
            print("=> no checkpoint found at '{}'".format(resume_non_converted))
    return model
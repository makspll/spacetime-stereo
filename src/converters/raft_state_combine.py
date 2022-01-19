import sys 
import os 
import torch 
import numpy as np 
import re 
if __name__ == "__main__":
    path_raft = sys.argv[1]
    path_ours = sys.argv[2]
    name_key = sys.argv[3]
    out_path = sys.argv[4]

    raft = torch.load(path_raft,map_location='cpu')
    ours = torch.load(path_ours,map_location='cpu')

    new_keys = {("module."+name_key+x[6:]):y for x,y in raft.items()}
    
    new_state_dict = {**ours["state_dict"],**new_keys}
    ours["state_dict"] = new_state_dict

    torch.save(ours,out_path)

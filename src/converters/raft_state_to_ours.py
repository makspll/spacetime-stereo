import sys 
import os 
import torch 
import numpy as np 

if __name__ == "__main__":
    path = sys.argv[1]
    out_path = sys.argv[2]

    c = torch.load(path,map_location='cpu')

    torch.save({"state_dict":c},out_path)

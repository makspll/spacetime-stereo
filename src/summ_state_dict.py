import sys 
import os 
import torch 
import numpy as np 

if __name__ == "__main__":
    path = sys.argv[1]

    c = torch.load(path,map_location='cpu')
    val_max = np.argmax(c['accuracies_val'])
    print(val_max, 1 - c['accuracies_val'][val_max])

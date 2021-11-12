from PIL import Image 
import numpy as np


def load_rgb_img(path,dtype=None):
    return np.asarray(Image.open(path),dtype=dtype)
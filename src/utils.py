from PIL import Image 
import numpy as np


def load_rgb_img(path):
    return np.asarray(Image.open(path))
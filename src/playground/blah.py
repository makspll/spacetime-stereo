import numpy as np
from models.augmentations.image_prep import crop_pad_to
from PIL import Image
import sys 


if __name__ == "__main__":
    img = np.asarray(Image.open(sys.argv[1]),dtype=np.uint8)   
    print(img.shape)
    img = crop_pad_to(img,int(sys.argv[2]),int(sys.argv[3]),start_corner=[int(sys.argv[4]),int(sys.argv[5])],padding_mode="replicate")
    print(img.shape)
    img = Image.fromarray(img.astype(np.uint8))
    img.save("2_" + sys.argv[1])
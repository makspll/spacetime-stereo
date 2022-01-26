import torch 
import sys 
from PIL import Image 
import numpy as np 
import torch.nn.functional as F 

def warp_with_flow(img,flow, mask=None, ):
    """
    warps given image with the given flow in reverse (brings image back in time)
    Args:
        img : B x C x H x W
        flow : B x H x W x 2
    Returns:
        warped img: B x C x H x W
    """
    
    # n,c,h,w = img.size()
    # hh,hw = (h)/2,(w)/2
    # grid = torch.Tensor(np.moveaxis(np.meshgrid(np.linspace(-1,1,num=w),np.linspace(-1,1,num=h)),0,-1))
    # print(n,c,h,w,grid.shape)
    # n_ofl = flow / torch.Tensor([hw,hh]) + grid

    # # blow up non-valid values, causes them to land outsid boundary
    # # important: invalid values for flow and disparity must be set to 0
    # n_ofl[flow == 0] = 1e10 

    # warped = F.grid_sample(img,n_ofl, mode="nearest",align_corners=True,padding_mode='zeros')
    
    # return warped
    h,w = img.shape[-2:]
    b = img.shape[0]
    nImg = torch.clone(img)
    nImg[:] = 0

    for B in range(b):
        for x in range(w):
            for y in range(h):
                flx = flow[B,y,x,0]
                fly = flow[B,y,x,1]

                nX = (x + flx).int()
                nY = (y + fly).int()

                if(nY < 0 or nY >= h or nX < 0 or nX >= w):
                    continue

                nImg[B,:,y,x] = img[B,:,nY,nX]
    print(nImg.shape)
    return nImg

def warp_with_flow_forward(img,flow, mask=None):
    h,w = img.shape[-2:]
    b = img.shape[0]
    nImg = torch.clone(img)
    nImg[:] = 0

    for B in range(b):
        for x in range(w):
            for y in range(h):
                flx = flow[B,y,x,0]
                fly = flow[B,y,x,1]

                nX = (x + flx).int()
                nY = (y + fly).int()

                if(nY < 0 or nY >= h or nX < 0 or nX >= w or (flx ==0 and fly == 0)):
                    continue

                nImg[B,:,nY,nX] = img[B,:,y,x]
    print(nImg.shape)
    return nImg

def open_kitti_flow(path):
    import cv2
    flo_file = cv2.imread(path, -1)
    flo_img = flo_file[:,:,3:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64 
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] =  0# MAX_FLOW_KITTI * 2
    return flo_img

if __name__ == "__main__":
    with Image.open(sys.argv[1]) as i:
        with Image.open(sys.argv[3]) as gt:
        
            img = np.asarray(i)[np.newaxis,:]
            gt = np.asarray(gt)[np.newaxis,:]
            fl = torch.Tensor(open_kitti_flow(sys.argv[2])[np.newaxis,:])
            print(img.shape,gt.shape)
            if(len(img.shape) == 3):
                img = torch.Tensor(img[np.newaxis,:])
            else:
                img = torch.Tensor(np.moveaxis(img,-1,1))

            if(len(gt.shape) == 3):
                gt = torch.Tensor(gt[np.newaxis,:])
            else:
                gt = torch.Tensor(np.moveaxis(gt,-1,1))

            print(img.shape,gt.shape)

            out = warp_with_flow_forward(img,fl)

            if(out.shape[1] == 1):
                out = out[0,0,:].numpy().astype(np.uint8)
            else:
                out = np.moveaxis(out.numpy()[0,:],0,-1).astype(np.uint8)

            Image.fromarray(out).show()
            
            print(out.shape)

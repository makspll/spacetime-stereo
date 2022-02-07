import torch 
import sys 
from PIL import Image 
import numpy as np 
import torch.nn.functional as F 

def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)

def invert_flow(flow):
    """ 
        flow : B x 2 x H x W 
    """
    h,w = flow.shape[0:2]
    b = flow.shape[0]
    n_flow = torch.zeros_like(flow,device=flow.device)
    for B in range(b):
        for x in range(w):
            for y in range(h):
                
                flx = flow[B,0,y,x]
                fly = flow[B,1,y,x]
                nX = (x + flx).int()
                nY = (y + fly).int()

                n_flow[B,0,nY,nX] = -flx
                n_flow[B,1,nY,nX] = -fly

    return flow 

def warp_with_flow(img,flow, mask=None, ):
    """
    warps given image with the given flow in reverse (brings image back in time)
    Args:
        img : B x C x H x W
        flow : B x 2 x H x W 
    Returns:
        warped img: B x C x H x W
    """
    flow = _moveaxis(flow,1,-1)
    n,c,h,w = img.size()
    hh,hw = (h)/2,(w)/2
    grid = torch.Tensor(np.moveaxis(np.meshgrid(np.linspace(-1,1,num=w),np.linspace(-1,1,num=h)),0,-1))
    n_ofl = flow / torch.Tensor([hw,hh]) + grid

    # blow up non-valid values, causes them to land outsid boundary
    # important: invalid values for flow and disparity must be set to 0
    n_ofl[flow == 0] = 1e10 

    warped = F.grid_sample(img,n_ofl, mode="nearest",align_corners=True,padding_mode='zeros')
    
    return warped
    # h,w = img.shape[-2:]
    # b = img.shape[0]
    # nImg = torch.clone(img)
    # nImg[:] = 0

    # for B in range(b):
    #     for x in range(w):
    #         for y in range(h):
    #             flx = flow[B,y,x,0]
    #             fly = flow[B,y,x,1]

    #             nX = (x + flx).int()
    #             nY = (y + fly).int()

    #             if(nY < 0 or nY >= h or nX < 0 or nX >= w):
    #                 continue

    #             nImg[B,:,y,x] = img[B,:,nY,nX]
    # return nImg

# def warp_with_flow_forward(img,flow, mask=None):
#     h,w = img.shape[-2:]
#     b = img.shape[0]
#     nImg = torch.clone(img)
#     nImg[:] = 0

#     for B in range(b):
#         for x in range(w):
#             for y in range(h):
#                 flx = flow[B,y,x,0]
#                 fly = flow[B,y,x,1]

#                 nX = (x + flx).int()
#                 nY = (y + fly).int()

#                 if(nY < 0 or nY >= h or nX < 0 or nX >= w or (flx ==0 and fly == 0)):
#                     continue

#                 nImg[B,:,nY,nX] = img[B,:,y,x]
#     print(nImg.shape)
#     return nImg

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
        
            img = np.asarray(i)[np.newaxis,:] / 256
            gt = np.asarray(gt)[np.newaxis,:] 
            fl = torch.Tensor(open_kitti_flow(sys.argv[2])[np.newaxis,:])

            if(len(img.shape) == 3):
                img = torch.Tensor(img[np.newaxis,:])
            else:
                img = torch.Tensor(np.moveaxis(img,-1,1))

            if(len(gt.shape) == 3):
                gt = torch.Tensor(gt[np.newaxis,:])
            else:
                gt = torch.Tensor(np.moveaxis(gt,-1,1))


            out = warp_with_flow(img,invert_flow(fl))
            print(out.shape)

            if(out.shape[1] == 1):
                out = out[0,0,:].numpy()
            else:
                out = np.moveaxis(out.numpy()[0,:],0,-1)
                print(out.shape)
            Image.fromarray((out*256).astype(np.uint8)).show()
            

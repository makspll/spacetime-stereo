import numpy as np
import torch
import random
import torch.nn.functional as F
import math 

def crop_pad_to(x, new_height, new_width, start_corner=None, padding_mode="constant"):
    if len(x.shape) > 2:    
        c = x.shape[-1] 
        h = x.shape[-3]
        w = x.shape[-2]
        gray = False
    else:
        x = x[:,:,np.newaxis]
        c = 1
        h = x.shape[0]
        w = x.shape[1]
        gray = True
    
    # pad to required dimensions if any axis is too short
    # TODO: check odd dimensions and distribute padding with floor

    width_pad = max(new_width - w,0) / 2
    height_pad = max(new_height - h,0) / 2
    pleft = math.floor(width_pad)
    pright = math.ceil(width_pad)
    ptop = math.floor(height_pad)
    pbottom = math.ceil(height_pad)
    if pleft or ptop:
        channel_first = torch.Tensor(np.moveaxis(x,-1,0))
        out = F.pad(channel_first[np.newaxis,:],[pleft,pright,ptop,pbottom],mode=padding_mode).numpy()[0,:]
        out = np.moveaxis(out,0,-1)
    else:
        if start_corner:
            start_x = start_corner[0]
            start_y = start_corner[1]
        else:
            start_x = int((w - new_width) / 2)
            start_y = int((h - new_height) / 2)

        assert((start_x + new_width) <= w)
        assert((start_y + new_height) <= h)

        out = x[start_y: start_y + new_height, start_x: start_x + new_width, :]

    if gray:
        return out[:,:,0]
    return out

def normalize(x, means, stds):
    c = x.shape[-1]
    h = x.shape[-3]
    w = x.shape[-2]

    out = np.zeros([h, w, c], 'float32')

    for c in range(c):
        out [:,:,c] = (x[:,:,c] - means[c]) / stds[c]
    return out 

def kitti_transform(x, new_height, new_width, start_corner=None, normalize_rgb=True, padding_mode="constant"):

    if normalize_rgb:

        means = []
        stds = []

        for c in range(x[0].shape[-1]):
            means.append(np.mean(x[:,:,c]))
            stds.append(np.std(x[:,:,c]))

        x = normalize(x,means,stds)

    x = crop_pad_to(x, new_height, new_width, start_corner=start_corner,padding_mode=padding_mode)

    if (len(x.shape) != 2):
        x = np.moveaxis(x,-1,0)

    # batch dimension
    x = x[np.newaxis,:]

    return x


def kitti_train_transform(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)
    
    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h+shift, crop_width + shift], 'float32')
        temp_data[6:7,:,:] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)
   
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height + shift, crop_width + shift], 'float32')
        temp_data[6: 7, :, :] = 1000
        temp_data[:, crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)
    if shift > 0:
        start_x = random.randint(0, w - crop_width)
        shift_x = random.randint(-shift, shift)
        if shift_x + start_x < 0 or shift_x + start_x + crop_width > w:
            shift_x = 0
        start_y = random.randint(0, h - crop_height)
        left = temp_data[0: 3, start_y: start_y + crop_height, start_x + shift_x: start_x + shift_x + crop_width]
        right = temp_data[3: 6, start_y: start_y + crop_height, start_x: start_x + crop_width]
        target = temp_data[6: 7, start_y: start_y + crop_height, start_x + shift_x : start_x+shift_x + crop_width]
        target = target - shift_x
        return left, right, target
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = random.randint(0, w - crop_width)
        start_y = random.randint(0, h - crop_height)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

import torch
import numpy as np
def np2torch(np_im):
    np_im = np_im / 255
    return  torch.from_numpy(np_im).permute(2,0,1).unsqueeze(0).float()
def torch2np(im):
    return np.uint8(im.squeeze().permute(1,2,0).cpu().numpy() * 255)
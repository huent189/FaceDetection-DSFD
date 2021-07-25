import cv2
import numpy as np
import torch
from utils.zerodce import ZeroDCE, ZeroDCE_ext
from utils.images import np2torch, torch2np
def clahe_prepocess(img):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lab[:,:,-1] = clahe.apply(lab[:,:,-1])
  img = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
  return img
#######################################
def image_agcwd(img, a=0.25, truncated_cdf=False):
    h,w = img.shape[:2]
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    intensity_max = unique_intensity.max()
    intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
    pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
    
    if truncated_cdf: 
        inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd
    
    img_new = img.copy()
    for i in unique_intensity:
        img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
   
    return img_new
def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed = 255 - agcwd
    return reversed

def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd
def iagwcd_preprocess(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0]
    threshold = 0.3
    exp_in = 112 # Expected global average intensity 
    M,N = img.shape[:2]
    mean_in = np.sum(Y/(M*N)) 
    t = (mean_in - exp_in)/ exp_in
    img_output = None
    if t < -threshold: # Dimmed Image
        result = process_dimmed(Y)
        YCrCb[:,:,0] = result
        img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
    elif t > threshold:
        result = process_bright(Y)
        YCrCb[:,:,0] = result
        img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
    else:
        img_output = img
    return img_output
class DLPreprocesor():
    def __init__(self, model_type, pretrain):
        if model_type == 'zerodce':
            self.model = ZeroDCE().cuda()
        elif model_type == 'zerodce_ext':
            self.model = ZeroDCE_ext(scale_factor=12).cuda()
        self.model.load_state_dict(torch.load(pretrain))
        self.model.eval()
    def preprocess(self, np_im):
        im = np2torch(np_im).cuda()
        with torch.no_grad():
            im = self.model(im)
        return torch2np(im)
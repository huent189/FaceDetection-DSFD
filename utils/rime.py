import torch
import shutil
import glob
import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import torch.nn as nn
from utils.images import np2torch, torch2np
import numpy as np
def get_type_by_gmm(im, is_im=False):
    if not is_im:
        im = cv2.imread(im)
    im = np.max(im, axis=2)
    # s = 32 / min(im.shape[0], im.shape[1])
    # if s < 1:
        # im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
    im = cv2.resize(im, (32,32))
    # print(im.shape)
    hist = im.ravel() / 256
    gmm = GaussianMixture(n_components=3, covariance_type='spherical', random_state=8)
    gmm.fit(np.expand_dims(hist, axis=1))
    mu = gmm.means_
    scores = np.exp(gmm.score_samples(gmm.means_.reshape(-1,1)))
    thresh = max(np.histogram(hist, bins=256, range=[0,1], density=True)[0]) * 0.05
    # thresh = max(thresh, scores.min())
    mu = mu[scores > thresh]
    # print(mu)
    if (mu < 0.4).all():
        return 'under'
    elif (mu > 0.6).all():
        return 'over'
    return 'mix'
def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se='SE', nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5, 7]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        # self.use_res_connect = False
        conv_layer = nn.Conv2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        elif nl == 'LeRE':
            nlin_layer = nn.LeakyReLU
        elif nl == 'HSig':
            nlin_layer = Hsigmoid
        elif nl == 'NegHSig':
            nlin_layer = NegHsigmoid
        else:
            raise NotImplementedError
        if se == 'SE':
            SELayer = SEModule
        if se == 'SENoBias':
            SELayer = SEModuleNoBias

        else:
            SELayer = Identity
        if exp != oup:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, exp, 1, 1, 0, bias=True),
                nlin_layer(inplace=True),
                # dw
                conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=True),
                SELayer(exp),
                nlin_layer(inplace=True),
                # pw-linear
                conv_layer(exp, oup, 1, 1, 0, bias=True),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, exp, 1, 1, 0, bias=True),
                nlin_layer(inplace=False),
                conv_layer(exp, oup, 1, 1, 0, bias=True),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1,1,0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1,1,0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y
class UNetLikeV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = MobileBottleneck(3, 3, 3, 1,6)
        base_number = 8
        self.conv1 = MobileBottleneck(3, base_number, 3, 2, int(base_number * 1.5), False, 'RE')
        self.conv2 = MobileBottleneck(base_number, base_number, 3, 1, int(base_number*1.5), False, 'RE')
        self.conv3 = MobileBottleneck(base_number, base_number*2, 3, 2, base_number*3, False, 'RE')
        self.conv5 = MobileBottleneck(base_number*2, base_number*2, 3, 1, base_number*3, False, 'RE')
        self.conv6 = MobileBottleneck(base_number*2, base_number, 3, 1, base_number*3, False, 'RE')
        self.conv7 = MobileBottleneck(base_number*2, base_number, 3, 1, base_number*3, False, 'RE')
        self.conv8 = MobileBottleneck(base_number, 3, 3, 1, int(base_number * 1.5), False, 'RE') 
        self.last_conv = MobileBottleneck(6, 3, 3,1,9)
        self.resize_mindim = 256
    def forward(self, x):
        min_dim = min(x.shape[0], x.shape[1])
        if self.resize_mindim is not None and self.resize_mindim < min_dim:
          x_down = F.interpolate(x, scale_factor=self.resize_mindim/min_dim, mode='bilinear', align_corners=True)
        else:
          x_down = x
        x_1 = self.first_conv(x)
        r = self.conv1(x_1)
        r = self.conv2(r)
        r_d2 = r
        r = self.conv3(r)
        r = self.conv5(r)
        r = self.conv6(r)
        r = F.interpolate(r, (r_d2.shape[2], r_d2.shape[3]))
        r = self.conv7(torch.cat([r_d2, r], dim=1))
        r = self.conv8(r)
        r = F.interpolate(r, (x_down.shape[2], x_down.shape[3]))
        r = self.last_conv(torch.cat([x_1,r], dim=1))
        r = torch.abs(r + 0.01)
        if self.resize_mindim is not None and self.resize_mindim < min_dim:
            r = F.interpolate(x, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        x = 1 - (1 - x) ** r
        return x
class RIME():
    def __init__(self, under_model, mix_model, over_model):
        self.models = {
            "under" : UNetLikeV2().cuda(),
            'over': UNetLikeV2().cuda(),
            'mix': UNetLikeV2().cuda()
        }
        self.models['under'].load_state_dict(torch.load(under_model))
        self.models['under'].eval()
        self.models['over'].load_state_dict(torch.load(over_model))
        self.models['over'].eval()
        self.models['mix'].load_state_dict(torch.load(mix_model))
        self.models['mix'].eval()
    def preprocess(self, np_im):
        dt = get_type_by_gmm(np_im, True)
        im = np2torch(np_im).cuda()
        with torch.no_grad():
            im = self.models[dt](im)
        return torch2np(im)
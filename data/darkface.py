from __future__ import division, print_function

import os.path as osp
import pdb
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
#from utils.augmentations import SSDAugmentation
import scipy.io
import torch
import torch.utils.data as data
import glob
import os
'''WIDER Face Dataset Classes
author: swordli
'''
#from .config import HOME
sys.path.append('/f/home/jianli/code/s3fd.180716/')
plt.switch_backend('agg')

WIDERFace_CLASSES = ['face']  # always index 0
# note: if you used our download scripts, this should be right
WIDERFace_ROOT = '/data2/lijian/widerface/data/'


class WIDERFaceAnnotationTransform(object):
    '''Transforms a WIDERFace annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    '''

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDERFace_CLASSES, range(len(WIDERFace_CLASSES))))

    def __call__(self, target, width, height):
        '''
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        '''
        for i in range(len(target)):

            '''
            if target[i][0] < 2 : target[i][0] = 2
            if target[i][1] < 2 : target[i][1] = 2
            if target[i][2] > width-2 : target[i][2] = width - 2
            if target[i][3] > height-2 : target[i][3] = height - 2
            '''
            target[i][0] = float(target[i][0]) / width 
            target[i][1] = float(target[i][1]) / height  
            target[i][2] = float(target[i][2]) / width 
            target[i][3] = float(target[i][3]) / height  
            '''
            if target[i][0] < 0.0001:
                target[i][0] = 0.0001 
            if target[i][1] < 0.0001:
                target[i][1] = 0.0001 
            if target[i][2] > 0.9999:
                target[i][2] = 0.9999
            if target[i][3] > 0.9999:
                target[i][3] = 0.9999
            '''
            # filter error bbox
            
            #if target[i][0] >= target[i][2] or target[i][1] >= target[i][3] or target[i][0] < 0 or target[i][1] < 0 or target[i][2] > 1 or target[i][3] > 1 :
            #    print ('error bbox: ' ,  target[i])
            
            '''
            assert target[i][0] >= 0.001
            assert target[i][1] >= 0.001
            assert target[i][2] <= 0.999
            assert target[i][3] <= 0.999
            assert target[i][0] < target[i][2]
            assert target[i][1] < target[i][3]
            '''
            #res.append( [ target[i][0], target[i][1], target[i][2], target[i][3], target[i][4] ] )
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
def read_label(file_name):
  with open(file_name, 'r') as f:
    total_bbox = int(next(f))
    bboxes = []
    for line in f: # read rest of lines
        bboxes.append([int(x) for x in line.split()]+ [0])
    assert len(bboxes) == total_bbox, f'total {file_name}'
    assert len(bboxes[0]) == 5, f'bbox {file_name}'
    return bboxes
class DarkFaceDataset(data.Dataset):
    def __init__(self, root,
                 image_sets='train',
                 transform=None, target_transform=WIDERFaceAnnotationTransform(),
                 dataset_name='WIDER Face',preprocess=None):
        self.img_ids = glob.glob(os.path.join(root, 'image/*'))
        self.label_ids = glob.glob(os.path.join(root, 'label/*'))
        self.target_transform = target_transform
        self.transform = transform
        self.preprocess = preprocess
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.img_ids)

    def pull_item(self, index):

        target = read_label(self.label_ids[index])
        img = cv2.imread(self.img_ids[index])
        if self.preprocess == 'clahe':
          hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
          clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
          hsv[-1] = clahe.apply(hsv[-1])
          img = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
        height, width, channels = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            # data augmentation
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            #self.vis_detections_v2(img , boxes , index)
            # to rgb
            #img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def vis_detections(self , im,  dets, image_name ):

        # cv2.imwrite('./tmp_res/'+str(image_name)+'ori.jpg' , im)
        # print (im)
        # size = im.shape[0]
        # dets = dets*size
        # '''Draw detected bounding boxes.'''
        # class_name = 'face'
        # #im = im[:, :, (2, 1, 0)]
        # fig, ax = plt.subplots(figsize=(12, 12))
        # ax.imshow(im, aspect='equal')

        # for i in range(len(dets)):
        #     bbox = dets[i, :4]
        #     ax.add_patch(
        #         plt.Rectangle((bbox[0], bbox[1]),
        #                       bbox[2] - bbox[0] + 1,
        #                       bbox[3] - bbox[1] + 1, fill=False,
        #                       edgecolor='red', linewidth=2.5)
        #         )
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('./tmp_res/'+str(image_name)+'.jpg', dpi=fig.dpi)
        pass
    def vis_detections_v2(self , im,  dets, image_name ):
        # size = im.shape[0]
        # dets = dets*size
        # '''Draw detected bounding boxes.'''
        # class_name = 'face'
        # for i in range(len(dets)):
        #     bbox = dets[i, :4]
        #     #print ((bbox[0],bbox[1]), (bbox[2],bbox[3]) )
        #     cv2.rectangle( im , (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0),5 )
        # cv2.imwrite('./tmp_res/'+str(image_name)+'.jpg', im)
        pass
    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        return cv2.imread(self.img_ids[index], cv2.IMREAD_COLOR)
        pass

    def pull_event(self, index):
        # return self.event_ids[index]
        return ''
        # pass

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.img_ids[index]
        anno = read_label(self.label_ids[index])
        gt = self.target_transform(anno, 1, 1)
        return img_id.split('/')[-1], gt
        # pass

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        # return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
        pass

'''
from utils.augmentations import SSDAugmentation
if __name__ == '__main__': 
    dataset = WIDERFaceDetection( root=WIDERFace_ROOT, transform=SSDAugmentation(640,(104,117,123) ) )
    for i in range(10000):
       img, tar = dataset.pull_item(i)
    print (sta_w)
'''

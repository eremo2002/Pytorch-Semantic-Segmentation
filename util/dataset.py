import glob
from logging import root
import cv2
import os
import numpy as np
import copy
import torch

from pycocotools.coco import COCO



class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, set_name, transform):
        super(Custom_Dataset, self).__init__()

        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_'+self.set_name+'.json'))
        
        whole_image_ids = self.coco.getImgIds()
        self.image_ids = []

        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) != 0:
                self.image_ids.append(idx)
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
      
        # load image
        image, file_name = self.load_image(idx)
        image = np.expand_dims(image, -1)

        # load annotation info
        annotation = self.load_annotations(idx)        
        
        # gt_mask_array = np.zeros((image.shape[0], image.shape[1], 4)) # 4 is number of class
        gt_mask_array = np.zeros((image.shape[0], image.shape[1], 1)) # single class
        

        # extract label, segm 
        for i, v in enumerate(annotation):
            label = annotation[i]['category_id']
            mask = self.coco.annToMask(annotation[i])            
            # gt_mask_array[:, :, int(label)+1] += mask # multi-class
            gt_mask_array[:, :, int(label)] += mask # single class
        

        data = {'image': image, 'mask': gt_mask_array}

        if self.transform:
            data = self.transform(data)
            
        return data
      
      

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]        
        path = os.path.join(self.root_dir, 'JPEGImages', self.set_name, image_info['file_name'])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
       
        return image, image_info['file_name']


    def load_annotations(self, image_index):
        return self.coco.loadAnns(self.coco.getAnnIds(image_index))

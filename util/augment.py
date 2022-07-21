import time
import copy
import cv2
import numpy as np
import torch


class ToTensor(object):
    def __call__(self, data):        
        image, mask = data['image'], data['mask']
        image = torch.from_numpy(image.copy()).type(torch.float32)
        mask = torch.from_numpy(mask.copy()).type(torch.float32)
                
        # (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)        

        data = {'image': image, 'mask': mask}

        return data

      
# coco에서 mask 영역은 255가 아닌 1로 normalize해서 로드하기 때문에 image만 normalize함
class Normalize(object):
    def __call__(self, data):
        image, mask = data['image'], data['mask']        
        image = image / 255.        

        data = {'image': image, 'mask': mask}

        return data


class Random_Gamma(object):
    def __init__(self, p, range=(0.5, 1.5)):
        self.p = p
        self.range = range

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if self.p >= np.random.random():
            self.gamma = np.random.uniform(low=self.range[0], high=self.range[1]) # e.g.  0.5 ~ 1.5
            if image.dtype == np.uint8:
                table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** self.gamma) * 255
                image = cv2.LUT(image, table.astype(np.uint8))
                image = np.expand_dims(image, -1)
            else:
                image = np.power(image, self.gamma)
        
        data = {'image': image, 'mask': mask}

        return data


class Random_Brightness(object):
    def __init__(self, p, range=0.5):
        self.p = p
        self.range = range

    def __call__(self, data):
        
        image, mask = data['image'], data['mask']
        
        if self.p >= np.random.random():            
            self.sigma = np.random.uniform(low=-(self.range), high=(self.range)) # e.g.  -0.3 ~ 0.3                        
            image = cv2.add(image, np.mean(image)*self.sigma)
            image = np.expand_dims(image, -1) # because of cv2.add()
        
        data = {'image': image, 'mask': mask}

        return data


class Random_Contrast(object):
    def __init__(self, p, range=0.5):
        self.p = p
        self.range = range

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if self.p >= np.random.random():
            self.sigma = np.random.uniform(low=1-self.range, high=1+self.range)            
            image = image * self.sigma
            image = np.clip(image , 0, 255).astype(np.uint8)            
            
        data = {'image': image, 'mask': mask}

        return data


class Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        
        if np.random.rand() <= self.p:
            image = image[:, ::-1, :]
            mask = mask[:, ::-1, :]
          
        data = {'image': image, 'mask': mask}

        return data


class Vertical_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data['image'], data['mask']
      
        if np.random.rand() <= self.p:
            image = image[::-1, :, :]
            mask = mask[::-1, :, :]            
           
        data = {'image': image, 'mask': mask}

        return data


class Blur(object):
    def __init__(self, p=0.5, blur_type='gaussian'):
        self.p = p
        self.blur_type = blur_type

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if np.random.rand() <= self.p:            
            if self.blur_type == 'gaussian':            
                sigma = np.random.randint(4)
                image = cv2.GaussianBlur(image, (5,5), sigma)
           
        data = {'image': image, 'mask': mask}

        return data


class Affine_Shear(object):
    def __init__(self, p=0.5, range_mx=0.5, range_my=0.3):
        self.p = p
        self.mx = np.random.uniform(low=0, high=range_mx)
        self.my = np.random.uniform(low=0, high=range_my)

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        h, w = image.shape[0], image.shape[1]

        if np.random.rand() <= self.p:

            if np.random.rand() <= self.p:
                # x
                # self.mx = np.random.uniform(low=0, high=self.mx)
                
                affine_matrix = np.array([[1, self.mx, 0],
                                        [0, 1, 0]]).astype(np.float32)
                image_affine = cv2.warpAffine(image, affine_matrix, (int(h+w*self.mx), w))
                mask_affine  = cv2.warpAffine(mask, affine_matrix, (int(h+w*self.mx), w))

                image_affine = cv2.resize(image_affine, (h, w))
                mask_affine = cv2.resize(mask_affine, (h, w))

                image_affine = np.expand_dims(image_affine, -1)
                mask_affine = np.expand_dims(mask_affine, -1)

                data = {'image': image_affine, 'mask': mask_affine}
            else:
                # y
                # self.my = np.random.uniform(low=0, high=self.my)
                affine_matrix = np.array([[1, 0, 0],
                                        [self.my, 1, 0]]).astype(np.float32)
                image_affine = cv2.warpAffine(image, affine_matrix, (h, int(h*self.my+w)))
                mask_affine = cv2.warpAffine(mask, affine_matrix, (h, int(h*self.my+w)))

                image_affine = cv2.resize(image_affine, (h, w))
                mask_affine = cv2.resize(mask_affine, (h, w))

                image_affine = np.expand_dims(image_affine, -1)
                mask_affine = np.expand_dims(mask_affine, -1)

           
                data = {'image': image_affine, 'mask': mask_affine}

        return data



class Rotation(object):   
    # input shape shuld be (h, w, c)
    def __init__(self, p=0.5, angle=(-30, 30)):
        self.p = p 
        self.angle = angle        

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        if self.p >= np.random.random():
            h, w = image.shape[0], image.shape[1]
            rotation_angle = np.random.randint(self.angle[0], self.angle[1])
            rotation_matrix = cv2.getRotationMatrix2D((h/2, w/2), rotation_angle, 1)
            
            image = cv2.warpAffine(image, rotation_matrix, (h, w))
            mask = cv2.warpAffine(mask, rotation_matrix, (h, w))

            image = np.expand_dims(image, -1)
            mask = np.expand_dims(mask, -1)
        data = {'image': image, 'mask': mask}
        
        return data





class Shift_X(object):
    def __init__(self, p, range=30):
        self.p = p        
        self.range = range

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if self.p >= np.random.random():
            h, w = image.shape[0], image.shape[1]
            shifted_image = np.zeros(image.shape).astype(np.uint8)
            shifted_mask = np.zeros(mask.shape).astype(np.uint8)
            
            self.dx = np.random.randint(low=-self.range, high=self.range)
            if self.dx > 0: # shift right
                shifted_image[:, self.dx:, :] = image[:, :w-self.dx, :]
                shifted_mask[:, self.dx:, :] = mask[:, :w-self.dx, :]
            else: # shift left                
                shifted_image[:, :w+self.dx, :] = image[:, (-self.dx):, :]
                shifted_mask[:, :w+self.dx, :] = mask[:, (-self.dx):, :]

            data = {'image': shifted_image, 'mask': shifted_mask}            
        else:
            data = {'image': image, 'mask': mask}

        return data


class Shift_Y(object):
    def __init__(self, p, range=30):
        self.p = p        
        self.range = range

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if self.p >= np.random.random():
            h, w = image.shape[0], image.shape[1]
            shifted_image = np.zeros(image.shape).astype(np.uint8)
            shifted_mask = np.zeros(mask.shape).astype(np.uint8)
            
            self.dy = np.random.randint(low=-self.range, high=self.range)
            if self.dy > 0: # shift up
                shifted_image[:h-self.dy, :, :] = image[self.dy:, :, :]
                shifted_mask[:h-self.dy, :, :] = mask[self.dy:, :, :]                
            else: # shift down
                shifted_image[-self.dy:, :, :] = image[:(h+self.dy), :, :]
                shifted_mask[-self.dy:, :, :] = mask[:(h+self.dy), :, :]                

            data = {'image': shifted_image, 'mask': shifted_mask}            
        else:
            data = {'image': image, 'mask': mask}

        return data


class Random_Crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size        

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        h, w = image.shape[0], image.shape[1]

        top = np.random.randint(0, h - self.patch_size[0])
        bottom = top + self.patch_size[0]        
        left = np.random.randint(0, w - self.patch_size[1])
        right = left + self.patch_size[1]
        
        image_patch = image[top:bottom, left:right]
        mask_patch = mask[top:bottom, left:right]
   
        data = {'image': image_patch, 'mask': mask_patch}

        return data


class Random_Cutout(object):    
    def __init__(self, p):
        self.p = p 
    def __call__(self, data):
        image, mask = data['image'], data['mask']

        if self.p >= np.random.random():
            h, w = image.shape[0], image.shape[1]

            top = np.random.randint(0, h - h//4)
            bottom = top + h//4       
            left = np.random.randint(0, w - w//4)
            right = left + w//4
            
            image[top:bottom, left:right] = 255
   
        data = {'image': image, 'mask': mask}

        return data

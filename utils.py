import random

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class DataHandler(object):
    def __init__(self, opt, input_img):
        super(DataHandler, self).__init__()
        
        #self.input_img = input_img
        self.input_img = Image.open(opt.input_img)
        self.input_pairs = self.create_pairs()
        
        self.scale_factor = opt.scale_factor
        
        # x[0] : hr, HR - test image I's size-ratio approximately 1
        size_ratio = np.float32([x[0].size[0]*x[0].size[1] / float(input_img.size[0] * input_img.size[1]) for x in self.input_pairs])
        
        self.sampling_probability = size_ratio / np.sum(size_ratio)
        self.transforms = transforms.Compose([
            RandomRotation([0, 90, 180, 270]),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomCrop(opt.crop_size),
            ToTensor()])
    
    def create_pairs(self):
        # Downsampling
        smaller_side = min(self.input_img[0:2])
        larger_side = max(self.input_img[0:2])
        
        factors = []
        for i in range(smaller_side//5, smaller_side+1):
            downsampled_smaller_side = i
            ratio = float(downsampled_smaller_side) / smaller_side
            downsampled_larger_side = round(larger_side * ratio)
            if downsampled_smaller_side % self.scale_factor == 0 and \
                downsampled_larger_side % self.scale_factor == 0:
                    factors.append(ratio)
        
        pairs = []
        for ratio in factors:
            hr_father = self.input_img.resize((int(self.input_img.size[0] * ratio), \
                                        int(self.input_img.size[1] * ratio)), \
                                        resample=Image.BICUBIC)
            
            lr_son = hr_father.resize((int(hr_father.size[0]/self.scale_factor), \
                                        int(hr_father.size[1]/self.scale_factor)),
                                        resample=Image.BICUBIC)
            
            lr_son = lr_son.resize(hr_father.size, resample=Image.BICUBIC)
            
            pairs.append((hr_father, lr_son))
        
        return pairs
            
    def preprocess_data(self):
        while True:
            hr, lr = random.choices(self.input_pairs, weights=self.sampling_probability, k=1)[0]
            hr_t, lr_t = self.transforms((hr, lr))
            hr_t = hr_t.unsqueeze(0)
            lr_t = lr_t.unsqueeze(0)
            
            yield hr_t, lr_t


class RandomRotation(object):
    def __init__(self, angles):
        self.angles = angles
    
    def __call__(self, data):
        hr, lr = data
        angle = random.choices(self.angles)
        
        return F.rotate(hr, angle), F.rotate(lr, angle)


class RandomVerticalFlip(object):
    def __call__(self, data):
        hr, lr = data
        if np.random.randn() < 0.5:
            return F.vflip(hr), F.vflip(lr)
        return hr, lr
    

class RandomHorizontalFlip(object):
    def __call__(self, data):
        hr, lr = data
        if np.random.randn() < 0.5:
            return F.hflip(hr), F.hflip(lr)
        
        return hr, lr


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = (int(crop_size), int(crop_size))
    
    @staticmethod
    def setting_window(hr, window_size):
        w, h = hr.size
        f_h, f_w = window_size
        
        if w == f_w or h == f_h:
            return 0, 0, h, w
        
        if w < f_w or h < f_h:
            f_h, f_w = h//2, w//2
        
        x = np.random.randint(0, w-f_w)
        y = np.random.randint(0, h-f_h)
        
        return x, y, f_h, f_w
        
    def __call__(self, data):
        hr, lr = data
        x, y, h, w = self.setting_window(hr, self.crop_size)
        
        return F.crop(hr, x, y, h, w), F.crop(lr, x, y, h, w)
        


class ToTensor(object):
    def __call__(self, data):
        hr, lr = data
        return F.to_tensor(hr), F.to_tensor(lr)
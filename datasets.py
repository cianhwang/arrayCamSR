from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np

class TrainSetLoaderFlickr1024(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoaderFlickr1024, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        
        h, w, c = img_lr_left.shape
        xxs = np.zeros((h, w, w))#, dtype=np.uint16)
        yys = np.zeros((h, w, w))#, dtype=np.uint16)
        for i in range(h):
            for j in range(w):
                xxs[i, j] = i*np.ones((w,))#, dtype=np.uint16)
                yys[i, j] = np.arange(w)#, dtype=np.uint16)
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    
    def __len__(self):
        return len(self.file_list)

class ValidSetLoaderFlickr1024(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(ValidSetLoaderFlickr1024, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        
        h, w, c = img_lr_left.shape
        xxs = np.zeros((h, w, w))#, dtype=np.uint16)
        yys = np.zeros((h, w, w))#, dtype=np.uint16)
        for i in range(h):
            for j in range(w):
                xxs[i, j] = i*np.ones((w,))#, dtype=np.uint16)
                yys[i, j] = np.arange(w)#, dtype=np.uint16)
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    
    def __len__(self):
        return len(self.file_list)

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        
        num = self.file_list[index].split('_')[-1]
        xxs = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
        yys = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    
    def __len__(self):
        return len(self.file_list)

class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(ValidSetLoader, self).__init__()
        self.dataset_dir = dataset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        
        num = self.file_list[index].split('_')[-1]
        xxs = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
        yys = np.load(f'{self.dataset_dir}/xxs_{num}.npy')
        Pos = (xxs, yys)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right), Pos
    
    def __len__(self):
        return len(self.file_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        hr_image_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')
        
        h, w = lr_image_left.height, lr_image_left.width
        jn = 80
        xxs = np.zeros((h, w, jn))#, dtype=np.uint16)
        yys = np.zeros((h, w, jn))#, dtype=np.uint16)
        for i in range(h):
            for j in range(w):
                xxs[i, j] = i*np.ones((jn,))#, dtype=np.uint16)
                if j < jn: #j > w - jn: #
                    a, b = 0, jn#w-jn, w #
                else:
                    a, b = j-jn, j #j, j+jn #
                yys[i, j] = np.arange(a, b)#, dtype=np.uint16)
        Pos = (xxs, yys)
        
        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right, Pos
    def __len__(self):
        return len(self.file_list)

def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]
        """"no rotation
        if random.random()<0.5:
            lr_image_left = lr_image_left.transpose(1, 0, 2)
            lr_image_right = lr_image_right.transpose(1, 0, 2)
            hr_image_left = hr_image_left.transpose(1, 0, 2)
            hr_image_right = hr_image_right.transpose(1, 0, 2)
        """
        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

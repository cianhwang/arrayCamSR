from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
from skimage import metrics
from torch.nn import init

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()
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

        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
    def __len__(self):
        return len(self.file_list)
    
class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(ValidSetLoader, self).__init__()
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

        #img_hr_left, img_hr_right, img_lr_left, img_lr_right = augumentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
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
        hr_image_left  = ToTensor()(hr_image_left)
        hr_image_right = ToTensor()(hr_image_right)
        lr_image_left  = ToTensor()(lr_image_left)
        lr_image_right = ToTensor()(lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
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

class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def cal_psnr(img1, img2):
    img1_np = np.array(img1.cpu())
    img2_np = np.array(img2.cpu())
    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def cal_ssim(img1, img2):
    img1_np = np.array(img1.squeeze().cpu()).transpose(1,2,0)
    img2_np = np.array(img2.squeeze().cpu()).transpose(1,2,0)
    return metrics.structural_similarity(img1_np, img2_np, multichannel=True, data_range=1.0)

def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

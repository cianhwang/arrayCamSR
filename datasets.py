import h5py
import numpy as np
from torch.utils.data import Dataset


import cv2
from scipy.special import j1
import glob
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.measure import block_reduce

from kernels import Kernels
import argparse

from dual_align import align_image, load_exr

xxs = np.load('xxs_BLENDER2K_DUAL_x2_downsample.npy').astype(np.int16)
yys = np.load('yys_BLENDER2K_DUAL_x2_downsample.npy').astype(np.int16)
        
class Camera:
    def __init__(self, lam=0.633e-6, f_num=16, n_photon=1e2, p=6.6e-6, unit=0.1e-6, kernel='jinc',scale=1):
        k_r = int(5*f_num*lam/unit)
        H = Kernels(lam=lam, f_num=f_num, unit=unit, k_r = k_r).select_kernel(kernel)
        block_size = int(p/unit)//2*2+1
        size = ((k_r*2+1)//block_size//2*2-1)*block_size
        bleed = (k_r*2+1-size)//2
        H_crop = H[bleed:-bleed, bleed:-bleed]
        H = block_reduce(H_crop, block_size=(block_size, block_size), func=np.sum)
        H = H/H.sum()
        assert H.shape[0]%2 == 1
        self.k_r = H.shape[0]//2
        
        self.scale = scale
        self.f_num = f_num
        self.kernel = kernel
        self.n_photon = n_photon
        self.H_t = torch.from_numpy(H).float().unsqueeze(0).unsqueeze(0)
        
    def jinc(self,rho):
        f = j1(rho)/rho
        f[rho == 0] = 0.5
        return f
    
    def forward(self, img):
        img_t = torch.from_numpy(img).float()
        img_t = F.pad(img_t, (self.k_r, self.k_r, self.k_r, self.k_r)).unsqueeze(0).unsqueeze(0)
        blurry_img = F.conv2d(img_t, self.H_t).squeeze().numpy()
        img_downsample = block_reduce(blurry_img, block_size=(self.scale, self.scale), func=np.sum)
        noisy_img = np.random.poisson(img_downsample * self.n_photon)
        img_sensor = (noisy_img).astype(np.float)
        norm_img_sensor = img_sensor/(self.scale**2*self.n_photon)
        return norm_img_sensor
        
class Train_or_Evalset_DUAL(Dataset):
    def __init__(self, args, patch_size = 32, is_train=True):
        if is_train:
            file_a, file_b = args.train_file.split(',')
            self.transform = transforms.Compose([
                    transforms.RandomCrop(patch_size+256)
            ])
        else:
            file_a, file_b = args.eval_file.split(',')
            self.transform = transforms.Compose([
                    transforms.CenterCrop(patch_size+256)
                ])
        
        self.files_a = sorted(glob.glob(file_a + '/*.exr'))
        self.files_b = sorted(glob.glob(file_b + '/*.exr'))
        self.camera = []
        self.scale = args.scale
        assert len(args.kernel.split(","))==1 and len(args.f_num.split(","))==1 and args.num_channels==1
        assert self.scale == 2
        for kernel in args.kernel.split(","):
            for f_num in args.f_num.split(","):
                self.camera.append(Camera(lam=args.lam,f_num=float(f_num), n_photon=args.n_photon, p=args.p, kernel=kernel, scale=args.scale))
    
    def __getitem__(self, idx):
        img1 = load_exr(self.files_a[idx])/4.0
        img2 = load_exr(self.files_b[idx])/4.0
        h, w = img1.shape
        gt = torch.from_numpy(img1)
        index_map = torch.arange(torch.numel(gt)).view(h, w)
        gt = torch.stack([gt, index_map], dim=0)
        gt = self.transform(gt).numpy() #gt[:, 234:234+32+256, 921:921+32+256].numpy()#
        index_map = gt[1].flatten()
        x_l,y_l = int(index_map[0])//w, int(index_map[0])%w
        x_r,y_r = int(index_map[-1])//w+1, int(index_map[-1])%w+1

        img1s = []
        img2s = []
        for i, camera in enumerate(self.camera):
            img1s.append(camera.forward(gt[0]))
            img2s.append(camera.forward(img2))
            
        img1s = np.stack(img1s,axis=0)
        img2s = np.stack(img2s,axis=0)
        
        edge = 128 
        gt_t = torch.from_numpy(gt[0]).float().unsqueeze(0)[:, edge:-edge, edge:-edge]
        img_1t =  torch.from_numpy(img1s).float()[:, edge//self.scale:-edge//self.scale, edge//self.scale:-edge//self.scale]
        img_2t =  torch.from_numpy(img2s).float()
        x_l_adj = (x_l//self.scale+edge//self.scale)
        x_r_adj = (x_r//self.scale-edge//self.scale)
        y_l_adj = (y_l//self.scale+edge//self.scale)
        y_r_adj = (y_r//self.scale-edge//self.scale)

        pos_mat = (xxs[x_l_adj:x_r_adj, y_l_adj:y_r_adj], 
                   yys[x_l_adj:x_r_adj, y_l_adj:y_r_adj])
        return img_1t, img_2t, gt_t, pos_mat

    def __len__(self):
        return len(self.files_a)
    
def print_stat(narray, narray_name = "array"):
    print(narray_name, "shape: ", narray.shape, "dtype:", narray.dtype)
    arr = narray.flatten()
    print(narray_name , "stat: max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--n_photon', type=int, default=1000)
    parser.add_argument('--f_num', type=str, default="48")
    parser.add_argument('--kernel', type=str, default="jinc")
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.633e-6)
    parser.add_argument('--p', type=float, default=6.6e-6)
    
    args = parser.parse_args()
    trainset = Train_or_Evalset_DUAL(args, 32, True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    img_1t, img_2t, gt_t, pos_mat = dataiter.next()
    print_stat(img_1t, "images1")
    print_stat(img_2t, "images2")
    print_stat(gt_t, "gts")

    plt.imshow(img_1t[0, 0].numpy(), cmap='gray')
    plt.show()
    plt.imshow(img_2t[0, 0].numpy(), cmap='gray')
    plt.show()
    plt.imshow(gt_t[0, 0].numpy(), cmap='gray')
    plt.show()

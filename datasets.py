import h5py
import numpy as np
from torch.utils.data import Dataset
import time

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

# xxs = np.load('xxs_BLENDER2K_DUAL_x2_downsample.npy').astype(np.int16)
# yys = np.load('yys_BLENDER2K_DUAL_x2_downsample.npy').astype(np.int16)

def rot_mat(t):
    theta = np.arccos(t[2]/np.linalg.norm(t))
    phi = np.arctan2(t[1], t[0])
    R = np.array([[np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(phi), np.cos(phi), 0],
               [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]])
    return R

def tx_mat(t):
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])

t1 = np.array([7, -7, 5])
R1 = rot_mat(t1)
t2 = np.array([7.5, -6.5, 5])
R2 = rot_mat(t2)
R = R2.dot(R1.T)
tx = tx_mat(R1.dot(t2-t1))
E = R.dot(tx)

res_w2, res_h2 = 1024,768
f2 = 64. # (mm)
sensor_width = 36.
pix_size = sensor_width/res_w2

image_coor2 = np.meshgrid(np.arange(res_h2), np.arange(res_w2))
image_coor2 = [image_coor2[0].transpose(), image_coor2[1].transpose()]
image_coor2_in_mm = [image_coor2[0] * pix_size - res_h2//2 * pix_size + 0.5 * pix_size, 
            image_coor2[1] * pix_size - res_w2//2 * pix_size + 0.5 * pix_size]
cam_coor2 = np.stack([image_coor2_in_mm[0], 
            image_coor2_in_mm[1],
            -f2*np.ones_like(image_coor2_in_mm[0])], axis=2)

cam_coor = cam_coor2.copy()

#E = np.array([[-3.86608326e-04,  3.86608326e-04,  5.34352595e-01],
#        [ 3.35060549e-04, -3.35060549e-04, -4.63105583e-01],
#        [-4.99999738e-01,  4.99999738e-01, -7.23507525e-04]])

def calc_lam(points):
    x_l, x_r, y_l, y_r = points
    p = cam_coor[x_l:x_r, y_l:y_r].reshape(-1, 3).T
    lam = E.dot(p)
    return lam

def calc_pos_mat(points, lam):
    x_l, x_r, y_l, y_r = points
    dist = np.abs(cam_coor2[x_l:x_r, y_l:y_r].reshape(-1, 3).dot(lam))/np.expand_dims((lam[0, :]**2+lam[1, :]**2)**0.5, axis=0)
    k = max(x_r - x_l, y_r - y_l) #dist < pix_size * 1.414 : but k for each point will be different
    pps = np.argpartition(dist, k, axis=0)[:k].T
    xxs,yys = pps//(y_r - y_l), pps%(y_r - y_l)
    return xxs, yys

def load_exr(path):
    img = cv2.imread(path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.mean(img, axis=2)
    return img

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
    def __init__(self, args, patch_size = (32, 96), is_train=True):
        if is_train:
            file_a, file_b = args.train_file.split(',')
            self.transform = transforms.Compose([
                    transforms.RandomCrop(patch_size)
            ])
        else:
            file_a, file_b = args.eval_file.split(',')
            self.transform = transforms.Compose([
                    transforms.CenterCrop(patch_size)
                ])
        self.patch_size = patch_size
        self.files_a = sorted(glob.glob(file_a + '/*.png'))#sorted(glob.glob(file_a + '/*.exr'))
        self.files_b = sorted(glob.glob(file_b + '/*.png'))#sorted(glob.glob(file_b + '/*.exr'))
        self.camera = []
        self.scale = args.scale
        assert len(args.kernel.split(","))==1 and len(args.f_num.split(","))==1 and args.num_channels==1
        assert self.scale == 2
        for kernel in args.kernel.split(","):
            for f_num in args.f_num.split(","):
                self.camera.append(Camera(lam=args.lam,f_num=float(f_num), n_photon=args.n_photon, p=args.p, kernel=kernel, scale=args.scale))
    
    def __getitem__(self, idx):
        gt1 = cv2.imread(self.files_a[idx], 0)/255.0#load_exr(self.files_a[idx])/4.0
        gt2 = cv2.imread(self.files_b[idx], 0)/255.0#load_exr(self.files_b[idx])/4.0
        img1s, img2s = [], []
        for i, camera in enumerate(self.camera):
            img1s.append(camera.forward(gt1))
            img2s.append(camera.forward(gt2))
        img1s,img2s = np.stack(img1s,axis=0), np.stack(img2s,axis=0)
        _, h, w = img1s.shape
        
        edge = 128//self.scale
        index_map = np.arange(h*w).reshape(1, h, w)
        mixed = np.concatenate([img1s, img2s, index_map], axis=0)
        assert mixed.shape[0]==3
        mixed = torch.from_numpy(mixed[..., edge:-edge, edge:-edge])
        mixed = self.transform(mixed)#mixed[:, 182:182+32, 123:123+96]#
        index_map = mixed[2].numpy().flatten()
        x_l,y_l = int(index_map[0])//w, int(index_map[0])%w
        x_r,y_r = int(index_map[-1])//w+1, int(index_map[-1])%w+1
        
        gt1_t = torch.from_numpy(gt1[x_l*self.scale:x_r*self.scale, y_l*self.scale:y_r*self.scale]).float().unsqueeze(0)
        gt2_t = torch.from_numpy(gt2[x_l*self.scale:x_r*self.scale, y_l*self.scale:y_r*self.scale]).float().unsqueeze(0)
        img1_t =  mixed[0:1].float()#.unsqueeze(0)
        img2_t =  mixed[1:2].float()#.unsqueeze(0)
        
        lam = calc_lam((x_l, x_r, y_l, y_r))
        pos_mat = calc_pos_mat((x_l, x_r, y_l, y_r), lam)
        
        return img1_t, img2_t, gt1_t, pos_mat

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
    parser.add_argument('--n_photon', type=int, default=100000)
    parser.add_argument('--f_num', type=str, default="9")
    parser.add_argument('--kernel', type=str, default="jinc")
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.55e-6)
    parser.add_argument('--p', type=float, default=1.5e-6)
    
    args = parser.parse_args()
#     trainset = Train_or_Evalset_DUAL(args, (32, 96), True)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
    evalset = Train_or_Evalset_DUAL(args, (32, 96), False)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1,
                                          shuffle=False, num_workers=1)
    dataiter = iter(evalloader)
    img_1t, img_2t, gt_t, pos_mat = dataiter.next()
    print_stat(img_1t, "images1")
    print_stat(img_2t, "images2")
    print_stat(gt_t, "gts")
    
    xxs, yys = pos_mat
    print(xxs.shape)
    
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(img_1t[0, 0].numpy(), cmap='gray')
    ax[0][0].plot(80, 28, marker='*', color='red')
    ax[0][1].imshow(img_2t[0, 0].numpy(), cmap='gray')
    ax[0][1].plot(sorted(yys[0, 28*96+80]), sorted(xxs[0, 28*96+80]), color='red')
    ax[1][0].imshow(gt_t[0, 0].numpy(), cmap='gray')
    plt.show()
    
###### test pos mat from patch of A -> local of B
#     start = time.time()
#     lam = calc_lam((800//2,580//2), 16)
#     xxs, yys = calc_pos_mat((800//2,580//2), lam, 128)
#     print("xxs.dtype", xxs.dtype)
#     print("elapsed time:", time.time()-start)
#     pos_mat = np.zeros((256, 256))
#     pos_mat[xxs[:, 0], yys[:, 0]] = 1
#     plt.imshow(pos_mat, cmap='gray')
#     plt.show()

    

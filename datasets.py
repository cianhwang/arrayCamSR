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
            f2*np.ones_like(image_coor2_in_mm[0])], axis=2)

cam_coor = cam_coor2.copy()

E = np.array([[-3.86608326e-04,  3.86608326e-04,  5.34352595e-01],
       [ 3.35060549e-04, -3.35060549e-04, -4.63105583e-01],
       [-4.99999738e-01,  4.99999738e-01, -7.23507525e-04]])

def calc_lam(point, patch_size=16):
    point_x, point_y = point
    p = cam_coor[point_x:point_x+patch_size, point_y:point_y+patch_size].reshape(-1, 3).T
    lam = E.dot(p)
    return lam

def calc_pos_mat(point, lam, disparity=64):
    point_x, point_y = point
    dist = np.abs(cam_coor2[point_x-disparity:point_x+disparity, point_y-disparity:point_y+disparity].reshape(-1, 3).dot(lam))/np.expand_dims((lam[0, :]**2+lam[1, :]**2)**0.5, axis=0)
    pps = np.argpartition(dist, 4*disparity-1, axis=0)[:4*disparity-1].T
    xxs,yys = pps//(2*disparity), pps%(2*disparity)
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
        img1 = cv2.imread(self.files_a[idx], 0)/255.0#load_exr(self.files_a[idx])/4.0
        img2 = cv2.imread(self.files_b[idx], 0)/255.0#load_exr(self.files_b[idx])/4.0
        h, w = img1.shape
        edge = 128 
        disparity = 128
        
        gt = torch.from_numpy(img1)
        index_map = torch.arange(torch.numel(gt)).view(h, w)
        gt = torch.stack([gt, index_map], dim=0)[:, disparity:-disparity, disparity:-disparity]
        gt = self.transform(gt).numpy() # gt[:, 800-128:800-128+32+256, 580-128:580-128+32+256].numpy() #
        index_map = gt[1].flatten()
        x_l,y_l = int(index_map[0])//w, int(index_map[0])%w
        x_r,y_r = int(index_map[-1])//w+1, int(index_map[-1])%w+1
        
        x_c, y_c = x_l + edge, y_l + edge

        img1s = []
        img2s = []
        for i, camera in enumerate(self.camera):
            img1s.append(camera.forward(gt[0]))
            img2s.append(camera.forward(img2[x_c-disparity-edge:x_c+disparity+edge, y_c-disparity-edge:y_c+disparity+edge]))
            
        img1s = np.stack(img1s,axis=0)
        img2s = np.stack(img2s,axis=0)
        

        gt_t = torch.from_numpy(gt[0]).float().unsqueeze(0)[:, edge:-edge, edge:-edge]
        img_1t =  torch.from_numpy(img1s).float()[:, edge//self.scale:-edge//self.scale, edge//self.scale:-edge//self.scale]
        img_2t =  torch.from_numpy(img2s).float()[:, edge//self.scale:-edge//self.scale, edge//self.scale:-edge//self.scale]
        x_l_adj = (x_l//self.scale+edge//self.scale)
        x_r_adj = (x_r//self.scale-edge//self.scale)
        y_l_adj = (y_l//self.scale+edge//self.scale)
        y_r_adj = (y_r//self.scale-edge//self.scale)

#         pos_mat = (xxs[x_l_adj:x_r_adj, y_l_adj:y_r_adj], 
#                    yys[x_l_adj:x_r_adj, y_l_adj:y_r_adj])
        lam = calc_lam((x_l_adj, y_l_adj), self.patch_size//self.scale)
        pos_mat = calc_pos_mat((x_l_adj, y_l_adj), lam, disparity//self.scale)
        
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
    parser.add_argument('--n_photon', type=int, default=100000)
    parser.add_argument('--f_num', type=str, default="9")
    parser.add_argument('--kernel', type=str, default="jinc")
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.55e-6)
    parser.add_argument('--p', type=float, default=1.5e-6)
    
    args = parser.parse_args()
    trainset = Train_or_Evalset_DUAL(args, 32, True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    evalset = Train_or_Evalset_DUAL(args, 32, False)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1,
                                          shuffle=False, num_workers=1)
    dataiter = iter(evalloader)
    img_1t, img_2t, gt_t, pos_mat = dataiter.next()
    print_stat(img_1t, "images1")
    print_stat(img_2t, "images2")
    print_stat(gt_t, "gts")
    
    xxs, yys = pos_mat
    print(xxs.shape)

    plt.imshow(img_1t[0, 0].numpy(), cmap='gray')
    plt.show()
    plt.imshow(img_2t[0, 0].numpy(), cmap='gray')
    plt.plot(yys[0, 0], xxs[0, 0], color='red')
    plt.show()
    plt.imshow(gt_t[0, 0].numpy(), cmap='gray')
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

    

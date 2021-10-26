from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
from datasets import Train_or_Evalset_DUAL
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='BLENDER2K_valid_DUAL')
    #parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--eval-file', type=str, required=True)
    #parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--n_photon', type=int, default=1000)
    parser.add_argument('--f_num', type=str, default="48")
    parser.add_argument('--kernel', type=str, default="jinc")
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.633e-6)
    parser.add_argument('--p', type=float, default=6.6e-6)
    return parser.parse_args()

def tensor2uint8(img):
    ## 1x1xhxw
    img = torch.clamp(img, 0, 1)
    img_uint8 = (img[0, 0].cpu().numpy()*255.0).astype(np.uint8)
    return img_uint8

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale).to(cfg.device)
    cudnn.benchmark = True
    pretrained_dict = torch.load('log/x' + str(cfg.scale) + '/PASSRnet_x' + str(cfg.scale) + '.pth')['state_dict']
    net.load_state_dict(pretrained_dict)

    psnr_list = []

    with torch.no_grad():
        for idx_iter, (LR_left, LR_right, HR_left, Pos) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            scene_name = str(idx_iter)#test_loader.dataset.file_list[idx_iter]
            
            SR_left, M_right_to_left = net(LR_left, LR_right, is_training=0, Pos=Pos)
            
            psnr_list.append(cal_psnr(HR_left.clamp(0, 1).data.cpu(), SR_left.clamp(0, 1).data.cpu()))

            ## save results
            if not os.path.exists('results/'+cfg.dataset):
                os.mkdir('results/'+cfg.dataset)
            if not os.path.exists('results/'+cfg.dataset+'/'+scene_name):
                os.mkdir('results/'+cfg.dataset+'/'+scene_name)
#             SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
#             SR_left_img.save('results/'+cfg.dataset+'/'+scene_name+'/img_0.png')
            cv2.imwrite('results/'+cfg.dataset+'/'+scene_name+'/left_sr.png', tensor2uint8(SR_left))
            cv2.imwrite('results/'+cfg.dataset+'/'+scene_name+'/left_lr.png', tensor2uint8(LR_left))
            cv2.imwrite('results/'+cfg.dataset+'/'+scene_name+'/right_lr.png', tensor2uint8(LR_right))
            cv2.imwrite('results/'+cfg.dataset+'/'+scene_name+'/left_hr.png', tensor2uint8(HR_left))
            np.save('results/'+cfg.dataset+'/'+scene_name+'/M_right_to_left.npy', M_right_to_left.cpu().numpy())

        ## print results
        print(cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))

def main(cfg):
    test_set = Train_or_Evalset_DUAL(cfg, 16, False)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

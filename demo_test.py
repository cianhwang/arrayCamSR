from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
from datasets import TestSetLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='KITTI2012')
    parser.add_argument('--model_dir', type=str, default='log')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    net = nn.DataParallel(net)
    net.eval()
    cudnn.benchmark = True
    pretrained_dict = torch.load(cfg.model_dir+'/best.pth.tar')
    net.load_state_dict(pretrained_dict['state_dict'])

    psnr_list = AverageMeter()
    ssim_list = AverageMeter()

    with torch.no_grad():
        for idx_iter, (HR_left, HR_right, LR_left, LR_right, Pos) in enumerate(test_loader):
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device),Variable(HR_right).to(cfg.device),Variable(LR_left).to(cfg.device),Variable(LR_right).to(cfg.device)
            scene_name = test_loader.dataset.file_list[idx_iter]
            if HR_left.size(2)+HR_left.size(3) > 3000:
                continue

#             LR_left, LR_right, HR_left, HR_right = LR_right, LR_left, HR_right, HR_left
            
            SR_left, M_right_to_left = net(LR_left, LR_right, is_training=0, Pos=Pos) # 1st param: anchor; 2nd: alternative
            SR_left = torch.clamp(SR_left, 0, 1)

            psnr_list.update(cal_psnr(HR_left, SR_left))
            ssim_list.update(cal_ssim(HR_left, SR_left))

#             ## save results
#             if not os.path.exists('results/'+cfg.dataset+'/'+scene_name):
#                 os.makedirs('results/'+cfg.dataset+'/'+scene_name)
            if not os.path.exists('results/'+cfg.dataset):
                os.makedirs('results/'+cfg.dataset)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('results/'+cfg.dataset+'/'+scene_name+'_L.png')
#             SR_left_img.save('results/'+cfg.dataset+'/'+scene_name+'/.png')
#             LR_left = transforms.ToPILImage()(torch.squeeze(LR_left.data.cpu(), 0))
#             LR_left.save('results/'+cfg.dataset+'/'+scene_name+'/left_lr.png')
#             LR_right = transforms.ToPILImage()(torch.squeeze(LR_right.data.cpu(), 0))
#             LR_right.save('results/'+cfg.dataset+'/'+scene_name+'/right_lr.png')
#             HR_left = transforms.ToPILImage()(torch.squeeze(HR_left.data.cpu(), 0))
#             HR_left.save('results/'+cfg.dataset+'/'+scene_name+'/left_hr.png')
#             np.save('results/'+cfg.dataset+'/'+scene_name+'/M_right_to_left.npy', M_right_to_left.cpu().numpy())
#             np.save('results/'+cfg.dataset+'/'+scene_name+'/xxs.npy', Pos[0])
#             np.save('results/'+cfg.dataset+'/'+scene_name+'/yys.npy', Pos[1])
#             break

    ## print results
    print('mean psnr: {:.2f} | ssim: {:.4f}'.format(psnr_list.avg, ssim_list.avg))
        

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

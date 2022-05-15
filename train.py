from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datasets import TrainSetLoader, ValidSetLoader

def trainer(cfg):
    
    outputs_dir = os.path.join(cfg.outputs_dir, f'x{cfg.scale_factor}')
    print('[*] Saving outputs to {}'.format(outputs_dir))
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    logs_dir = os.path.join(cfg.logs_dir, f'x{cfg.scale_factor}')
    print('[*] Saving tensorboard logs to {}'.format(logs_dir))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    writer = SummaryWriter(logs_dir)
    
    net = PASSRnet(cfg.scale_factor)
    net = nn.DataParallel(net).to(cfg.device)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    valid_set = ValidSetLoader(dataset_dir=cfg.validset_dir, cfg=cfg)
    valid_loader = DataLoader(dataset=valid_set, num_workers=1, batch_size=1, shuffle=False)

    best_psnr = 0.0
    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        train(idx_epoch, net, train_loader, criterion_mse, optimizer, cfg.device, writer)
        val_psnr = valid(idx_epoch, net, valid_loader, cfg.device, writer)

        if val_psnr >= best_psnr:
            print("[*]")
            best_psnr = val_psnr
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'psnr': val_psnr,
            }, save_path = outputs_dir, filename='best.pth.tar')
            
def train(epoch, model, train_loader, criterion, optimizer, device, writer):
    model.train()
    loss_epoch = AverageMeter()
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'epoch {epoch+1}')
        for idx_iter, (HR_left, _, LR_left, LR_right, Pos) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right  = HR_left.to(device), LR_left.to(device), LR_right.to(device)

            SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = model(LR_left, LR_right, is_training=1, Pos=Pos)

            ### loss_SR
            loss = criterion(SR_left, HR_left)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.update(loss.item())
            t.set_postfix(loss='{:.6f}'.format(loss_epoch.avg))
            t.update(1)
    
    writer.add_scalar('Stats/training_loss', loss_epoch.avg, epoch+1)

def valid(epoch, model, valid_loader, device, writer):
    model.eval()
    psnr_epoch = AverageMeter()
    
    with tqdm(total=len(valid_loader)) as t:
        t.set_description('validate')
        for idx_iter, (HR_left, _, LR_left, LR_right, Pos) in enumerate(valid_loader):
            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right = HR_left.to(device), LR_left.to(device), LR_right.to(device)

            with torch.no_grad():
                SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                (V_left_to_right, V_right_to_left) = model(LR_left, LR_right, is_training=1, Pos=Pos)

            psnr_epoch.update(cal_psnr(HR_left.data.cpu(), SR_left.data.cpu()))

            t.set_postfix(psnr='{:.2f}'.format(psnr_epoch.avg))
            t.update(1)
            
    writer.add_scalar('Stats/valid_psnr', psnr_epoch.avg, epoch+1)

    return psnr_epoch.avg

def main(cfg):
    trainer(cfg)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/groups/djbrady/Qian/blender_patches_corrected')
    parser.add_argument('--validset_dir', type=str, default='/groups/djbrady/Qian/blender_patches_valid_corrected')
    parser.add_argument('--outputs_dir', type=str, default='log/')
    parser.add_argument('--logs_dir', type=str, default='ckpt/')
    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)


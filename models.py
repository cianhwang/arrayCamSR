import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

xxs = np.load('xxs_flickr1024.npy')
yys = np.load('yys_flickr1024.npy')
Pos = (xxs, yys)

class PASSRnet(nn.Module):
    def __init__(self, upscale_factor):
        super(PASSRnet, self).__init__()
        ### feature extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        ### paralax attention
        self.pam = PAM(64)
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_right, is_training):
        ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        if is_training == 1:
            ### parallax attention
            buffer, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                   (V_left_to_right, V_right_to_left)
        if is_training == 0:
            ### parallax attention
            buffer = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3
    
class fePAM(nn.Module):
    def __init__(self):
        super(fePAM, self).__init__()
        self.softmax = nn.Softmax(-1)
    def forward(self, Q, S, R, Pos):
        ## Q, S, R: n_batch x C x H x W
        ## Pos: Pos_x: nparray, H x W x k; Pos_y: nparray, H x W x k
        n_batch, n_channel, H, W = Q.size()
        Pos_x, Pos_y = Pos
        Pos_x, Pos_y = Pos_x.flatten(), Pos_y.flatten() #(H*W*k, )
        Key = S[:, :, Pos_x, Pos_y] #n_batch x C x H*W*k
        Key = Key.view(n_batch, n_channel, H*W, -1).permute(0, 2, 1, 3) #n_batch x H*W x C x k
        Q = Q.permute(0, 2, 3, 1).view(n_batch, H*W, n_channel).unsqueeze(2) # n_batch x H*W x 1 x C
        score = torch.matmul(Q, Key) #n_batch x H*W x 1 x k
        M_right_to_left = self.softmax(score) #n_batch x H*W x 1 x k

        Value = R[:, :, Pos_x, Pos_y] #n_batch x C x H*W*k
        Value = Value.view(n_batch, n_channel, H*W, -1).permute(0, 2, 3, 1) #n_batch x H*W x k x C
        buffer = torch.matmul(M_right_to_left, Value) #n_batch x H*W x 1 x C
        buffer = buffer.squeeze().view(n_batch, H, W, n_channel).permute(0, 3, 1, 2)

        return buffer

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fe_pam = fePAM()
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)# + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_left, x_right, is_training):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)

        ### M_{right_to_left}
        Q = self.b1(buffer_left)#.permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right)#.permute(0, 2, 1, 3)                                               # B * H * C * W
#         score = torch.bmm(Q.contiguous().view(-1, w, c),
#                           S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
#         M_right_to_left = self.softmax(score)

        ### fusion
        R = self.b3(x_right)
#         buffer = R.permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
#         buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        buffer = self.fe_pam(Q, S, R, Pos)
        out = self.fusion(torch.cat((buffer, x_left), 1))#, V_left_to_right), 1))

        ## output
        if is_training == 1:
            return out, (None, None), (None, None), (None, None)#\
               #(M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               #(M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w)), \
               #(V_left_to_right, V_right_to_left)
        if is_training == 0:
            return out

def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = 1-mask.float()
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)

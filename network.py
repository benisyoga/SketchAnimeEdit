import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from network_module import *

def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """
    ネットワークの重みを初期化
    net (network)       初期化するネットワーク
    init_type (str)     初期化する方法(normal, xavier, kaiming, orthogonal)
    init_gain (float)   初期化のパラメータ
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    net.apply(init_func)

#------------------------------
#         マスク推定器
#------------------------------
'''
入力:  画像[B,3,H,W] + スケッチ[B,1,H,W]
              ↓
出力:マスク[B,1,H,W](+ 画像[B,3,H,W])
'''
class MaskEstimator(nn.Module):
    def __init__(self, opt):
        super(MaskEstimator, self).__init__()
        self.mask_encoder = nn.Sequential(
            # エンコーダ
            GatedConv2d(opt.in_channels + 0,     opt.latent_channels,     5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), 
            GatedConv2d(opt.latent_channels,     opt.latent_channels * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #downsample /2
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #downsample /2
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )

        self.mask_bottleneck = nn.Sequential(
            # ボトルネック
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2,  dilation = 2,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4,  dilation = 4,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8,  dilation = 8,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )

        self.mask_decoder1 = nn.Sequential(
            # デコーダ
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #upsample *2
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels,     3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #upsample *2
            GatedConv2d(opt.latent_channels,     opt.latent_channels//2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2,  1,                       3, 1, 1, pad_type = opt.pad_type, activation = 'none',         norm = opt.norm),
            nn.Sigmoid()
        )

        self.mask_decoder2 = nn.Sequential(
            # デコーダ
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #upsample *2
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels,     3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm), #upsample *2
            GatedConv2d(opt.latent_channels,     opt.latent_channels//2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2,  opt.out_channels,        3, 1, 1, pad_type = opt.pad_type, activation = 'none',         norm = opt.norm),
            nn.Tanh()
        )

        #U-Net
        self.TCB1 = TwoConvBlock(opt.in_channels + 1,      opt.latent_channels,      3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB2 = TwoConvBlock(opt.latent_channels,      opt.latent_channels * 2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB3 = TwoConvBlock(opt.latent_channels * 2,  opt.latent_channels * 4,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB4 = TwoConvBlock(opt.latent_channels * 4,  opt.latent_channels * 8,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB5 = TwoConvBlock(opt.latent_channels * 8,  opt.latent_channels * 16, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')

        self.TCB6 = TwoConvBlock(opt.latent_channels * 16, opt.latent_channels * 8,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB7 = TwoConvBlock(opt.latent_channels * 8,  opt.latent_channels * 4,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB8 = TwoConvBlock(opt.latent_channels * 4,  opt.latent_channels * 2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB9 = TwoConvBlock(opt.latent_channels * 2,  opt.latent_channels,      3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.conv = nn.Conv2d(opt.latent_channels, 1, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(opt.latent_channels * 16, opt.latent_channels * 8) 
        self.UC2 = UpConv(opt.latent_channels * 8,  opt.latent_channels * 4) 
        self.UC3 = UpConv(opt.latent_channels * 4,  opt.latent_channels * 2) 
        self.UC4 = UpConv(opt.latent_channels * 2,  opt.latent_channels)

        self.TCB6_mg  = TwoConvBlock(opt.latent_channels * 16,  opt.latent_channels * 8,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB7_mg  = TwoConvBlock(opt.latent_channels * 8,   opt.latent_channels * 4,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB8_mg  = TwoConvBlock(opt.latent_channels * 4,   opt.latent_channels * 2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.TCB9_mg  = TwoConvBlock(opt.latent_channels * 2,   opt.latent_channels    ,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = 'gn')
        self.conv_mg = nn.Conv2d(opt.latent_channels, opt.out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.UC1_mg = UpConv(opt.latent_channels * 16, opt.latent_channels * 8) 
        self.UC2_mg = UpConv(opt.latent_channels * 8,  opt.latent_channels * 4) 
        self.UC3_mg = UpConv(opt.latent_channels * 4,  opt.latent_channels * 2) 
        self.UC4_mg = UpConv(opt.latent_channels * 2,  opt.latent_channels)
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, img, sketch, mask):
        '''
          img :入力画像
        sketch:入力スケッチ
        '''

        '''マスク推定'''
        #'''
        x = torch.cat((img, mask, sketch), dim=1)

        x1 = self.TCB1(x)
        p1 = self.maxpool(x1)
        x2 = self.TCB2(p1)
        p2 = self.maxpool(x2)
        x3 = self.TCB3(p2)
        p3 = self.maxpool(x3)
        x4 = self.TCB4(p3)
        p4 = self.maxpool(x4)
        x5 = self.TCB5(p4)

        p6 = self.UC1(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.TCB6(x6)
        p7 = self.UC2(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.TCB7(x7)
        p8 = self.UC3(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.TCB8(x8)
        p9 = self.UC4(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.TCB9(x9)

        mask_out = self.conv(x9)
        mask_out = self.sig(mask_out)

        p6_mg = self.UC1_mg(x5)
        x6_mg = torch.cat([p6_mg, x4], dim=1)
        x6_mg = self.TCB6_mg(x6_mg)
        p7_mg = self.UC2_mg(x6_mg)
        x7_mg = torch.cat([p7_mg, x3], dim=1)
        x7_mg = self.TCB7_mg(x7_mg)
        p8_mg = self.UC3_mg(x7_mg)
        x8_mg = torch.cat([p8_mg, x2], dim=1)
        x8_mg = self.TCB8_mg(x8_mg)
        p9_mg = self.UC4_mg(x8_mg)
        x9_mg = torch.cat([p9_mg, x1], dim=1)
        x9_mg = self.TCB9_mg(x9_mg)

        mg_img_out = self.conv_mg(x9_mg)
        mg_img_out = self.tanh(mg_img_out)
        #'''

        '''
        mask_in = torch.cat((img, sketch), dim=1)

        mask_nw = self.mask_encoder(mask_in)
        mask_nw = self.mask_bottleneck(mask_nw)

        mask_out = self.mask_decoder1(mask_nw)
        mask_out = F.interpolate(mask_out, (img.shape[2], img.shape[3]))

        mg_img_out = self.mask_decoder2(mask_nw)
        mg_img_out = F.interpolate(mg_img_out, (img.shape[2], img.shape[3]))
        #'''

        return mask_out, mg_img_out

#------------------------------
#            生成器
#------------------------------
'''
入力:  画像[B,3,H,W] + スケッチ[B,1,H,W]
              ↓
中間:マスク[B,1,H,W](+ 画像[B,3,H,W])
              ↓
出力:  画像[B,3,H,W]
'''
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        '''生成器１段階目'''
        self.style_encoder = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels + 1,     opt.latent_channels,     5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )
        self.style_bottleneck = nn.Sequential(
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2,  dilation = 2,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4,  dilation = 4,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8,  dilation = 8,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )
        self.coarse_encoder = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels + 1,     opt.latent_channels,     5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )
        self.coarse_bottleneck = nn.Sequential(
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2,  dilation = 2,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4,  dilation = 4,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8,  dilation = 8,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
        )
        self.coarse_decoder = nn.Sequential(
            # decoder
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels,     3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels//2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2,  opt.out_channels,        3, 1, 1, pad_type = opt.pad_type, activation = 'none',         norm = opt.norm),
            nn.Tanh()
        )
        '''生成器２段階目'''
        self.refine_conv = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels + 1,     opt.latent_channels,     5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels,     3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2,  dilation = 2,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4,  dilation = 4,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8,  dilation = 8,  pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(opt.in_channels + 1,     opt.latent_channels,     5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels,     3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = 'relu',         norm = opt.norm)
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels,   3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels,     opt.latent_channels//2,  3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2,  opt.out_channels,        3, 1, 1, pad_type = opt.pad_type, activation = 'none',         norm = opt.norm),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,fuse=True)

        
    def forward(self, img, mask, sketch):
        '''
          img :入力画像
        sketch:入力スケッチ
        '''

        '''生成１段階目'''
        noise = torch.rand(img.shape[0], 3, img.shape[2], img.shape[3]).cuda()
        first_masked_img = img * (1 - mask) + noise * mask
        first_unmasked_img = img * mask

        first_in1 = torch.cat((first_masked_img, mask, sketch), dim=1)      #  in: [B, 5, H, W]
        first_in2 = torch.cat((first_unmasked_img, mask, sketch), dim=1)

        first_out1 = self.coarse_encoder(first_in1)                         # out: [B, 3, H, W]
        first_out1 = self.coarse_bottleneck(first_out1)

        first_out2 = self.style_encoder(first_in2)
        first_out2 = self.style_bottleneck(first_out2)

        _,_,hs,ws = first_out2.shape
        first_out2 = F.max_pool2d(first_out2, kernel_size=(hs, ws))
        first_out2 = F.interpolate(first_out2, (hs,ws), mode='nearest')

        first_out = torch.cat((first_out1, first_out2), dim=1)
        first_out = self.coarse_decoder(first_out)
        first_out = F.interpolate(first_out, (img.shape[2], img.shape[3]))

        #return first_out

        second_masked_img = img * (1 - mask) + first_out * mask

        '''生成２段階目'''
        # Refinement
        second_in = torch.cat([second_masked_img, mask, sketch], dim=1)
        # encode
        refine_conv = self.refine_conv(second_in)     
        refine_atten = self.refine_atten_1(second_in)
        # bottleneck
        mask_s = F.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        # decode
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = F.interpolate(second_out, (img.shape[2], img.shape[3]))

        return first_out, second_out

#------------------------------
#            推定器
#------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.block1 = Conv2dLayer(opt.in_channels,         opt.latent_channels,     7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True) #downsample /2
        self.block2 = Conv2dLayer(opt.latent_channels,     opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True) #downsample /2
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True) #downsample /2
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True) #downsample /2
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True) #downsample /2
        self.block6 = Conv2dLayer(opt.latent_channels * 4,                       1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none',         norm = 'none',   sn = True) #downsample /2
        
    def forward(self, img, sketch):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, sketch), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x
    
# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x
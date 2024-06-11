import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import network
import test_dataset
import utils


def tester(opt):
    
    # Save the model if pre_train == True
    def load_model(net, epoch, opt, type):
        if type == 'G':
            model_name = 'MG_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'D':
            model_name = 'Di_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'E':
            model_name = 'Es_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)

        model_name = os.path.join(path_folder, model_name)
        pretrained_dict = torch.load(model_name)
        #restored_ckpt = {}
        #for k,v in pretrained_dict.items():
        #    restored_ckpt[k.replace('_orig_mod.', '')] = v
        net.load_state_dict(pretrained_dict, strict=False)

    def psnr(pred, target, pixel_max_cnt = 255):
        mse = torch.mul(target - pred, target - pred)
        rmse_avg = (torch.mean(mse).item()) ** 0.5
        p = 20 * np.log10(pixel_max_cnt / rmse_avg)
        return p

    def ssim(pred, target):
        pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        target = target[0]
        pred = pred[0]
        ssim = structural_similarity(target, pred, multichannel = True)
        return ssim

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    path_folder = opt.saved_path
    save_folder = opt.results_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Build networks
    net_E = utils.create_estimator(opt).eval()
    net_G = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model(net_E, opt.epoch, opt, type='E')
    load_model(net_G, opt.epoch, opt, type='G')
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    net_E = net_E.cuda()
    net_G = net_G.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = test_dataset.TestImg(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    L1 = nn.L1Loss()

    total_L1 = 0
    total_PSNR= 0
    total_SSIM= 0

    # Testing loop
    for batch_idx, (input, truth, Sketch, mask) in enumerate(dataloader):
        input = input.cuda()
        truth = truth.cuda()
        Sketch = Sketch.cuda()
        mask = mask.cuda()

        # Generator output
        with torch.no_grad():
            mask_out, _ = net_E(input, Sketch, mask)
            mask_out = (mask_out>0.5).float()
            first_out, second_out = net_G(input, mask_out, Sketch)

            # forward propagation
            first_out_wholeimg = input * (1 - mask_out) + first_out * mask_out        # in range [0, 1]
            second_out_wholeimg = input * (1 - mask_out) + second_out * mask_out      # in range [0, 1]

            noise = torch.rand(opt.batch_size, 3, opt.img_height, opt.img_width).cuda()
            masked_img = input * (1 - mask_out) + noise * mask_out
            mask_out = torch.cat((mask_out, mask_out, mask_out), 1)

            img_list = [truth, mask_out, masked_img, first_out, first_out_wholeimg, second_out, second_out_wholeimg]
            name_list = ['truth','mask_out', 'masked_img', 'first_out', 'first_out_whole', 'second_out', 'second_out_whole']
            utils.save_sample_png(sample_folder = opt.results_path, sample_name = '%d' % (batch_idx + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
            print('----------------------batch_idx%d' % (batch_idx + 1) + ' has been finished----------------------')

            score_L1 = L1(second_out_wholeimg, truth).item()
            score_PSNR = psnr(second_out_wholeimg, truth)
            score_SSIM = ssim(second_out_wholeimg, truth)
        

        total_L1 += score_L1
        total_PSNR += score_PSNR
        total_SSIM += score_SSIM

    print(total_L1/len(dataloader))
    print(total_PSNR/len(dataloader))
    print(total_SSIM/len(dataloader))
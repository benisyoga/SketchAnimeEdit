import os
import time
import random
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import itertools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator

import train_dataset
import test_dataset
import utils

'''
# batch_idx = ペア画像数 / バッチサイズ
for batch_idx, (img_A, img_B, Sketch_A, Sketch_B) in enumerate(train_loader):
    print(batch_idx)
    print('   img_A.shape:', img_A.shape)
    print('   img_B.shape:', img_B.shape)
    print('Sketch_A.shape:', Sketch_A.shape)
    print('Sketch_B.shape:', Sketch_B.shape)

    #print(img_A)
'''

def trainer(opt):
    #------------------------------
    #     パラメータの初期設定
    #------------------------------

    '''cudnn.benchmark:Trueで高速化'''
    cudnn.benchmark = opt.cudnn_benchmark

    '''
     ./save_paths:モデルが保存されるフォルダ
    ./save_images:画像が保存されるフォルダ
    ※存在しなかったら作る
    '''
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    '''
      M_Generator(MG):マスク推定ネットワーク
      I_Generator(IG):画像生成ネットワーク
    Discriminator(Di):真偽推定ネットワーク
    '''
    net_E = utils.create_estimator(opt)
    net_G = utils.create_generator(opt)
    net_D = utils.create_discriminator(opt)
    net_P = utils.create_perceptualnet()

    '''オプティマイザ'''
    optimizerE = torch.optim.Adam(net_E.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizerG = torch.optim.Adam(net_G.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizerD = torch.optim.Adam(net_D.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    pose_estimator = init_pose_estimator(opt.pose_config, opt.pose_checkpoint, device='cuda:0', cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True))))

    '''キーポイント推定'''
    def estimate_keypoints(img):
        img = img * 255

        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./kploss_test.jpg', img_copy)

        imagebox = [[0, 0, 256, 256]]

        pose_results = inference_topdown(pose_estimator, './kploss_test.jpg', imagebox)

        keypoints = pose_results[0].pred_instances.keypoints

        result = np.squeeze(keypoints)

        result = torch.from_numpy(result.astype(np.float32) / 256.0)
        #print(result)

        return result

    '''学習率を減少させる'''
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    '''モデルを保存する'''
    def save_model(net, epoch, opt, type):
        if type == 'G':
            model_name = 'MG_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'D':
            model_name = 'Di_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'E':
            model_name = 'Es_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)

        model_name = os.path.join(save_folder, model_name)
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_name)
            print('訓練済みモデルが保存されました。 epoch = %d' % (epoch))
            
    '''モデルを読み込む'''
    def load_model(net, epoch, opt, type):
        if type == 'G':
            model_name = 'MG_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'D':
            model_name = 'Di_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        elif type == 'E':
            model_name = 'Es_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)

        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(net_G, opt.resume_epoch, type='G_A2B')
        load_model(net_D, opt.resume_epoch, type='D_B')
        print('学習済みモデルが読み込まれました')

    '''GPUにデータを渡す'''
    net_E = net_E.cuda()
    net_G = net_G.cuda()
    net_D = net_D.cuda()
    net_P = net_P.cuda()

    #------------------------------
    #     データセットの初期設定
    #------------------------------

    '''データセットを作成する'''
    train_set = train_dataset.PairImgs(opt)
    test_set = test_dataset.TestImg(opt)

    '''作成したデータセットをデータローダ―に読み込ませる'''
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,  num_workers=opt.num_workers, pin_memory=True)
    test_loader =  DataLoader(test_set,  batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    #------------------------------
    #　           訓練
    #------------------------------

    '''テンソルの型指定(float)'''
    Tensor = torch.cuda.FloatTensor

    loss_train_L1 = []
    loss_train_GAN = []
    loss_train_BMR = []
    loss_train_P = []
    loss_train_KP = []
    loss_train_D = []

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    es_counter = 0
    es_patience = 50
    es_best_score = None

    '''訓練ループ'''
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (img_A, img_B, Sketch_A, Sketch_B, edge_A, edge_B, mask_A, mask_B) in enumerate(train_loader):
            flag =  random.random()

            loss_train_L1_epoch = 0
            loss_train_GAN_epoch = 0
            loss_train_BMR_epoch = 0
            loss_train_P_epoch = 0
            #loss_train_KP_epoch = 0
            loss_train_D_epoch = 0

            '''画像をGPUに'''
            img_A = img_A.cuda()
            img_B = img_B.cuda()
            Sketch_A = Sketch_A.cuda()
            Sketch_B = Sketch_B.cuda()
            edge_A = edge_A.cuda()
            edge_B = edge_B.cuda()
            mask_A = mask_A.cuda()
            mask_B = mask_B.cuda()

            '''GANの正解ラベル'''
            real = Tensor(np.ones((opt.batch_size, 1, opt.img_height//32, opt.img_width//32)))
            fake = Tensor(np.zeros((opt.batch_size, 1, opt.img_height//32, opt.img_width//32)))

            #------------------------------
            #　         Stage A
            #
            #      Im B + Sk A => Im A
            #------------------------------
            '''出力'''
            mask_out_B, mg_out_A = net_E(img_B, Sketch_A, mask_B)
            _, mg_out_B = net_E(img_A, Sketch_B, mask_A)

            #mask = mask_out_B.detach()

            #'''
            mask = mask_out_B

            if flag > epoch / opt.epochs:
                mask = mask.detach()
            else:
                mask = (mask>0.5).float().detach()
            #'''
            
            #first_out_A, second_out_A = net_G(img_B, mask_out_B.detach(), Sketch_A)
            first_out_A, second_out_A = net_G(img_B, mask, Sketch_A)

            '''識別器'''
            optimizerD.zero_grad()

            #second_out_wholeimg_A = img_B * (1 - mask_out_B) + second_out_A * mask_out_B
            second_out_wholeimg_A = img_B * (1 - mask) + second_out_A * mask

            fake_scalar = net_D(second_out_wholeimg_A.detach(), mask)
            true_scalar = net_D(img_A, mask)

            loss_fake = MSE(fake_scalar, fake)
            loss_true = MSE(true_scalar, real)

            lossD = 0.5 * (loss_fake + loss_true)
            loss_train_D_epoch += lossD.item()

            lossD.backward()
            optimizerD.step()

            '''画像生成器'''
            optimizerG.zero_grad()

            #second_out_wholeimg_A = img_B * (1 - mask_out_B.detach()) + second_out_A * mask_out_B.detach()
            second_out_wholeimg_A = img_B * (1 - mask.detach()) + second_out_A * mask.detach()

            img_featuremaps = net_P(img_A)
            second_out_featuremaps = net_P(second_out_A)

            # L1 Loss
            first_L1Loss  = L1(img_A, first_out_A)
            second_L1Loss = L1(img_A, second_out_A)
            L1Loss = first_L1Loss + second_L1Loss
            
            # GAN Loss
            fake_scalar = net_D(second_out_wholeimg_A, mask)
            GANLoss = 0.5*MSE(fake_scalar, real)

            # Perceptual Loss
            PLoss = L1(second_out_featuremaps, img_featuremaps)

            loss_train_L1_epoch += L1Loss.item()
            loss_train_GAN_epoch += GANLoss.item()
            loss_train_P_epoch += PLoss.item()
            

            # Compute losses
            loss_G = opt.lambda_l1 * L1Loss + opt.lambda_gan * GANLoss + opt.lambda_perceptual * PLoss
            loss_G.backward()

            optimizerG.step()

            '''マスク推定器'''
            optimizerE.zero_grad()

            mg_out_wholeimg_A = img_B * (1 - mask_out_B.detach()) + mg_out_A.detach() * mask_out_B
            mg_out_wholeimg_B = img_A * (1 - mask_out_B.detach()) + mg_out_B.detach() * mask_out_B

            second_out_wholeimg_A = img_B * (1 - mask_out_B.detach()) + second_out_A.detach() * mask_out_B

            #img_featuremaps = net_P(img_A)
            #second_out_featuremaps = net_P(second_out_wholeimg_A)

            noise = torch.rand(opt.batch_size, 3, opt.img_height, opt.img_width).cuda()
            #maskedimg_B = img_B * (1 - mask_out_B) + noise * mask_out_B
            maskedimg_B = img_B * (1 - mask) + noise * mask

            #true_kp = estimate_keypoints(img_A)
            #fake_kp = estimate_keypoints(second_out_wholeimg_A)

            # L1 Loss
            whole_L1Loss  = L1(img_A, second_out_wholeimg_A)
            L1Loss = whole_L1Loss

            # GAN Loss
            fake_scalar = net_D(second_out_wholeimg_A, mask)
            GANLoss = 0.5*MSE(fake_scalar, real)

            # BMR Loss
            BMRLoss_A1 = L1(img_A, mg_out_A)
            BMRLoss_B1 = L1(img_B, mg_out_B)
            BMRLoss_A2 = L1(img_A, mg_out_wholeimg_A)
            BMRLoss_B2 = L1(img_B, mg_out_wholeimg_B)
            BMRLoss = BMRLoss_A1 + BMRLoss_B1 + BMRLoss_A2 + BMRLoss_B2

            # Perceptual Loss
            #PLoss = L1(second_out_featuremaps, img_featuremaps)

            #kp loss
            #KpLoss = L1(true_kp, fake_kp)

            # Record losses
            loss_train_BMR_epoch += BMRLoss.item()
            #loss_train_KP_epoch += KpLoss.item()

            # Compute losses
            loss_E = opt.lambda_l1 * L1Loss + opt.lambda_gan * GANLoss + opt.lambda_BMR * BMRLoss
            loss_E.backward()

            optimizerE.step()

            print("\r train [epoch : %d/%d][idx : %d/%d] まで完了" % ((epoch+1), opt.epochs, (batch_idx+1), len(train_loader)), end="")

        #loss_train_L1.append(loss_train_L1_epoch/len(train_loader))
        #loss_train_GAN.append(loss_train_GAN_epoch/len(train_loader))
        #loss_train_BMR.append(loss_train_BMR_epoch/len(train_loader))
        #loss_train_P.append((loss_train_P_epoch/len(train_loader))/10)
        #loss_train_KP.append((loss_train_P_epoch/len(train_loader))/1000)
        #loss_train_D.append(loss_train_D_epoch/len(train_loader))

        adjust_learning_rate(opt.lr_g, optimizerE, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_g, optimizerG, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizerD, (epoch + 1), opt)

        if (epoch + 1) % 50 == 0 or (epoch + 1) <= 10:
            img_list = [img_B, Sketch_A, mg_out_A, mg_out_wholeimg_A, mask, maskedimg_B, first_out_A, second_out_A, second_out_wholeimg_A, img_A]
            name_list = ['input_im', 'input_sk', 'mg_out', 'mg_out_wholeimg', 'mask', 'maskedinput', 'first_out', 'second_out', 'second_out_wholeimg', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        '''
        total_L1 = 0
        total_PSNR= 0
        total_SSIM= 0

        #Test Loop
        for idx, (input, truth, Sketch, mask) in enumerate(test_loader):
            input = input.cuda()
            truth = truth.cuda()
            Sketch = Sketch.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                mask_out, _ = net_E(input, Sketch, mask)
                mask_out = (mask_out>0.5).float()
                _, second_out = net_G(input, mask_out, Sketch)

                second_out_wholeimg = input * (1 - mask_out) + second_out * mask_out

                score_L1 = L1(second_out_wholeimg, truth).item()

            total_L1 += score_L1

            print("\r valid [epoch : %d/%d][idx : %d/%d] まで完了" % ((epoch+1), opt.epochs, (idx+1), len(test_loader)), end="")

        if es_best_score is None:
            es_best_score = total_L1
        elif total_L1 > es_best_score:
            es_counter += 1
            if es_counter >= es_patience:
                print("\n Early Stoped at epoch%d"% (epoch+1))
                break
        else:
            es_best_score = total_L1
            es_counter = 0 
        #'''

        save_model(net_G, (epoch + 1), opt, type='G')
        save_model(net_D, (epoch + 1), opt, type='D')
        save_model(net_E, (epoch + 1), opt, type='E')
    
'''
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    x = [i for i in range(len(loss_train_L1))]
    plt.plot(x, loss_train_L1, color="r", label='L1')
    plt.plot(x, loss_train_GAN, color="g", label='GAN')
    plt.plot(x, loss_train_BMR, color="b", label='BMR')
    plt.plot(x, loss_train_P, color="y", label='P')
    #plt.plot(x, loss_train_KP, color="m", label='KP')
    plt.plot(x, loss_train_D, color="k", label='D')
    fig.legend()
    fig.savefig(os.path.join(opt.baseroot, "saved_paths/loss_train.png"))
#'''


    
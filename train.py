import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # mmlab
    parser.add_argument('--pose_config', type = str, default = '../mmpose/datas/kp42/config/custom_config_kp42.py', help = 'config of mmpose')
    parser.add_argument('--pose_checkpoint', type = str, default = '../mmpose/datas/kp42/checkpoint/epoch_60.pth', help = 'checkpoint of mmpose')
    parser.add_argument('--det_config', type = str, default = '../mmdetection/datas/FaceDetect/det_config/config.py', help = 'config of mmdet')
    parser.add_argument('--det_checkpoint', type = str, default = '../mmdetection/datas/FaceDetect/det_checkpoint/300epoch_100img.pth', help = 'checkpoint of mmdet')
    # General parameters
    parser.add_argument('--save_path', type = str, default = './saved_paths', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './saved_samples', help = 'training samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name', type = str, default = '', help = 'load model name')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs of training')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_epoch', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 100, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_gan', type = float, default = 10, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    parser.add_argument('--lambda_perceptual', type = float, default = 100, help = 'the parameter of Perceptual Loss')
    parser.add_argument('--lambda_BMR', type = float, default = 50, help = 'the parameter of BMR loss')
    parser.add_argument('--lambda_KP', type = float, default = 50, help = 'the parameter of KP loss')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = 'C:/Users/owner/program_myself/pytorch_study/SketchAnimeEdit', help = 'the training folder')
    parser.add_argument('--img_path', type = str, default = './dataset_imgs', help = 'the image folder')
    parser.add_argument('--img_height', type = int, default = 640, help = 'height of image')
    parser.add_argument('--img_width', type = int, default = 640, help = 'width of image')
    opt = parser.parse_args()
    print(opt)
    
    
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    
    # Enter main function
    import trainer
    if opt.gan_type == 'WGAN':
        trainer.trainer(opt)
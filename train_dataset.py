import os
import cv2
import glob
import numpy as np
import torch
import random

from torch.utils.data import Dataset

class PairImgs(Dataset):
    """
    self.img_paths   画像ペアの入ってるフォルダの一つ上のディレクトリへのパス
    self.imgs_list   画像ペアの入ってるフォルダすべてのList
    """

    def __init__(self, opt):
        '''画像ペアを入れたディレクトリの親へのパスを指定する'''
        self.opt = opt
        self.img_paths = opt.train_img_path
        self.imgs_list = glob.glob(os.path.join(self.img_paths, '*'), recursive=True)

        '''ペア画像が入っているディレクトリのリストを出力'''
        #print(glob.glob(os.path.join(self.img_paths, '*'), recursive=True))

        '''ペア画像が入っているディレクトリの数を出力'''
        #print(len(glob.glob(os.path.join(self.img_paths, '*'), recursive=True)))


    def __getitem__(self, index):
        '''indexで指定したディレクトリ以下のファイルを返す'''
        num = random.randint(0,1)
        flip = random.randint(0,1)

        set_A = random.randint(0,63)
        set_B = random.randint(0,63)
        filename_A = str(set_A)+'.jpg'
        filename_B = str(set_B)+'.jpg'
        #'''
        img_A    = cv2.imread(os.path.join(self.imgs_list[index], 'image/'+filename_A))
        img_B    = cv2.imread(os.path.join(self.imgs_list[index], 'image/'+filename_B))
        Sketch_A = cv2.imread(os.path.join(self.imgs_list[index], 'sketch/'+filename_A))
        Sketch_B = cv2.imread(os.path.join(self.imgs_list[index], 'sketch/'+filename_B))
        edge_A   = cv2.imread(os.path.join(self.imgs_list[index], 'edge/'+filename_A))
        edge_B   = cv2.imread(os.path.join(self.imgs_list[index], 'edge/'+filename_B))
        mask_A   = cv2.imread(os.path.join(self.imgs_list[index], 'mask/'+filename_A))
        mask_B   = cv2.imread(os.path.join(self.imgs_list[index], 'mask/'+filename_B))
        #'''

        '''画像をcv2として読み込む'''
        '''
        if num == 0:
            img_A = cv2.imread(os.path.join(self.imgs_list[index], "img_A.jpg"))
            img_B = cv2.imread(os.path.join(self.imgs_list[index], "img_B.jpg"))
            Sketch_A = cv2.imread(os.path.join(self.imgs_list[index], "cleanedge_A.jpg"))
            Sketch_B = cv2.imread(os.path.join(self.imgs_list[index], "cleanedge_B.jpg"))
            edge_A = cv2.imread(os.path.join(self.imgs_list[index], "edge_A.jpg"))
            edge_B = cv2.imread(os.path.join(self.imgs_list[index], "edge_B.jpg"))
            mask_A = cv2.imread(os.path.join(self.imgs_list[index], "mask_A.jpg"))
            mask_B = cv2.imread(os.path.join(self.imgs_list[index], "mask_B.jpg"))
        else:
            img_A = cv2.imread(os.path.join(self.imgs_list[index], "img_B.jpg"))
            img_B = cv2.imread(os.path.join(self.imgs_list[index], "img_A.jpg"))
            Sketch_A = cv2.imread(os.path.join(self.imgs_list[index], "cleanedge_B.jpg"))
            Sketch_B = cv2.imread(os.path.join(self.imgs_list[index], "cleanedge_A.jpg"))
            edge_A = cv2.imread(os.path.join(self.imgs_list[index], "edge_B.jpg"))
            edge_B = cv2.imread(os.path.join(self.imgs_list[index], "edge_A.jpg"))
            mask_A = cv2.imread(os.path.join(self.imgs_list[index], "mask_B.jpg"))
            mask_B = cv2.imread(os.path.join(self.imgs_list[index], "mask_A.jpg"))
        '''

        '''BGRをRGBに変更'''
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        '''BGRをグレースケールに変更'''
        Sketch_A = cv2.cvtColor(Sketch_A, cv2.COLOR_BGR2GRAY)
        Sketch_B = cv2.cvtColor(Sketch_B, cv2.COLOR_BGR2GRAY)
        edge_A = cv2.cvtColor(edge_A, cv2.COLOR_BGR2GRAY)
        edge_B = cv2.cvtColor(edge_B, cv2.COLOR_BGR2GRAY)
        mask_A = cv2.cvtColor(mask_A, cv2.COLOR_BGR2GRAY)
        mask_B = cv2.cvtColor(mask_B, cv2.COLOR_BGR2GRAY)

        '''
        画像をテンソルにして返す
        サイズ = (C*H*W)
        C:色[0,1] H:高さ W:幅
        '''
        img_A = torch.from_numpy(img_A.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        Sketch_A = torch.from_numpy(Sketch_A.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        Sketch_B = torch.from_numpy(Sketch_B.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        edge_A = torch.from_numpy(edge_A.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        edge_B = torch.from_numpy(edge_B.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        mask_A = torch.from_numpy(mask_A.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        mask_B = torch.from_numpy(mask_B.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()

        if flip == 0:
            img_A = torch.flip(img_A, dims=[2])
            img_B = torch.flip(img_B, dims=[2])
            Sketch_A = torch.flip(Sketch_A, dims=[2])
            Sketch_B = torch.flip(Sketch_B, dims=[2])
            edge_A = torch.flip(edge_A, dims=[2])
            edge_B = torch.flip(edge_B, dims=[2])
            mask_A = torch.flip(mask_A, dims=[2])
            mask_B = torch.flip(mask_B, dims=[2])

        return img_A, img_B, Sketch_A, Sketch_B, edge_A, edge_B, mask_A, mask_B

    def __len__(self):
        '''画像ペアの入ってるフォルダの数(=データセット数)を返す'''
        return len(glob.glob(os.path.join(self.img_paths, '*')))
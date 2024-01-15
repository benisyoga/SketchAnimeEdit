import os
import cv2
import glob
import numpy as np
import torch

from torch.utils.data import Dataset

class TestImg(Dataset):
    """
    self.img_paths   画像ペアの入ってるフォルダの一つ上のディレクトリへのパス
    self.imgs_list   画像ペアの入ってるフォルダすべてのList
    """

    def __init__(self, opt):
        '''画像ペアを入れたディレクトリの親へのパスを指定する'''
        self.opt = opt
        self.img_paths = opt.img_path
        self.imgs_list = glob.glob(os.path.join(self.img_paths, '*'), recursive=True)

    def __getitem__(self, index):
        '''画像をcv2として読み込む'''
        input = cv2.imread(os.path.join(self.imgs_list[index], "input.jpg"))
        truth = cv2.imread(os.path.join(self.imgs_list[index], "truth.jpg"))
        Sketch = cv2.imread(os.path.join(self.imgs_list[index], "sketch.jpg"))
        mask = cv2.imread(os.path.join(self.imgs_list[index], "mask.jpg"))

        '''BGRをRGBに変更'''
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        truth = cv2.cvtColor(truth, cv2.COLOR_BGR2RGB)
        '''BGRをグレースケールに変更'''
        Sketch = cv2.cvtColor(Sketch, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        '''
        画像をテンソルにして返す
        サイズ = (C*H*W)
        C:色[0,1] H:高さ W:幅
        '''
        input = torch.from_numpy(input.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        truth = torch.from_numpy(truth.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        Sketch = torch.from_numpy(Sketch.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).view(1, self.opt.img_height, self.opt.img_width).contiguous()

        return input, truth, Sketch, mask

    def __len__(self):
        '''画像ペアの入ってるフォルダの数(=データセット数)を返す'''
        return len(glob.glob(os.path.join(self.img_paths, '*')))
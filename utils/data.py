import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import cv2
import os
import os.path as osp
import sys
import random
import scipy.io as scio
import numpy as np

__all__ = ['SirstAugDataset', 'IRSTD1kDataset', 'NUDTDataset', 'SirstDataset']

class SirstAugDataset(Data.Dataset):
    '''
    Return: Single channel
    '''
    def __init__(self, base_dir=r'/Users/tianfangzhang/Program/DATASETS/sirst_aug',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']
        self.mode = mode
        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.tranform = augumentation()
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        # ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        if self.mode == 'train':
            img, mask = self.tranform(img, mask)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask

    def __len__(self):
        return len(self.names)

class IRSTD1kDataset(Data.Dataset):
    '''
    Return: Single channel
    '''

    def __init__(self, base_dir=r'D:/WFY/datasets/IRSTD-1k',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        # self.tranform = augumentation()

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        # ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        # img, mask = self.tranform(img, mask)
        img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)

class NUDTDataset(Data.Dataset):
    '''
    Return: Single channel
    '''

    def __init__(self, base_dir=r'D:/WFY/datasets/NUDT',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.mode = mode
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        # ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)
        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)
    

class SirstDataset(Data.Dataset):
    def __init__(self, base_dir=r'/opt/data/private/xzh/RPCANet-main/datasets/SIRSTdevkit',
                 mode='train', base_size=256):
        if mode == 'train':
            txtfile = 'trainval_v1.txt'
        elif mode == 'val':
            txtfile = 'test_v1.txt'

        self.list_dir = osp.join(base_dir, 'Splits', txtfile)
        self.imgs_dir = osp.join(base_dir, 'PNGImages')
        self.label_dir = osp.join(base_dir, 'SIRST/BinaryMask')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.base_size = base_size
        self.tranform = augumentation()
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        # ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)


        if self.mode == 'train':
            img, mask = self.tranform(img, mask)
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
            img = img.reshape(1, self.base_size, self.base_size) / 255.
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            return img, mask

        elif self.mode == 'val':
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
            img = img.reshape(1, self.base_size, self.base_size) / 255.
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
            _, h, w = img.shape
            # print(img.shape)
            img = PadImg(img)
            mask = PadImg(mask)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            return img, mask



    def __len__(self):
        return len(self.names)

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()

def PadImg(img, times=32):
    _, h, w = img.shape
    
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch
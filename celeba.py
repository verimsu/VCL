import csv
import torch
import os
import PIL
import lightly
from lightly.data.collate import imagenet_normalize 
from typing import Any, Callable, List, Optional, Union, Tuple
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
import numpy as np

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, train_percent=100, mode='train'):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att + 1 for att in range(40)]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=int)
        train_percent = int(162770 * (train_percent / 100))

        self._transform = T.Compose([
                            T.Resize(128),
                            T.ToTensor(),        
                            T.Normalize(lightly.data.collate.imagenet_normalize['mean'], lightly.data.collate.imagenet_normalize['std'])
                        ])        
        
        if mode == 'train':
            self.images = images[:train_percent]
            self.labels = labels[:train_percent]
            
            self._transform = T.Compose(
                                    [
                                        T.Resize(128),
                                        T.RandomResizedCrop(size=128, scale=(0.75, 1.0)),
                                        T.RandomHorizontalFlip(p=0.5),
                                        T.RandomApply([T.ColorJitter(0.7, 0.7, 0.7, 0.2)], p=0.5),
                                        T.RandomGrayscale(p=0.2),
                                        T.ToTensor(),
                                        T.Normalize(imagenet_normalize['mean'], imagenet_normalize['std']),
                                    ]
                                )            

        if mode == 'valid':
            self.images = images[162770:182637]
            self.labels = labels[162770:182637]

        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]

        self.length = len(self.images)

    def __getitem__(self, index):
        image = self._transform(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return {'data': image, 'target': att}

    def __len__(self):
        return self.length
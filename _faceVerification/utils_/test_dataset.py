"""
@author: Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import logging as logger
import cv2
import numpy as np
import lightly
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list_file(str): path of the image list file.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, image_list_file, crop_eye=False):
        self.image_root = image_root
        self.image_list = []
        image_list_buf = open(image_list_file)
        line = image_list_buf.readline().strip()
        while line:
            self.image_list.append(line)
            line = image_list_buf.readline().strip()
        self._transform = T.Compose([
                    T.Resize(128),
                    T.ToTensor(),        
                    T.Normalize(lightly.data.collate.imagenet_normalize['mean'], lightly.data.collate.imagenet_normalize['std'])
                ]) 
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)
        image = self._transform(image)        
        return image, short_image_path

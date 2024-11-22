import copy
import os

import time
import lightly
import numpy as np
import math
from PIL import Image
from pathlib import Path
import argparse
import json
import random
import signal
import sys
import time
import urllib
import shutil
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
from lightly.models import utils
from lightly.data import LightlyDataset
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MultilabelAccuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from celeba import CelebA
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on CelebA"
    )
    
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent"
    )

    parser.add_argument("--best",
        type=str,
        default=None,
        help="name to best saved model"
    )

    # Optim
    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    return parser

class TuningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.vgg16_bn(pretrained=False)
        self.backbone.classifier[0] = torch.nn.Linear(in_features=8192, out_features=4096, bias=True)
        self.backbone.avgpool = torch.nn.Identity()        
        self.backbone.classifier[6] = nn.Linear(4096, 40)
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.accuracy = MultilabelAccuracy(average='micro', num_labels=40)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        logits = self(x)
        loss = self.classification_loss(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch['data'], batch['target']
        logits = self(x)
        loss = self.classification_loss(logits, y.float())
        acc = self.accuracy(logits, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
        )
        return [optim]

def main():
    parser = get_arguments()
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    print("######### ", args.best, " #########")
    
    num_workers = 10
    batch_size = 128

    path = '/DATA/mehmetcanyavuz/celeba/CelebA128/'
    a_path = '/DATA/mehmetcanyavuz/celeba/list_attr_celeba.txt'    

    dataset_train_ssl = CelebA(path, a_path, args.train_percent, mode='train')
    dataset_valid_ssl = CelebA(path, a_path, mode='valid')
    dataset_test_ssl = CelebA(path, a_path, mode='test')   

    def get_data_loaders(batch_size: int):
        dataloader_train_ssl = torch.utils.data.DataLoader(
            dataset_train_ssl,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers
        )

        dataloader_valid_ssl = torch.utils.data.DataLoader(
            dataset_valid_ssl,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )    

        dataloader_test_ssl = torch.utils.data.DataLoader(
            dataset_test_ssl,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )        

        return dataloader_train_ssl, dataloader_valid_ssl, dataloader_test_ssl    
    
    dataloader_train_ssl, dataloader_valid_ssl, dataloader_test_ssl = get_data_loaders(
            batch_size=batch_size, 
        )

    benchmark_model = TuningModel()
    if args.best:
        msg = benchmark_model.load_state_dict(torch.load(args.best)['state_dict'], strict=False)
        print("Loading: ", msg)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", filename="supervised")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10, verbose=False, mode="min") 
    
    trainer = pl.Trainer(max_epochs=args.epochs, 
            accelerator="gpu", devices=1,
            logger=False, callbacks=[checkpoint_callback, early_stop_callback])
        
    trainer.fit(benchmark_model, train_dataloaders=dataloader_train_ssl, val_dataloaders=dataloader_valid_ssl)
    
    _path = 'checkpoints/supervised.ckpt'
    benchmark_model.load_from_checkpoint(_path, strict=True)
    
    print(str(round(100*trainer.test(benchmark_model, dataloaders=dataloader_test_ssl)[0]['test_acc'],2)))

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from lightly.data.collate import imagenet_normalize 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import optuna
from optuna.samplers import TPESampler
import random
import math
import os
import platform
import shutil
import sys
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Subset
sys.path.append("/home/mehmetyavuz/_face/_relativeAttribute")
from bable.models.drn import DRN
from bable.builders import datasets_builder, loss_builder, models_builder
from bable.utils.metrics_utils import MeanTool, ScoresAccuracyTool
from bable.utils.opts_utils import parse_args
from bable.utils.training_utils import get_optimizer_and_lr_schedule
from bable.utils.transforms_utils import get_default_transforms_config

from sklearn.model_selection import KFold


def _get_datasets(args):
    train_dataset_config = get_default_transforms_config()
    train_dataset_config['is_rgb'] = not args.is_bgr
    train_dataset_config['brightness'] = args.argument_brightness
    train_dataset_config['contrast'] = args.argument_contrast
    train_dataset_config['saturation'] = args.argument_saturation
    train_dataset_config['hue'] = args.argument_hue
    train_dataset_config['random_resized_crop_size'] = (
        args.argument_crop_height, args.argument_crop_width)
    train_dataset_config['random_resized_crop_scale'] = (
        args.argument_min_scale, args.argument_max_scale)
    train_dataset_config['random_resized_crop_ratio'] = (
        args.argument_min_ratio, args.argument_max_ratio)
    train_dataset = datasets_builder.build_dataset(
        dataset_type=args.dataset_type,
        split='train',
        category_id=args.category_id,
        trans_config=train_dataset_config,
        include_equal=args.include_equal,
    )

    val_dataset_config = get_default_transforms_config(False)
    val_dataset_config['random_resized_crop_size'] = (
        args.argument_crop_height, args.argument_crop_width)
    val_dataset_config['random_resized_crop_scale'] = (
        args.argument_min_scale, args.argument_max_scale)
    val_dataset_config['random_resized_crop_ratio'] = (
        args.argument_min_ratio, args.argument_max_ratio)    
    val_dataset_config['resize_size'] = (args.val_reisze_height,
                                         args.val_reisze_width)
    val_split = 'test'
    val_dataset = datasets_builder.build_dataset(
        dataset_type=args.dataset_type,
        split=val_split,
        category_id=args.category_id,
        trans_config=val_dataset_config,
        include_equal=args.include_equal,
    )

    return train_dataset, val_dataset


def train_one_epoch(model, train_loader, device, optimizer, loss_fn, epoch,
                    args):
    model.train()
    max_iters = math.ceil(len(train_loader.dataset) / args.batch_size)
    accuracy_tool = ScoresAccuracyTool()
    loss_tool = MeanTool()
    base_global_step = (epoch - 1) * max_iters

    for idx, d in enumerate(train_loader):
        # get data
        img1 = d[0][0]
        img2 = d[0][1]
        labels = d[1]

        # training
        optimizer.zero_grad()
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)
        s1_logits, s2_logits = model(img1, img2)
        loss = loss_fn(s1_logits, s2_logits, labels)
        loss.backward()
        optimizer.step()

        # update metrics
        accuracy_tool.update(s1_logits.cpu().detach().numpy(),
                             s2_logits.cpu().detach().numpy(),
                             labels.cpu().numpy())
        loss_tool.update(loss.cpu().detach().numpy())

        # update index
        idx += 1


def eval_once(model, val_loader, device, loss_fn, epoch, global_step,
              batch_size):
    model.eval()

    total_loss = .0
    accuracy_tool = ScoresAccuracyTool()
    batch_idx = 0
    with torch.no_grad():
        for d in val_loader:
            batch_idx += 1
            img1 = d[0][0].to(device)
            img2 = d[0][1].to(device)
            labels = d[1].to(device)
            s1_logits, s2_logits = model(img1, img2)
            total_loss += loss_fn(s1_logits, s2_logits, labels)
            accuracy_tool.update(s1_logits.cpu().numpy(),
                                 s2_logits.cpu().numpy(),
                                 labels.cpu().numpy())

    total_loss = total_loss / batch_idx
    accuracy = accuracy_tool.accuracy()

    return total_loss, accuracy


def _get_model_dir_name(args, category_name):
    return "logs-{}_{}_{}-{}_{}-loss_{}-{}_{}_{}-wd{}-{}".format(
        args.dataset_type,
        category_name,
        args.batch_size,
        args.model_type,
        args.extractor_type,
        args.loss_type,
        args.optimizer_type,
        args.lr,
        args.extractor_lr,
        args.weight_decay,
        args.logs_name,
    )


def train(model, train_loader, val_loader, model_dir, args):
    # remove/create dirs
    eval_dir = os.path.join(model_dir, args.eval_dir_name)
    if args.clean_model_dir and os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(eval_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = loss_builder.build_loss(args.loss_type)
    optimizer, lr_scheduler = get_optimizer_and_lr_schedule(
        model
    )

    min_loss = 1e10
    max_accuracy = .0
    early_stopping_cnt = 0
    for i in range(args.epochs):
        # train
        train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            loss_fn,
            i + 1,
            args,
        )

        # eval
        global_step = (i+1) * \
            math.ceil(len(train_loader.dataset)/args.batch_size)
        cur_loss, cur_accuracy = eval_once(model, val_loader, device, loss_fn,
                                           i + 1, global_step,
                                           args.val_batch_size)

        # save model
        if min_loss > cur_loss:
            file_names = os.listdir(eval_dir)
            prefix = args.min_loss_ckpt_name[:6]
            for file_name in file_names:
                if prefix in file_name:
                    os.remove(os.path.join(eval_dir, file_name))
                    break
            min_loss = cur_loss
            torch.save(
                model,
                os.path.join(eval_dir,
                             args.min_loss_ckpt_name.format(min_loss)))
        else:
            if args.early_stopping_mode == 'min_loss':
                early_stopping_cnt += 1
                if early_stopping_cnt >= args.early_stopping_epochs:
                    break

        if max_accuracy < cur_accuracy:
            file_names = os.listdir(eval_dir)
            prefix = args.max_accuracy_ckpt_name[:6]
            for file_name in file_names:
                if prefix in file_name:
                    os.remove(os.path.join(eval_dir, file_name))
                    break
            max_accuracy = cur_accuracy
            torch.save(
                model,
                os.path.join(eval_dir,
                             args.max_accuracy_ckpt_name.format(max_accuracy)))
            early_stopping_cnt = 0
        else:
            if args.early_stopping_mode == 'max_accuracy':
                early_stopping_cnt += 1
                if early_stopping_cnt >= args.early_stopping_epochs:
                    break

        # save one step model
        torch.save(model.state_dict(),
                   os.path.join(model_dir, args.step_ckpt_name))

        # lr schedule
        if lr_scheduler is not None:
            if args.lr_decay == 'minloss':
                lr_scheduler.step(min_loss)
            else:
                lr_scheduler.step()

    return max_accuracy

args = parse_args()

train_dataset, test_dataset = _get_datasets(args)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.val_batch_size,
    num_workers=args.num_workers,
)

CATEGORIES = (
    'baldhead', 'darkhair', 'eyesopen', 'goodlooking',
    'masculinelooking', 'mouthopen', 'smile', 'vforehead',
    'v_teeth', 'young'
)



model = DRN(args)
_w = torch.load(args.best)
_w = {f"ranker.{key}": value for key, value in _w.items()}
model.load_state_dict(_w,strict=False)
accuracy = train(model, train_loader, test_loader, "model", args)

print("## Algorithm: ", args.best)
print("## Category: ", CATEGORIES[args.category_id])
print("## Accuracy: ", accuracy)
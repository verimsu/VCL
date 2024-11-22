import os
import platform
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if 'Windows' in platform.platform():
    sys.path.append("E:\\vscprojects\\pytorch-relative-attributes")
else:
    sys.path.append("..")
from bable.builders import datasets_builder
from bable.utils.gradcam_util import (GradCam, get_merged_heatmap_image,
                                      preprocess_image)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_type = 'baidu_street_view_predict'
dataset_params = {'min_width': 224, 'min_height': 224, 'is_bgr': False}
ckpt_path = "logs/logs-baidu_street_view_beautiful_16-drn_vgg16-loss_ranknet-RMSprop_0.0001_1e-05-wd0.0001-default/eval/max_accuracy_0.7545.pth"
target_score_txt = 'test.txt'
heatmap_dir = './data/test'


def init_dataloader(dataset_type='baidu_street_view_predict',
                    dataset_params={
                        'min_width': 224,
                        'min_height': 224,
                        'is_bgr': False
                    }):
    dataset = datasets_builder.build_dataset(dataset_type, **dataset_params)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=20,
        pin_memory=True,
    )


def init_model(ckpt_path):
    model = torch.load(ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda:0")
    return model


def socres_statistics(dataloader, model):
    image_list = []
    scores = []
    with torch.no_grad():
        for d in tqdm(dataloader):
            if torch.cuda.is_available():
                d[1] = d[1].to("cuda:0")
            r = model.ranker(d[1])
            image_list.append(d[0][0])
            scores.append(r.cpu().numpy()[0][0])

    return image_list, scores


def show_images_with_increasing_score(image_list, scores, model):
    scores = np.array(scores)
    s = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    idx = []
    for i in np.arange(0, 1, 0.1):
        min_score = i
        max_score = i + 0.1
        ids = np.where((s >= min_score) & (s < max_score))[0]
        np.random.shuffle(ids)
        idx.append(ids[0])

    # show images example
    fig_img, axes = plt.subplots(2, 5, figsize=(30, 10))
    for i, id in enumerate(idx):
        ax = axes[i // 5][i % 5]
        ax.axis('off')
        ax.imshow(cv2.imread(image_list[id])[:, :, ::-1])
    fig_img.savefig("img.png")

    # show heatmap example
    fig_heatmap, axes = plt.subplots(2, 5, figsize=(30, 10))
    grad_cam = GradCam(model.ranker, ["28"], torch.cuda.is_available())
    for i, id in enumerate(idx):
        ax = axes[i // 5][i % 5]
        ax.axis('off')
        img = cv2.imread(image_list[id])
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input, None)
        ax.imshow(get_merged_heatmap_image(img, mask))
    fig_heatmap.savefig("heatmap.png")


def generate_scores_txt(image_list, scores, target_path):
    scores = np.array(scores)
    s = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    image_names = [n[n.rfind('/') + 1:] for n in image_list]
    data = {'image_names': image_names, 'scores': s}
    df = pd.DataFrame(data)
    df.to_csv(target_path, index=False)


def generate_heapmaps(model, image_list, heatmap_dir):
    image_names = [n[n.rfind('/') + 1:] for n in image_list]
    grad_cam = GradCam(model.ranker, ["28"], True)
    for id in tqdm(range(len(image_list))):
        img = cv2.imread(image_list[id])
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input, None)
        target_img = get_merged_heatmap_image(img, mask)
        target_img = cv2.resize(target_img, (1024, 512))
        cv2.imwrite(os.path.join(heatmap_dir, image_names[id]),
                    target_img[:, :, ::-1])


if __name__ == '__main__':
    dataloader = init_dataloader(dataset_type, dataset_params)
    print('successfully init dataloader...')
    model = init_model(ckpt_path)
    print('successfully init model...')

    image_list, scores = socres_statistics(dataloader, model)
    show_images_with_increasing_score(image_list, scores, model)

    if target_score_txt is not None:
        generate_scores_txt(image_list, scores, target_score_txt)

    if heatmap_dir is not None:
        generate_heapmaps(model, image_list, heatmap_dir)

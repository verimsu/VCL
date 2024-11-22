import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# PlacePulse 数据集的文件结构
# 处理前：
#    images 文件夹，包含所有谷歌街景图片；
#    votes.csv 文件，包含所有标注结果
# 处理后：
#   生成 broken_image_list.txt，包含所有无效街景图片文件名
#   生成 annos.csv 文件，在votes.csv的基础上去掉所有包含无效街景图片的样本
#   生成 {category}_{train/val/test}_equal.csv 文件，分别对应每一类的训练集/验证集/测试集
base_path = '/ssd01/zhangyiyang/pytorch-relative-attributes/data/PlacePulse/'
src_file_name = "votes.csv"
target_file_name = "annos.csv"
broken_file_list_name = "broken_image_list.txt"
train_percent = 0.65
val_percent = 0.15
categories = [
    'beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy'
]


def generate_broken_image_list(broken_file_list_name):
    """
    生成无效的 place pulse 图片列表，比如 514135dcfdc9f04926004b61

    返回一个list，每个元素是无效图片的文件名
    同时生成了一个 broken_image_list.txt 文件，保存所有无效图片文件名
    """
    image_dir = os.path.join(base_path, 'images')
    files = os.listdir(image_dir)
    sample_image_file_name = os.path.join(image_dir,
                                          '514135dcfdc9f04926004b61.jpg')
    sample_image = np.array(Image.open(sample_image_file_name))
    plt.imshow(sample_image)
    broken_list = []
    half_broken_list = []
    cnt = 0
    for file in tqdm(files):
        try:
            cur_img = np.array(Image.open(os.path.join(image_dir, file)))
            if np.max(np.abs(cur_img - sample_image)) == 0:
                broken_list.append(file)
        except Exception:
            half_broken_list.append(file)
        cnt += 1

    # 共 663 张无效图片，下面保存无效图片名称
    all_broken_list = broken_list + half_broken_list
    print(len(broken_list), len(half_broken_list), len(all_broken_list))
    with open(os.path.join(base_path, broken_file_list_name), 'w') as f:
        for line in all_broken_list:
            f.write(line + '\n')

    return all_broken_list


def remove_samples_with_broken_image(all_broken_list,
                                     src_file_name="votes.csv",
                                     target_file_name="annos.csv"):
    """
    更新csv

    去掉所有包含
    """
    src_csv_file = os.path.join(os.path.join(base_path, src_file_name))
    target_csv_file = os.path.join(os.path.join(base_path, target_file_name))
    broken_names = [name[:name.find('.')] for name in all_broken_list]
    df = pd.read_csv(src_csv_file)
    origin_cnt = len(df)
    df = df[~df.left_id.isin(broken_names)]
    df = df[~df.right_id.isin(broken_names)]
    cur_cnt = len(df)
    print('{} has {} error samples'.format(src_csv_file,
                                           (origin_cnt - cur_cnt)))
    df.to_csv(target_csv_file, index=False)


def generate_anno_file_for_each_category(csv_file,
                                         train_percent=0.65,
                                         val_percent=0.15,
                                         base_file_name='{}_{}_equal.csv'):
    for category in tqdm(categories):
        df = pd.read_csv(csv_file)
        df = df[df.category == category]
        not_equal_df = df[df.winner != 'equal']

        num_train = int(len(not_equal_df) * train_percent)
        num_val = int(len(not_equal_df) * val_percent)
        num_test = len(not_equal_df) - num_val - num_train

        not_equal_df = not_equal_df.sample(frac=1.0)  # shuffle
        train_df = not_equal_df[:num_train]
        val_df = not_equal_df[num_train:-num_test]
        test_df = not_equal_df[-num_test:]

        equal_df = df[df.winner == 'equal']
        train_df = pd.concat([train_df, equal_df])

        target_format = os.path.join(base_path, base_file_name)

        train_df.to_csv(target_format.format(category, 'train'), index=False)
        val_df.to_csv(target_format.format(category, 'val'), index=False)
        test_df.to_csv(target_format.format(category, 'test'), index=False)


if __name__ == '__main__':
    broken_file_list = generate_broken_image_list(broken_file_list_name)
    remove_samples_with_broken_image(broken_file_list, src_file_name,
                                     target_file_name)
    generate_anno_file_for_each_category(target_file_name,
                                         train_percent=train_percent,
                                         val_percent=val_percent)

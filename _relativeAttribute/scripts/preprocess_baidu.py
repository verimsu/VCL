import os
from tqdm import tqdm
import pandas as pd

train_percent = 0.9
val_percent = 0.1
target_dir = "F:\\data\\BaiduStreetView"
base_file_name = '{}_{}_equal.csv'
for category in tqdm(['beautiful']):
    df = pd.read_csv(os.path.join(target_dir, "{}.csv".format(category)))
    df = df[df.category == category]
    not_equal_df = df[df.winner != 'equal']

    num_train = int(len(not_equal_df) * train_percent)
    num_val = int(len(not_equal_df) * val_percent)

    not_equal_df = not_equal_df.sample(frac=1.0)  # shuffle
    train_df = not_equal_df[:num_train]
    val_df = not_equal_df[num_train:]

    equal_df = df[df.winner == 'equal']
    train_df = pd.concat([train_df, equal_df])

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_format = os.path.join(target_dir, base_file_name)

    train_df.to_csv(target_format.format(category, 'train'), index=False)
    val_df.to_csv(target_format.format(category, 'val'), index=False)
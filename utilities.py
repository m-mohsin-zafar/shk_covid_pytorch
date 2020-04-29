import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# root = os.getcwd()
# dataset_root = os.path.join(root, 'data')
#
# corona_fnames = sorted(os.listdir(os.path.join(dataset_root, 'corona')))
# non_corona_fnames = sorted(os.listdir(os.path.join(dataset_root, 'non_corona')))

# print(len(corona_fnames))
# print(len(non_corona_fnames))

gen_labels = False
if gen_labels:
    labels = []
    fnames = []
    for f in os.listdir(os.path.join(dataset_root, 'covid_data')):

        if f.startswith('Corona'):
            labels.append(1)
        elif f.startswith('Non_'):
            labels.append(0)

        fnames.append(f)

    x = fnames
    y = labels

    data = {
        'x': x,
        'y': y
    }

    df = pd.DataFrame(data=data)
    df.to_csv('labels.csv', index=False)


def train_validate_test_split(df, train_percent=.6, validate_percent=.25, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def get_stratified_train_val_test_sets(inp_df, tr_frac=0.75, val_frac=0.15, ts_frac=0.10):

    dataset_length = len(inp_df)
    tr_size = int(dataset_length * tr_frac)
    val_size = dataset_length * val_frac
    ts_size = dataset_length * ts_frac

    X_train, X_test, y_train, y_test = train_test_split(inp_df.x, inp_df.y, test_size=ts_frac, stratify=inp_df.y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=tr_size, stratify=y_train)

    assert val_size == len(X_val), 'Validation Size does not match'

    train = {
        'x': X_train,
        'y': y_train
    }

    val = {
        'x': X_val,
        'y': y_val
    }

    test = {
        'x': X_test,
        'y': y_test
    }

    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)
    test_df = pd.DataFrame(test)

    return train_df, val_df, test_df


copy_all_data_to_new_folder = False

if copy_all_data_to_new_folder:

    for direc in [os.path.join(dataset_root, 'corona'), os.path.join(dataset_root, 'non_corona')]:

        print('DIR = ', direc)

        dirName = os.path.join(dataset_root, 'covid_data')

        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")

        for x in tqdm(os.listdir(direc)):

            src_fpath = os.path.join(direc, x)
            dst_fpath = os.path.join(dirName, x)

            if os.path.exists(src_fpath):
                newPath = shutil.copy(src_fpath, dst_fpath)
                # print("Path of copied file : ", newPath)

import geopandas as gpd
import numpy as np
import pandas as pd
import os

from data_lib.haiti.config import DATA_DIR


def load_all():
    return gpd.read_file(os.path.join(DATA_DIR, "SHAPE", "BTB_Points.shp"))


def load_xy():
    x_df = load_all()[["Longitude", "Latitude", "Area", "Floors", "Constructi", "FloorType", "Structure"]]
    y_df = load_all()[["Building_s"]]
    one_hot_list = ["Constructi", "FloorType", "Structure"]
    for col in one_hot_list:
        new = pd.get_dummies(x_df[col])
        x_df = x_df.drop(col, axis=1)
        x_df = x_df.join(new)

    return x_df.values, y_df.Building_s


def normalize(x, y):
    return (x - np.nanmean(x, axis=1, keepdims=True)) / np.nanstd(x, axis=1, keepdims=True), y


def make_splits(y, seed, train_ratio, val_ratio):
    """buildings divided at random - labels have Green/Yellow/Red label (0, 1, 2) or 'None' (row of zeros)"""
    np.random.seed(seed)
    numb = y.shape[0]
    splits = np.array(["empty"] * numb)
    labelled_idxs = np.where(np.sum(y, axis=1) != 0)[0].flatten()
    idxs = np.random.choice(labelled_idxs, size=len(labelled_idxs), replace=False)
    train_idxs = idxs[: int(train_ratio * len(labelled_idxs))]
    val_idxs = idxs[int(train_ratio * len(labelled_idxs)): int((train_ratio + val_ratio) * len(labelled_idxs))]
    test_idxs = idxs[int((train_ratio + val_ratio) * len(labelled_idxs)):]
    splits[train_idxs] = "train"
    splits[val_idxs] = "val"
    splits[test_idxs] = "test"

    return splits

import h5py
from h5py._hl.group import Group
from h5py._hl.dataset import Dataset
import numpy as np
from glob import glob
import os

from data_lib.soilmoisture.config import DATA_DIR, AOI_INDEX
from data_lib.soilmoisture.Data import Data


def load_xy(aoi=AOI_INDEX):
    """
    returns x (numb_samples, numb_timesteps) and y (numb_samples, numb_timesteps) with np.nan values if no data
    """
    file_paths = sorted(glob(os.path.join(DATA_DIR, "*", "*.h5")), reverse=True)
    d = {
        Data.strip_date(fp): Data(fp, aoi) for fp in file_paths
    }
    x = np.stack(
        [getattr(data, attr) for _, data in sorted(d.items(), key=lambda data: data[1].date) for attr in ["am", "pm"]],
        axis=-1
    )
    x[x == -9999] = np.nan  # replace -9999 (no data) with np.nan
    x = x.reshape(-1, x.shape[-1])
    return x, x


def make_splits(y_cont, seed, train_ratio, val_ratio):
    """either train, val, test or empty - empty corresponds to unobserved samples"""
    np.random.seed(seed)
    numb = np.sum((~np.isnan(y_cont)))  # y_cont is (#samples, #timesteps)
    idxs = np.random.choice(np.argwhere(~np.isnan(y_cont)).flatten(), size=numb, replace=False)
    train = idxs[:int(numb * train_ratio)]
    val = idxs[int(numb * train_ratio):int(numb * (train_ratio + val_ratio))]
    test = idxs[int(numb * (train_ratio + val_ratio)):]
    splits = np.array(["empty"] * np.prod(y_cont.shape))
    splits[train] = 'train'
    splits[val] = 'val'
    splits[test] = 'test'

    return splits


def one_hot_classify(y_cont, numb_classes, splits):
    """
    Takes continuously-valued soil moisture measurements and bins them into classes

    The class boundaries are computed to give a roughly equal number of samples in each class
    These boundaries are computed using the soil moisture measurements in the training set only

    Returns: one-hot encoded matrix (numb_samples, numb_classes) with any all-zero rows corresponding to missing data
    """
    percentiles = np.linspace(0, 100, numb_classes + 1)
    # compute class boundaries based on observed (training) data only
    boundaries = np.percentile(y_cont[splits == "train"], q=percentiles)

    columns = []
    for i in range(numb_classes):
        if i == 0:
            # handles case when y_val or y_test below minimum in y_train
            columns.append(y_cont < boundaries[i + 1])
        elif i == numb_classes - 1:
            # handles case when y_val or y_test above maximum in y_train
            columns.append(y_cont >= boundaries[i])
        else:
            columns.append((y_cont < boundaries[i + 1]) * (y_cont >= boundaries[i]))

    y_one_hot = np.stack(columns, axis=-1)
    return y_one_hot.astype(int)


def compress_soil_moisture_file(ip_path, op_path):
    keep_keys = {"Metadata": "all",
                 "Soil_Moisture_Retrieval_Data_AM": ["soil_moisture", "longitude", "latitude"],
                 "Soil_Moisture_Retrieval_Data_PM": ["soil_moisture_pm", "longitude_pm", "latitude_pm"]}
    with h5py.File(ip_path, "r") as input_h5:
        with h5py.File(op_path, "w") as output_h5:

            def add_groups(output_h5, obj, group=None, path=""):
                for key in obj.keys():  # a string representing a group or dataset
                    if obj[key].name.split('/')[1] in keep_keys.keys():
                        if isinstance(obj[key], Group):  # if it's a group
                            if path == "":  # don't add preceding '/' on first groups
                                group_path = key
                            else:
                                group_path = path + "/{}".format(key)
                            group = output_h5.create_group(group_path)  # create a group
                            add_groups(output_h5, obj[key], group, group_path)  # recursion but with group passed
                        if isinstance(obj[key], Dataset):  # if it's a dataset
                            if key in keep_keys[path]:  # if it's a dataset we chose to keep
                                group.create_dataset(key, data=np.array(obj[key]), compression="gzip",
                                                     compression_opts=9)

            add_groups(output_h5, input_h5)


def _compress_folder():
    for fp in glob(os.path.join(DATA_DIR, "*", "*.h5")):
        folder = os.path.join(os.path.join(*fp.split("/")[:-3]), "SPL3SMP_E_imr27_compressed", fp.split('/')[-2])
        folder = '/' + folder
        if not os.path.isdir(folder):
            os.makedirs(folder)
            compress_soil_moisture_file(fp, os.path.join(folder, fp.split('/')[-1]))


def imageify(array, never_seen):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not array.ndim == 1:
        raise ValueError("'array' arg must be 1-dimensional but was %s-dimensional" % array.ndim)

    image = np.nan * np.zeros((AOI_INDEX["bottom"] - AOI_INDEX["top"], AOI_INDEX["right"] - AOI_INDEX["left"]))
    image[~never_seen] = array
    return image

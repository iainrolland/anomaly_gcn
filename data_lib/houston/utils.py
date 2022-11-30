from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from scipy.ndimage import label
import numpy as np

from data_lib.houston import ground_truth


def get_training_array(tif):
    array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                     out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                     resampling=Resampling.bilinear)
    return normalise(array, tif.nodata)


def normalise(array, nodata):
    """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
    array[array == nodata] = 0
    return (array - array.min(axis=(1, 2))[:, None, None]) / (
        (array.max(axis=(1, 2)) - array.min(axis=(1, 2)))[:, None, None])


def read_tif_channels(tif, channel_index):
    if not isinstance(channel_index, list):
        channel_index = list(channel_index)
    if channel_index[0] == 0:  # rasterio indexes channels starting from 1 not 0...
        channel_index = list(np.array(channel_index) + 1)
    profile = tif.profile
    profile.update({'count': len(channel_index)})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(tif.read(channel_index))
    return memory_file


def resample_tif(tif, scale_factor, mode=Resampling.bilinear):
    data = tif.read(
        out_shape=(
            tif.count,
            int(tif.height * scale_factor),
            int(tif.width * scale_factor)
        ),
        resampling=mode
    )
    # scale image transform
    transform = tif.transform * tif.transform.scale(
        (tif.width / data.shape[-1]),
        (tif.height / data.shape[-2])
    )
    profile = tif.profile
    profile.update({'width': int(tif.width * scale_factor),
                    'height': int(tif.height * scale_factor),
                    'transform': transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(data)
    return memory_file


def make_splits(gt, seed, train_ratio=0.4, val_ratio=0.3):
    np.random.seed(seed)
    splits = np.array(["empty"] * np.prod(gt.shape[:2]))
    for class_id in range(1, gt.max() + 1):  # for each output class (other than "Unclassified")
        labels, numb_objects = label(gt == class_id)  # label each contiguous block with a number from 1,...,N
        num_pixels = (labels != 0).sum()
        # if the labelled area for a class is small, a finer grid chop might be needed to get pixels into each split
        if num_pixels < 1000:
            labels = grid_chop(labels, grid_size=25)
        elif num_pixels < 10000:
            labels = grid_chop(labels, grid_size=150)
        else:
            labels = grid_chop(labels, grid_size=250)
        numb_objects = labels.max()
        objects = np.random.choice(np.arange(1, numb_objects + 1), numb_objects, replace=False)
        numb_per_split = np.ceil(
            np.array([numb_objects * train_ratio, numb_objects * (train_ratio + val_ratio), numb_objects])).astype(int)
        training_objects = objects[:numb_per_split[0]]
        val_objects = objects[numb_per_split[0]:numb_per_split[1]]
        test_objects = objects[numb_per_split[1]:]

        splits[np.isin(labels, training_objects).flatten()] = "train"
        splits[np.isin(labels, val_objects).flatten()] = "val"
        splits[np.isin(labels, test_objects).flatten()] = "test"
        
    return splits


def grid_chop(labels, grid_size=150):
    x, y = np.array([np.arange(labels.shape[1])] * labels.shape[0]), np.array(
        [np.arange(labels.shape[0])] * labels.shape[1]).T
    div_x, div_y = x // grid_size, y // grid_size
    grid_idxs = div_x + div_y * div_x.max()
    class_mask = labels != 0
    grid_id_nums = set(grid_idxs[class_mask])
    object_numb = 1
    for grid_id in grid_id_nums:
        grid_mask = grid_idxs == grid_id
        object_nums = set(labels[grid_mask]).difference({0})
        for obj_num in object_nums:
            labels[grid_mask & (labels == obj_num)] = object_numb
            object_numb += 1
    return labels



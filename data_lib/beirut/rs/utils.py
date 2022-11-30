import numpy as np
from rasterio import transform, MemoryFile
from rasterio.mask import mask
from rasterio.windows import Window
from scipy.stats import describe
from tqdm import tqdm
import warnings

from data_lib.beirut.rs.process_traffic_light import get_labels

warnings.filterwarnings("error")


def get_x(df, tif, with_coords=True):
    labels = df.copy()
    if labels.crs.to_epsg() != tif.crs.to_epsg():
        labels.to_crs(epsg=tif.crs.to_epsg(), inplace=True)
    if with_coords:
        descriptors = [[] for _ in range(tif.count + 1)]
    else:
        descriptors = [[] for _ in range(tif.count)]
    for building in tqdm(labels.geometry.values):
        masked = mask(tif, [building], all_touched=True, crop=True)[0]
        masked = [m[m != 0] for m in masked]
        for i, ch in enumerate(masked):
            if len(ch) != 0:
                try:
                    stats = describe(ch)
                except RuntimeWarning:  # RuntimeWarning produced when only one pixel describing building
                    stats = describe([ch, ch])
                description = [float(value) for value in stats.minmax]
                description.extend([float(value) for value in stats[2:]])
                descriptors[i].append(np.array(description).flatten())
            else:
                descriptors[i].append(np.array([0] * len(descriptors[i][-1])))
        if with_coords:
            x_centroid, y_centroid = building.centroid.coords[0]
            descriptors[-1].append(np.array([x_centroid, y_centroid]))  # building position
    return np.concatenate([np.array(d) for d in descriptors], axis=-1)


def expand_tif(tif, bounds, clip_values):
    extremities = [f(a, b) for a, b, f in zip(tif.bounds, bounds, [min, min, max, max])]
    left_idx, bottom_idx = ~tif.transform * (extremities[0], extremities[1])
    right_idx, top_idx = ~tif.transform * (extremities[2], extremities[3])
    width, height = right_idx - left_idx, bottom_idx - top_idx

    profile = tif.profile
    profile.update({'width': width, 'height': height, 'transform': transform.from_bounds(*extremities, width, height)})

    with MemoryFile() as memory_file:
        dst = memory_file.open(**profile)
        for i, (ch, clip_val) in enumerate(zip(tif.read(), clip_values)):
            ch_clipped = ch.copy()
            ch_clipped[ch_clipped > clip_val] = clip_val
            dst.write(ch_clipped, window=Window(-left_idx, -top_idx, tif.shape[1], tif.shape[0]),
                      indexes=i + 1)
    return dst


def get_splits():
    labels_df = get_labels()
    random = np.random.choice(labels_df.index, len(labels_df), replace=False)
    train_validate, validate_test = int(len(random) * .3), int(len(random) * .7)  # 30/40/30 train/validate/test ratio
    train = labels_df.iloc[random[:train_validate]]
    validate = labels_df.iloc[random[train_validate:validate_test]]
    test = labels_df.iloc[random[validate_test:]]
    return train, validate, test

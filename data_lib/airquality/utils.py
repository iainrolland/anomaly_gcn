from glob import glob
import numpy as np
import os
import rasterio
import rasterio.sample
import geopandas as gpd
import pandas as pd

from data_lib.airquality.config import DATA_DIR
import data_lib.airquality.path_utils as pu


def open_ground_stations_shp(region, datatype):
    return gpd.read_file(pu.get_ground_station_path(region, datatype))


def get_ground_stations(region, datatype):
    return open_ground_stations_shp(region, datatype).geometry.unique()


def CAMS_NO2_fpath(day, hour, region="Italy"):
    fname = "CAMS_NO2_day{}_h{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "CAMS", "NO2_surface", fname)


def CAMS_PM25_fpath(day, hour, region="Italy"):
    fname = "CAMS_PM2_5_day{}_h{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "CAMS", "PM2_5", fname)


def SEN5P_NO2_fpath(day, hour, region="Italy"):
    fname = "S5P_NO2_OFFL_L2_day{}_T{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "sentinel5P", "NO2", fname)


def file_exists(fpath):
    return os.path.isfile(fpath)


def sample_tif(tif, point, low=-900, high=900):
    if tif is None:
        return np.nan
    else:
        val = list(rasterio.sample.sample_gen(tif, point.coords[:]))[0][0]
        if low < val < high:
            return val
        else:
            return np.nan


def load_xy(with_coords=False):
    """
    returns x (numb_samples, numb_features) and y (numb_samples,) with np.nan values if no data
    """
    files = glob(os.path.join(DATA_DIR, "Italy", "CAMS", "PM2_5", "CAMS*.tif"))
    days = sorted(np.unique([int(f.split("day")[-1].split('_')[0]) for f in files]))
    station_measurements = gpd.read_file(os.path.join(DATA_DIR, "Italy", "ground_air_quality", "PM25",
                                                      "PM25_italy_ground.shp"))
    stations = get_ground_stations("Italy", "PM25")

    # cross-reference stations with measurements
    data_dict = {}
    for i, station in enumerate(stations):
        sub_df = station_measurements[station_measurements.geometry == station]
        data_dict[i] = pd.Series(sub_df.AirQuality.values, index=sub_df.Date.values)

    df = pd.DataFrame(data_dict)
    missing_days = [day for day in days if day not in df.index.values]
    df = pd.concat([df, pd.DataFrame(np.nan, index=missing_days, columns=df.columns)])
    df.sort_index(inplace=True)
    station_measurements = df.to_numpy(dtype=np.float64, na_value=np.nan)

    x = []
    y = []

    camsNO2_path = None
    sen5pNO2_path = None

    for i, day in enumerate(days):
        for hour in range(24):
            path = CAMS_PM25_fpath(day, hour)
            if file_exists(path):
                camsPM25_path = path

                if file_exists(CAMS_NO2_fpath(day, hour)):
                    camsNO2_path = CAMS_NO2_fpath(day, hour)
                if file_exists(SEN5P_NO2_fpath(day, hour)):
                    sen5pNO2_path = SEN5P_NO2_fpath(day, hour)

                # outputs
                for j in range(len(stations)):
                    y.append([station_measurements[i, j]])

                # inputs
                with rasterio.open(camsPM25_path) as camsPM25_tif:
                    if camsNO2_path is not None:
                        with rasterio.open(camsNO2_path) as camsNO2_tif:
                            if sen5pNO2_path is not None:
                                with rasterio.open(sen5pNO2_path) as sen5pNO2_tif:
                                    for s in stations:
                                        x.append([])
                                        x[-1].append(sample_tif(camsPM25_tif, s))
                                        x[-1].append(sample_tif(camsNO2_tif, s))
                                        x[-1].append(sample_tif(sen5pNO2_tif, s))
                                        if with_coords:
                                            x[-1].append(s.x)
                                            x[-1].append(s.y)
                            else:
                                for s in stations:
                                    x.append([])
                                    x[-1].append(sample_tif(camsPM25_tif, s))
                                    x[-1].append(sample_tif(camsNO2_tif, s))
                                    x[-1].append(np.nan)
                                    if with_coords:
                                        x[-1].append(s.x)
                                        x[-1].append(s.y)
                    else:
                        for s in stations:
                            x.append([])
                            x[-1].append(sample_tif(camsPM25_tif, s))
                            x[-1].append(np.nan)
                            x[-1].append(np.nan)
                            if with_coords:
                                x[-1].append(s.x)
                                x[-1].append(s.y)

    return np.array(x), np.array(y)


def make_splits(y_cont, seed, train_ratio, val_ratio):
    np.random.seed(seed)
    # since ground stations measure only once per day and get repeated 24 times (to give per hour labels), sample every
    # 24th value when reshaped to give time in axis 0 (this allows us to split entire days into train/test rather than
    # by hour which would allow specific ground station measurements to exist across dataset splits)
    y_daily = y_cont.reshape(-1, 50)[::24].flatten()
    numb = np.sum((~np.isnan(y_daily)))  # number of ground stations with measurements (not nan)
    idxs = np.random.choice(np.argwhere(~np.isnan(y_daily)).flatten(), size=numb, replace=False)
    train = idxs[:int(numb * train_ratio)]
    val = idxs[int(numb * train_ratio):int(numb * (train_ratio + val_ratio))]
    test = idxs[int(numb * (train_ratio + val_ratio)):]
    splits = np.array(["empty"] * y_daily.shape[0])
    splits[train] = 'train'
    splits[val] = 'val'
    splits[test] = 'test'

    # reverse the thinning by 24 to get back to value for every hour
    splits = np.repeat(splits.reshape(-1, 50), 24, axis=0).flatten()
    return splits


def one_hot_classify(y_cont, numb_classes, splits):
    """
    Takes the continuously-valued air quality measurements and bins them into classes

    The class boundaries are computed to give a roughly equal number of samples in each class
    These boundaries are computed using the air quality measurements in the training set only

    Returns: one-hot encoded matrix (numb_samples, numb_classes) with any all-zero rows corresponding to missing data
    """
    percentiles = np.linspace(0, 100, numb_classes + 1)
    # compute class boundaries based on observed (training) data only
    boundaries = np.percentile(y_cont[splits == "train"], q=percentiles)

    y_one_hot = np.stack([((y_cont > low) * (y_cont < high)).flatten() for low, high in zip(boundaries[:-1],
                                                                                            boundaries[1:])],
                         axis=1)
    return y_one_hot.astype(int)

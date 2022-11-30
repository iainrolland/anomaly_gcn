from spektral_datasets.AirQuality import AirQuality
from spektral_datasets.BeirutDataset import BeirutDataset
from spektral_datasets.HaitiDamage import HaitiDamage
from spektral_datasets.HoustonDatasetMini import *
from spektral_datasets.LondonSAR import LondonSAR
from spektral_datasets.SoilMoisture import SoilMoisture


def get_dataset(data_name):
    supported_datasets = dict(
        zip(["AirQuality", "BeirutDataset", "HaitiDamage", "HoustonDatasetMini", "LondonTimeSeries", "SoilMoisture"],
            [AirQuality, BeirutDataset, HaitiDamage, HoustonDatasetMini, LondonSAR, SoilMoisture]))
    try:
        return supported_datasets[data_name]
    except KeyError:
        raise ValueError(
            "{} was not a recognised dataset. Must be one of {}.".format(data_name, "/".join(supported_datasets)))

import os
from glob import glob

from data_lib.airquality.config import DATA_DIR


def get_ground_station_path(region, datatype):
    if region not in ["Italy", "California", "South Africa"]:
        raise ValueError("Region '{}' data does not exist".format(region))
    if datatype not in ["NO2", "PM25"]:
        raise ValueError("Datatype '{}' data does not exist".format(datatype))

    folder = os.path.join(DATA_DIR, '_'.join(region.split(' ')), "ground_air_quality", datatype)
    path_list = glob(os.path.join(folder, "*.shp"))
    if len(path_list) == 0:
        raise FileNotFoundError("No shapefile found for region '{}' and datatype '{}'".format(region, datatype))
    elif len(path_list) > 1:
        raise ValueError("Multiple shapefiles found for region '{}' and datatype '{}'".format(region, datatype))
    else:
        return path_list[0]

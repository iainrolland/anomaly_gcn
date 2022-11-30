import datetime
import numpy as np
import h5py


class Data:
    def __init__(self, file_path, aoi=None):
        self._h5 = self.open_h5(file_path)
        self.date = self.strip_date(file_path)
        self.aoi = aoi

    @property
    def am(self):
        return self.crop(self.get_attr(self._h5["Soil_Moisture_Retrieval_Data_AM"], "soil_moisture"))

    @property
    def pm(self):
        return self.crop(self.get_attr(self._h5["Soil_Moisture_Retrieval_Data_PM"], "soil_moisture_pm"))

    @property
    def lat(self):
        return self.crop(self.get_attr(self._h5["Soil_Moisture_Retrieval_Data_AM"], "latitude"))

    @property
    def lon(self):
        return self.crop(self.get_attr(self._h5["Soil_Moisture_Retrieval_Data_AM"], "longitude"))

    def crop(self, ip):
        if self.aoi is not None:
            return ip[self.aoi["top"]:self.aoi["bottom"], self.aoi["left"]:self.aoi["right"]]
        else:
            return ip

    @staticmethod
    def get_attr(h5, attr):
        array = h5[attr][...]
        try:
            array[array == h5[attr].attrs["_FillValue"]] = np.nan
        except KeyError:
            pass
        return array

    @staticmethod
    def strip_date(file_path):
        date_string = file_path.split("/")[-2]
        return datetime.datetime.strptime(date_string, "%Y.%m.%d").date()

    @staticmethod
    def open_h5(file_path):
        return h5py.File(file_path, "r")


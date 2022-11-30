import os
import mat73


DATA_PATH = "data_lib/london_time_series/data"


def pca_path():
    return os.path.join(DATA_PATH, "result-Satsense-PCA.mat")


def satsense_path():
    return os.path.join(DATA_PATH, "Satsense-bank-bridge2.mat")


if __name__ == '__main__':
    mat73.loadmat(pca_path(),)

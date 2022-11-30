import rasterio
import numpy as np

from data_lib.houston.utils import get_training_array, read_tif_channels


def training_array():
    tif = rasterio.open("data_lib/houston/data/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix")
    # only the first 48 channels of the HS tif are actually HS channels (we don't use channel 49/50)
    tif = read_tif_channels(tif, np.arange(48) + 1)
    return get_training_array(tif)

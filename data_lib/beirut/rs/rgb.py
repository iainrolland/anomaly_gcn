import rasterio

from data_lib.beirut.rs.process_traffic_light import get_labels


def tif_before():
    before = rasterio.open("data_lib/beirut/data/20JUL31083035-10300500A5F95600-Clipped.tif")
    before_clip_values = [406, 638, 406]
    # tif_before = expand_tif(before, bigger_bounds, before_clip_values)
    return before


def tif_after():
    after = rasterio.open("data_lib/beirut/data/20AUG05084637-10300500A6F8AA00-Clipped.tif")
    after_clip_values = [405, 612, 393]
    # tif_after = expand_tif(after, bigger_bounds, after_clip_values)
    return after


def labels():
    return get_labels().to_crs(tif_before().crs.to_epsg())


def bigger_bounds():
    return labels().total_bounds

import rasterio

from data_lib.beirut.rs.process_traffic_light import get_labels

before = rasterio.open("data_lib/beirut/data/20JUL31083035-10300500A5F95600-Clipped.tif")
before_clip_values = [406, 638, 406]
after = rasterio.open("data_lib/beirut/data/20AUG05084637-10300500A6F8AA00-Clipped.tif")
after_clip_values = [405, 612, 393]
labels = get_labels().to_crs(before.crs.to_epsg())

bigger_bounds = labels.total_bounds
# tif_before = expand_tif(before, bigger_bounds, before_clip_values)
# tif_after = expand_tif(after, bigger_bounds, after_clip_values)
tif_before = before
tif_after = after

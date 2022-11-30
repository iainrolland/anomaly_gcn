import rasterio

IFG_PATH = "data_lib/beirut/data/20200730_20200805_IW1_Stack_ifg_deb_dinsar_flt_ML_TC_big.tif"

ifg = rasterio.open(IFG_PATH)

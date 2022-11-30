import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

SURVEY_PATH = "data_lib/beirut/data/UN/Classified_Parcels_Survey_20200909.shp"
MOB_PATH = "data_lib/beirut/data/UN/Classified_Parcels_MoB_20200907.shp"


def get_labels():
    mob = gpd.read_file(MOB_PATH)
    survey = gpd.read_file(SURVEY_PATH)
    mob.rename(columns={'Category': 'decision'}, inplace=True)
    survey.rename(columns={'decision__': 'decision'}, inplace=True)
    combined = gpd.GeoDataFrame(
        pd.concat([mob[['decision', 'geometry']].dropna(), survey[['decision', 'geometry']].dropna()],
                  ignore_index=True),
        crs="epsg:32636").to_crs(epsg=4326)
    combined['decision'] = combined.decision.str.split().str.get(0)
    combined.to_crs("epsg:4326")

    for i, val in enumerate(combined.geometry):
        if "MultiPolygon" in str(type(val)):
            polys = []
            for p in val.geoms:
                polys.append(Polygon(list(zip(p.exterior.coords.xy[0], p.exterior.coords.xy[1]))))
                combined.geometry[i] = MultiPolygon(polys)
        else:
            combined.geometry[i] = Polygon(list(zip(val.exterior.coords.xy[0], val.exterior.coords.xy[1])))
    return combined


def bounds():
    return get_labels().total_bounds


def save_labels():
    df = get_labels()
    df.to_file("./data/BeirutDamages.shp")

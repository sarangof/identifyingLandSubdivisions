import dask_geopandas as dgpd
import pandas as pd
from dask import delayed, compute, visualize
import geopandas as gpd
from dask.diagnostics import ProgressBar
from citywide_calculation import get_utm_crs
from metrics_calculation import calculate_minimum_distance_to_roads_option_B
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize, nearest_points
#from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from scipy.optimize import fminbound, minimize
from metrics_groupby import metrics



MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
CITY_INFO_PATH = f'{INPUT_PATH}/city_info'
EXTENTS_PATH = f'{CITY_INFO_PATH}/extents'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
SEARCH_BUFFER_PATH = f'{INPUT_PATH}/city_info/search_buffers'
OUTPUT_PATH = f'{MAIN_PATH}/output'
OUTPUT_PATH_CSV = f'{OUTPUT_PATH}/csv'
OUTPUT_PATH_RASTER = f'{OUTPUT_PATH}/raster'
OUTPUT_PATH_PNG = f'{OUTPUT_PATH}/png'
OUTPUT_PATH_RAW = f'{OUTPUT_PATH}/raw_results'


@delayed
def get_epsg(city_name):
    search_buffer = f'{SEARCH_BUFFER_PATH}/{city_name}/{city_name}_search_buffer.geoparquet'
    extent = dgpd.read_parquet(search_buffer)
    geometry = extent.geometry[0].compute()
    epsg = get_utm_crs(geometry)
    print(f'{city_name} EPSG: {epsg}')
    return epsg

def load_dataset(path, epsg=None):
    dataset = dgpd.read_parquet(path, npartitions=4)
    
    # Only assign if the file has no CRS
    if epsg:
        if dataset.crs is None:
            dataset = dataset.set_crs("EPSG:4326")  # assume WGS84 if missing
        dataset = dataset.to_crs(epsg)

    return dataset

@delayed
def row_count(dgdf):
    """Count the rows in a dataframe"""
    row_count = dgdf.map_partitions(len).compute().sum()

    return row_count


def test_math(input):
    return input + input


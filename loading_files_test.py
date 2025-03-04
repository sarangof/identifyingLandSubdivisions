import time
start_time = time.time()  # Start the timer

import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from metrics_calculation import *
#from create_rectangles import *
from standardize_metrics import *
import matplotlib.pyplot as plt
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from datetime import datetime
import shutil
#import pyproj
from pyproj import CRS, Geod
import json
from dask import delayed, compute, visualize
import dask_geopandas as dgpd  
import dask.dataframe as dd
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from shapely.errors import TopologicalError
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from functools import partial
from cloudpathlib import S3Path
import s3fs
from shapely.strtree import STRtree  
from dask.distributed import CancelledError
from dask_geopandas import read_parquet as dgpd_read_parquet


MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
OUTPUT_PATH = f'{MAIN_PATH}/output'
OUTPUT_PATH_CSV = f'{OUTPUT_PATH}/csv'
OUTPUT_PATH_RASTER = f'{OUTPUT_PATH}/raster'
OUTPUT_PATH_PNG = f'{OUTPUT_PATH}/png'
OUTPUT_PATH_RAW = f'{OUTPUT_PATH}/raw_results'


fs = s3fs.S3FileSystem(anon=False)

# Define important parameters for this run
grid_size = 200
row_epsilon = 0.01


# Function to load data for a single city
@delayed
def load_city_data(city_name):
    """Loads buildings, roads, and intersections for a city in parallel."""
    def load_parquet(path):
        return gpd.read_parquet(path) if fs.exists(path) else None

    print(f"📥 Loading data for {city_name}...")

    buildings = load_parquet(f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet')
    roads = load_parquet(f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet')
    intersections = load_parquet(f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet')

    if buildings is not None and "dataset" in buildings.columns:
        buildings = buildings[buildings["dataset"] != "OpenStreetMap"]

    return city_name, buildings, roads, intersections
    
def load_all_cities(cities):
    """Load geographic data for all cities in parallel and return a dictionary."""
    delayed_results = [load_city_data(city) for city in cities]
    
    return delayed_results

def load_all_city_grids(cities, grid_size=200):
    """Loads the grids for all cities and creates a global processing queue."""
    grid_paths = {city: f"{GRIDS_PATH}/{city}/{city}_{grid_size}m_grid.geoparquet" for city in cities}
    
    # Read all grids in parallel
    grids = {city: dgpd_read_parquet(path) for city, path in grid_paths.items() if fs.exists(path)}
    
    # Add city name column for tracking
    for city, grid in grids.items():
        grid["city_name"] = city

    # Concatenate all city grids into a single Dask DataFrame
    return dd.concat(list(grids.values()))


def process_all_cells(global_grid, city_data):
    """Creates a global queue of all grid cells and processes them in parallel."""
    delayed_tasks = [
        process_cell(grid_id, row, city_data) #THIS NEEDS MODIFICATION AND A WHOLE OTHER FUNCTION
        for grid_id, row in global_grid.iterrows()
    ]
    return compute(*delayed_tasks)

def run_all(cities, sample_prop, grid_size=200):
    tasks = []
    delayed_cities = load_all_cities(cities)
    delayed_grids = load_all_city_grids(cities)
    visualize(delayed_cities, delayed_grids, 'dask_file.svg')
    city_data_list, grid_data_list = compute(delayed_cities, delayed_grids)
    return city_data, grid_data_list

cities = ["Belo_Horizonte", "Campinas", "Bogota", "Nairobi"]
city_data, grid_data_list = run_all(cities, sample_prop=0.05)


end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Tasks completed in {elapsed_time:.2f} seconds.")
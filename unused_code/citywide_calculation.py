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

# Helper function to get UTM CRS based on city geometry centroid
def get_utm_crs(geometry):
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_crs = CRS.from_user_input(f"+proj=utm +zone={(int((lon + 180) // 6) + 1)} +{'south' if lat < 0 else 'north'} +datum=WGS84")
    authority_code = utm_crs.to_authority()
    if authority_code is not None:
        epsg_code = int(authority_code[1])
    else:
        epsg_code = None
    return epsg_code
    
@delayed
#def process_cell(grid_id, geod, rectangle, rectangle_projected, buildings, blocks_intersecting, blocks_clipped, roads, roads_union_extended, intersections, utm_proj_city):
def process_cell(grid_id, row, city_data):
    """
    Processes a single cell using Dask Delayed.
    """
    #print(f"\U0001F539 Processing cell {grid_id} with {len(buildings)} buildings, {len(roads)} roads, {len(intersections)} intersections")

    # CASE 2: Invalid rectangle → Completely remove from dataset (exit early)
    if rectangle is None or rectangle.is_empty or not rectangle.is_valid:
        #print(f"\U0001F6A8 Skipping cell {grid_id}: Invalid rectangle {rectangle}")
        return  # <<< No return value at all (Dask ignores it)

    if rectangle_projected.is_empty or not rectangle_projected.is_valid:
        #print(f"\U0001F6A8 Cell {grid_id} has an invalid rectangle_projected! Geometry: {rectangle_projected}")
        return  # <<< No return value at all (Dask ignores it)

    #print(f"📌 DEBUG: Converting rectangle {grid_id} to WGS84")
    rectangle_wgs84 = gpd.GeoSeries(rectangle_projected, crs=f"EPSG:{utm_proj_city}").to_crs(epsg=4326).iloc[0]
    #print(f"📌 DEBUG: rectangle_wgs84 for cell {grid_id}: {rectangle_wgs84}")

    rectangle_area, _ = geod.geometry_area_perimeter(rectangle_wgs84)
    #print(f"📌 DEBUG: rectangle_area for cell {grid_id}: {rectangle_area}")

    if np.isnan(rectangle_area) or rectangle_area <= 0:
        #print(f"\U0001F6D1 Cell {grid_id} has invalid area {rectangle_area}, skipping.")
        return  # <<< No return value at all (Dask ignores it)

    try:
        # Preparatory calculations
        if not buildings.empty:
            building_area = buildings.area.sum()
            n_buildings = len(buildings)
            building_density = (1000.0 * 1000 * n_buildings) / rectangle_area if rectangle_area > 0 else np.nan
        else:
            building_area, building_density, n_buildings = np.nan, np.nan, np.nan

        # Clip intersections
        if not intersections.empty:
            intersections_bool = True
            n_intersections = len(intersections.drop_duplicates('osmid'))
        else:
            intersections_bool = False
            n_intersections = 0

        roads_bool = not roads.empty

        # CASE 1: Valid but empty cell → Return DataFrame filled with NaNs
        if not roads_bool and not intersections_bool:
            #print(f"⚠️ Assigning NAs to {grid_id}: No roads or intersections present.")
            return pd.DataFrame([{
                'grid_id': grid_id,
                'metric_1': np.nan, 'metric_2': np.nan, 'metric_3': np.nan,
                'metric_4': np.nan, 'metric_5': np.nan, 'metric_6': np.nan,
                'metric_7': np.nan, 'metric_8': np.nan, 'metric_9': np.nan,
                'metric_10': np.nan, 'metric_11': np.nan, 'metric_12': np.nan,
                'metric_13': np.nan,
                'buildings_bool': False,
                'intersections_bool': False,
                'roads_bool': False,
                'rectangle_area': rectangle_area,
                'building_area': np.nan,
                'share_tiled_by_blocks': np.nan,
                'road_length': np.nan,
                'n_intersections': np.nan,
                'n_buildings': np.nan,
                'building_density': np.nan
            }])

        # Otherwise, proceed with normal metric calculations
        if not buildings.empty and not roads.empty:
            #print(f"📌 DEBUG: Running metric_1 for cell {grid_id}")
            #print(f"📌 DEBUG: road_union geometry: {roads_union_extended if isinstance(roads_union_extended, gpd.GeoSeries) else 'Not a GeoSeries'}")
            m1, buildings = metric_1_distance_less_than_20m(buildings, roads_union_extended, utm_proj_city)
            #print(f"📌 DEBUG: Running metric_2 for cell {grid_id}")
            m2 = metric_2_average_distance_to_roads(buildings)
        else:
            m1, m2 = np.nan, np.nan

        #print(f"📌 DEBUG: metric_1 for cell {grid_id} = {m1}")
        #print(f"📌 DEBUG: metric_2 for cell {grid_id} = {m2}")

        #print(f"📌 DEBUG: Running metric_3 for cell {grid_id}")
        m3 = metric_3_road_density(rectangle_area, roads) if not roads.empty else 0
        #print(f"📌 DEBUG: metric_3 for cell {grid_id} = {m3}")

        if not intersections.empty:
            #print(f"📌 DEBUG: Running metric_4 for cell {grid_id}")
            m4 = metric_4_share_4way_intersections(intersections)
            #print(f"📌 DEBUG: Running metric_5 for cell {grid_id}")
            m5 = metric_5_intersection_density(intersections, rectangle_area)
        else:
            m4, m5 = (np.nan if not roads.empty else 0), 0

        #print(f"📌 DEBUG: metric_4 for cell {grid_id} = {m4}")
        #print(f"📌 DEBUG: metric_5 for cell {grid_id} = {m5}")
        #print(f"📌 DEBUG: Running metric_6 for cell {grid_id}")
        m6 = (
            metric_6_entropy_of_building_azimuth(buildings, rectangle_id=1, bin_width_degrees=5, plot=False)[0]
            if not buildings.empty else np.nan
        )
        #print(f"📌 DEBUG: metric_6 for cell {grid_id} = {m6}")

        if not blocks_intersecting.empty:
            area_tiled_by_blocks = blocks_clipped.area.sum()
            share_tiled_by_blocks = area_tiled_by_blocks / rectangle_area
            #print(f"📌 DEBUG: Running metric_7 for cell {grid_id}")
            m7, blocks_clipped = metric_7_average_block_width(blocks_intersecting, blocks_clipped, rectangle_projected, rectangle_area)
            #print(f"📌 DEBUG: Running metric_8 for cell {grid_id}")
            m8, _, _ = metric_8_two_row_blocks(blocks_intersecting, buildings, utm_proj_city, row_epsilon=row_epsilon)
        else:
            m7, m8, share_tiled_by_blocks = np.nan, np.nan, 0

        #print(f"📌 DEBUG: metric_7 for cell {grid_id} = {m7}")
        #print(f"📌 DEBUG: metric_8 for cell {grid_id} = {m8}")
        #print(f"📌 DEBUG: Running metric_9 for cell {grid_id}")
        m9 = metric_9_tortuosity_index(roads) if not roads.empty else np.nan
        #print(f"📌 DEBUG: metric_9 for cell {grid_id} = {m9}")
        #print(f"📌 DEBUG: Running metric_10 for cell {grid_id}")
        m10 = metric_10_average_angle_between_road_segments(intersections, roads) if not roads.empty and not intersections.empty else np.nan
        #print(f"📌 DEBUG: metric_10 for cell {grid_id} = {m10}")

        road_length = roads.length.sum() if not roads.empty else np.nan

        if not buildings.empty:
            #print(f"📌 DEBUG: Running metric_11 for cell {grid_id}")
            m11 = metric_11_building_density(n_buildings, rectangle_area)
            #print(f"📌 DEBUG: metric_11 for cell {grid_id} = {m11}")
            #print(f"📌 DEBUG: Running metric_12 for cell {grid_id}")
            m12 = metric_12_built_area_share(building_area, rectangle_area)
            #print(f"📌 DEBUG: metric_12 for cell {grid_id} = {m12}")
            #print(f"📌 DEBUG: Running metric_13 for cell {grid_id}")
            m13 = metric_13_average_building_area(building_area, n_buildings)
            #print(f"📌 DEBUG: metric_13 for cell {grid_id} = {m13}")
        else:
            m11, m12, m13 = 0, 0, np.nan

        # Final result
        result_df = pd.DataFrame([{
            'grid_id': grid_id,
            'metric_1': float(m1), 'metric_2': float(m2), 'metric_3': float(m3),
            'metric_4': float(m4), 'metric_5': float(m5), 'metric_6': float(m6),
            'metric_7': float(m7), 'metric_8': float(m8), 'metric_9': float(m9),
            'metric_10': float(m10), 'metric_11': float(m11), 'metric_12': float(m12),
            'metric_13': float(m13),
            'buildings_bool': bool(not buildings.empty),
            'intersections_bool': bool(intersections_bool),
            'roads_bool': bool(roads_bool),
            'rectangle_area': float(rectangle_area) if not np.isnan(rectangle_area) else 0.0,
            'building_area': float(building_area) if not np.isnan(building_area) else 0.0,
            'share_tiled_by_blocks': float(share_tiled_by_blocks) if not np.isnan(share_tiled_by_blocks) else 0.0,
            'road_length': float(road_length) if not np.isnan(road_length) else 0.0,
            'n_intersections': int(n_intersections) if not np.isnan(n_intersections) else 0,
            'n_buildings': int(n_buildings) if not np.isnan(n_buildings) else 0,
            'building_density': float(building_density) if not np.isnan(building_density) else 0.0
        }])
        #print(f"📌 DEBUG: Preparing final DataFrame for cell {grid_id}")
        #print(f"✅ Successfully calculated all metrics for cell {grid_id}")
        #print(f"📌 DEBUG: Returning final DataFrame for cell {grid_id}:")
        #print(result_df)
        return result_df

    except Exception as e:
        import traceback
        print(f"❌ Error processing cell {grid_id}: {e}")
        print(f"🔍 Debug Info: rectangle={rectangle}, rectangle_projected={rectangle_projected}")
        traceback.print_exc()
        raise  # <<< No return value at all (Dask ignores it)

def extract_confidence_and_dataset(df):
    """Extract confidence and dataset from the sources column."""
    
    def extract_first_value(x, key):
        """Extract the key from the first dictionary inside an array."""
        if isinstance(x, np.ndarray) and len(x) > 0:  # Check if it's a NumPy array
            first_entry = x[0]  # Get the first dictionary
            if isinstance(first_entry, dict) and key in first_entry:
                return first_entry[key]
        return np.nan  # Return NaN if missing or incorrect format

    df["confidence"] = df["sources"].apply(lambda x: extract_first_value(x, "confidence"))
    df["dataset"] = df["sources"].apply(lambda x: extract_first_value(x, "dataset"))
    
    return df

@delayed
def project_and_process(buildings, roads, intersections):   
    print(f"👾 Entering project and process.") 
    # Check if any data are missing
    if buildings is None or buildings.empty:
        print("❌ Error: No building data available.")
        return None
    else:
        print(f"✅ Buildings data loaded with {len(buildings)} records.")
    
    if roads is None or roads.empty:
        print("❌ Error: No road data available.")
        return None
    else:
        # Get UTM projection for the city
        first_row_df = roads.head(1)  
        if isinstance(first_row_df, pd.Series):
            first_row_df = first_row_df.to_frame().T  

        if not first_row_df.empty:
            first_row = first_row_df.iloc[0]  
        else:
            print("❌ Error: Unable to determine projection based on road network")
            return None

        utm_proj_city = get_utm_crs(first_row.geometry)  
        if utm_proj_city is None:
            print("❌ Error: Unable to determine EPSG code for city.")
            return None  

    if intersections is None or intersections.empty:
        print("❌ Error: No intersections data available.")
        return None

    try:
        OSM_roads_all_projected = roads.to_crs(epsg=utm_proj_city)
        OSM_intersections_all_projected = intersections.to_crs(epsg=utm_proj_city) if intersections is not None else None
        Overture_data_all_projected = buildings.to_crs(epsg=utm_proj_city) if buildings is not None else None
        print(f"✅ Succesfuly projected buildings, roads and intersections.")
    except Exception as e:
        print(f"❌ Error reprojecting data for city: {e}")
        return None

    try: 
        road_union = OSM_roads_all_projected.unary_union
        if road_union.is_empty:
            print(f"❌ Road union is empty.")
            return None
        else:
            print(f"✅ Succesfully performed unary union on road network.")
    except:
        print(f"❌ Error performing unary unary union on road network.")
        return None

    try:
        if not OSM_roads_all_projected.empty:
            blocks = get_blocks(road_union, OSM_roads_all_projected)
            if blocks.empty:
                print(f"❌ Resulting blocks are empty.")
                return None
            else:
                print(f"✅ Succesfully calculated blocks.")
        else:
            print(f"❌ Roads were not correctly projected and blocks cannot be built.")
            return None
    except:
        print(f"❌ Error calculating blocks: {e}")
        return None

    # Debugging: Print What is Being Returned
    print(f"📦 Returning from project_and_process() for city:")
    print(f"   - Overture: {type(Overture_data_all_projected)}")
    print(f"   - Blocks: {type(blocks)}")
    print(f"   - Roads: {type(OSM_roads_all_projected)}")
    print(f"   - Intersections: {type(OSM_intersections_all_projected)}")
    print(f"   - Road union: {type(road_union)}")
    print(f"   - UTM Projection: {utm_proj_city}")

    print('ROAD UNION')
    print(road_union)
    # **Ensure all dictionary keys exist**
    result = {
        "overture": Overture_data_all_projected,
        "blocks": blocks,
        "roads": OSM_roads_all_projected,
        "intersections": OSM_intersections_all_projected,
        "road_union": road_union,
        "utm_proj": utm_proj_city
    }

    del buildings
    del roads
    del intersections

    return result


def ensure_valid_geodata(data):
    """Ensures the data is a valid Dask GeoDataFrame before accessing attributes like `.empty`."""
    if isinstance(data, (dgpd.GeoDataFrame, dd.DataFrame)):  # Already lazy? Keep it.
        return data
    elif isinstance(data, gpd.GeoDataFrame):  # Not lazy? Convert it.
        return dgpd.from_geopandas(data, npartitions=10)
    else:  # If None or unexpected type, return an empty lazy GeoDataFrame.
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=["geometry"], geometry="geometry"), npartitions=1)

def clip_features_to_rectangles(rectangles, city_data, buffer_size=300):
    print(f"👾 Entering clip_features_to_rectangles.") 

    print(f"📊 Checking `rectangles`: {type(rectangles)}, npartitions={getattr(rectangles, 'npartitions', 'Unknown')}")
    assert isinstance(rectangles, (dd.DataFrame, dgpd.GeoDataFrame)), "❌ `rectangles` was computed too early!"

    def lazy_clip_partition(df):
        """Ensure `city_name` stays lazy and process partition correctly."""
        if "city_name" not in df.columns:
            print(f"⚠️ `city_name` missing in DataFrame: {df.columns}")
            return pd.DataFrame()

        unique_cities = df["city_name"].unique()
        if len(unique_cities) != 1:
            print(f"🚨 Error: Multiple cities found in partition! {unique_cities}")
            return pd.DataFrame()  

        city_name = unique_cities[0]

        # Get city-specific data lazily
        city = city_data.get(city_name, None)
        if city is None:
            print(f"⚠️ No data found for {city_name}, returning empty DataFrame.")
            return pd.DataFrame()

        return df.assign(
            buildings=ensure_valid_geodata(city.get("overture")),
            roads=ensure_valid_geodata(city.get("roads")),
            roads_union_extended=delayed(lambda x, y: x.intersection(y))(city.get("road_union", MultiLineString([])), df["geometry"]),
            intersections=ensure_valid_geodata(city.get("intersections")),
            blocks_clipped=ensure_valid_geodata(city.get("blocks")),
        )

    feature_mapping = rectangles.map_partitions(
        lazy_clip_partition,
        meta=pd.DataFrame({
            "grid_id": pd.Series(dtype="int"),
            "city_name": pd.Series(dtype="str"),
            "buildings": pd.Series(dtype="object"),
            "roads": pd.Series(dtype="object"),
            "roads_union_extended": pd.Series(dtype="object"),
            "intersections": pd.Series(dtype="object"),
            "blocks_clipped": pd.Series(dtype="object"),
        })
    )

    return feature_mapping



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

    return project_and_process(buildings, roads, intersections)

# Load all cities in parallel
def load_all_cities(cities):
    """Returns a Dask-delayed dictionary mapping city names to their processed data."""
    return {city: load_city_data(city) for city in cities}


def load_city_grid(city, grid_size):
    path = f"{GRIDS_PATH}/{city}/{city}_{grid_size}m_grid.geoparquet"
    
    if not fs.exists(path):
        print(f"🚨 File not found: {path}")
        return dd.from_pandas(gpd.GeoDataFrame(columns=["grid_id", "geometry"], geometry="geometry"), npartitions=1)  
    
    grid = dgpd_read_parquet(path)  # Read as Dask GeoDataFrame

    # Lazily add a city name column
    grid = grid.assign(city_name=city)

    # Reset index and rename columns in a **Dask-friendly** way
    grid = grid.map_partitions(lambda df: df.reset_index(drop=False).rename(columns={"index": "grid_id"}))
    grid = grid.map_partitions(lambda df: df.assign(grid_id=df["grid_id"].astype(int)))


    return grid  # Always returns a Dask GeoDataFrame


def load_all_city_grids(cities, grid_size=200):
    grids = [load_city_grid(city, grid_size) for city in cities]
    grids = [g for g in grids if g is not None]  # Remove any None entries
    global_grid = dd.concat(grids) if len(grids) > 1 else (grids[0] if grids else None)
    return global_grid


def process_all_cells(global_feature_mapping):
    """Processes all grid cells in parallel without using iterrows()."""
    
    def process_partition(df_partition):
        return df_partition.apply(lambda row: process_cell(row.grid_id, row), axis=1)

    return global_feature_mapping.map_partitions(process_partition)


def load_global_feature_mapping(city_data, global_grid, sample_prop):
    """Creates a global feature mapping using Dask parallelization."""

    sampled_grid = global_grid.sample(frac=sample_prop, random_state=42)  # Ensure distributed sampling

    meta = pd.DataFrame({
        "grid_id": pd.Series(dtype="int"),
        "city_name": pd.Series(dtype="str"),
        "buildings": pd.Series(dtype="object"),
        "roads": pd.Series(dtype="object"),
        "roads_union_extended": pd.Series(dtype="object"),
        "intersections": pd.Series(dtype="object"),
        "blocks_clipped": pd.Series(dtype="object"),
    })

    def lazy_clip_partition(df, city_data):
        city_name = df["city_name"].values[0] 
        return clip_features_to_rectangles(city_name, city_data, df)

    # Ensure partitions remain lazy
    global_feature_mapping = sampled_grid.map_partitions(
        lazy_clip_partition,
        city_data,  # Pass separately so Dask does NOT compute it too soon
        meta=meta
    )

    return global_feature_mapping


def run_all(cities,sample_prop):
    print("📥 Loading and processing city data...")
    city_data = load_all_cities(cities)  # Now delayed!

    print("📊 Loading global city grids...")
    global_grid = load_all_city_grids(cities)  # Still delayed!

    print("📊 Creating global feature mapping...")
    global_feature_mapping = load_global_feature_mapping(city_data, global_grid, sample_prop)
    visualize(global_feature_mapping,'global_feature_mapping.svg')

    #return compute(global_feature_mapping)

    #print("⚡ Processing all cells in parallel...")
    #processed_cells = process_all_cells(global_feature_mapping)

    # Compute everything **at the last step**
    #with Profiler() as prof, ResourceProfiler(dt=1) as rprof, ProgressBar():
    #    compute(processed_cells, timeout=1200)

    #prof.visualize(filename="profiler_graph.svg")



if __name__ == "__main__":
    cities = ["Belo_Horizonte", "Campinas", "Bogota"]
    run_all(cities, sample_prop=0.05)
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
def process_cell(grid_id, geod, rectangle, rectangle_projected, buildings, blocks_intersecting, blocks_clipped, roads, roads_union_extended, intersections, utm_proj_city):
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

def load_buildings(city_name):
    path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet'

    if not fs.exists(path):
        print(f"Missing buildings data for city {city_name}. Skipping.")
        buildings = None
    else:
        # Load directly as a GeoDataFrame
        buildings = gpd.read_parquet(path)

        if "sources" in buildings.columns:
            buildings = extract_confidence_and_dataset(buildings)

            if "dataset" in buildings.columns:
                buildings = buildings[buildings["dataset"] != "OpenStreetMap"]

        print(f"✅ {city_name}: Successfully loaded Overture buildings.")
    return buildings

def load_roads(city_name):
    path_parquet = f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet'

    if not fs.exists(path_parquet):
        print(f"⚠️ No roads data found for {city_name}. Skipping.")
        roads = None
    else:
        try:
            print(f"📂 Loading Parquet roads data for {city_name}...")
            roads = gpd.read_parquet(path_parquet)  # Directly load as GeoDataFrame
            print(f"✅ Successfully loaded roads data for {city_name}")
            print(f"   - Columns: {list(roads.columns)}")
            #print(f"   - CRS: {roads.crs}")

        except Exception as e:
            print(f"❌ Error loading Parquet roads data for {city_name}: {e}")
            roads = None

    return roads

def load_intersections(city_name):
    
    path = f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'

    if not fs.exists(path):
        print(f"Missing intersections data for city {city_name}. Skipping.")
        intersections = None
    else:
        try:
           intersections = gpd.read_parquet(path)  # Directly load as GeoDataFrame
           intersections['osmid'] = intersections['osmid'].astype('int64')
           intersections = intersections[intersections.street_count>2]
           print(f"✅ Successfully loaded intersections data for {city_name}")
        except Exception as e:
            print(f"❌ Error loading intersections data for {city_name}: {e}")
            intersections = None
    return intersections

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
        first_row_df = roads.head(1)  # This may return a Series if only one column exists
        if isinstance(first_row_df, pd.Series):
            first_row_df = first_row_df.to_frame().T  # Convert Series to DataFrame

        if not first_row_df.empty:
            first_row = first_row_df.iloc[0]  # Safely access first row
        else:
            print("❌ Error: Unable to determine projection based on road network")
            return None

        utm_proj_city = get_utm_crs(first_row.geometry)  
        if utm_proj_city is None:
            print("❌ Error: Unable to determine EPSG code for city.")
            return None  # Fail early if we can't determine the UTM CRS

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

    # **Ensure all dictionary keys exist**
    result = {
        "overture": Overture_data_all_projected,
        "blocks": blocks,
        "roads": OSM_roads_all_projected,
        "intersections": OSM_intersections_all_projected,
        "road_union": road_union,
        "utm_proj": utm_proj_city
    }

    return result

def clip_features_to_rectangles(city_data, rectangles, buffer_size=300):
    """
    Clips buildings, roads, and intersections to each rectangle.
    Returns a dictionary associating each rectangle ID with its features.
    """

    print(f"👾 Entering project and process.") 
    buildings, roads, intersections, blocks, road_union = (
        city_data['overture'], city_data['roads'], city_data['intersections'], city_data['blocks'], city_data['road_union']
    )

    # Create spatial indexes for efficient lookup
    building_index = STRtree(buildings.geometry) if not buildings.empty else None
    road_index = STRtree(roads.geometry) if not roads.empty else None
    intersection_index = STRtree(intersections.geometry) if not intersections.empty else None
    blocks_index = STRtree(blocks.geometry) if not blocks.empty else None

    rectangle_features = {}

    for rect_id, rect_geom in zip(rectangles.index, rectangles.geometry):
        #print(f"🟡 Clipping features for rectangle {rect_id}")
        rect_id = int(rect_id)

        rect_box = rect_geom.bounds  # Bounding box for spatial index lookup
        rect_buffered = rect_geom.buffer(buffer_size)  # Expanded area for roads

        # **Buildings: Retrieve all buildings that intersect the rectangle**
        if building_index:
            building_candidates_idx = building_index.query(rect_geom)
            buildings_in_rect = buildings.iloc[building_candidates_idx]
            #print(f"📊 Before Clipping: {len(buildings_in_rect)} buildings in rectangle {rect_id}")

        else:
            buildings_in_rect = gpd.GeoDataFrame(columns=buildings.columns, crs=buildings.crs)

        # **Blocks: Retrieve all blocks that intersect the rectangle**
        if blocks_index:
            blocks_candidates_idx = blocks_index.query(rect_geom)
            blocks_intersecting_rect = blocks.iloc[blocks_candidates_idx]
            #print(f"📊 Before Clipping: {len(blocks_intersecting_rect)} blocks in rectangle {rect_id}")
        else:
            blocks_intersecting_rect = gpd.GeoDataFrame(columns=blocks.columns, crs=blocks.crs)

        # **Blocks: Retrieve cookie-cutter blocks inside the rectangle**
        if blocks_index: 
            blocks_candidates_idx = blocks_index.query(rect_geom)
            blocks_within_rect = gpd.clip(blocks.iloc[blocks_candidates_idx], rect_geom)
            #print(f"📊 After Clipping: {len(blocks_within_rect)} blocks in rectangle {rect_id}")
        else:
            blocks_within_rect = gpd.GeoDataFrame(columns=blocks.columns, crs=blocks.crs)

        # **Roads: Retrieve and clip roads inside the rectangle**
        if road_index:
            road_candidates_idx = road_index.query(rect_geom)
            roads_in_rect = roads.iloc[road_candidates_idx]
            #print(f"📊 Before Clipping: {len(roads_in_rect)} roads in rectangle {rect_id}")
            
            roads_in_rect = gpd.clip(roads_in_rect, rect_geom)
            #print(f"📊 After Clipping: {len(roads_in_rect)} roads in rectangle {rect_id}")
        else:
            roads_in_rect = gpd.GeoDataFrame(columns=roads.columns, crs=roads.crs)

        # **Expanded Roads: Retrieve and clip roads in the buffered region**
        roads_union_extended = road_union.intersection(rect_buffered)
        if not roads_union_extended.is_empty:
            roads_union_extended = roads_union_extended  # Keep as MultiLineString
        else:
            roads_union_extended = None

        # **Intersections: Retrieve and clip intersections inside the rectangle**
        if intersection_index:
            intersection_candidates_idx = intersection_index.query(rect_geom)
            intersections_in_rect = intersections.iloc[intersection_candidates_idx]
            #print(f"📊 Before Clipping: {len(intersections_in_rect)} intersections in rectangle {rect_id}")
            
            intersections_in_rect = gpd.clip(intersections_in_rect, rect_geom)
            #print(f"📊 After Clipping: {len(intersections_in_rect)} intersections in rectangle {rect_id}")
        else:
            intersections_in_rect = gpd.GeoDataFrame(columns=intersections.columns, crs=intersections.crs)

        # Store the results
        rectangle_features[rect_id] = {
            "buildings": buildings_in_rect if not buildings_in_rect.empty else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=buildings.crs),
            "roads": roads_in_rect if not roads_in_rect.empty else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=roads.crs),
            "roads_union_extended": roads_union_extended,  # Now correctly stored as a geometry
            "intersections": intersections_in_rect if not intersections_in_rect.empty else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=intersections.crs),
            "blocks_clipped": blocks_within_rect if not blocks_within_rect.empty else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=blocks.crs),
            "blocks_intersecting": blocks_intersecting_rect if not blocks_intersecting_rect.empty else gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=blocks.crs)
        }


    #print(f"🧐 Clipped {len(rectangle_features)} cells")

    return rectangle_features

@delayed
def process_city(city_name, city_data, sample_prop, override_processed=False, grid_size=200):
    print(f"👾 Entering process_city in {city_name}.") 
    if city_data is not None:
        try:
            print(f"📌 Starting processing for {city_name}")

            # Load city grid as Dask GeoDataFrame
            city_grid = dgpd.read_parquet(f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet')

            # Debug: Print initial columns and index
            print(f"📂 {city_name} city grid columns: {list(city_grid.columns)}")
            print(f"🆔 {city_name} city grid index name: {city_grid.index.name}")

            # Convert index to column if 'grid_id' is missing
            if "grid_id" not in city_grid.columns:
                print(f"🔄 Assigning index as 'grid_id' for {city_name}")
                city_grid = city_grid.reset_index()
                if "index" in city_grid.columns:  # Check if reset_index() created "index"
                    city_grid = city_grid.rename(columns={"index": "grid_id"})
                else:
                    raise ValueError(f"🚨 Failed to generate 'grid_id' for {city_name}. Check Parquet file structure.")

            # Ensure `city_grid` is not empty
            if city_grid.compute().empty:
                raise ValueError(f"🚨 No grid data found for {city_name}. Check if the file exists or is corrupt.")

            print(f"🗺 CRS Check for {city_name}:")
            print(f"   📌 Grid CRS: {city_grid.crs}")
            print(f"   🏢 Buildings CRS: {city_data['overture'].crs if 'overture' in city_data else 'Missing'}")
            print(f"   🛣 Roads CRS: {city_data['roads'].crs if 'roads' in city_data else 'Missing'}")
            print(f"   🚦 Intersections CRS: {city_data['intersections'].crs if 'intersections' in city_data else 'Missing'}")

            # If CRS is inconsistent, convert everything to UTM
            utm_proj_city = city_data.get("utm_proj", None)
            if utm_proj_city and city_grid.crs and city_grid.crs.to_epsg() != utm_proj_city:
                #print(f"🔄 Reprojecting {city_name} to {utm_proj_city}")
                city_grid = city_grid.to_crs(epsg=utm_proj_city)

            # Ensure city_grid has a geometry column
            if 'geometry' not in city_grid.columns:
                raise ValueError(f"🚨 {city_name}: 'geometry' column is missing in city grid!")

            # OJO: AN EXTRA STEP WILL BE NEEDED HERE BECAUSE THE PROCESSED FLAG WILL BE HANDLED SEPARATELY.
            # Initialize 'processed' column in a Dask-friendly way
            city_grid["grid_id"] = city_grid["grid_id"].astype(int)
            city_grid = city_grid.assign(processed=False)

            # Sample unprocessed cells
            unprocessed_grid = city_grid[~city_grid['processed']]
            sampled_grid = unprocessed_grid.sample(frac=sample_prop, random_state=42) if sample_prop < 1.0 else unprocessed_grid
            sampled_grid["grid_id"] = sampled_grid["grid_id"].astype(int)

            # Ensure sampled_grid is not empty
            if sampled_grid.npartitions == 0:
                print(f"⚠️ Skipping {city_name}: No unprocessed cells left after sampling.")
                return

            # Ensure sampled_grid has `grid_id`
            if "grid_id" not in sampled_grid.columns:
                raise ValueError(f"🚨 'grid_id' column missing after sampling in {city_name}. Columns present: {list(sampled_grid.columns)}")

            # Mark sampled cells as processed
            city_grid = city_grid.assign(processed=city_grid['grid_id'].isin(sampled_grid['grid_id']))

            sampled_grid = sampled_grid.set_index("grid_id")

            # Clip features for each rectangle
            #print(f"✂️ Clipping features for {city_name}...")
            rectangle_features = clip_features_to_rectangles(city_data, rectangles=sampled_grid, buffer_size=300)

            # Ensure clipping worked correctly
            if not rectangle_features:
                raise ValueError(f"🚨 Clipping returned empty results for {city_name}. Check input data.")
            
            #print(f"🔍 city_grid['grid_id'] sample: {list(city_grid['grid_id'].compute()[:10])}")
            #print(f"🔍 rectangle_features.keys(): {list(rectangle_features.keys())[:10]}")

            # Extract processed city data
            geod = Geod(ellps="WGS84")
            sampled_grid["geometry_projected"] = sampled_grid["geometry"].to_crs(epsg=utm_proj_city)

            #print(f"🧐 Available rectangle_features keys: {list(rectangle_features.keys())[:10]}")
            #print(f"🧐 Sampled grid IDs: {list(sampled_grid.reset_index()['grid_id'].compute()[:10])}")
            #print(f"🧐 Sampled grid index: {sampled_grid.index}, grid_id: {sampled_grid.reset_index()['grid_id'].unique()}")

            # Parallelize Cell Processing

            sampled_grid.index = sampled_grid.index.astype(int)  # Ensure it's an integer index
            rectangle_features = {int(k): v for k, v in rectangle_features.items()}  # Match types

            #print(f"🧐 Rectangle feature keys: {list(rectangle_features.keys())[:10]}")

            delayed_results = []
            for grid_id, row in sampled_grid.iterrows():
                if grid_id in rectangle_features:
                    rectangle_projected = gpd.GeoSeries([row["geometry_projected"]], crs=f"EPSG:{utm_proj_city}").iloc[0]
                    rectangle = row["geometry"]
                    rectangle = sampled_grid.loc[grid_id, 'geometry'].compute()
                    if isinstance(rectangle, pd.Series):
                        rectangle = rectangle.iloc[0]
                    cell_features = rectangle_features.get(grid_id, {})

                    #print(f"🔍 Checking cell {grid_id}: Features found? {bool(cell_features)}")

                    if not cell_features:
                        print(f"⚠️ No features found for cell {grid_id} in {city_name}. Skipping.")
                        continue  # Skip empty cells

                    #print(f"✅ Adding cell {grid_id} to delayed_results for {city_name}")

                    if rectangle is None or rectangle.is_empty or not rectangle.is_valid:
                        print(f"🚨 Skipping cell {grid_id}: Invalid rectangle {rectangle}")
                        continue  # Skip this cell

                    #print(f"📏 Checking cell {grid_id}: rectangle={rectangle}")
                    if rectangle is None or rectangle.is_empty:
                        print(f"🚨 Skipping cell {grid_id}: Invalid or empty rectangle")
                        continue
                    else:
                        processed_cell = delayed(process_cell)(
                            grid_id, geod, rectangle, rectangle_projected,
                            cell_features.get("buildings", gpd.GeoDataFrame()),
                            cell_features.get("blocks_intersecting", gpd.GeoDataFrame()),
                            cell_features.get("blocks_clipped", gpd.GeoDataFrame()),
                            cell_features.get("roads", gpd.GeoDataFrame()),
                            cell_features.get("roads_union_extended", gpd.GeoDataFrame()),
                            cell_features.get("intersections", gpd.GeoDataFrame()),
                            utm_proj_city
                        )
                        delayed_results.append(processed_cell)
                else:
                    print(f" ⚠️ Cell ID NOT in rectangle features")

            # Before calling from_delayed(), print metadata
            #print(f"🔍 DEBUG: Checking delayed_results before creating Dask DataFrame for {city_name}")

            # Ensure that delayed_results is not empty
            if not delayed_results:
                print(f"🚨 No valid processed cells for {city_name}. Check if all cells were skipped.")
                
            # Print the first few elements to check their types
            for i, result in enumerate(delayed_results[:5]):  # Checking first 5
                print(f"   - Cell {i}: Type={type(result)}")

            # DEBUGGING: Check what is inside `delayed_results` before calling `from_delayed()`
            #print(f"🔍 DEBUG: Checking `delayed_results` for {city_name}")
            if not delayed_results:
                print(f"🚨 No valid processed cells for {city_name}. Check if all cells were skipped.")

            first_result = delayed_results[0].compute()

            # Now, try creating Dask DataFrame with only valid results
            final_geo_df = dd.from_delayed(delayed_results, meta={
                'grid_id': int, 'metric_1': float, 'metric_2': float, 'metric_3': float, 'metric_4': float,
                'metric_5': float, 'metric_6': float, 'metric_7': float, 'metric_8': float,
                'metric_9': float, 'metric_10': float, 'metric_11': float, 'metric_12': float, 'metric_13': float,
                'buildings_bool': bool, 'intersections_bool': bool, 'roads_bool': bool,
                'rectangle_area': float, 'building_area': float, 'share_tiled_by_blocks': float,
                'road_length': float, 'n_intersections': int, 'n_buildings': int, 'building_density': float
            })

            final_geo_df = final_geo_df.persist()

            # Merge back with city grid

            if not isinstance(final_geo_df, dgpd.GeoDataFrame):
                print("🔄 Converting final_geo_df to Dask GeoDataFrame")
                final_geo_df = dgpd.from_dask_dataframe(final_geo_df)

            # Save to S3
            output_name = f"{city_name}_{grid_size}m_results"
            remote_path = f"{OUTPUT_PATH_RAW}/{city_name}/raw_results_{grid_size}"
            output_temp_path = "."

            # Temporary save for computation
            temporary_folder_for_computation = f"{output_temp_path}/{output_name}/"
            final_geo_df.to_parquet(temporary_folder_for_computation, 
                                    engine="pyarrow", 
                                    compute=True)
            print(f"✅ Temporary save for computation for city {city_name} in folder {temporary_folder_for_computation}")

            # Upload to S3
            output_path = S3Path(remote_path)
            output_path.upload_from(temporary_folder_for_computation)

            print(f"✅ Successfully processed and saved {city_name} to {output_path}")

            if os.path.exists(temporary_folder_for_computation):
                shutil.rmtree(temporary_folder_for_computation)

            print(f"✅ (And temporary file was deleted) {city_name} to {output_path}")

        except Exception as e:
            print(f"❌ Error processing {city_name}: {e}")
            raise
    else:
        print(f"❌ Received empty geographic features")  

def run_all_citywide_calculation(cities, sample_prop, grid_size=200):
    tasks = []
    for city in cities:
        
        # Load geographic features
        buildings = load_buildings(city)
        roads = load_roads(city)
        intersections = load_intersections(city)

        # Project data 
        city_data = project_and_process(buildings, roads, intersections).compute()
        if city_data is not None:
            task = process_city(city, city_data, sample_prop=sample_prop, grid_size=grid_size)
        else:
            print(f"❌ Error processing {city}: some geographic features are missing.")
            continue

        tasks.append(task)
    tasks[0].visualize(filename="dask_graph.svg", format="svg")
    with Profiler() as prof, ResourceProfiler(dt=1) as rprof, ProgressBar():
        #compute(*tasks)
        compute(*tasks, timeout=1200)
    prof.visualize(filename="profiler_graph.svg")

# This function allows external cluster calls like Code C.
if __name__ == "__main__":
    cities = ["Belo_Horizonte", "Campinas", "Bogota"]
    run_all_citywide_calculation(cities, sample_prop=0.05)

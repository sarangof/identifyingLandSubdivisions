import dask_geopandas as dgpd
import geopandas as gpd
from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely.errors import ShapelyError
import dask
from dask import delayed
import pandas as pd
import numpy as np
from shapely.wkb import loads as wkb_loads
from dask import compute
import s3fs
import fsspec
import traceback
import os
from shapely.ops import unary_union, polygonize
from auxiliary_functions import *

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
CITY_INFO_PATH = f'{INPUT_PATH}/city_info'
EXTENTS_PATH = f'{CITY_INFO_PATH}/extents'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
BUILDINGS_DISTANCES_PATH = f'{INPUT_PATH}/buildings_with_distances'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
SEARCH_BUFFER_PATH = f'{INPUT_PATH}/city_info/search_buffers'
BLOCKS_PATH = f'{INPUT_PATH}/blocks'
OUTPUT_PATH = f'{MAIN_PATH}/output'
OUTPUT_PATH_CSV = f'{OUTPUT_PATH}/csv'
OUTPUT_PATH_RASTER = f'{OUTPUT_PATH}/raster'
OUTPUT_PATH_PNG = f'{OUTPUT_PATH}/png'
OUTPUT_PATH_RAW = f'{OUTPUT_PATH}/raw_results'



max_distance = 200.
default_distance = 500.

'''
AUX FUNCTIONS TO CALCULATE AND SAVE BUILDING DISTANCE TO CLOSEST ROAD 
(FOR METRICS 1 AND 2)
'''

def compute_distance_partition(buildings_df, roads_geom_list, max_distance, default_distance):
    tree = STRtree(roads_geom_list)

    def distance_fn(bgeom):
        try:
            bgeom = shape(bgeom) if not isinstance(bgeom, BaseGeometry) else bgeom
            nearby_indices = tree.query(bgeom.buffer(max_distance))
            if nearby_indices is None or len(nearby_indices) == 0:
                return default_distance
            nearby_geoms = [roads_geom_list[i] for i in nearby_indices]
            return min(bgeom.distance(road) for road in nearby_geoms)
        except Exception:
            return default_distance

    buildings_df = buildings_df.copy()
    buildings_df['geometry'] = buildings_df['geometry'].apply(shape)  # extra safe
    buildings_df["distance_to_nearest_road"] = buildings_df.geometry.apply(distance_fn)
    return buildings_df


@delayed
def calculate_building_distances_to_roads(city_name, grid_size=200):
    paths = {
    'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
    'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
    'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet'
    }
    epsg = get_epsg(city_name).compute()  
    # Load and prepare roads for spatial index
    roads = load_dataset(paths['roads'], epsg=epsg).compute()
    included_road_types = ['trunk','motorway','primary','secondary','tertiary','primary_link','secondary_link','tertiary_link','trunk_link','motorway_link','residential','unclassified','road','living_street']
    def highway_filter(highway_value):
        # If highway_value is missing, return False
        if pd.isna(highway_value):
            return False
        # Split the string by commas, and strip any whitespace from each part
        types = [part.strip() for part in highway_value.split(',')]
        # Return True if any of the types is in our included list
        return any(t in included_road_types for t in types)

    # Now filter the roads GeoDataFrame:
    roads = roads[roads['highway'].apply(highway_filter)]

    roads_geom_list = [geom for geom in roads.geometry]

    # Load buildings lazily
    buildings = load_dataset(paths['buildings'], epsg=epsg)

    meta = buildings._meta.assign(distance_to_nearest_road='f8')

    # Apply distance computation per partition
    buildings_with_dist = buildings.map_partitions(
        compute_distance_partition,
        roads_geom_list,
        max_distance,
        default_distance,
        meta=meta
    )

    # Write output
    columns_to_keep = ['id', 'geometry','distance_to_nearest_road']
    buildings_with_dist = buildings_with_dist[columns_to_keep].set_index('id')
    out_path = paths['buildings'].replace(".geoparquet", "_with_distances.geoparquet")
    buildings_with_dist.to_parquet(out_path)
    return out_path

'''
AUX FUNCTIONS TO CREATE BLOCKS 
(USED BY METRICS 6, 7 AND 8)
'''

@delayed
def produce_blocks(city_name,YOUR_NAME,grid_size):
    # Construct file paths for the city
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }
    
    epsg = get_epsg(city_name).compute()
    
    roads = load_dataset(paths['roads'], epsg=epsg).compute()
    included_road_types = ['trunk','motorway','primary','secondary','tertiary','primary_link','secondary_link','tertiary_link','trunk_link','motorway_link','residential','unclassified','road','living_street']
    
    def highway_filter(highway_value):
        # If highway_value is missing, return False
        if pd.isna(highway_value):
            return False
        # Split the string by commas, and strip any whitespace from each part
        types = [part.strip() for part in highway_value.split(',')]
        # Return True if any of the types is in our included list
        return any(t in included_road_types for t in types)

    # Now filter the roads GeoDataFrame:
    roads = roads[roads['highway'].apply(highway_filter)]
    
    blocks = get_blocks(roads)

    # Now add the inscribed circle information.
    blocks = add_inscribed_circle_info(blocks)
    
    # Define the output path for the blocks geoparquet
    path_blocks = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet'

    blocks = blocks.set_crs(epsg)

    # Convert the geometry column to WKT before saving
    #blocks["geometry"] = blocks["geometry"].apply(lambda geom: geom.wkt if geom is not None else None)
    
    # Save the blocks dataset. 
    blocks.to_parquet(path_blocks)
    
    # Optionally, return the output path or any summary info.
    return blocks


import time
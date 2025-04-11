from auxiliary_functions import *

import dask_geopandas as dgpd
import pandas as pd
from dask import delayed, compute, visualize
import geopandas as gpd
from dask.diagnostics import ProgressBar
from metrics_calculation import calculate_minimum_distance_to_roads_option_B
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize, nearest_points
#from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from scipy.optimize import fminbound, minimize
from metrics_groupby import metrics

YOUR_NAME = 'sara'
grid_size = 200


@delayed
def building_and_intersection_metrics(city_name):
    grid_cell_count = 0
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }
    # Get EPSG
    epsg = get_epsg(city_name).compute()
    # Load grid
    grid = load_dataset(paths['grid'], epsg=epsg)#.compute()
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid['cell_area'] = grid.geometry.area

    cells = grid.index.size
    grid_cell_count += cells

    # Load buildings and perform relevant calculations on it
    buildings = load_dataset(paths['buildings'], epsg=epsg)#.compute()
    buildings['area'] = buildings.geometry.area
    joined_buildings = dgpd.sjoin(buildings, grid, predicate='within')  
    counts_buildings = joined_buildings.groupby('index_right').size()
    grid['n_buildings'] = grid.index.map(counts_buildings).fillna(0.).astype(int)
    built_area_buildings = joined_buildings.groupby('index_right')['area'].sum()
    grid['built_area'] = grid.index.map(built_area_buildings).fillna(0.).astype(float)

    #total_buildings = row_count(buildings).compute()
    #print(total_buildings)
    # Load roads
    roads = load_dataset(paths['roads'], epsg=epsg)#.compute()
    
    #road_union = roads.unary_union.compute()
    #roads = roads.compute()

    # Load intersections
    intersections = load_dataset(paths['intersections'], epsg=epsg)#.compute()

    intersections_3plus = intersections[intersections.street_count >= 3]
    intersections_4way = intersections[intersections.street_count == 4]

    grid['cell_area_km2'] = grid['cell_area']/1000000.
    
    roads_grid_joined = dgpd.sjoin(roads, grid, predicate='within')
    road_length_km = roads_grid_joined.groupby('index_right')['length'].sum()/1000.
    grid['road_length'] = grid.index.map(road_length_km).fillna(0.).astype(float)


    joined_intersections_3plus = dgpd.sjoin(intersections_3plus, grid, predicate='within')
    counts_intersections_3plus = joined_intersections_3plus.groupby('index_right').size()
    grid['intersections_3plus'] = grid.index.map(counts_intersections_3plus).fillna(0).astype(int)

    joined_intersections_4way = dgpd.sjoin(intersections_4way, grid, predicate='within')
    counts_intersections_4way = joined_intersections_4way.groupby('index_right').size()
    grid['intersections_4way'] = grid.index.map(counts_intersections_4way).fillna(0).astype(int) # OJO: NEED TO CHANGE NA


    grid['m3'] = grid['road_length']/grid['cell_area_km2']
    grid['m4'] = grid['intersections_4way'] / grid['intersections_3plus']
    grid['m5'] =  (1000.**2)*(grid['intersections_4way']/grid['cell_area']) #make sure this is equivalent to the meter calculation

    

    grid['m11'] = 1.0*grid['n_buildings'] / grid['cell_area'] # Building density
    grid['m12'] = grid['built_area'] / grid['cell_area'] # Built area share
    grid['m13'] = grid['built_area'] / grid['n_buildings'] # Average building area

    path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_metrics_3_4_5_11_12_13_grid_{YOUR_NAME}.geoparquet'

    if 'geom' in grid.columns:
        grid = grid.drop(columns='geom')

    grid.to_parquet(path)
    return grid_cell_count, path


@delayed
def building_distance_metrics(city_name):

     grid_cell_count = 0
     paths = {
         'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
         'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
         'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
         'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
         'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
     }
     # Get EPSG
     epsg = get_epsg(city_name).compute()
     # Load grid
     grid = load_dataset(paths['grid'], epsg=epsg)#.compute()
     if 'geom' in grid.columns:
         grid = grid.drop(columns=['geom'])
     
     buildings = load_dataset(paths['buildings_with_distances'], epsg=epsg)#.compute()
     buildings['distance_to_nearest_road'] = buildings['distance_to_nearest_road'].astype(float)
     buildings['area'] = buildings.geometry.area
     joined_buildings = dgpd.sjoin(buildings, grid, predicate='intersects')  
     counts_buildings = joined_buildings.groupby('index_right').size()
     grid['n_buildings'] = grid.index.map(counts_buildings).fillna(0).astype(int)
     average_distance = joined_buildings.groupby('index_right')['distance_to_nearest_road'].mean()
     grid['average_distance_nearest_building'] = grid.index.map(average_distance).fillna(0).astype(float)
    
    
     buildings_closer_than_20m = buildings[buildings['distance_to_nearest_road'] <= 20]
     joined_buildings_closer_than_20m = dgpd.sjoin(buildings_closer_than_20m, grid, predicate='intersects') 
     n_buildings_closer_than_20m = joined_buildings_closer_than_20m.groupby('index_right').size()
     grid['n_buildings_closer_than_20m'] = grid.index.map(n_buildings_closer_than_20m).fillna(0).astype(float)
    
     grid['m1'] = grid['n_buildings_closer_than_20m'] / grid['n_buildings']
     grid['m2'] = grid['average_distance_nearest_building']
    
     path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}_metrics_1_2.geoparquet'
    
     if 'geom' in grid.columns:
         grid = grid.drop(columns='geom')
    
     grid.to_parquet(path)



@delayed
def compute_m6_m7_m8(city_name):
    """
    Computes:
    - M6: KL divergence (building orientation)
    - M7: Average block width
    - M8: Building density ratio (inner vs. outer buffer)
    """

    epsilon = 0.001
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'blocks': f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'buildings_with_distances_azimuths': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet',
        'buildings_to_blocks':f'{BLOCKS_PATH}/{city_name}/{city_name}_buildings_to_blocks_{YOUR_NAME}.geoparquet'
    }

    epsg = get_epsg(city_name).compute()
    grid = load_dataset(paths['grid'], epsg=epsg)
    blocks = load_dataset(paths['blocks'], epsg=epsg).persist()
    buildings = load_dataset(paths['buildings_with_distances_azimuths'], epsg=epsg).persist()
    buildings['azimuth'] = buildings['azimuth'].map_partitions(pd.to_numeric, errors='coerce')


    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    
    blocks['block_id'] = blocks.index
    blocks['epsilon_buffer'] = blocks['geometry'].buffer(-(1.- epsilon) * blocks['max_radius'])
    blocks['width_buffer'] = blocks['geometry'].buffer(-0.2 * blocks['max_radius'])

    buildings_blocks = dgpd.sjoin(buildings, blocks, predicate='intersects').persist() #,how='right'
    buildings_blocks = buildings_blocks[['block_id', 'geometry', 'epsilon_buffer','width_buffer','azimuth']]
    buildings_blocks = buildings_blocks.set_index('block_id').repartition(npartitions=4)

    block_grid_overlap = compute_block_grid_weights(blocks, grid)
    block_grid_overlap = block_grid_overlap.compute()

    # Metric 6
    kl_df = compute_block_kl_metrics(buildings_blocks)
    m6_grid = aggregate_m6(kl_df.compute(), block_grid_overlap)
 
    # Metric 7
    block_grid_overlap['weighted_max_radius'] = (
        block_grid_overlap['max_radius'] * block_grid_overlap['area_weight']
    )

    grid_m7 = block_grid_overlap.groupby('grid_id').agg(
        total_weighted_max_radius=('weighted_max_radius', 'sum'),
        total_weight=('area_weight', 'sum')
    )
    grid_m7['m7'] = grid_m7['total_weighted_max_radius'] / grid_m7['total_weight']

    # Metric 8
    width_buffer_ratios = buildings_blocks.map_partitions(clip_buildings_by_buffer, buffer_type='width_buffer')
    epsilon_buffer_ratios = buildings_blocks.map_partitions(clip_buildings_by_buffer, buffer_type='epsilon_buffer')
    clipped_buildings_area_to_buffer_ratio = epsilon_buffer_ratios / width_buffer_ratios
    clipped_buildings_area_to_buffer_ratio = clipped_buildings_area_to_buffer_ratio.replace([np.inf, -np.inf], np.nan).fillna(999)
    ratio_df = clipped_buildings_area_to_buffer_ratio.to_frame(name='m8')
    blocks_with_m8 = blocks.merge(ratio_df, left_on='block_id', right_index=True, how='left').compute()
    block_grid_overlap = block_grid_overlap.merge(blocks_with_m8, how='left',left_on='block_id',right_index=True)
    block_grid_overlap['weighted_m8'] = (
        block_grid_overlap['m8'] * block_grid_overlap['area_weight']
    )
    grid_m8 = block_grid_overlap.groupby('grid_id').agg(
        total_weighted_m8=('weighted_m8', 'sum'),
        total_weight=('area_weight', 'sum')
    )
    grid_m8['m8'] = grid_m8['total_weighted_m8'] / grid_m8['total_weight']

    # Merge all metrics
    grid = grid.merge(m6_grid, left_index=True, right_index=True, how='left')
    grid = grid.merge(grid_m7[['m7']], left_index=True, right_index=True, how='left')
    grid = grid.merge(grid_m8[['m8']], left_index=True, right_index=True, how='left')

    '''
    # Fill NaNs
    '''

    grid['m6'] = grid['m6'].fillna(0)
    grid['m7'] = grid['m7'].fillna(0)
    grid['m8'] = grid['m8'].fillna(-999.)
    
    # Save Output
    grid = grid.compute()  
    path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}_metrics_6_7_8.geoparquet'
    grid.to_parquet(path)
    
    #path = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_with_m8_{YOUR_NAME}.geoparquet'#f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}_metrics_6_7_8.geoparquet'
    #blocks_with_m8.to_parquet(path)
    return  path

@delayed
def metrics_roads_intersections(city_name):

    paths = {
    'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
    'blocks': f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
    'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
    'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
    'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    epsg = get_epsg(city_name).compute()
    grid = load_dataset(paths['grid'], epsg=epsg)
    roads = load_dataset(paths['roads'], epsg=epsg)
    intersections = load_dataset(paths['intersections'], epsg=epsg).compute()

    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    intersections['osmid'] = intersections['osmid'].astype(int)
    intersection_angles = compute_intersection_angles(roads, intersections)
    street_count_mapping = intersections.set_index('osmid')['street_count'].to_dict()
    intersection_angle_mapping = compute_intersection_mapping(intersection_angles, street_count_mapping)
    intersection_angle_mapping = intersection_angle_mapping.compute()  


    intersections_with_angles_metric = intersections.merge(
        intersection_angle_mapping.rename("average_angle"), left_on="osmid", right_index=True, how="left"
    )

    joined_intersection_angles_grid = dgpd.sjoin(intersections_with_angles_metric, grid, predicate="within")
    average_angle_between_roads = joined_intersection_angles_grid.groupby('index_right')['average_angle'].mean()


    roads_with_tortuosity = calculate_tortuosity(roads.compute(), intersections)
    joined_tortuosity_grid = dgpd.sjoin(roads_with_tortuosity.set_geometry('road_geometry'), grid, predicate="within")
    average_tortuosity = joined_tortuosity_grid.groupby('index_right')['tortuosity'].mean()


    grid['metric_9'] = grid.index.map(average_tortuosity).fillna(-999.).astype(float)
    grid['metric_10'] = grid.index.map(average_angle_between_roads).fillna(-999.).astype(float)

    path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_metrics_9_10_{YOUR_NAME}.geoparquet'

    if 'geom' in grid.columns:
        grid = grid.drop(columns='geom')

    grid.to_parquet(path)

    return path


import time
from dask import compute

start_time = time.time()  # Start the timer

cities = ['Nairobi']
cities = [city.replace(' ', '_') for city in cities]

tasks = [building_and_intersection_metrics(city_name) for city_name in cities]
tasks.append([building_distance_metrics(city_name) for city_name in cities])
tasks.append([compute_m6_m7_m8(city_name) for city_name in cities])
tasks.append([metrics_roads_intersections(city_name) for city_name in cities])
results = compute(*tasks)  

end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Tasks completed in {elapsed_time:.2f} seconds.")


'''
from auxiliary_functions import load_dataset
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from dask import delayed, compute
from citywide_calculation import get_utm_crs, get_epsg
from metrics_calculation import calculate_minimum_distance_to_roads_option_B
from dask.diagnostics import ProgressBar
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize, nearest_points
from scipy.optimize import fminbound, minimize
from metrics_groupby import metrics
import numpy as np

# Assume the following global paths are defined elsewhere:
# GRIDS_PATH, BUILDINGS_PATH, ROADS_PATH, INTERSECTIONS_PATH, BLOCKS_PATH, OUTPUT_PATH_RASTER

YOUR_NAME = 'sara'
grid_size = 200

# ------------------------------------------------------------------------------
# 1. Load common datasets (for metrics m1, m2, m3, m4, m5, m9, m10, m11, m12, m13)
# ------------------------------------------------------------------------------
@delayed
def load_common_datasets(city_name):
    epsg = get_epsg(city_name).compute()
    
    grid_path = f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet'
    grid = load_dataset(grid_path, epsg=epsg)
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid['cell_area'] = grid.geometry.area
    
    buildings_path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet'
    buildings = load_dataset(buildings_path, epsg=epsg)
    
    roads_path = f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet'
    roads = load_dataset(roads_path, epsg=epsg)
    
    intersections_path = f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    intersections = load_dataset(intersections_path, epsg=epsg)
    
    bld_with_dist_path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet'
    buildings_with_dist = load_dataset(bld_with_dist_path, epsg=epsg)
    
    return {
         'grid': grid,
         'buildings': buildings,
         'buildings_with_dist': buildings_with_dist,
         'roads': roads,
         'intersections': intersections,
         'epsg': epsg,
    }

# ------------------------------------------------------------------------------
# 2. Metric functions using the common datasets
# ------------------------------------------------------------------------------

# Metric 1: Ratio of buildings within 20 m to total buildings
@delayed
def compute_metric_m1(city_name, datasets):
    grid = datasets['grid']
    buildings = datasets['buildings_with_dist']
    # Ensure distances are floats
    buildings['distance_to_nearest_road'] = buildings['distance_to_nearest_road'].astype(float)
    
    joined = dgpd.sjoin(buildings, grid, predicate='intersects')
    total_buildings = joined.groupby('index_right').size()
    grid = grid.assign(n_buildings=grid.index.map(total_buildings).fillna(0).astype(int))
    
    buildings_close = buildings[buildings['distance_to_nearest_road'] <= 20]
    joined_close = dgpd.sjoin(buildings_close, grid, predicate='intersects')
    count_close = joined_close.groupby('index_right').size()
    
    m1 = grid.index.map(count_close).fillna(0).astype(float) / grid['n_buildings']
    return m1

# Metric 2: Average distance to the nearest road
@delayed
def compute_metric_m2(city_name, datasets):
    grid = datasets['grid']
    buildings = datasets['buildings_with_dist']
    joined = dgpd.sjoin(buildings, grid, predicate='intersects')
    avg_distance = joined.groupby('index_right')['distance_to_nearest_road'].mean()
    m2 = grid.index.map(avg_distance).fillna(0).astype(float)
    return m2

# Metric 3: Road length density (road length per cell area in km²)
@delayed
def compute_metric_m3(city_name, datasets):
    grid = datasets['grid']
    roads = datasets['roads']
    grid = grid.assign(cell_area_km2=grid['cell_area'] / 1e6)
    roads_joined = dgpd.sjoin(roads, grid, predicate='within')
    road_length_km = roads_joined.groupby('index_right')['length'].sum() / 1000.
    m3 = grid.index.map(road_length_km).fillna(0).astype(float) / grid['cell_area_km2']
    return m3

# Metric 4: Ratio of 4-way intersections to intersections with 3+ roads
@delayed
def compute_metric_m4(city_name, datasets):
    grid = datasets['grid']
    intersections = datasets['intersections'].compute()  # force computation if needed
    intersections_3plus = intersections[intersections.street_count >= 3]
    intersections_4way = intersections[intersections.street_count == 4]
    
    joined_3plus = dgpd.sjoin(intersections_3plus, grid, predicate='within')
    count_3plus = joined_3plus.groupby('index_right').size()
    
    joined_4way = dgpd.sjoin(intersections_4way, grid, predicate='within')
    count_4way = joined_4way.groupby('index_right').size()
    
    m4 = grid.index.map(count_4way).fillna(0).astype(float) / grid.index.map(count_3plus).fillna(0)
    return m4

# Metric 5: Intersection density per cell area (using 4-way intersections)
@delayed
def compute_metric_m5(city_name, datasets):
    grid = datasets['grid']
    intersections = datasets['intersections'].compute()
    count_4way = dgpd.sjoin(intersections[intersections.street_count == 4], grid, predicate='within')\
                    .groupby('index_right').size()
    m5 = (1000.**2) * grid.index.map(count_4way).fillna(0).astype(float) / grid['cell_area']
    return m5

# Metric 11: Building density (number of buildings per cell area)
@delayed
def compute_metric_m11(city_name, datasets):
    grid = datasets['grid']
    buildings = datasets['buildings']
    buildings = buildings.assign(area=buildings.geometry.area)
    joined = dgpd.sjoin(buildings, grid, predicate='within')
    count_buildings = joined.groupby('index_right').size()
    m11 = grid.index.map(count_buildings).fillna(0).astype(float) / grid['cell_area']
    return m11

# Metric 12: Built area share (sum of building areas divided by cell area)
@delayed
def compute_metric_m12(city_name, datasets):
    grid = datasets['grid']
    buildings = datasets['buildings']
    buildings = buildings.assign(area=buildings.geometry.area)
    joined = dgpd.sjoin(buildings, grid, predicate='within')
    sum_area = joined.groupby('index_right')['area'].sum()
    m12 = grid.index.map(sum_area).fillna(0).astype(float) / grid['cell_area']
    return m12

# Metric 13: Average building area
@delayed
def compute_metric_m13(city_name, datasets):
    grid = datasets['grid']
    buildings = datasets['buildings']
    buildings = buildings.assign(area=buildings.geometry.area)
    joined = dgpd.sjoin(buildings, grid, predicate='within')
    avg_area = joined.groupby('index_right')['area'].mean()
    m13 = grid.index.map(avg_area).fillna(0).astype(float)
    return m13

# Metric 9: Average road tortuosity
@delayed
def compute_metric_m9(city_name, datasets):
    grid = datasets['grid']
    roads = datasets['roads'].compute()
    intersections = datasets['intersections'].compute()
    # calculate_tortuosity is assumed to be defined elsewhere.
    roads_with_tortuosity = calculate_tortuosity(roads, intersections)
    joined_tortuosity = dgpd.sjoin(roads_with_tortuosity.set_geometry('road_geometry'), grid, predicate="within")
    average_tortuosity = joined_tortuosity.groupby('index_right')['tortuosity'].mean()
    m9 = grid.index.map(average_tortuosity).fillna(-999.).astype(float)
    return m9

# Metric 10: Average intersection angle
@delayed
def compute_metric_m10(city_name, datasets):
    grid = datasets['grid']
    intersections = datasets['intersections'].compute()
    intersection_angles = compute_intersection_angles(datasets['roads'], intersections)
    street_count_mapping = intersections.set_index('osmid')['street_count'].to_dict()
    intersection_angle_mapping = compute_intersection_mapping(intersection_angles, street_count_mapping).compute()
    intersections_with_angles = intersections.merge(
         intersection_angle_mapping.rename("average_angle"),
         left_on="osmid", right_index=True, how="left"
    )
    joined_angles = dgpd.sjoin(intersections_with_angles, grid, predicate="within")
    average_angle = joined_angles.groupby('index_right')['average_angle'].mean()
    m10 = grid.index.map(average_angle).fillna(-999.).astype(float)
    return m10

# ------------------------------------------------------------------------------
# 3. Metric functions for block-based calculations (m6, m7, m8)
# These load additional datasets (blocks, buildings with azimuths) and follow your original logic.
# ------------------------------------------------------------------------------

# Metric 6: KL divergence for building orientation
@delayed
def compute_metric_m6(city_name):
    epsilon = 0.001
    epsg = get_epsg(city_name).compute()
    
    grid_path = f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet'
    grid = load_dataset(grid_path, epsg=epsg)
    if 'geom' in grid.columns:
         grid = grid.drop(columns=['geom'])
    
    blocks_path = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet'
    blocks = load_dataset(blocks_path, epsg=epsg).persist()
    
    buildings_az_path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet'
    buildings = load_dataset(buildings_az_path, epsg=epsg).persist()
    buildings['azimuth'] = buildings['azimuth'].map_partitions(pd.to_numeric, errors='coerce')
    
    blocks['block_id'] = blocks.index
    blocks['epsilon_buffer'] = blocks['geometry'].buffer(-(1.- epsilon) * blocks['max_radius'])
    blocks['width_buffer'] = blocks['geometry'].buffer(-0.2 * blocks['max_radius'])
    
    buildings_blocks = dgpd.sjoin(buildings, blocks, predicate='intersects').persist()
    buildings_blocks = buildings_blocks[['block_id', 'geometry', 'epsilon_buffer', 'width_buffer', 'azimuth']]
    buildings_blocks = buildings_blocks.set_index('block_id').repartition(npartitions=4)
    
    block_grid_overlap = compute_block_grid_weights(blocks, grid)
    block_grid_overlap = block_grid_overlap.compute()
    
    kl_df = compute_block_kl_metrics(buildings_blocks)
    m6_grid = aggregate_m6(kl_df.compute(), block_grid_overlap)
    
    grid = grid.merge(m6_grid, left_index=True, right_index=True, how='left')
    m6 = grid['m6'].fillna(0)
    return m6

# Metric 7: Average block width
@delayed
def compute_metric_m7(city_name):
    epsilon = 0.001
    epsg = get_epsg(city_name).compute()
    
    grid_path = f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet'
    grid = load_dataset(grid_path, epsg=epsg)
    if 'geom' in grid.columns:
         grid = grid.drop(columns=['geom'])
    
    blocks_path = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet'
    blocks = load_dataset(blocks_path, epsg=epsg).persist()
    
    block_grid_overlap = compute_block_grid_weights(blocks, grid)
    block_grid_overlap = block_grid_overlap.compute()
    block_grid_overlap['weighted_max_radius'] = block_grid_overlap['max_radius'] * block_grid_overlap['area_weight']
    
    grid_m7 = block_grid_overlap.groupby('grid_id').agg(
         total_weighted_max_radius=('weighted_max_radius', 'sum'),
         total_weight=('area_weight', 'sum')
    )
    grid_m7['m7'] = grid_m7['total_weighted_max_radius'] / grid_m7['total_weight']
    
    grid = grid.merge(grid_m7[['m7']], left_index=True, right_index=True, how='left')
    m7 = grid['m7'].fillna(0)
    return m7

# Metric 8: Building density ratio (inner vs. outer buffer)
@delayed
def compute_metric_m8(city_name):
    epsilon = 0.001
    epsg = get_epsg(city_name).compute()
    
    grid_path = f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet'
    grid = load_dataset(grid_path, epsg=epsg)
    if 'geom' in grid.columns:
         grid = grid.drop(columns=['geom'])
    
    blocks_path = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet'
    blocks = load_dataset(blocks_path, epsg=epsg).persist()
    
    buildings_az_path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet'
    buildings = load_dataset(buildings_az_path, epsg=epsg).persist()
    buildings['azimuth'] = buildings['azimuth'].map_partitions(pd.to_numeric, errors='coerce')
    
    blocks['block_id'] = blocks.index
    blocks['epsilon_buffer'] = blocks['geometry'].buffer(-(1.- epsilon) * blocks['max_radius'])
    blocks['width_buffer'] = blocks['geometry'].buffer(-0.2 * blocks['max_radius'])
    
    buildings_blocks = dgpd.sjoin(buildings, blocks, predicate='intersects').persist()
    buildings_blocks = buildings_blocks[['block_id', 'geometry', 'epsilon_buffer', 'width_buffer', 'azimuth']]
    buildings_blocks = buildings_blocks.set_index('block_id').repartition(npartitions=4)
    
    block_grid_overlap = compute_block_grid_weights(blocks, grid)
    block_grid_overlap = block_grid_overlap.compute()
    
    width_buffer_ratios = buildings_blocks.map_partitions(clip_buildings_by_buffer, buffer_type='width_buffer')
    epsilon_buffer_ratios = buildings_blocks.map_partitions(clip_buildings_by_buffer, buffer_type='epsilon_buffer')
    clipped_buildings_area_to_buffer_ratio = epsilon_buffer_ratios / width_buffer_ratios
    clipped_buildings_area_to_buffer_ratio = clipped_buildings_area_to_buffer_ratio.replace([np.inf, -np.inf], np.nan).fillna(999)
    ratio_df = clipped_buildings_area_to_buffer_ratio.to_frame(name='m8')
    
    blocks_with_m8 = blocks.merge(ratio_df, left_on='block_id', right_index=True, how='left').compute()
    block_grid_overlap = block_grid_overlap.merge(blocks_with_m8, how='left', left_on='block_id', right_index=True)
    block_grid_overlap['weighted_m8'] = block_grid_overlap['m8'] * block_grid_overlap['area_weight']
    
    grid_m8 = block_grid_overlap.groupby('grid_id').agg(
         total_weighted_m8=('weighted_m8', 'sum'),
         total_weight=('area_weight', 'sum')
    )
    grid_m8['m8'] = grid_m8['total_weighted_m8'] / grid_m8['total_weight']
    
    grid = grid.merge(grid_m8[['m8']], left_index=True, right_index=True, how='left')
    m8 = grid['m8'].fillna(-999.)
    return m8

# ------------------------------------------------------------------------------
# 4. Merge all metrics into the grid and save
# ------------------------------------------------------------------------------
def calculate_metrics(city_name):
    # Load common datasets (for m1, m2, m3, m4, m5, m9, m10, m11, m12, m13)
    datasets = load_common_datasets(city_name)
    
    # Compute metrics that use the common datasets
    m1   = compute_metric_m1(city_name, datasets)
    m2   = compute_metric_m2(city_name, datasets)
    m3   = compute_metric_m3(city_name, datasets)
    m4   = compute_metric_m4(city_name, datasets)
    m5   = compute_metric_m5(city_name, datasets)
    m11  = compute_metric_m11(city_name, datasets)
    m12  = compute_metric_m12(city_name, datasets)
    m13  = compute_metric_m13(city_name, datasets)
    m9   = compute_metric_m9(city_name, datasets)
    m10  = compute_metric_m10(city_name, datasets)
    
    # Compute block-based metrics (m6, m7, m8) which load their own extra datasets
    m6   = compute_metric_m6(city_name)
    m7   = compute_metric_m7(city_name)
    m8   = compute_metric_m8(city_name)
    
    @delayed
    def merge_metrics(grid, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13):
         grid['m1'] = m1
         grid['m2'] = m2
         grid['m3'] = m3
         grid['m4'] = m4
         grid['m5'] = m5
         grid['m6'] = m6
         grid['m7'] = m7
         grid['m8'] = m8
         grid['m9'] = m9
         grid['m10'] = m10
         grid['m11'] = m11
         grid['m12'] = m12
         grid['m13'] = m13
         return grid
    
    final_grid = merge_metrics(datasets['grid'], m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13)
    
    @delayed
    def save_grid(grid, path):
         if 'geom' in grid.columns:
              grid = grid.drop(columns=['geom'])
         grid.to_parquet(path)
         return path
    
    out_path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_grid_all_metrics_{YOUR_NAME}.geoparquet'
    saved = save_grid(final_grid, out_path)
    result = compute(saved)
    return result[0]

# ------------------------------------------------------------------------------
# 5. Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    start_time = time.time()
    city = 'Nairobi'
    city = city.replace(' ', '_')
    final_path = calculate_metrics(city)
    elapsed_time = time.time() - start_time
    print(f"All metrics computed and saved to {final_path}")
    print(f"Tasks completed in {elapsed_time:.2f} seconds.")

'''
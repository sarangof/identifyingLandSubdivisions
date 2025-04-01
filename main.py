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
def metrics(city_name,YOUR_NAME,grid_size):
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
    grid['n_buildings'] = grid.index.map(counts_buildings).fillna(0).astype(int)
    built_area_buildings = joined_buildings.groupby('index_right')['area'].sum()
    grid['built_area'] = grid.index.map(built_area_buildings).fillna(0).astype(float)

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
    grid['road_length'] = grid.index.map(road_length_km).fillna(0).astype(float)


    joined_intersections_3plus = dgpd.sjoin(intersections_3plus, grid, predicate='within')
    counts_intersections_3plus = joined_intersections_3plus.groupby('index_right').size()
    grid['intersections_3plus'] = grid.index.map(counts_intersections_3plus).fillna(0).astype(int)

    joined_intersections_4way = dgpd.sjoin(intersections_4way, grid, predicate='within')
    counts_intersections_4way = joined_intersections_4way.groupby('index_right').size()
    grid['intersections_4way'] = grid.index.map(counts_intersections_4way).fillna(0).astype(int)


    grid['m3'] = grid['road_length']/grid['cell_area_km2']
    grid['m4'] = grid['intersections_4way'] / grid['intersections_3plus']
    grid['m5'] =  (1000.**2)*(grid['intersections_4way']/grid['cell_area']) #make sure this is equivalent to the meter calculation

    

    grid['m11'] = 1.0*grid['n_buildings'] / grid['cell_area'] # Building density
    grid['m12'] = grid['built_area'] / grid['cell_area'] # Built area share
    grid['m13'] = grid['built_area'] / grid['n_buildings'] # Average building area

    path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}.geoparquet'

    if 'geom' in grid.columns:
        grid = grid.drop(columns='geom')

    grid.to_parquet(path)
    return grid_cell_count, path



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

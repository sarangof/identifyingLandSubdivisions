import dask_geopandas as dgpd
from dask import delayed
from metrics_calculation import calculate_minimum_distance_to_roads_option_B

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
CITY_INFO_PATH = f'{INPUT_PATH}/city_info'
EXTENTS_PATH = f'{CITY_INFO_PATH}/extents'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
OUTPUT_PATH = f'{MAIN_PATH}/output'
OUTPUT_PATH_CSV = f'{OUTPUT_PATH}/csv'
OUTPUT_PATH_RASTER = f'{OUTPUT_PATH}/raster'
OUTPUT_PATH_PNG = f'{OUTPUT_PATH}/png'
OUTPUT_PATH_RAW = f'{OUTPUT_PATH}/raw_results'


@delayed
def get_epsg(city_name):
    urban_extent = f'{EXTENTS_PATH}/{city_name}/{city_name}_urban_extent.geoparquet'
    extent = dgpd.read_parquet(urban_extent)
    geometry = extent.geometry[0].compute()
    epsg = get_utm_crs(geometry)
    print(f'{city_name} EPSG: {epsg}')
    return epsg

@delayed
def load_dataset(path, epsg=None):
    """Load a single parquet dataset"""
    dataset = dgpd.read_parquet(path, npartitions=2)
    if epsg:
        dataset = dataset.to_crs(epsg=epsg)
    return dataset

@delayed
def row_count(dgdf):
    """Count the rows in a dataframe"""
    row_count = dgdf.map_partitions(len).compute().sum()

    return row_count




@delayed
def metrics(city_name, YOUR_NAME, grid_size=200):
    grid_cell_count = 0
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }
    # Get EPSG
    epsg = get_epsg(city_name)

    # Load grid
    grid = load_dataset(paths['grid'], epsg=epsg).compute()

    cells = grid.index.size
    grid_cell_count += cells.compute()

    # Load buildings
    buildings = load_dataset(paths['buildings'], epsg=epsg)

    # Calculate building count per grid cell
    #joined_buildings = dgpd.sjoin(buildings, grid, predicate='within')
    #counts_buildings = joined_buildings.groupby('index_right').size()
    #grid['building_count'] = grid.index.map(counts_buildings).fillna(0).astype(int)
 
    # Load roads
    roads = load_dataset(paths['roads'], epsg=epsg)

    # Make road unions
    road_union = roads.union_all
    
    # Load intersections
    intersections = load_dataset(paths['intersections'], epsg=epsg).compute()
    intersections_3plus = intersections[intersections.street_count >= 3]
    intersections_4way = intersections[intersections.street_count == 4]

    # Calculate intersections_3plus
    joined_intersections_3plus = dgpd.sjoin(intersections_3plus, grid, predicate='within')
    counts_intersections_3plus = joined_intersections_3plus.groupby('index_right').size()
    grid['intersections_3plus'] = grid.index.map(counts_intersections_3plus).fillna(0).astype(int)

    # Calculate intersections_4way
    joined_intersections_4way = dgpd.sjoin(intersections_4way, grid, predicate='within')
    counts_intersections_4way = joined_intersections_4way.groupby('index_right').size()
    grid['intersections_4way'] = grid.index.map(counts_intersections_4way).fillna(0).astype(int)

    # Calculate minimum distance to roads
    #buildings['distance_to_road'] = calculate_minimum_distance_to_roads_option_B(buildings, roads)

    # Count building within 10 meters of road per grid cell
    #buildings_near_road = buildings[buildings.distance_to_road <= 20]



    #1 Share of building footprints that are less than 10-meters away from the nearest road
    #m1 = 1.*((sum(buildings['distance_to_road']<=20))/len(buildings))
    #grid['m1'] = 


    #4 Share of 3-way and 4-way intersections 
    grid['m4'] = grid['intersections_4way'] / grid['intersections_3plus']


    # Write out results
    path = f'{OUTPUT_PATH}/city_info/grids/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}.geoparquet'
    grid.to_parquet(path)
    return grid_cell_count, path

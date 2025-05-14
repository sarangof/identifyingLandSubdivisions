import dask_geopandas as dgpd
import dask.dataframe as dd
import pandas as pd
from dask import delayed, compute, visualize
import geopandas as gpd
from dask.diagnostics import ProgressBar
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize, nearest_points
#from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from scipy.optimize import fminbound, minimize
#from unused_code.metrics_groupby import metrics
from dask import delayed
import dask_geopandas as dgpd
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkb
from scipy.stats import entropy

from pre_processing import *
from auxiliary_functions import *
from standardize_metrics import *

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
CITY_INFO_PATH = f'{INPUT_PATH}/city_info'
EXTENTS_PATH = f'{CITY_INFO_PATH}/extents'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
BLOCKS_PATH = f'{INPUT_PATH}/blocks'
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
def building_and_intersection_metrics(city_name,grid_size,YOUR_NAME):
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
    # slim to just geometry & persist
    roads = load_dataset(paths['roads'], epsg=epsg)
    grid = load_dataset(paths['grid'], epsg=epsg)#.compute()

    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid['cell_area'] = grid.geometry.area

    cells = grid.index.size
    grid_cell_count += cells



    # 1. ensure buildings have an area field
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    buildings['area'] = buildings.geometry.area

    # 2. tiny grid for broadcast
    grid_small = (
        grid
        .reset_index()[['index','geometry']]
        .rename(columns={'index':'index_right'})
    )

    # 3. per-partition clipping & aggregation
    def building_area_partition(bldg_part, grid_sm):
        # clip to each cell
        clipped = gpd.overlay(bldg_part, grid_sm, how='intersection')
        clipped['clipped_area'] = clipped.geometry.area
        # sum built area and count buildings per cell
        out = clipped.groupby('index_right').agg(
            built_area=('clipped_area', 'sum'),
            n_buildings=('geometry', 'size')
        )
        return out

    # 4. minimal meta for Dask
    meta_ba = pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'built_area': pd.Series(dtype='float64'),
        'n_buildings': pd.Series(dtype='int64')
    }).set_index('index_right')

    # 5. map_partitions + aggregate
    parts = buildings.map_partitions(
        building_area_partition,
        grid_small,
        meta=meta_ba
    ).persist()

    agg = parts.groupby('index_right').sum()

    # 6. assign back to grid
    grid['n_buildings'] = agg['n_buildings'].fillna(0).astype(int)
    grid['built_area']  = agg['built_area'].fillna(0.0)


    roads_geo = roads[['geometry']]#.persist()

    # tiny grid dataframe for overlay
    grid_small = (
        grid.reset_index()[['index','geometry']]
            .rename(columns={'index':'index_right'})
    )

    # per-partition overlay + length
    def road_length_partition(df, grid_sm):
        clipped = gpd.overlay(df, grid_sm, how='intersection')
        L = clipped.geometry.length.values
        return pd.DataFrame({
            'index_right': clipped['index_right'].values,
            'length_in_cell': L
        }, index=clipped.index)

    meta_rl = pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'length_in_cell': pd.Series(dtype='float64')
    })

    road_parts = roads_geo.map_partitions(
        road_length_partition, grid_small, meta=meta_rl
    ).persist()

    agg_rl = road_parts.groupby('index_right').agg(
        total_len_m=('length_in_cell','sum')
    )

    grid['cell_area_km2'] = grid['cell_area']/1000000.
    
    grid['road_length'] = (agg_rl['total_len_m'] / 1000.)
    grid['road_length'] = grid['road_length'].fillna(0.0)
    grid['has_roads'] = grid['road_length'] > 0

    # Intersection metrics
    intersections = load_dataset(paths['intersections'], epsg=epsg)
    ji = dgpd.sjoin(intersections, grid, predicate='intersects')
    counts2 = ji[ji.street_count>=2].groupby('index_right').size()
    counts3 = ji[ji.street_count>=3].groupby('index_right').size()
    counts4 = ji[ji.street_count==4].groupby('index_right').size()
    
    grid['n_intersections'] = counts2
    grid['n_intersections'] = grid['n_intersections'].fillna(0).astype(int)

    
    # right after you build n_intersections:
    grid['has_intersections'] = (grid['n_intersections'] > 0).astype('bool')


    
    grid['intersections_3plus'] = counts3
    grid['intersections_3plus'] = grid['intersections_3plus'].fillna(0).astype(int)
    
    grid['intersections_4way']  = counts4
    grid['intersections_4way']  = grid['intersections_4way'].fillna(0).astype(int)

    # Downstream metrics
    grid['m3_raw'] = grid['road_length'] / grid['cell_area_km2']
    grid['m3_std'] = grid['m3_raw'].map_partitions(standardize_metric_3, meta=('m3','float64'))

    grid['m4_raw'] = grid['intersections_4way'] / grid['intersections_3plus']

    '''
    grid['m4_raw'] = grid['m4_raw'].mask(
        grid['m4_raw'].isna() & grid['has_roads'],
        0.0
    )
    '''

    m4_median = grid['m4_raw'].dropna().quantile(0.5).compute()

    # now fill *all* remaining NaNs with that constant
    grid['m4_raw'] = grid['m4_raw'].fillna(m4_median)

    # continue as before
    grid['m4_std'] = grid['m4_raw'].map_partitions(
        standardize_metric_4, meta=('m4','float64')
    )
    
    grid['m5_raw'] = (1000**2) * (grid['n_intersections'] / grid['cell_area'])
    grid['m5_raw'] = grid['m5_raw'].mask(
        grid['has_roads'] & grid['m5_raw'].isna(), 
        0.0)
    grid['m5_std'] = grid['m5_raw'].map_partitions(standardize_metric_5, meta=('m5','float64'))
    grid['m10_raw'] = grid['n_buildings'] / grid['cell_area_km2']
    grid['m10_std'] = grid['m10_raw'].map_partitions(standardize_metric_10, meta=('m10','float64'))
    grid['m11_raw'] = grid['built_area'] / grid['cell_area']
    grid['m11_std'] = grid['m11_raw'].map_partitions(standardize_metric_11, meta=('m11','float64'))
    grid['m12_raw'] = grid['built_area'] / grid['n_buildings']
    grid['m12_raw'] = grid['m12_raw'].fillna(0.0)
    grid['m12_std'] = grid['m12_raw'].map_partitions(standardize_metric_12, meta=('m12','float64'))

    # … write out and return …
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_metrics_3_4_5_10_11_12_grid_{YOUR_NAME}.geoparquet'
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out)
    return out

@delayed
def building_distance_metrics(city_name, grid_size, YOUR_NAME):
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
     grid['n_buildings'] = counts_buildings
     grid['n_buildings'] = grid['n_buildings'].fillna(0).astype(int)

     grid['has_buildings'] = grid['n_buildings']    > 0
     average_distance = joined_buildings.groupby('index_right')['distance_to_nearest_road'].mean()
     grid['average_distance_nearest_building'] = average_distance
     grid['average_distance_nearest_building'] = grid['average_distance_nearest_building'].fillna(0.0)

     buildings_closer_than_20m = buildings[buildings['distance_to_nearest_road'] <= 20]
     joined_buildings_closer_than_20m = dgpd.sjoin(buildings_closer_than_20m, grid, predicate='intersects') 
     n_buildings_closer_than_20m = joined_buildings_closer_than_20m.groupby('index_right').size()
     grid['n_buildings_closer_than_20m'] = n_buildings_closer_than_20m
     grid['n_buildings_closer_than_20m'] = grid['n_buildings_closer_than_20m'].fillna(0.0)
     grid = grid.assign(
    n_buildings_closer_than_20m = grid['n_buildings_closer_than_20m'].mask(
        (grid['n_buildings'] > 0) & (grid['n_buildings_closer_than_20m'].isna()),
        0))
     grid = grid.assign(
         m1_raw = grid['n_buildings_closer_than_20m'] / grid['n_buildings']
         )
     grid['m1_raw'] = grid['m1_raw'].fillna(grid['m1_raw'].median())#grid['n_buildings_closer_than_20m'] / grid['n_buildings']
     grid['m1_std'] = grid['m1_raw'].map_partitions(standardize_metric_1, meta=('m1', 'float64'))
     grid['m2_raw'] = grid['average_distance_nearest_building']
     grid['m2_raw'] = grid['m2_raw'].fillna(grid['m2_raw'].median())
     grid['m2_std'] = grid['m2_raw'].map_partitions(standardize_metric_2, meta=('m2', 'float64'))
     
     path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}_metrics_1_2.geoparquet'
    
     if 'geom' in grid.columns:
         grid = grid.drop(columns='geom')
    
     grid.to_parquet(path)

@delayed
def compute_m6_m7(city_name, grid_size, YOUR_NAME):
    """
    Computes:
    - M6: KL divergence (building orientation), weighted by:
         (a) block’s proportional overlap with each grid cell
         (b) number of buildings inside that block∩cell
      fallback: unweighted KL for cells with ≥2 buildings but no blocks
    - M7: Average block width
    """

    # 0) paths & load
    epsg = get_epsg(city_name).compute()
    grid = load_dataset(f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
                        epsg=epsg)
    blocks    = load_dataset(f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet', epsg=epsg).persist()
    buildings = load_dataset(f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet', epsg=epsg).persist()
    buildings['azimuth'] = buildings['azimuth']\
        .map_partitions(pd.to_numeric,
                        meta=('azimuth','float64'),
                        errors='coerce')


    # drop stray geometry column if present
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # 1) prepare blocks
    epsilon = 0.001
    blocks = blocks.assign(
        block_id=blocks.index,
        epsilon_buffer=blocks.geometry.buffer(-(1-epsilon)*blocks.max_radius),
        width_buffer  =blocks.geometry.buffer(-0.2*blocks.max_radius)
    )

    # 2) compute block→grid overlaps (with area_weight)
    #    compute_block_grid_weights returns a **pandas** GeoDataFrame after compute()
    bgo = compute_block_grid_weights(blocks, grid).compute()

    # 3) count **per (block, cell)** buildings inside each overlap
    #    (so blocks only contribute where they actually contain buildings)
    buildings_pdf = buildings.compute()[['geometry']]
    # join buildings → block_cell overlaps
    join = gpd.sjoin(buildings_pdf,
                     bgo[['block_id','grid_id','geometry']],
                     predicate='intersects')
    # count
    n_bc = (
        join
        .groupby(['block_id','grid_id'])
        .size()
        .rename('n_buildings_cell')
        .reset_index()
    )
    # merge back, fill zero
    bgo = (
        bgo
        .merge(n_bc, on=['block_id','grid_id'], how='left')
        .fillna({'n_buildings_cell': 0})
    )

    # 4) block-level KL & per-cell weighted m6
    kl_df   = compute_block_kl_metrics(
                  # still uses your buildings_blocks → block-level KL
                  dgpd.sjoin(buildings, blocks, predicate='intersects')
                     [['block_id','geometry','epsilon_buffer','width_buffer','azimuth']]
                     .set_index('block_id')
                     .repartition(npartitions=4)
              ).compute()
    # adjust aggregate to use n_buildings_cell
    df = (
        bgo
        .merge(kl_df, on='block_id', how='left')
        .dropna(subset=['standardized_kl'])
        .assign(weight = lambda d: d.area_weight * d.n_buildings_cell,
                weighted_kl = lambda d: d.standardized_kl * d.weight)
    )
    grid_m6 = (
        df
        .groupby('grid_id')
        .agg(total_weighted_kl=('weighted_kl','sum'),
             total_weight=('weight','sum'))
    )
    grid_m6['m6'] = grid_m6.total_weighted_kl / grid_m6.total_weight

    # 5) compute M7 as before
    bgo['weighted_max_radius'] = bgo.max_radius * bgo.area_weight
    grid_m7 = (
        bgo
        .groupby('grid_id')
        .agg(total_weighted_max_radius=('weighted_max_radius','sum'),
             total_weight=('area_weight', 'sum'))
    )
    grid_m7['m7'] = grid_m7.total_weighted_max_radius / grid_m7.total_weight

    # 6) unweighted KL fallback for cells with ≥2 buildings (no blocks)
    def kl_divergence(arr, bins=18):
        hist,_ = np.histogram(arr, bins=bins, range=(0,180))
        P = hist/hist.sum() if hist.sum()>0 else np.ones(bins)/bins
        Q = np.ones(bins)/bins
        return entropy(P, Q) / np.log(bins)

    b2g = dgpd.sjoin(buildings[['geometry','azimuth']],
                     grid[['geometry']],
                     predicate='intersects'
                    ).persist()
    m6_unw = (
        b2g.groupby('index_right')['azimuth']
           .apply(lambda s: kl_divergence(s.to_numpy()),
                  meta=('azimuth','float64'))
           .rename('m6_unweighted')
    )

    # 7) stitch everything back onto the full grid
    #    name all indexes “grid_id” so reset_index() yields a column to join
    grid        = grid       .rename_axis("grid_id")
    grid_m6     = grid_m6    .rename_axis("grid_id")
    m6_unw      = m6_unw     .rename_axis("grid_id")
    grid_m7     = grid_m7    .rename_axis("grid_id")

    grid = (
        grid
        .reset_index()                                # now has “grid_id” column
        .merge(grid_m6.reset_index(),   on="grid_id", how="left")
        .merge(m6_unw.reset_index(),    on="grid_id", how="left")
        .merge(grid_m7.reset_index()[['grid_id','m7']],
               on="grid_id", how="left")
        .set_index("grid_id")
    )

    # 8) build & standardize raw columns
    grid['m6_raw'] = grid['m6'].fillna(grid['m6_unweighted'])
    m6_median = grid['m6_raw'].median().compute()
    grid['m6_raw'] = grid['m6_raw'].fillna(m6_median)
    grid['m6_std'] = grid['m6_raw'].map_partitions(
                        standardize_metric_6, meta=('m6','float64')
                     )

    grid['m7_raw'] = grid['m7'].fillna(200)
    grid['m7_std'] = grid['m7_raw'].map_partitions(
                        standardize_metric_7, meta=('m7','float64')
                     )

    grid = grid.drop(columns=['m6','m7'])

    # 9) write out
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_grid_{YOUR_NAME}_metrics_6_7.geoparquet'
    grid.to_parquet(out)#, engine='pyarrow', index=False)
    return out

@delayed
def metrics_roads_intersections(city_name, grid_size, YOUR_NAME):

    paths = {
    'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
    'blocks': f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
    'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
    'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
    'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # LOAD
    epsg = get_epsg(city_name).compute()
    grid = load_dataset(paths['grid'], epsg=epsg).persist()
    roads = load_dataset(paths['roads'], epsg=epsg).persist()
    intersections = load_dataset(paths['intersections'], epsg=epsg).compute()

    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Prep for metric 10
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

    # Prep for metric 9
    
    roads_simple = roads[['geometry']]
    
    grid_small = (
        grid
        .reset_index()[["index","geometry"]]
        .rename(columns={"index":"index_right"})
    )
    
    # inside metrics_roads_intersections, before overlay:
    overlay_meta = gpd.GeoDataFrame(
        {
            "index_right": pd.Series(dtype="int64"),
            "geometry":    gpd.GeoSeries(dtype="geometry")
        },
        geometry="geometry"
    )

    roads_cells = roads_simple.map_partitions(
        overlay_partition,
        grid_small,
        meta=overlay_meta
    ).persist()

    # roads_cells now has:
    #  - geometry   = clipped road piece
    #  - index_right = the cell it belongs to

    # 3) compute wt & length in one pass per partition
    out = roads_cells.map_partitions(
        partition_tortuosity_clipped,   
        meta=pd.DataFrame({
            "index_right": pd.Series(dtype="int64"),
            "wt":           pd.Series(dtype="float64"),
            "length":      pd.Series(dtype="float64")
        })
    ).persist()

    # 4) aggregate back into grid
    agg = out.groupby("index_right").agg(
        total_len=("length","sum"),
        sum_wt   =("wt",    "sum")
    )
    grid["m8_raw"] = grid.index.map(agg["sum_wt"]/agg["total_len"]).astype(float)

    grid['m8_raw'] = grid['m8_raw'].fillna(0.0)

    grid["m8_std"] = grid["m8_raw"].map_partitions(
        standardize_metric_8, meta=("m8","float64")
    )
    
    grid['m9_raw'] = grid.index.map(average_angle_between_roads).astype(float)#.fillna(np.mean(average_angle_between_roads)).astype(float)
    m9_median = grid['m9_raw'].dropna().quantile(0.5).compute()
    grid['m9_raw'] = grid['m9_raw'].fillna(m9_median)

    grid['m9_std'] = grid['m9_raw'].map_partitions(standardize_metric_9, meta=('m9', 'float64'))
    
    path = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_grid_metrics_8_9_{YOUR_NAME}.geoparquet'

    if 'geom' in grid.columns:
        grid = grid.drop(columns='geom')

    grid.to_parquet(path)

    return path
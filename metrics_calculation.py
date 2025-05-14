'''
Module: Building and Intersection Metrics Pipeline

This module defines Dask-delayed functions to compute spatial metrics
for city grids, including building counts, road lengths, intersection
counts, and various standardized metrics (m1–m12). It reads geospatial
data from S3, processes it with GeoPandas and Dask-Geopandas, and
writes results back to Parquet.
'''

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# Numerical and data libraries
import numpy as np
import pandas as pd

# Geospatial libraries for vector data handling
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize, nearest_points
from shapely import wkb

# Dask for parallel computation
import dask.dataframe as dd
import dask_geopandas as dgpd
from dask import delayed, compute, visualize
from dask.diagnostics import ProgressBar

# Scientific utilities
from scipy.stats import entropy
from scipy.optimize import fminbound, minimize

# Project-specific preprocessing and metric functions
from pre_processing import *
from auxiliary_functions import *
from standardize_metrics import *

# -----------------------------------------------------------------------------
# Constants and S3 Paths
# -----------------------------------------------------------------------------
MAIN_PATH           = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH          = f"{MAIN_PATH}/input"
CITY_INFO_PATH      = f"{INPUT_PATH}/city_info"
EXTENTS_PATH        = f"{CITY_INFO_PATH}/extents"
BUILDINGS_PATH      = f"{INPUT_PATH}/buildings"
BLOCKS_PATH         = f"{INPUT_PATH}/blocks"
ROADS_PATH          = f"{INPUT_PATH}/roads"
INTERSECTIONS_PATH  = f"{INPUT_PATH}/intersections"
GRIDS_PATH          = f"{INPUT_PATH}/city_info/grids"
SEARCH_BUFFER_PATH  = f"{INPUT_PATH}/city_info/search_buffers"
OUTPUT_PATH         = f"{MAIN_PATH}/output"
OUTPUT_PATH_CSV     = f"{OUTPUT_PATH}/csv"
OUTPUT_PATH_RASTER  = f"{OUTPUT_PATH}/raster"
OUTPUT_PATH_PNG     = f"{OUTPUT_PATH}/png"
OUTPUT_PATH_RAW     = f"{OUTPUT_PATH}/raw_results"

# -----------------------------------------------------------------------------
# 1. BUILDING AND INTERSECTION METRICS (m3, m4, m5, m10, m11, m12)
# -----------------------------------------------------------------------------
@delayed
def building_and_intersection_metrics(city_name, grid_size, YOUR_NAME):
    """
    Computes multiple cell-level metrics based on buildings, roads, and
    intersections. Outputs a GeoParquet with raw and standardized metrics:
      - Building count and built area
      - Road length and presence
      - Intersection counts and types
      - Metrics m3, m4, m5, m10, m11, m12
    """
    # --- Define input file paths for the city and grid size ---
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # --- Determine projection (EPSG) for accurate area and length calculations ---
    epsg = get_epsg(city_name).compute()

    # --- Load grid and roads into GeoDataFrames ---
    grid  = load_dataset(paths['grid'], epsg=epsg)
    roads = load_dataset(paths['roads'], epsg=epsg)

    # Remove any stray 'geom' column inherited from previous operations
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Compute and store cell area in native units
    grid['cell_area'] = grid.geometry.area
    grid_cell_count = grid.shape[0]

    # -------------------------------------------------------------------------
    # Building-based metrics: built area and building count per cell
    # -------------------------------------------------------------------------
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    buildings['area'] = buildings.geometry.area  # individual building footprint

    # Create a small grid GeoDataFrame for spatial join
    grid_small = (
        grid.reset_index()[['index','geometry']]
            .rename(columns={'index':'index_right'})
    )

    # Partition function: clip building parts to cells and aggregate
    def building_area_partition(bldg_part, grid_sm):
        clipped = gpd.overlay(bldg_part, grid_sm, how='intersection')
        clipped['clipped_area'] = clipped.geometry.area
        return clipped.groupby('index_right').agg(
            built_area=('clipped_area', 'sum'),
            n_buildings=('geometry', 'size')
        )

    # Define minimal metadata for Dask aggregation
    meta_ba = pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'built_area':  pd.Series(dtype='float64'),
        'n_buildings': pd.Series(dtype='int64')
    }).set_index('index_right')

    # Compute and persist building metrics across partitions
    parts = buildings.map_partitions(
        building_area_partition,
        grid_small,
        meta=meta_ba
    ).persist()
    agg = parts.groupby('index_right').sum()

    # Assign results back to main grid, filling missing with zeros
    grid['n_buildings'] = agg['n_buildings'].fillna(0).astype(int)
    grid['built_area']  = agg['built_area'].fillna(0.0)

    # -------------------------------------------------------------------------
    # Road-based metrics: total road length and presence per cell
    # -------------------------------------------------------------------------
    roads_geo = roads[['geometry']]

    def road_length_partition(df, grid_sm):
        clipped = gpd.overlay(df, grid_sm, how='intersection')
        return pd.DataFrame({
            'index_right':     clipped['index_right'].values,
            'length_in_cell':  clipped.geometry.length.values
        }, index=clipped.index)

    meta_rl = pd.DataFrame({
        'index_right':     pd.Series(dtype='int64'),
        'length_in_cell':  pd.Series(dtype='float64')
    })

    road_parts = roads_geo.map_partitions(
        road_length_partition,
        grid_small,
        meta=meta_rl
    ).persist()
    agg_rl = road_parts.groupby('index_right').agg(total_len_m=('length_in_cell','sum'))

    # Convert cell area to km² and road length to km
    grid['cell_area_km2'] = grid['cell_area'] / 1e6
    grid['road_length']   = (agg_rl['total_len_m'] / 1000.).fillna(0.0)
    grid['has_roads']     = grid['road_length'] > 0

    # -------------------------------------------------------------------------
    # Intersection-based metrics: counts of intersections by connectivity
    # -------------------------------------------------------------------------
    intersections = load_dataset(paths['intersections'], epsg=epsg)
    ji = dgpd.sjoin(intersections, grid, predicate='intersects')

    counts2 = ji[ji.street_count >= 2].groupby('index_right').size()
    counts3 = ji[ji.street_count >= 3].groupby('index_right').size()
    counts4 = ji[ji.street_count == 4].groupby('index_right').size()

    grid['n_intersections']     = counts2.fillna(0).astype(int)
    grid['has_intersections']   = grid['n_intersections'] > 0
    grid['intersections_3plus'] = counts3.fillna(0).astype(int)
    grid['intersections_4way']  = counts4.fillna(0).astype(int)

    # -------------------------------------------------------------------------
    # Downstream metrics: m3, m4, m5, m10, m11, m12
    # -------------------------------------------------------------------------
    # M3: road length density (km per km²)
    grid['m3_raw'] = grid['road_length'] / grid['cell_area_km2']
    grid['m3_std'] = grid['m3_raw'].map_partitions(standardize_metric_3, meta=('m3','float64'))

    # M4: ratio of 4-way to 3+-way intersections
    grid['m4_raw'] = grid['intersections_4way'] / grid['intersections_3plus']
    m4_median      = grid['m4_raw'].dropna().quantile(0.5).compute()
    grid['m4_raw'] = grid['m4_raw'].fillna(m4_median)
    grid['m4_std'] = grid['m4_raw'].map_partitions(standardize_metric_4, meta=('m4','float64'))

    # M5: intersection density per km²
    grid['m5_raw'] = (1e6) * (grid['n_intersections'] / grid['cell_area'])
    grid['m5_raw'] = grid['m5_raw'].mask(grid['has_roads'] & grid['m5_raw'].isna(), 0.0)
    grid['m5_std'] = grid['m5_raw'].map_partitions(standardize_metric_5, meta=('m5','float64'))

    # M10: building count density per km²
    grid['m10_raw'] = grid['n_buildings'] / grid['cell_area_km2']
    grid['m10_std'] = grid['m10_raw'].map_partitions(standardize_metric_10, meta=('m10','float64'))

    # M11: built area ratio
    grid['m11_raw'] = grid['built_area'] / grid['cell_area']
    grid['m11_std'] = grid['m11_raw'].map_partitions(standardize_metric_11, meta=('m11','float64'))

    # M12: average building footprint size
    grid['m12_raw'] = (grid['built_area'] / grid['n_buildings']).fillna(0.0)
    grid['m12_std'] = grid['m12_raw'].map_partitions(standardize_metric_12, meta=('m12','float64'))

    # --- Write result to GeoParquet ---
    out_path = (
        f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_metrics_"
        f"3_4_5_10_11_12_grid_{YOUR_NAME}.geoparquet"
    )
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out_path)
    return out_path


# -----------------------------------------------------------------------------
# 2. BUILDING DISTANCE METRICS (m1, m2)
# -----------------------------------------------------------------------------
@delayed
def building_distance_metrics(city_name, grid_size, YOUR_NAME):
    """
    Computes distance-based building metrics:
      - m1: proportion of buildings within 20m of any road
      - m2: average distance to nearest road
    """
    # --- Define input file paths ---
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # Determine projection for accurate metrics
    epsg = get_epsg(city_name).compute()

    # Load and clean grid
    grid = load_dataset(paths['grid'], epsg=epsg)
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Load buildings with precomputed distances
    buildings = load_dataset(paths['buildings_with_distances'], epsg=epsg)
    buildings['distance_to_nearest_road'] = buildings['distance_to_nearest_road'].astype(float)
    buildings['area'] = buildings.geometry.area

    # Spatial join: assign buildings to grid cells
    joined = dgpd.sjoin(buildings, grid, predicate='intersects')
    counts = joined.groupby('index_right').size()
    grid['n_buildings'] = counts.fillna(0).astype(int)
    grid['has_buildings'] = grid['n_buildings'] > 0

    # Average distance per cell
    avg_dist = joined.groupby('index_right')['distance_to_nearest_road'].mean()
    grid['average_distance_nearest_building'] = avg_dist.fillna(0.0)

    # Buildings within 20m: count and proportion
    close = buildings[buildings['distance_to_nearest_road'] <= 20]
    joined_close = dgpd.sjoin(close, grid, predicate='intersects')
    close_counts = joined_close.groupby('index_right').size()
    grid['n_buildings_closer_than_20m'] = close_counts.fillna(0).astype(int)

    # Metric m1: fill missing proportions with median
    grid['m1_raw'] = grid['n_buildings_closer_than_20m'] / grid['n_buildings']
    grid['m1_raw'] = grid['m1_raw'].fillna(grid['m1_raw'].median())
    grid['m1_std'] = grid['m1_raw'].map_partitions(standardize_metric_1, meta=('m1','float64'))

    # Metric m2: fill missing average distances with median
    grid['m2_raw'] = grid['average_distance_nearest_building'].fillna(grid['average_distance_nearest_building'].median())
    grid['m2_std'] = grid['m2_raw'].map_partitions(standardize_metric_2, meta=('m2','float64'))

    # Write result to GeoParquet
    out_path = f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_grid_{YOUR_NAME}_metrics_1_2.geoparquet"
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out_path)
    return out_path


# -----------------------------------------------------------------------------
# 3. BLOCK-BASED METRICS (m6, m7)
# -----------------------------------------------------------------------------
@delayed
def compute_m6_m7(city_name, grid_size, YOUR_NAME):
    """
    Computes block-based metrics:
      - m6: KL divergence of building orientations per cell, weighted by block overlap
      - m7: weighted average block width per cell
      Fallback: unweighted KL for cells with ≥2 buildings but no block overlap.
    """
    # Determine projection and load datasets
    epsg      = get_epsg(city_name).compute()
    grid      = load_dataset(f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet', epsg=epsg)
    blocks    = load_dataset(f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet', epsg=epsg).persist()
    buildings = load_dataset(f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet', epsg=epsg).persist()
    buildings['azimuth'] = buildings['azimuth'].map_partitions(pd.to_numeric, meta=('azimuth','float64'), errors='coerce')

    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Prepare block buffers for overlap and width calculations
    epsilon = 0.001
    blocks = blocks.assign(
        block_id=blocks.index,
        epsilon_buffer=blocks.geometry.buffer(-(1-epsilon)*blocks.max_radius),
        width_buffer  =blocks.geometry.buffer(-0.2*blocks.max_radius)
    )

    # Compute block-grid overlap weights
    bgo = compute_block_grid_weights(blocks, grid).compute()

    # Count buildings per (block, cell)
    buildings_pdf = buildings.compute()[['geometry']]
    join = gpd.sjoin(buildings_pdf, bgo[['block_id','grid_id','geometry']], predicate='intersects')
    n_bc = join.groupby(['block_id','grid_id']).size().rename('n_buildings_cell').reset_index()
    bgo  = bgo.merge(n_bc, on=['block_id','grid_id'], how='left').fillna({'n_buildings_cell':0})

    # Compute block-level KL metrics and weight per cell
    kl_df = compute_block_kl_metrics(
        dgpd.sjoin(buildings, blocks, predicate='intersects')[['block_id','geometry','epsilon_buffer','width_buffer','azimuth']]
        .set_index('block_id').repartition(npartitions=4)
    ).compute()
    df = (bgo.merge(kl_df, on='block_id', how='left')
          .dropna(subset=['standardized_kl'])
          .assign(weight=lambda d: d.area_weight * d.n_buildings_cell,
                  weighted_kl=lambda d: d.standardized_kl * d.weight)
    )
    grid_m6 = df.groupby('grid_id').agg(total_weighted_kl=('weighted_kl','sum'), total_weight=('weight','sum'))
    grid_m6['m6'] = grid_m6.total_weighted_kl / grid_m6.total_weight

    # Compute weighted average block width (m7)
    bgo['weighted_max_radius'] = bgo.max_radius * bgo.area_weight
    grid_m7 = bgo.groupby('grid_id').agg(total_weighted_max_radius=('weighted_max_radius','sum'), total_weight=('area_weight','sum'))
    grid_m7['m7'] = grid_m7.total_weighted_max_radius / grid_m7.total_weight

    # Fallback unweighted KL for cells without blocks but ≥2 buildings
    def kl_divergence(arr, bins=18):
        hist,_ = np.histogram(arr, bins=bins, range=(0,180))
        P = hist/hist.sum() if hist.sum()>0 else np.ones(bins)/bins
        Q = np.ones(bins)/bins
        return entropy(P, Q) / np.log(bins)

    b2g = dgpd.sjoin(buildings[['geometry','azimuth']], grid[['geometry']], predicate='intersects').persist()
    m6_unw = (b2g.groupby('index_right')['azimuth']
               .apply(lambda s: kl_divergence(s.to_numpy()), meta=('azimuth','float64'))
               .rename('m6_unweighted')
    )

    # Merge m6 and m7 back into grid
    grid = (grid.rename_axis('grid_id').reset_index()
                .merge(grid_m6.reset_index(), on='grid_id', how='left')
                .merge(m6_unw.reset_index(),   on='grid_id', how='left')
                .merge(grid_m7.reset_index()[['grid_id','m7']], on='grid_id', how='left')
                .set_index('grid_id')
    )

    # Finalize raw and standardized columns for m6, m7
    grid['m6_raw'] = grid['m6'].fillna(grid['m6_unweighted'])
    m6_median     = grid['m6_raw'].median().compute()
    grid['m6_raw'] = grid['m6_raw'].fillna(m6_median)
    grid['m6_std'] = grid['m6_raw'].map_partitions(standardize_metric_6, meta=('m6','float64'))

    grid['m7_raw'] = grid['m7'].fillna(200)
    grid['m7_std'] = grid['m7_raw'].map_partitions(standardize_metric_7, meta=('m7','float64'))

    grid = grid.drop(columns=['m6','m7'])

    # Write result to GeoParquet
    out = f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_grid_{YOUR_NAME}_metrics_6_7.geoparquet"
    grid.to_parquet(out)
    return out


# -----------------------------------------------------------------------------
# 4. ROADS & INTERSECTIONS METRICS (m8, m9)
# -----------------------------------------------------------------------------
@delayed
def metrics_roads_intersections(city_name, grid_size, YOUR_NAME):
    """
    Computes connectivity and tortuosity metrics:
      - m8: average road tortuosity per cell
      - m9: average intersection angle per cell
    """
    # Load grid, roads, and intersections
    epsg         = get_epsg(city_name).compute()
    grid         = load_dataset(f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet', epsg=epsg).persist()
    roads        = load_dataset(f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet', epsg=epsg).persist()
    intersections= load_dataset(f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet', epsg=epsg).compute()

    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Compute average intersection angles
    intersections['osmid'] = intersections['osmid'].astype(int)
    intersection_angles    = compute_intersection_angles(roads, intersections)
    street_count_map       = intersections.set_index('osmid')['street_count'].to_dict()
    angle_map              = compute_intersection_mapping(intersection_angles, street_count_map).compute()
    intersections_with_ang = intersections.merge(angle_map.rename('average_angle'), left_on='osmid', right_index=True, how='left')
    joined_angles          = dgpd.sjoin(intersections_with_ang, grid, predicate='within')
    average_angle_between_roads = joined_angles.groupby('index_right')['average_angle'].mean()

    # Compute road tortuosity per cell via overlay and partition function
    roads_simple = roads[['geometry']]
    grid_small   = grid.reset_index()[['index','geometry']].rename(columns={'index':'index_right'})

    overlay_meta = gpd.GeoDataFrame({'index_right': pd.Series(dtype='int64'),
                                     'geometry':    gpd.GeoSeries(dtype='geometry')}, geometry='geometry')

    roads_cells = roads_simple.map_partitions(overlay_partition, grid_small, meta=overlay_meta).persist()
    tort_parts  = roads_cells.map_partitions(partition_tortuosity_clipped, meta=pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'wt':           pd.Series(dtype='float64'),
        'length':      pd.Series(dtype='float64')
    })).persist()

    agg = tort_parts.groupby('index_right').agg(total_len=('length','sum'), sum_wt=('wt','sum'))
    grid['m8_raw'] = grid.index.map(agg['sum_wt'] / agg['total_len']).astype(float).fillna(0.0)
    grid['m8_std'] = grid['m8_raw'].map_partitions(standardize_metric_8, meta=('m8','float64'))

    # Finalize m9 and standardize
    grid['m9_raw'] = grid.index.map(average_angle_between_roads).astype(float)
    m9_median      = grid['m9_raw'].dropna().quantile(0.5).compute()
    grid['m9_raw'] = grid['m9_raw'].fillna(m9_median)
    grid['m9_std'] = grid['m9_raw'].map_partitions(standardize_metric_9, meta=('m9','float64'))

    # Write output GeoParquet
    out_path = f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{grid_size}m_grid_metrics_8_9_{YOUR_NAME}.geoparquet"
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out_path)
    return out_path
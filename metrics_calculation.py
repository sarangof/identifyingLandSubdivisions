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

# Delayed task to compute building and road intersection metrics for each grid cell
# - Calculates building counts and built area per cell
# - Computes road length per cell and related flags
# - Derives intersection counts and standardizes various metrics (m3, m4, m5, m10, m11, m12)
@delayed
def building_and_intersection_metrics(city_name, grid_size, YOUR_NAME):
    # Initialize counter for total grid cells processed
    grid_cell_count = 0

    # Define file paths for inputs based on city name and grid size
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # Retrieve the appropriate EPSG code for spatial operations
    epsg = get_epsg(city_name).compute()

    # Load and prepare road geometries
    roads = load_dataset(paths['roads'], epsg=epsg)
    # Load and prepare grid cells
    grid = load_dataset(paths['grid'], epsg=epsg)  # geometry and index

    # Remove any stray 'geom' column if present
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    # Compute area of each grid cell
    grid['cell_area'] = grid.geometry.area

    # Update grid cell count
    cells = grid.index.size
    grid_cell_count += cells

    # --- Building area and count per cell ---
    # Load building footprints and compute their area
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    buildings['area'] = buildings.geometry.area

    # Create a lightweight copy of grid for spatial joins
    grid_small = (
        grid
        .reset_index()[['index', 'geometry']]
        .rename(columns={'index': 'index_right'})
    )

    # Define function for per-partition clipping and aggregation of building areas
    def building_area_partition(bldg_part, grid_sm):
        # Clip building geometries to grid cells
        clipped = gpd.overlay(bldg_part, grid_sm, how='intersection')
        # Compute clipped building area
        clipped['clipped_area'] = clipped.geometry.area
        # Aggregate built area and count buildings by cell
        out = clipped.groupby('index_right').agg(
            built_area=('clipped_area', 'sum'),
            n_buildings=('geometry', 'size')
        )
        return out

    # Define minimal metadata DataFrame for Dask
    meta_ba = pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'built_area': pd.Series(dtype='float64'),
        'n_buildings': pd.Series(dtype='int64')
    }).set_index('index_right')

    # Apply building_area_partition across Dask partitions and persist
    parts = buildings.map_partitions(
        building_area_partition,
        grid_small,
        meta=meta_ba
    ).persist()

    # Sum results across partitions
    agg = parts.groupby('index_right').sum()

    # Attach aggregated building metrics back to grid
    grid['n_buildings'] = agg['n_buildings'].fillna(0).astype(int)
    grid['built_area']  = agg['built_area'].fillna(0.0)

    # --- Road length per cell ---
    # Extract only geometry column for roads
    roads_geo = roads[['geometry']]

    # Prepare lightweight grid copy again for overlay
    grid_small = (
        grid.reset_index()[['index', 'geometry']]
            .rename(columns={'index': 'index_right'})
    )

    # Define partition function to compute clipped road lengths
    def road_length_partition(df, grid_sm):
        clipped = gpd.overlay(df, grid_sm, how='intersection')
        # Extract length of each clipped segment
        L = clipped.geometry.length.values
        return pd.DataFrame({
            'index_right': clipped['index_right'].values,
            'length_in_cell': L
        }, index=clipped.index)

    # Metadata for Dask overlay of road lengths
    meta_rl = pd.DataFrame({
        'index_right': pd.Series(dtype='int64'),
        'length_in_cell': pd.Series(dtype='float64')
    })

    # Compute road lengths per partition and persist
    road_parts = roads_geo.map_partitions(
        road_length_partition, grid_small, meta=meta_rl
    ).persist()

    # Aggregate total road length per cell
    agg_rl = road_parts.groupby('index_right').agg(
        total_len_m=('length_in_cell', 'sum')
    )

    # Convert cell area to km^2 and road length to km, add a boolean to indicate whether a grid cell has roads.
    grid['cell_area_km2'] = grid['cell_area'] / 1e6
    grid['road_length'] = (agg_rl['total_len_m'] / 1000.0).fillna(0.0)
    grid['has_roads'] = grid['road_length'] > 0

    # --- Intersection counts per cell ---
    intersections = load_dataset(paths['intersections'], epsg=epsg)
    ji = dgpd.sjoin(intersections, grid, predicate='intersects')
    # Count intersections by minimum street counts
    counts2 = ji[ji.street_count >= 2].groupby('index_right').size()
    counts3 = ji[ji.street_count >= 3].groupby('index_right').size()
    counts4 = ji[ji.street_count == 4].groupby('index_right').size()

    # Attach intersection counts and flags
    grid['n_intersections'] = counts2.fillna(0).astype(int)
    grid['has_intersections'] = (grid['n_intersections'] > 0).astype(bool)
    grid['intersections_3plus'] = counts3.fillna(0).astype(int)
    grid['intersections_4way']  = counts4.fillna(0).astype(int)

    # --- Compute raw and standardized metrics ---
    # M3: Road density (km per km^2)
    grid['m3_raw'] = grid['road_length'] / grid['cell_area_km2']
    grid['m3_std'] = grid['m3_raw'].map_partitions(
        standardize_metric_3, meta=('m3', 'float64')
    )

    # M4: Fraction of 4-way intersections among 3+ intersections
    grid['m4_raw'] = grid['intersections_4way'] / grid['intersections_3plus']
    # Fill missing values with median of non-null
    m4_median = grid['m4_raw'].dropna().quantile(0.5).compute()
    grid['m4_raw'] = grid['m4_raw'].fillna(m4_median)
    grid['m4_std'] = grid['m4_raw'].map_partitions(
        standardize_metric_4, meta=('m4', 'float64')
    )

    # M5: Intersection density per m^2, scaled to per km^2
    grid['m5_raw'] = (1000 ** 2) * (grid['n_intersections'] / grid['cell_area'])
    grid['m5_raw'] = grid['m5_raw'].mask(
        grid['has_roads'] & grid['m5_raw'].isna(),
        0.0
    )
    grid['m5_std'] = grid['m5_raw'].map_partitions(
        standardize_metric_5, meta=('m5', 'float64')
    )

    # M10: Building density (count per km^2)
    grid['m10_raw'] = grid['n_buildings'] / grid['cell_area_km2']
    grid['m10_std'] = grid['m10_raw'].map_partitions(
        standardize_metric_10, meta=('m10', 'float64')
    )

    # M11: Built-area fraction of cell area
    grid['m11_raw'] = grid['built_area'] / grid['cell_area']
    grid['m11_std'] = grid['m11_raw'].map_partitions(
        standardize_metric_11, meta=('m11', 'float64')
    )

    # M12: Average building size (built area per building)
    grid['m12_raw'] = (grid['built_area'] / grid['n_buildings']).fillna(0.0)
    grid['m12_std'] = grid['m12_raw'].map_partitions(
        standardize_metric_12, meta=('m12', 'float64')
    )

    # --- Output results to Parquet file ---
    out = (
        f'{OUTPUT_PATH_RASTER}/{city_name}/'
        f'{city_name}_{grid_size}m_metrics_3_4_5_10_11_12_grid_{YOUR_NAME}.geoparquet'
    )
    # Drop stray geometry if present before writing
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out)
    return out

# Delayed task to compute building distance metrics for each grid cell
# - Counts buildings and computes average distance to nearest road per cell
# - Calculates proportion of buildings within 20m and standardizes metrics m1 and m2
@delayed
def building_distance_metrics(city_name, grid_size, YOUR_NAME):
    # Define file paths for inputs based on city name and grid size
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # Retrieve the EPSG code for spatial operations
    epsg = get_epsg(city_name).compute()

    # Load and prepare grid cells
    grid = load_dataset(paths['grid'], epsg=epsg)
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # Load building data with precomputed distances to nearest road
    buildings = load_dataset(paths['buildings_with_distances'], epsg=epsg)
    # Ensure distance field is float
    buildings['distance_to_nearest_road'] = buildings['distance_to_nearest_road'].astype(float)
    # Compute building footprint area
    buildings['area'] = buildings.geometry.area

    # Spatial join buildings to grid cells
    joined_buildings = dgpd.sjoin(buildings, grid, predicate='intersects')

    # Count total buildings per cell
    counts_buildings = joined_buildings.groupby('index_right').size()
    grid['n_buildings'] = counts_buildings.fillna(0).astype(int)
    grid['has_buildings'] = grid['n_buildings'] > 0

    # Compute average distance to nearest road per cell
    average_distance = joined_buildings.groupby('index_right')['distance_to_nearest_road'].mean()
    grid['average_distance_nearest_building'] = average_distance.fillna(0.0)

    # Identify buildings closer than 20m to roads
    buildings_closer_than_20m = buildings[buildings['distance_to_nearest_road'] <= 20]
    joined_closer = dgpd.sjoin(buildings_closer_than_20m, grid, predicate='intersects')
    n_closer = joined_closer.groupby('index_right').size()
    grid['n_buildings_closer_than_20m'] = n_closer.fillna(0).astype(int)

    # Assign zero where cells have buildings but none are within 20m
    grid = grid.assign(
        n_buildings_closer_than_20m =
            grid['n_buildings_closer_than_20m'].mask(
                (grid['n_buildings'] > 0) &
                (grid['n_buildings_closer_than_20m'].isna()),
                0
            )
    )

    # M1: Proportion of buildings within 20m of a road
    grid = grid.assign(
        m1_raw = grid['n_buildings_closer_than_20m'] / grid['n_buildings']
    )
    grid['m1_raw'] = grid['m1_raw'].fillna(grid['m1_raw'].median())
    grid['m1_std'] = grid['m1_raw'].map_partitions(
        standardize_metric_1, meta=('m1', 'float64')
    )

    # M2: Average building-to-road distance metric
    grid['m2_raw'] = grid['average_distance_nearest_building'].fillna(
        grid['average_distance_nearest_building'].median()
    )
    grid['m2_std'] = grid['m2_raw'].map_partitions(
        standardize_metric_2, meta=('m2', 'float64')
    )

    # Output results to Parquet file
    path = (
        f'{OUTPUT_PATH_RASTER}/{city_name}/'
        f'{city_name}_{str(grid_size)}m_grid_{YOUR_NAME}_metrics_1_2.geoparquet'
    )
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(path)


# Delayed task to compute KL divergence and average block width metrics per grid cell
# - M6: Orientation-based KL divergence weighted by block overlap and building counts
#       Fallback: unweighted KL for cells with ≥2 buildings and no associated blocks
# - M7: Average block width, weighted by block area share within each grid cell
@delayed
def compute_m6_m7(city_name, grid_size, YOUR_NAME):
    """
    Computes:
    - M6 (weighted): KL divergence of building orientations per cell
        * Weight by block-area share within cell and building count per block-cell overlap
    - M6 (fallback): Unweighted KL for cells with ≥2 buildings but no block overlap
    - M7: Mean block width per cell, weighted by block-area share
    """

    # 0) Determine projection and load inputs
    epsg = get_epsg(city_name).compute()
    grid = load_dataset(
        f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
        epsg=epsg
    )
    blocks = load_dataset(
        f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        epsg=epsg
    ).persist()
    buildings = load_dataset(
        f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet',
        epsg=epsg
    ).persist()
    # Ensure azimuth is numeric
    buildings['azimuth'] = buildings['azimuth'].map_partitions(
        pd.to_numeric, meta=('azimuth','float64'), errors='coerce'
    )

    # Remove any extraneous geometry column
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # 1) Prepare block buffers for overlap weighting
    epsilon = 0.001
    blocks = blocks.assign(
        block_id=blocks.index,
        epsilon_buffer=blocks.geometry.buffer(-(1-epsilon) * blocks.max_radius),
        width_buffer=blocks.geometry.buffer(-0.2 * blocks.max_radius)
    )

    # 2) Compute block-to-grid overlap weights (area_weight per record)
    bgo = compute_block_grid_weights(blocks, grid).compute()

    # 3) Count buildings within each block-cell overlap
    buildings_pdf = buildings.compute()[['geometry']]
    join = gpd.sjoin(
        buildings_pdf,
        bgo[['block_id', 'grid_id', 'geometry']],
        predicate='intersects'
    )
    n_bc = (
        join.groupby(['block_id','grid_id']).size()
            .rename('n_buildings_cell')
            .reset_index()
    )
    bgo = (
        bgo.merge(n_bc, on=['block_id','grid_id'], how='left')
           .fillna({'n_buildings_cell': 0})
    )

    # 4) Compute block-level KL and aggregate to weighted m6 per cell
    kl_df = compute_block_kl_metrics(
        dgpd.sjoin(buildings, blocks, predicate='intersects')
            [['block_id','geometry','epsilon_buffer','width_buffer','azimuth']]
            .set_index('block_id')
            .repartition(npartitions=4)
    ).compute()
    df = (
        bgo.merge(kl_df, on='block_id', how='left')
           .dropna(subset=['standardized_kl'])
           .assign(
               weight=lambda d: d.area_weight * d.n_buildings_cell,
               weighted_kl=lambda d: d.standardized_kl * d.weight
           )
    )
    grid_m6 = (
        df.groupby('grid_id')
          .agg(
              total_weighted_kl=('weighted_kl','sum'),
              total_weight=('weight','sum')
          )
    )
    grid_m6['m6'] = grid_m6.total_weighted_kl / grid_m6.total_weight

    # 5) Compute weighted average block width per cell (m7)
    bgo['weighted_max_radius'] = bgo.max_radius * bgo.area_weight
    grid_m7 = (
        bgo.groupby('grid_id')
           .agg(
               total_weighted_max_radius=('weighted_max_radius','sum'),
               total_weight=('area_weight','sum')
           )
    )
    grid_m7['m7'] = grid_m7.total_weighted_max_radius / grid_m7.total_weight

    # 6) Fallback: unweighted orientation KL for cells lacking block overlaps
    def kl_divergence(arr, bins=18):
        hist,_ = np.histogram(arr, bins=bins, range=(0,180))
        P = hist/hist.sum() if hist.sum()>0 else np.ones(bins)/bins
        Q = np.ones(bins)/bins
        return entropy(P, Q) / np.log(bins)

    b2g = dgpd.sjoin(
        buildings[['geometry','azimuth']],
        grid[['geometry']],
        predicate='intersects'
    ).persist()
    m6_unw = (
        b2g.groupby('index_right')['azimuth']
           .apply(lambda s: kl_divergence(s.to_numpy()),
                  meta=('azimuth','float64'))
           .rename('m6_unweighted')
    )

    # 7) Merge weighted and unweighted results back into grid
    grid = grid.rename_axis('grid_id')
    grid_m6 = grid_m6.rename_axis('grid_id')
    m6_unw = m6_unw.rename_axis('grid_id')
    grid_m7 = grid_m7.rename_axis('grid_id')
    grid = (
        grid.reset_index()
            .merge(grid_m6.reset_index(), on='grid_id', how='left')
            .merge(m6_unw.reset_index(), on='grid_id', how='left')
            .merge(grid_m7.reset_index()[['grid_id','m7']], on='grid_id', how='left')
            .set_index('grid_id')
    )

    # 8) Standardize raw metrics and drop intermediates
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

    # 9) Write results to Parquet and return path
    out = (
        f'{OUTPUT_PATH_RASTER}/{city_name}/'
        f'{city_name}_{grid_size}m_grid_{YOUR_NAME}_metrics_6_7.geoparquet'
    )
    grid.to_parquet(out)
    return out


# Delayed task to compute tortuosity (M8) and intersection angle (M9) per grid cell
@delayed
def metrics_roads_intersections(city_name, grid_size, YOUR_NAME):
    """
    Computes:
    - M8: Weighted road tortuosity per grid cell
    - M9: Average intersection angle per grid cell
    """
    # Define file paths for required inputs
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
        'blocks': f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'
    }

    # Load spatial data with correct coordinate reference system
    epsg = get_epsg(city_name).compute()
    grid = load_dataset(paths['grid'], epsg=epsg).persist()
    roads = load_dataset(paths['roads'], epsg=epsg).persist()
    intersections = load_dataset(paths['intersections'], epsg=epsg).compute()

    # Clean grid of extraneous geometry field
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])

    # --- M9: Intersection angle metric ---
    intersections['osmid'] = intersections['osmid'].astype(int)
    # Compute angles between connecting roads at each intersection
    intersection_angles = compute_intersection_angles(roads, intersections)
    street_count_mapping = intersections.set_index('osmid')['street_count'].to_dict()
    intersection_angle_mapping = compute_intersection_mapping(
        intersection_angles, street_count_mapping
    ).compute()
    intersections_with_angles = intersections.merge(
        intersection_angle_mapping.rename('average_angle'),
        left_on='osmid', right_index=True, how='left'
    )
    joined_angles = dgpd.sjoin(
        intersections_with_angles, grid, predicate='within'
    )
    average_angle_between_roads = (
        joined_angles.groupby('index_right')['average_angle'].mean()
    )

    # --- M8: Road tortuosity metric ---
    roads_simple = roads[['geometry']]
    grid_small = (
        grid.reset_index()[['index','geometry']]
            .rename(columns={'index':'index_right'})
    )
    overlay_meta = gpd.GeoDataFrame(
        {'index_right': pd.Series(dtype='int64'),
         'geometry': gpd.GeoSeries(dtype='geometry')},
        geometry='geometry'
    )
    # Clip road segments to grid cells
    roads_cells = roads_simple.map_partitions(
        overlay_partition, grid_small, meta=overlay_meta
    ).persist()
    # Compute weighted tortuosity and segment length
    tort = roads_cells.map_partitions(
        partition_tortuosity_clipped,
        meta=pd.DataFrame({
            'index_right': pd.Series(dtype='int64'),
            'wt': pd.Series(dtype='float64'),
            'length': pd.Series(dtype='float64')
        })
    ).persist()
    agg = tort.groupby('index_right').agg(
        total_len=('length','sum'),
        sum_wt=('wt','sum')
    )
    grid['m8_raw'] = (
        grid.index.map(agg['sum_wt']/agg['total_len'])
             .fillna(0.0)
             .astype(float)
    )
    grid['m8_std'] = grid['m8_raw'].map_partitions(
        standardize_metric_8, meta=('m8','float64')
    )

    # Standardize M9 and fill missing values
    grid['m9_raw'] = (
        grid.index.map(average_angle_between_roads)
             .fillna(0.0)
             .astype(float)
    )
    m9_median = grid['m9_raw'].dropna().quantile(0.5).compute()
    grid['m9_raw'] = grid['m9_raw'].fillna(m9_median)
    grid['m9_std'] = grid['m9_raw'].map_partitions(
        standardize_metric_9, meta=('m9','float64')
    )

    # Write and return
    out = (
        f'{OUTPUT_PATH_RASTER}/{city_name}/'
        f'{city_name}_{grid_size}m_grid_metrics_8_9_{YOUR_NAME}.geoparquet'
    )
    if 'geom' in grid.columns:
        grid = grid.drop(columns=['geom'])
    grid.to_parquet(out)
    return out
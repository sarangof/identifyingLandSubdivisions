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
from shapely.geometry import (MultiLineString, LineString, Point,Polygon, MultiPolygon, MultiPoint)
from shapely.ops import (polygonize, nearest_points, voronoi_diagram, linemerge, unary_union)
from shapely.validation import make_valid
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

# Delayed task to compute building and road intersection metrics for each block
# - Calculates building counts and built area per block
# - Computes road length per block and related flags
# - Derives intersection counts and standardizes various metrics (m3, m4, m5, m10, m11, m12)
@delayed
def building_and_intersection_metrics(city_name, YOUR_NAME, boundary_eps=0.75):
    """
    Block-native metrics (M3, M4, M5, M10, M11, M12), with *roads associated to a block*
    defined as: interior segments + the segments that run along the block boundary
    (enclosing). Intersections are counted only if they lie on those associated
    road segments (and only on the valid portions).

    dask-expr compatible; writes a Parquet dataset dir and returns its path.
    boundary_eps is in CRS units (meters if projected).
    """
    # -------- Paths --------
    paths = {
        'buildings':     f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads':         f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet',
        'blocks':        f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
    }

    # -------- Load base layers --------
    epsg = get_epsg(city_name).compute()
    roads = load_dataset(paths['roads'], epsg=epsg)
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    intersections = load_dataset(paths['intersections'], epsg=epsg)
    blocks = load_dataset(paths['blocks'], epsg=epsg)

    # Clean stray columns, ensure unique id
    if 'geom' in blocks.columns:
        blocks = blocks.drop(columns=['geom'])
    if 'fid' in blocks.columns and blocks['fid'].dropna().is_unique:
        blocks = blocks.rename(columns={'fid': 'block_id'})
    else:
        blocks = blocks.reset_index(drop=True)
        blocks['block_id'] = blocks.index.astype('int64')

    # IMPORTANT: make block_id the INDEX ONLY (no same-named column)
    blocks = blocks.set_index('block_id', drop=True)

    # Denominators
    blocks['block_area'] = blocks.geometry.area
    blocks['block_area_km2'] = blocks['block_area'] / 1e6

    # Optional
    buildings['area'] = buildings.geometry.area

    # -------- Small pandas block frames --------
    # For polygon overlays (interior roads/buildings): need block_id as COLUMN
    blocks_small_overlay = (
        blocks[['geometry']].compute().reset_index()[['block_id','geometry']]
    )

    # ---- Blocks overlay for intersection counting (tolerance) ----
    COUNT_BUFFER_M = 0.75  # meters; start with 0.5–1.0 in projected CRS
    blocks_for_ic = blocks_small_overlay.copy()
    blocks_for_ic["geometry"] = blocks_for_ic.geometry.buffer(COUNT_BUFFER_M)


    # For boundary overlays: boundary buffer polygon per block (block_id as COLUMN)
    blocks_boundary_overlay = blocks_small_overlay.copy()
    # buffer the boundary a tiny amount to capture co-linear boundary segments robustly
    blocks_boundary_overlay['geometry'] = (blocks_boundary_overlay.geometry.boundary)


    # For sjoin (points on segments): use index = block_id ONLY
    blocks_small_sjoin = blocks[['geometry']].compute()  # index = block_id
    blocks_small_sjoin.index.name = 'block_id'

    # -------- Partition fns --------
    def building_area_partition(
        bldg_part: gpd.GeoDataFrame,
        blocks_sm_ov: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        clipped = gpd.overlay(bldg_part, blocks_sm_ov, how="intersection")

        if clipped.empty:
            return pd.DataFrame(
                columns=["built_area", "n_buildings"],
                index=pd.Index([], name="block_id")
            )

        # clean tiny self-intersections before union
        clipped["geometry"] = clipped.geometry.buffer(0)

        # union fragments per block BEFORE measuring area 
        union_geom = clipped.groupby("block_id")["geometry"].apply(lambda s: s.unary_union)
        built_area = union_geom.area.rename("built_area")

        # Building count: If there is a stable building id column, use it; otherwise keep fragment-count
        if "id" in clipped.columns:
            n_buildings = clipped.groupby("block_id")["id"].nunique().rename("n_buildings")
        elif "building_id" in clipped.columns:
            n_buildings = clipped.groupby("block_id")["building_id"].nunique().rename("n_buildings")
        else:
            n_buildings = clipped.groupby("block_id").size().rename("n_buildings")

        out = pd.concat([built_area, n_buildings], axis=1)
        out.index.name = "block_id"
        return out


    def road_segments_partition(roads_part: gpd.GeoDataFrame,
                                blocks_sm_ov: gpd.GeoDataFrame,
                                blocks_bd_ov: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Return *associated road segments* per block:
          - interior clips (roads ∩ polygon)
          - boundary clips (roads ∩ buffered boundary)
        Columns: ['block_id','geometry']
        """
        inside = gpd.overlay(roads_part, blocks_sm_ov, how='intersection')
        inside = inside[['block_id','geometry']]
        return inside[~inside.geometry.is_empty & (inside.geometry.length > 0)]

    # -------- Metadata (Dask) --------
    meta_ba = pd.DataFrame({
        'block_id': pd.Series(dtype='int64'),
        'built_area': pd.Series(dtype='float64'),
        'n_buildings': pd.Series(dtype='int64')
    }).set_index('block_id')

    # for road segments: we return geometries with block_id (not aggregated yet)
    meta_rseg = gpd.GeoDataFrame({
        'block_id': pd.Series(dtype='int64'),
        'geometry': gpd.GeoSeries(dtype='geometry')
    }, geometry='geometry')

    meta_ic = pd.DataFrame({
        'block_id': pd.Series(dtype='int64'),
        'n_intersections': pd.Series(dtype='int64'),
        'intersections_3plus': pd.Series(dtype='int64'),
        'intersections_4way': pd.Series(dtype='int64')
    }).set_index('block_id')

    # -------- Buildings → blocks (built area / counts) --------
    parts_b = buildings.map_partitions(building_area_partition, blocks_small_overlay, meta=meta_ba).persist()
    agg_bldg = (
        parts_b.reset_index()
               .groupby('block_id')
               .sum()
               .reset_index()
               .set_index('block_id')
    )
    blocks = blocks.join(agg_bldg, how='left')
    blocks['n_buildings'] = blocks['n_buildings'].fillna(0).astype('int64')
    blocks['built_area']  = blocks['built_area'].fillna(0.0)

    # -------- Roads → associated segments (interior + boundary) --------
    roads_simple = roads[['geometry']]
    assoc_segments = roads_simple.map_partitions(
        road_segments_partition,
        blocks_small_overlay,
        blocks_boundary_overlay,
        meta=meta_rseg
    ).persist()

    # Compute total length per block from associated segments
    # Compute total length per block from associated segments
    # IMPORTANT: keep block_id as a COLUMN (never index) until AFTER the groupby
    def seglen_partition(df: gpd.GeoDataFrame) -> pd.DataFrame:
        # Return a plain pandas DF with RangeIndex
        if df.empty:
            return pd.DataFrame(
                {
                    "block_id": pd.Series([], dtype="int64"),
                    "seg_len_m": pd.Series([], dtype="float64"),
                }
            )

        out = pd.DataFrame(
            {
                "block_id": df["block_id"].astype("int64", errors="ignore"),
                "seg_len_m": df.geometry.length.astype("float64"),
            }
        )
        # Make 100% sure index is not named 'block_id'
        out = out.reset_index(drop=True)
        out.index.name = None
        return out

    meta_len = pd.DataFrame(
        {
            "block_id": pd.Series(dtype="int64"),
            "seg_len_m": pd.Series(dtype="float64"),
        }
    )

    seglens = assoc_segments.map_partitions(seglen_partition, meta=meta_len).persist()

    # Aggregate: result is index=block_id ONLY (safe for joining to blocks)
    agg_len = (
        seglens.groupby("block_id")["seg_len_m"]
        .sum()
        .rename("total_len_m")
        .to_frame()
    )
    agg_len = agg_len.rename_axis("block_id")

    blocks = blocks.join(agg_len, how="left")


    blocks['total_len_m'] = blocks['total_len_m'].fillna(0.0)
    blocks['road_length'] = blocks['total_len_m'] / 1000.0  # km
    blocks['has_roads']   = blocks['road_length'] > 0

    def intersections_in_block_partition(pnts_part: gpd.GeoDataFrame, blocks_ov: gpd.GeoDataFrame) -> pd.DataFrame:
        if pnts_part.empty:
            cols = ['n_intersections', 'intersections_3plus', 'intersections_4way']
            return pd.DataFrame(columns=cols, index=pd.Index([], name='block_id'))

        left = pnts_part[['geometry', 'street_count']].copy()
        if 'osmid' in pnts_part.columns:
            left['osmid'] = pnts_part['osmid']

        j = gpd.sjoin(left, blocks_ov[['block_id','geometry']], predicate='intersects', how='inner')

        # Dedup within block
        if 'osmid' in j.columns:
            j = j.drop_duplicates(subset=['block_id','osmid'])
        else:
            j = j.drop_duplicates(subset=['block_id','geometry'])

        g2 = j[j.street_count >= 2].groupby('block_id').size().rename('n_intersections')
        g3 = j[j.street_count >= 3].groupby('block_id').size().rename('intersections_3plus')
        g4 = j[j.street_count == 4].groupby('block_id').size().rename('intersections_4way')
        out = pd.concat([g2, g3, g4], axis=1)
        out.index.name = 'block_id'
        return out


    parts_i = intersections.map_partitions(
        intersections_in_block_partition,
        blocks_for_ic,
        meta=meta_ic
    ).persist()


    agg_i = (
        parts_i.reset_index()
               .groupby('block_id')
               .sum()
               .reset_index()
               .set_index('block_id')
    )
    blocks = blocks.join(agg_i, how='left')
    for c in ['n_intersections', 'intersections_3plus', 'intersections_4way']:
        blocks[c] = blocks[c].fillna(0).astype('int64')
    blocks['has_intersections'] = blocks['n_intersections'] > 0

    # -------- Metrics (raw + std) --------
    blocks['m3_raw'] = blocks['road_length'] / blocks['block_area_km2']                        # road density
    blocks['m3_std'] = blocks['m3_raw'].map_partitions(standardize_metric_3, meta=('m3','float64'))

    blocks['m4_raw'] = blocks['intersections_4way'] / blocks['intersections_3plus']            # 4-way share
    blocks['m4_raw'] = blocks['m4_raw'].replace([np.inf, -np.inf], np.nan)
    m4_med = blocks['m4_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m4_med):
        m4_med = 0.0
    blocks['m4_raw'] = blocks['m4_raw'].fillna(m4_med)
    blocks['m4_std'] = blocks['m4_raw'].map_partitions(standardize_metric_4, meta=('m4','float64'))

    blocks['m5_raw'] = (blocks['n_intersections'] / blocks['block_area']) * (1000 ** 2)        # intersections / km²
    blocks['m5_raw'] = blocks['m5_raw'].mask(blocks['has_roads'] & blocks['m5_raw'].isna(), 0.0)
    blocks['m5_std'] = blocks['m5_raw'].map_partitions(standardize_metric_5, meta=('m5','float64'))

    blocks['m10_raw'] = blocks['n_buildings'] / blocks['block_area_km2']                       # building density
    blocks['m10_std'] = blocks['m10_raw'].map_partitions(standardize_metric_10, meta=('m10','float64'))

    blocks['m11_raw'] = blocks['built_area'] / blocks['block_area']                             # built fraction
    blocks['m11_std'] = blocks['m11_raw'].map_partitions(standardize_metric_11, meta=('m11','float64'))

    blocks['m12_raw'] = (blocks['built_area'] / blocks['n_buildings']).fillna(0.0)             # avg building size
    blocks['m12_std'] = blocks['m12_raw'].map_partitions(standardize_metric_12, meta=('m12','float64'))

    # -------- Write out (dataset directory) --------
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_metrics_3_4_5_10_11_12_{YOUR_NAME}'

    # Dask-safe duplicate check (index is block_id ONLY)
    tmp = blocks.reset_index()[['block_id']]
    n_unique = tmp['block_id'].nunique().compute()
    n_rows   = tmp['block_id'].count().compute()
    if n_unique != n_rows:
        raise ValueError(f"Duplicate block_id detected: n_unique={n_unique}, n_rows={n_rows}")

    # Column-name duplicates check via pandas Index
    import pandas as _pd
    if _pd.Index(blocks.columns).duplicated().any():
        raise ValueError("Duplicate column names found.")

    # Eager write
    blocks.to_parquet(out, write_index=True, compute=True)
    return out

# Delayed task to compute building distance metrics for each grid cell
# - Counts buildings and computes average distance to nearest road per cell
# - Calculates proportion of buildings within 20m and standardizes metrics m1 and m2
@delayed
def building_distance_metrics(city_name, YOUR_NAME):
    """
    Block-native M1 & M2 (using buildings_with_distances):
      - M1: share of buildings within 20m of a road (per block)
      - M2: average building→nearest-road distance (per block)
    Avoids dask-expr pitfalls:
      * 'block_id' is the index ONLY on the dask frame
      * sjoin uses pandas frames inside map_partitions
      * groupby folds via reset_index() → groupby('block_id') → sum
    Writes a Parquet dataset dir and returns its path.
    """

    # ---------- Paths ----------
    paths = {
        'blocks':        f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings_dist':f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
    }

    # ---------- Load bases ----------
    epsg = get_epsg(city_name).compute()

    # Blocks (dask-geopandas)
    blocks = load_dataset(paths['blocks'], epsg=epsg)
    if 'geom' in blocks.columns:
        blocks = blocks.drop(columns=['geom'])

    # Ensure unique block_id, as index only
    if 'fid' in blocks.columns and blocks['fid'].dropna().is_unique:
        blocks = blocks.rename(columns={'fid': 'block_id'})
    elif 'block_id' not in blocks.columns:
        blocks = blocks.reset_index(drop=True)
        blocks['block_id'] = blocks.index.astype('int64')
    blocks = blocks.set_index('block_id', drop=True)

    def _ensure_block_id_index_only(df):
        # block_id must exist ONLY as the index, never as a column
        if 'block_id' in df.columns:
            df = df.drop(columns=['block_id'])
        df.index.name = 'block_id'
        return df

    blocks = _ensure_block_id_index_only(blocks)


    # Buildings with precomputed distances
    buildings = load_dataset(paths['buildings_dist'], epsg=epsg)
    # Robust dtypes
    buildings['distance_to_nearest_road'] = buildings['distance_to_nearest_road'].astype('float64')

    # ---------- Make small pandas block frame for sjoin ----------
    blocks_small_sjoin = blocks[['geometry']].compute()  # pandas GeoDF; index is block_id

    # ---------- Partition function ----------
    def building_dist_partition(bldg_part: gpd.GeoDataFrame,
                                blocks_sm_sj: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Per-partition: sjoin buildings→blocks; aggregate count, sum(distance), and <=20m count.
        Returns a pandas DataFrame indexed by block_id with columns:
          n_buildings, sum_distance, n_closer_20m
        """
        if bldg_part.empty:
            return pd.DataFrame(columns=['n_buildings','sum_distance','n_closer_20m'],
                                index=pd.Index([], name='block_id'))

        # Keep only needed columns to reduce memory
        bb = bldg_part[['geometry', 'distance_to_nearest_road']].copy()
        # sjoin (right index becomes 'index_right')
        j = gpd.sjoin(bb, blocks_sm_sj, predicate='intersects', how='inner')
        if j.empty:
            return pd.DataFrame(columns=['n_buildings','sum_distance','n_closer_20m'],
                                index=pd.Index([], name='block_id'))

        # Move right index into a real column named block_id
        if 'index_right' in j.columns:
            j = j.rename(columns={'index_right': 'block_id'})

        # Aggregate per block
        j['is_close'] = (j['distance_to_nearest_road'] <= 20.0).astype('int8')
        grp = j.groupby('block_id')
        out = grp.agg(
            n_buildings = ('geometry', 'size'),
            sum_distance = ('distance_to_nearest_road', 'sum'),
            n_closer_20m = ('is_close', 'sum'),
        )
        out.index.name = 'block_id'
        return out

    # ---------- Dask metadata ----------
    meta = (
        pd.DataFrame({
            'block_id': pd.Series(dtype='int64'),
            'n_buildings': pd.Series(dtype='int64'),
            'sum_distance': pd.Series(dtype='float64'),
            'n_closer_20m': pd.Series(dtype='int64'),
        }).set_index('block_id')
    )

    # ---------- Map-reduce across partitions ----------
    parts = buildings.map_partitions(building_dist_partition, blocks_small_sjoin, meta=meta).persist()
    agg_all = (
        parts.reset_index()
             .groupby('block_id')     # dask-expr safe (no as_index=)
             .sum()
             .reset_index()
             .set_index('block_id')
    )

    # ---------- Join back to blocks (safe alignment) ----------
    blocks = blocks.join(agg_all, how='left')
    for c, dtype in [('n_buildings','int64'), ('n_closer_20m','int64')]:
        blocks[c] = blocks[c].fillna(0).astype(dtype)
    blocks['sum_distance'] = blocks['sum_distance'].fillna(0.0)

    blocks['has_buildings'] = blocks['n_buildings'] > 0

    # Average distance (handle 0/0)
    with pd.option_context('mode.use_inf_as_na', True):
        blocks['average_distance_nearest_building'] = (
            (blocks['sum_distance'] / blocks['n_buildings'].where(blocks['n_buildings'] > 0))
            .fillna(0.0)
        )

    # ---------- Metrics ----------
    # M1: share of buildings within 20m (fill NA with median of non-NA)
    blocks['m1_raw'] = (blocks['n_closer_20m'] / blocks['n_buildings'].where(blocks['n_buildings'] > 0))
    m1_med = blocks['m1_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m1_med):
        m1_med = 0.0
    blocks['m1_raw'] = blocks['m1_raw'].fillna(m1_med)
    blocks['m1_std'] = blocks['m1_raw'].map_partitions(standardize_metric_1, meta=('m1', 'float64'))

    # M2: average distance (fill NA with median)
    blocks['m2_raw'] = blocks['average_distance_nearest_building']
    m2_med = blocks['m2_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m2_med):
        m2_med = 0.0
    blocks['m2_raw'] = blocks['m2_raw'].fillna(m2_med)
    blocks['m2_std'] = blocks['m2_raw'].map_partitions(standardize_metric_2, meta=('m2', 'float64'))

    # ---------- Write out ----------
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_metrics_1_2_{YOUR_NAME}'

    # Dask-safe duplicate id check (index is block_id ONLY → this is safe)
    tmp = blocks.reset_index()[['block_id']]
    n_unique = tmp['block_id'].nunique().compute()
    n_rows   = tmp['block_id'].count().compute()
    if n_unique != n_rows:
        raise ValueError(f"Duplicate block_id detected: n_unique={n_unique}, n_rows={n_rows}")

    # No duplicate column names
    if pd.Index(blocks.columns).duplicated().any():
        raise ValueError("Duplicate column names found.")

    # Eager write so the delayed task really produces files now
    blocks.to_parquet(out, write_index=True, compute=True)
    return out


# Delayed task to compute KL divergence and average block width metrics per grid cell
# - M6: Orientation-based KL divergence weighted by block overlap and building counts
#       Fallback: unweighted KL for cells with ≥2 buildings and no associated blocks
# - M7: Average block width, weighted by block area share within each grid cell
@delayed
def compute_m6_m7(city_name, YOUR_NAME):
    """
    Block-level metrics:
      - M6: KL divergence of building orientations per block (0..1)
      - M7: Block width proxy per block (≈ 2 * max_radius)
    No grid dependency. dask-expr compatible.
    Writes a Parquet dataset directory and returns its path.
    """
    # ---------- Paths ----------
    paths = {
        'blocks':    f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet',
    }

    # ---------- Load ----------
    epsg = get_epsg(city_name).compute()

    blocks = load_dataset(paths['blocks'], epsg=epsg)
    if 'geom' in blocks.columns:
        blocks = blocks.drop(columns=['geom'])

    # Ensure unique block_id as INDEX ONLY (avoid name collisions later)
    if 'fid' in blocks.columns and blocks['fid'].dropna().is_unique:
        blocks = blocks.rename(columns={'fid': 'block_id'})
    elif 'block_id' not in blocks.columns:
        blocks = blocks.reset_index(drop=True)
        blocks['block_id'] = blocks.index.astype('int64')
    blocks = blocks.set_index('block_id', drop=True)

    # Need max_radius for M7
    if 'max_radius' not in blocks.columns:
        raise ValueError("blocks is missing 'max_radius' column required for M7. Please add it in pre-processing.")

    # Buildings (with azimuths)
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    # Make sure azimuth is numeric float in [0,180)
    buildings['azimuth'] = buildings['azimuth'].map_partitions(
        pd.to_numeric, errors='coerce', meta=('azimuth', 'float64')
    )
    def _clamp_azimuth(s):
        s = s.astype('float64')
        s = s.where(~s.notnull(), s % 180.0)
        return s
    buildings['azimuth'] = buildings['azimuth'].map_partitions(_clamp_azimuth, meta=('azimuth','float64'))

    # ---------- Small pandas frame for sjoin (index = block_id ONLY) ----------
    blocks_small_sjoin = blocks[['geometry']].compute()
    # Explicit index name helps some GeoPandas versions return 'block_id' column
    blocks_small_sjoin.index.name = 'block_id'

    # ---------- KL helper ----------
    def kl_divergence(arr, bins=18):
        if arr.size == 0:
            return np.nan
        hist, _ = np.histogram(arr, bins=bins, range=(0.0, 180.0))
        total = hist.sum()
        if total == 0:
            return np.nan
        P = hist / total
        mask = P > 0
        kl = np.sum(P[mask] * (np.log(P[mask]) - np.log(1.0 / bins)))
        return float(kl / np.log(bins))

    # ---------- Join buildings→blocks and compute M6 per block ----------
    b2b = dgpd.sjoin(
        buildings[['geometry', 'azimuth']],
        blocks_small_sjoin[['geometry']],
        predicate='intersects',
        how='inner'
    ).persist()

    # Harmonize the right key name across GeoPandas/Dask versions
    if 'index_right' in b2b.columns:
        join_key = 'index_right'
    elif 'block_id' in b2b.columns:
        join_key = 'block_id'
    else:
        # As a last resort, try to recover from index names
        # (very rare; helps if sjoin kept right index unnamed)
        # Convert to pandas for a tiny sample to inspect columns
        sample_cols = list(b2b._meta.columns)
        raise KeyError(f"sjoin result lacks 'index_right'/'block_id'. Columns: {sample_cols}")

    m6_series = (
        b2b.groupby(join_key)['azimuth']
           .apply(lambda s: kl_divergence(s.to_numpy()), meta=('azimuth', 'float64'))
           .rename('m6_raw')
    )

    # If key != 'block_id', rename index to 'block_id' for alignment with blocks
    if join_key != 'block_id':
        m6_series = m6_series.rename_axis('block_id')

    # Join back to blocks
    blocks = blocks.join(m6_series, how='left')

    # Fill NA with median (or 0 if all NA), then standardize
    m6_med = blocks['m6_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m6_med):
        m6_med = 0.0
    blocks['m6_raw'] = blocks['m6_raw'].fillna(m6_med)
    blocks['m6_std'] = blocks['m6_raw'].map_partitions(standardize_metric_6, meta=('m6', 'float64'))

    # ---------- M7: width proxy per block ----------
    blocks['m7_raw'] = 2.0 * blocks['max_radius']
    m7_med = blocks['m7_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m7_med):
        m7_med = 200.0
    blocks['m7_raw'] = blocks['m7_raw'].fillna(m7_med)
    blocks['m7_std'] = blocks['m7_raw'].map_partitions(standardize_metric_7, meta=('m7', 'float64'))

    # ---------- Write out ----------
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_metrics_6_7_{YOUR_NAME}'

    # Dask-safe duplicate id check (index is block_id ONLY → reset_index won’t clash)
    tmp = blocks.reset_index()[['block_id']]
    n_unique = tmp['block_id'].nunique().compute()
    n_rows   = tmp['block_id'].count().compute()
    if n_unique != n_rows:
        raise ValueError(f"Duplicate block_id detected: n_unique={n_unique}, n_rows={n_rows}")

    if pd.Index(blocks.columns).duplicated().any():
        raise ValueError("Duplicate column names found.")

    blocks.to_parquet(out, write_index=True, compute=True)
    return out


# Delayed task to compute tortuosity (M8) and intersection angle (M9) per grid cell
@delayed
def metrics_roads_intersections(city_name, YOUR_NAME):
    """
    Block-level metrics:
      - M8: length-weighted road tortuosity per block
      - M9: average intersection angle per block
    No grid dependency. dask-expr compatible.
    Writes a Parquet dataset directory and returns its path.
    """
    # -------- Paths --------
    paths = {
        'blocks':        f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'roads':         f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet',
    }

    # -------- Load --------
    epsg = get_epsg(city_name).compute()
    roads = load_dataset(paths['roads'], epsg=epsg).persist()
    intersections = load_dataset(paths['intersections'], epsg=epsg).compute()  # pandas for angle calc
    blocks = load_dataset(paths['blocks'], epsg=epsg)

    # Clean & ensure unique block_id
    if 'geom' in blocks.columns:
        blocks = blocks.drop(columns=['geom'])
    if 'fid' in blocks.columns and blocks['fid'].dropna().is_unique:
        blocks = blocks.rename(columns={'fid': 'block_id'})
    elif 'block_id' not in blocks.columns:
        blocks = blocks.reset_index(drop=True)
        blocks['block_id'] = blocks.index.astype('int64')
    # IMPORTANT: index ONLY, no same-named column
    blocks = blocks.set_index('block_id', drop=True)

    # -------- Prepare small pandas block frames to avoid sjoin/overlay name clashes --------
    # For overlay_partition (expects a column named 'index_right')
    blocks_small_overlay = (
        blocks[['geometry']].compute()
        .reset_index().rename(columns={'index': 'block_id'})  # ensure explicit name
        .rename(columns={'block_id': 'index_right'})          # overlayPartition expects 'index_right'
        [['index_right', 'geometry']]
    )

    # Improve intersection counting due to small mismatches
    blocks_small_overlay["geometry"] = blocks_small_overlay.geometry.buffer(0)


    # For sjoin: use index=block_id ONLY (no 'block_id' column)
    blocks_small_sjoin = blocks[['geometry']].compute()
    blocks_small_sjoin.index.name = 'block_id'

    # ======================================================================
    # M9: Average intersection angle per block
    # ======================================================================

    # --- Keep only 3-way / 4-way intersections up front (strict metric definition) ---
    intersections['osmid'] = intersections['osmid'].astype('int64')
    intersections_sc = intersections[intersections['street_count'].isin([3, 4])].copy()

    # Bearings from incident edges (roads is Dask; intersections_sc is pandas)
    angles_df = compute_intersection_angles(roads, intersections_sc)

    # STRICT: mapping only for intersections where observed bearings count == street_count
    street_count_mapping = intersections_sc.set_index('osmid')['street_count'].to_dict()
    intersection_angle_mapping = compute_intersection_mapping(angles_df, street_count_mapping).compute()

    intersections_with_angles = intersections_sc.merge(
        intersection_angle_mapping.rename('average_angle'),
        left_on='osmid', right_index=True, how='left'
    )

    # --- Assign intersections to blocks (avoid losing boundary points) ---
    # within() excludes points exactly on the boundary; buffer fixes that.
    blocks_for_m9 = blocks_small_sjoin[['geometry']].copy()
    blocks_for_m9['geometry'] = blocks_for_m9.geometry.buffer(0.5)  # meters (epsg is projected here)

    j_int = gpd.sjoin(
        intersections_with_angles,
        blocks_for_m9,
        predicate='within',
        how='inner'
    )

    if 'index_right' in j_int.columns:
        j_int = j_int.rename(columns={'index_right': 'block_id'})

    # Keep only intersections that actually got a strict angle value
    j_ok = j_int[j_int['average_angle'].notna()].copy()

    # Per-block mean angle metric
    m9_per_block = (
        j_ok.groupby('block_id')['average_angle']
            .mean()
            .rename('m9_raw')
    )

    # Per-block count of contributing intersections
    m9_n_per_block = (
        j_ok.groupby('block_id')
            .size()
            .rename('m9_n_intersections')
    )

    m9_out = pd.concat([m9_per_block, m9_n_per_block], axis=1)
    m9_df = dd.from_pandas(m9_out.reset_index().set_index('block_id'), npartitions=1)


    # ======================================================================
    # M8: Road tortuosity per block
    # ======================================================================
    roads_simple = roads[['geometry']]

    # Meta for overlay_partition output (GeoDataFrame with 'index_right' and geometry)
    overlay_meta = gpd.GeoDataFrame(
        {'index_right': pd.Series(dtype='int64'),
         'geometry': gpd.GeoSeries(dtype='geometry')},
        geometry='geometry'
    )

    # Clip road segments to blocks (Dask left, pandas right)
    roads_blocks = roads_simple.map_partitions(
        overlay_partition, blocks_small_overlay, meta=overlay_meta
    ).persist()

    # Compute weighted tortuosity + segment length on clipped segments
    tort = roads_blocks.map_partitions(
        partition_tortuosity_clipped,
        meta=pd.DataFrame({
            'index_right': pd.Series(dtype='int64'),
            'wt': pd.Series(dtype='float64'),
            'length': pd.Series(dtype='float64')
        })
    ).persist()

    # Aggregate to per-block
    agg = (
        tort.groupby('index_right')
            .agg(total_len=('length', 'sum'),
                 sum_wt=('wt', 'sum'))
    )
    # m8_raw = sum_wt / total_len; rename index to block_id for alignment
    agg = agg.assign(m8_raw=agg['sum_wt'] / agg['total_len']).rename_axis('block_id')

    # ======================================================================
    # Join results back to blocks & standardize
    # ======================================================================
    # Join M8
    blocks = blocks.join(agg[['m8_raw']], how='left')
    blocks['m8_raw'] = blocks['m8_raw'].fillna(0.0)
    blocks['m8_std'] = blocks['m8_raw'].map_partitions(standardize_metric_8, meta=('m8', 'float64'))

    # Join M9
    blocks = blocks.join(m9_df[['m9_raw', 'm9_n_intersections']], how='left')
    blocks['m9_n_intersections'] = blocks['m9_n_intersections'].fillna(0).astype('int64')

    # Keep m9_raw as NaN when nothing valid contributed (your preference)
    # Standardize on an imputed series so std exists, but raw preserves missingness
    m9_median = blocks['m9_raw'].dropna().quantile(0.5).compute()
    if pd.isna(m9_median):
        m9_median = 0.0

    m9_for_std = blocks['m9_raw'].fillna(m9_median)
    blocks['m9_std'] = m9_for_std.map_partitions(standardize_metric_9, meta=('m9', 'float64'))


    # ======================================================================
    # Write out (dataset directory)
    # ======================================================================
    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_metrics_8_9_{YOUR_NAME}'

    # Dask-safe duplicate id check
    tmp = blocks.reset_index()[['block_id']]
    n_unique = tmp['block_id'].nunique().compute()
    n_rows   = tmp['block_id'].count().compute()
    if n_unique != n_rows:
        raise ValueError(f"Duplicate block_id detected: n_unique={n_unique}, n_rows={n_rows}")

    # Column-name dup check
    if pd.Index(blocks.columns).duplicated().any():
        raise ValueError("Duplicate column names found.")

    # Eager write so this delayed task actually produces files
    blocks.to_parquet(out, write_index=True, compute=True)
    return out


# =====================================================================
# Block-level K-complexity (no grid dependency)
# =====================================================================

@delayed
def metrics_k_blocks(city_name, YOUR_NAME, buffer_radius=100.0):
    """
    Block-level K-complexity metrics.

    Uses the helpers defined in auxiliary_functions.py:

        compute_k_single_block(block_id,
                               block_geom,
                               buildings_geoms,
                               streets_geoms,
                               buffer_radius)

    Strategy (mirrors compute_m6_m7 style):

    1. Load blocks, buildings, roads with load_dataset (Dask-GeoPandas).
    2. Compute a small GeoPandas blocks frame for spatial joins.
    3. Do ONE buildings→blocks spatial join (Dask) and ONE roads→blocks
       spatial join with buffered blocks.
    4. Convert those joins to pandas and group them by block_id.
    5. For each block_id, call compute_k_single_block with *only*
       the buildings/roads relevant to that block.
    6. Join results back to blocks and write a Parquet dataset.
    """

    # ---------- Paths ----------
    paths = {
        'blocks':    f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads':     f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
    }

    # ---------- Load (Dask GeoDataFrames) ----------
    epsg = get_epsg(city_name).compute()
    blocks = load_dataset(paths['blocks'], epsg=epsg)
    buildings = load_dataset(paths['buildings'], epsg=epsg)
    roads = load_dataset(paths['roads'], epsg=epsg)

    # Clean stray columns, ensure unique block_id (same logic as other blocks)
    if 'geom' in blocks.columns:
        blocks = blocks.drop(columns=['geom'])

    if 'fid' in blocks.columns and blocks['fid'].dropna().is_unique:
        blocks = blocks.rename(columns={'fid': 'block_id'})
    elif 'block_id' not in blocks.columns:
        blocks = blocks.reset_index(drop=True)
        blocks['block_id'] = blocks.index.astype('int64')

    blocks = blocks.set_index('block_id', drop=True)

    # ---------- Small GeoPandas frame for sjoins ----------
    blocks_small = blocks[['geometry']].compute()
    blocks_small.index.name = 'block_id'
    blocks_small_gdf = gpd.GeoDataFrame(blocks_small, geometry='geometry', crs=epsg)

    # ---------- 1) Buildings → blocks (sjoin) ----------
    b2b = dgpd.sjoin(
        buildings[['geometry']],
        dgpd.from_geopandas(blocks_small_gdf[['geometry']], npartitions=1),
        predicate='intersects',     
        how='inner'
    ).persist()

    # Identify right-hand key name (GeoPandas/Dask version differences)
    if 'index_right' in b2b.columns:
        b_join_key = 'index_right'
    elif 'block_id' in b2b.columns:
        b_join_key = 'block_id'
    else:
        sample_cols = list(b2b._meta.columns)
        raise KeyError(
            f"Buildings sjoin lacks 'index_right'/'block_id'. Columns: {sample_cols}"
        )

    b2b_pd = b2b[['geometry', b_join_key]].compute()
    b2b_pd = b2b_pd.rename(columns={b_join_key: 'block_id'})

    # Group buildings by block_id once 
    b_groups = {
        bid: grp['geometry'].tolist()
        for bid, grp in b2b_pd.groupby('block_id')
    }

    # ---------- 2) Roads → blocks with buffer ----------
    # Buffer blocks in GeoPandas, then wrap as Dask for sjoin
    blocks_buffered = blocks_small_gdf.copy()
    blocks_buffered['geometry'] = blocks_buffered['geometry'].buffer(buffer_radius)

    r2b = dgpd.sjoin(
        roads[['geometry']],
        dgpd.from_geopandas(blocks_buffered[['geometry']], npartitions=1),
        predicate='intersects',
        how='inner'
    ).persist()

    if 'index_right' in r2b.columns:
        r_join_key = 'index_right'
    elif 'block_id' in r2b.columns:
        r_join_key = 'block_id'
    else:
        sample_cols = list(r2b._meta.columns)
        raise KeyError(
            f"Roads sjoin lacks 'index_right'/'block_id'. Columns: {sample_cols}"
        )

    r2b_pd = r2b[['geometry', r_join_key]].compute()
    r2b_pd = r2b_pd.rename(columns={r_join_key: 'block_id'})

    r_groups = {
        bid: grp['geometry'].tolist()
        for bid, grp in r2b_pd.groupby('block_id')
    }

    # ---------- 3) Compute K per block (pandas loop, but on pre-subset data) ----------
    results = []
    for bid, block_geom in blocks_small_gdf['geometry'].items():
        # pull pre-subset lists (or empty lists)
        b_geoms = b_groups.get(bid, [])
        r_geoms = r_groups.get(bid, [])

        info = compute_k_single_block(
            block_id=int(bid),
            block_geom=block_geom,
            buildings_geoms=b_geoms,
            streets_geoms=r_geoms,
            buffer_radius=float(buffer_radius),
        )
        results.append(info)

    # ---------- 4) Join back to blocks & write ----------
    if results:
        k_df = pd.DataFrame(results).set_index('block_id')
        out_df = blocks_small_gdf.join(k_df, how='left')
    else:
        # No blocks / no buildings; produce empty but consistent schema
        out_df = blocks_small_gdf.copy()
        out_df['on_network_street_length'] = np.nan
        out_df['off_network_street_length'] = np.nan
        out_df['nearest_external_street'] = np.nan
        out_df['building_count'] = 0
        out_df['building_layers'] = ""
        out_df['k_complexity'] = np.nan

    out = f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_metrics_k_{YOUR_NAME}.geoparquet'

    # Duplicate id check (pandas – like in compute_m6_m7 but non-Dask)
    tmp = out_df.reset_index()[['block_id']]
    n_unique = tmp['block_id'].nunique()
    n_rows = tmp['block_id'].shape[0]
    if n_unique != n_rows:
        raise ValueError(
            f"Duplicate block_id in K-metrics: n_unique={n_unique}, n_rows={n_rows}"
        )

    out_df.to_parquet(out, index=True)
    return out

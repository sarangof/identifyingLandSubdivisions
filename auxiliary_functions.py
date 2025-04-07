import dask_geopandas as dgpd
import pandas as pd
import dask.dataframe as dd
import numpy as np
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
from scipy.stats import entropy
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, mapping, Point
from polylabel import polylabel 


MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
MAIN_PATH = MAIN_PATH
INPUT_PATH = f"{MAIN_PATH}/input"
CITY_INFO_PATH = f"{INPUT_PATH}/city_info"
EXTENTS_PATH = f"{INPUT_PATH}/city_info/extents"
BUILDINGS_PATH = f"{INPUT_PATH}/buildings"
ROADS_PATH = f"{INPUT_PATH}/roads"
BLOCKS_PATH = f"{INPUT_PATH}/blocks"
INTERSECTIONS_PATH = f"{INPUT_PATH}/intersections"
GRIDS_PATH = f"{INPUT_PATH}/city_info/grids"
SEARCH_BUFFER_PATH = f"{INPUT_PATH}/city_info/search_buffers"
OUTPUT_PATH = f"{MAIN_PATH}/output"
OUTPUT_PATH_CSV = f"{OUTPUT_PATH}/csv"
OUTPUT_PATH_RASTER = f"{OUTPUT_PATH}/raster"
OUTPUT_PATH_PNG = f"{OUTPUT_PATH}/png"
OUTPUT_PATH_RAW = f"{OUTPUT_PATH}/raw_results"

"""
BASIC FUNCTIONS
"""

@delayed
def get_epsg(city_name):
    search_buffer = f'{SEARCH_BUFFER_PATH}/{city_name}/{city_name}_search_buffer.geoparquet'
    extent = dgpd.read_parquet(search_buffer)
    geometry = extent.geometry[0].compute()
    epsg = get_utm_crs(geometry)
    print(f'{city_name} EPSG: {epsg}')
    return epsg

def load_dataset(path, epsg=None):
    dataset = dgpd.read_parquet(path, npartitions=4)
    
    # Only assign if the file has no CRS
    if epsg:
        if dataset.crs is None:
            dataset = dataset.set_crs("EPSG:4326")  # assume WGS84 if missing
        dataset = dataset.to_crs(epsg)

    return dataset

@delayed
def row_count(dgdf):
    """Count the rows in a dataframe"""
    row_count = dgdf.map_partitions(len).compute().sum()

    return row_count


def test_math(input):
    return input + input



"""
PRE-PROCESSING FOR BLOCKS
"""

def to_geojson_dict(geom):
    """
    Convert a Shapely geometry to a GeoJSON-like dict with lists instead of tuples.
    """
    geojson = mapping(geom)
    def recursive_convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        else:
            return obj
    return recursive_convert(geojson)

def compute_largest_inscribed_circle(geom):
    """
    Compute the largest inscribed circle for a given polygon or multipolygon.

    Parameters:
      geom (shapely.geometry): A Polygon or MultiPolygon.
    
    Returns:
      tuple: (optimal_point, max_radius) where optimal_point is a shapely Point and max_radius is a float.
    """
    if geom is None or geom.is_empty:
        return None, None

    if geom.geom_type == 'Polygon':
        geojson_poly = to_geojson_dict(geom)
        # Pass in the coordinates list instead of the entire dict.
        optimal_coords = polylabel(geojson_poly["coordinates"])
        optimal = Point(optimal_coords)
        radius = geom.boundary.distance(optimal)
        return optimal, radius

    elif geom.geom_type == 'MultiPolygon':
        best_point = None
        best_radius = 0
        for poly in geom.geoms:
            geojson_poly = to_geojson_dict(poly)
            optimal_coords = polylabel(geojson_poly["coordinates"])
            candidate = Point(optimal_coords)
            radius = poly.boundary.distance(candidate)
            if radius > best_radius:
                best_radius = radius
                best_point = candidate
        return best_point, best_radius

    else:
        return None, None

def add_inscribed_circle_info(blocks_gdf):
    """
    Adds two new columns to a blocks GeoDataFrame: 'optimal_point' and 'max_radius'
    which indicate the center and radius of the largest inscribed circle for each block.
    Converts the optimal_point geometries to WKT strings for Parquet compatibility.
    
    Parameters:
      blocks_gdf (GeoDataFrame): A GeoDataFrame with block polygons.
      
    Returns:
      GeoDataFrame: The input GeoDataFrame with two new columns.
    """
    # Apply the computation for each geometry
    results = blocks_gdf.geometry.apply(lambda geom: compute_largest_inscribed_circle(geom))
    
    # Unpack the tuple results into two new columns
    blocks_gdf["optimal_point"] = results.apply(lambda x: x[0])
    blocks_gdf["max_radius"] = results.apply(lambda x: x[1])
    
    # Convert the 'optimal_point' column from Shapely objects to WKT strings
    blocks_gdf["optimal_point"] = blocks_gdf["optimal_point"].apply(
        lambda geom: geom.wkt if geom is not None else None
    )
    
    return blocks_gdf


def get_blocks(roads):
    """
    Create urban blocks from a grid and road network.

    Parameters:
      grid (GeoDataFrame): A GeoDataFrame of grid polygons defining the city extent.
      roads (GeoDataFrame): A GeoDataFrame of road line geometries.
    
    Returns:
      GeoDataFrame: A GeoDataFrame of block polygons.
    """
    # Merge all road geometries into a single geometry
    roads_union = unary_union(roads.geometry)
    
    # Polygonize the road network to generate blocks.
    # The polygonize function returns an iterator of Polygons.
    blocks_polygons = list(polygonize(roads_union))
    
    # Create a GeoDataFrame for blocks
    blocks_gdf = gpd.GeoDataFrame(geometry=blocks_polygons, crs=roads.crs)
    
    # Remove any empty geometries resulting from the intersection.
    blocks_gdf = blocks_gdf[~blocks_gdf.is_empty]
    
    return blocks_gdf

'''
PRE-PROCESSING: CALCULATE BUILDING AZIMUTH
'''

def compute_azimuth_partition(df):
    def azimuth(geom):
        if geom is None or geom.is_empty:
            return np.nan
        oriented = geom.minimum_rotated_rectangle
        coords = list(oriented.exterior.coords)
        edge = LineString([coords[0], coords[1]])
        dx, dy = edge.xy[0][1] - edge.xy[0][0], edge.xy[1][1] - edge.xy[1][0]
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        return angle % 90

    df = df.copy()
    df['azimuth'] = df['geometry'].map(azimuth)
    return df


def calculate_azimuths(city_name, CITY_NAME, grid_size):
    paths = {
        'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet',
        'blocks': f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet',
        'buildings_with_distances': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances.geoparquet',
        'buildings_with_distances_azimuths': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}_with_distances_and_azimuths.geoparquet',
        'buildings_to_blocks':f'{BLOCKS_PATH}/{city_name}/{city_name}_buildings_to_blocks_{YOUR_NAME}.geoparquet'
    }
    epsg = get_epsg(city_name).compute()
    buildings = load_dataset(paths['buildings_with_distances'], epsg=epsg)
    meta = buildings._meta.copy()
    meta['azimuth'] = 'f8'
    buildings = buildings.map_partitions(compute_azimuth_partition, meta=meta)
    path = paths['buildings_with_distances_azimuths']
    buildings.to_parquet(path)

    return path

"""
METRIC 6 CALCULATION
"""

def calculate_standardized_kl_azimuth(buildings_df, bin_width_degrees=5):
    azimuths = buildings_df['azimuth'].to_numpy()
    num_bins = int(90 / bin_width_degrees)
    histogram, _ = np.histogram(azimuths, bins=num_bins, range=(0, 90))
    P = histogram / histogram.sum() if histogram.sum() > 0 else np.ones(num_bins) / num_bins
    Q = np.ones(num_bins) / num_bins
    kl_divergence = entropy(P, Q)
    max_kl_divergence = np.log(num_bins)
    return kl_divergence / max_kl_divergence


@delayed
def compute_block_kl_metrics(buildings_blocks):
    grouped = buildings_blocks.groupby('block_id')
    kl_data = grouped.apply(lambda g: pd.Series({
        'standardized_kl': calculate_standardized_kl_azimuth(g),
        'n_buildings': len(g),
    })).reset_index()
    return kl_data

def compute_block_grid_weights(blocks, grid):
    """
    Computes the proportional overlap of blocks in each grid cell.
    Returns a Dask DataFrame containing block_id, index_right (grid ID), and area_weight.
    """


    #blocks = blocks.rename_axis(index='block_id').reset_index()
    grid = grid.rename_axis(index='grid_id').reset_index()

    def overlay_partition(blocks_df, grid_df):
        """Computes intersection between blocks and grid."""
        return gpd.overlay(blocks_df, grid_df, how='intersection')

    #meta = blocks._meta.merge(grid._meta, how="outer")

    block_grid_overlap = blocks.map_partitions(overlay_partition, grid)#, meta=meta


    # Step 2: Compute area for each block-grid overlap
    block_grid_overlap = block_grid_overlap.assign(
        overlap_area=block_grid_overlap.map_partitions(lambda df: df.geometry.area, meta=('overlap_area', 'f8'))
    )

    # Step 3: Compute the total area of each grid cell
    grid_areas = grid.assign(grid_area=grid.map_partitions(lambda df: df.geometry.area, meta=('grid_area', 'f8')))


    # Step 4: Merge grid cell areas into block-grid overlap
    block_grid_overlap = block_grid_overlap.merge(grid_areas[['grid_id','grid_area']], left_on='grid_id', right_on='grid_id', how='left')

    # Step 5: Compute area weight as the ratio of overlap to grid cell area
    block_grid_overlap = block_grid_overlap.assign(
        area_weight=block_grid_overlap['overlap_area'] / block_grid_overlap['grid_area']
    )
    block_grid_overlap = block_grid_overlap.map_partitions(
        lambda df: df.assign(
            area_weight=df['area_weight'] / df.groupby(df['grid_id'])['area_weight'].transform('sum')
        ),
        meta=block_grid_overlap._meta  # Preserve original structure
    )

    return block_grid_overlap[['block_id', 'optimal_point', 'max_radius', 'grid_id', 'geometry', 'overlap_area', 'grid_area', 'area_weight']]


def aggregate_m6(kl_df, overlap_df):
    df = overlap_df.merge(kl_df, on='block_id', how='left')
    df = df.dropna(subset=['standardized_kl'])

    # Compute weights
    df['weight'] = df['area_weight'] * df['n_buildings']
    df['weighted_kl'] = df['standardized_kl'] * df['weight']

    # Aggregate directly at the GRID level
    grid_aggregated = df.groupby('grid_id').agg(
        total_weighted_kl=('weighted_kl', 'sum'),
        total_weight=('weight', 'sum')
    )

    # Compute final KL divergence for each grid cell
    grid_aggregated['m6'] = grid_aggregated['total_weighted_kl'] / grid_aggregated['total_weight']

    return grid_aggregated[['m6']]

"""
BLOCK WIDTH CALCULATION
"""

def get_internal_buffer_with_target_area(geom, target_area, tolerance=1e-6, max_iter=100):
    """
    Iteratively finds an internal buffer that results in the target area.

    Parameters:
    - geom: Shapely Polygon geometry.
    - target_area: Desired area for the internal buffer.
    - tolerance: Error tolerance for area difference.
    - max_iter: Maximum iterations to refine buffer.

    Returns:
    - Buffered geometry (Polygon) or the original geometry if buffering is not possible.
    """
    if geom.is_empty or geom.area <= target_area:
        return geom  # Return original if no valid buffer can be made

    buffer_dist = -0.1 * (geom.area ** 0.5)  # Start with a fraction of block size
    iteration = 0

    while iteration < max_iter:
        buffered_geom = geom.buffer(buffer_dist)

        if buffered_geom.is_empty:
            return geom  # If buffering fails, return original geometry

        new_area = buffered_geom.area
        area_diff = abs(new_area - target_area)

        if area_diff < tolerance:
            return buffered_geom  # Found a good enough buffer

        # Adjust buffer distance using a binary search-like approach
        if new_area > target_area:
            buffer_dist *= 1.1  # Increase buffer distance
        else:
            buffer_dist *= 0.9  # Decrease buffer distance

        iteration += 1

    return buffered_geom  # Return best-found buffer

"""
AUXULIARY FUNCTIONS FOR METRIC 8
"""

def clip_group(df_group, buffer_type):
    # Use the buffer from the specified column.
    buffer_geom = df_group[buffer_type].iloc[0]
    # Use the original building footprint column for intersection.
    clipped = np.vectorize(lambda geom: geom.intersection(buffer_geom))(df_group['geometry'].values)
    return pd.DataFrame({
        'building_id': df_group['building_id'],
        'block_id': df_group['block_id'],
        'clipped_geometry': clipped
    }, index=df_group.index)


def clip_buildings_by_buffer(buildings_blocks_df, buffer_type):
    # Copy the input and reset the index.
    gdf = buildings_blocks_df.copy().reset_index()
    # If an 'id' column exists, rename it; otherwise, create 'building_id' from the index.
    if 'id' in gdf.columns:
        gdf = gdf.rename(columns={'id': 'building_id'})
    else:
        gdf['building_id'] = gdf.index

    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError("GeoDataFrame must have a projected CRS for efficient clipping.")

    # Group by block_id and apply the clipping function.
    clipped_series = gdf.groupby('block_id', group_keys=False).apply(clip_group, buffer_type)
    
    # Create a GeoDataFrame from the result.
    clipped_geo = gpd.GeoDataFrame(clipped_series, geometry='clipped_geometry', crs=buildings_blocks_df.crs)
    
    # Merge the clipped geometries back into the original GeoDataFrame.
    gdf = gdf.merge(clipped_geo[['building_id', 'block_id', 'clipped_geometry']], 
                    on=['building_id', 'block_id'], how='left')
    
    gdf_clipped = gpd.GeoDataFrame(gdf.copy(), geometry='clipped_geometry', crs=buildings_blocks_df.crs)
    gdf_clipped['clipped_area'] = gdf_clipped['clipped_geometry'].area
    gdf_clipped['buffer_area'] = gdf_clipped[buffer_type].area
    
    # Aggregate the areas by block.
    clipped_building_area = gdf_clipped.groupby('block_id')['clipped_area'].sum()
    total_buffer_area = gdf_clipped.groupby('block_id')['buffer_area'].sum()
    ratio = clipped_building_area / total_buffer_area
    return ratio


"""
FOR ROAD ANGLES AT INTERSECTIONS
"""


def compute_bearing_vectorized(x1, y1, x2, y2):
    """Compute bearing (in degrees) from (x1,y1) to (x2,y2) in vectorized form."""
    # Compute differences
    dx = x2 - x1
    dy = y2 - y1
    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad) % 360
    return angles_deg

def compute_intersection_angles(roads_df, intersections_df):
    """
    For each road, compute the bearing at the intersection.
    
    For each road, we assume:
      - If an intersection is the start (u), we compute the bearing from that intersection
        (using its coordinates from intersections_df) to the road's centroid.
      - Similarly for the end (v).
    
    Returns a DataFrame with columns:
      intersection_id, bearing
    """
    # Merge for start intersections
    roads_u = roads_df.merge(intersections_df[['osmid', 'geometry']], left_on='u', right_on='osmid', how='left', suffixes=('', '_u'))
    # Extract coordinates: intersection (start) and road centroid
    roads_u['x_u'] = roads_u.geometry_u.x
    roads_u['y_u'] = roads_u.geometry_u.y
    roads_u['centroid_x'] = roads_u.geometry.centroid.x
    roads_u['centroid_y'] = roads_u.geometry.centroid.y
    roads_u['bearing'] = compute_bearing_vectorized(roads_u['x_u'], roads_u['y_u'],
                                                      roads_u['centroid_x'], roads_u['centroid_y'])
    roads_u = roads_u[['u', 'bearing']].rename(columns={'u': 'intersection_id'})
    
    # Merge for end intersections
    roads_v = roads_df.merge(intersections_df[['osmid', 'geometry']], left_on='v', right_on='osmid', how='left', suffixes=('', '_v'))
    roads_v['x_v'] = roads_v.geometry_v.x
    roads_v['y_v'] = roads_v.geometry_v.y
    roads_v['centroid_x'] = roads_v.geometry.centroid.x
    roads_v['centroid_y'] = roads_v.geometry.centroid.y
    roads_v['bearing'] = compute_bearing_vectorized(roads_v['x_v'], roads_v['y_v'],
                                                      roads_v['centroid_x'], roads_v['centroid_y'])
    roads_v = roads_v[['v', 'bearing']].rename(columns={'v': 'intersection_id'})
    
    # Combine both: Each row corresponds to a road connected to an intersection, with its computed bearing.
    intersection_angles = dd.concat([roads_u, roads_v], interleave_partitions=True)
    return intersection_angles

# --- Step 2: Compute Sequential Differences for Each Intersection ---


def compute_sequential_differences(angles):
    """
    Given a sorted array of angles (in degrees), compute the circular differences
    between consecutive angles (including the wrap-around).
    """
    # Append the first angle plus 360 to account for wrap-around
    extended = np.concatenate([angles, [angles[0] + 360]])
    return np.diff(extended)

def compute_intersection_metric(group, street_count_mapping):
    """
    Computes the metric for one intersection.
    
    Parameters:
      group: DataFrame group (with a column 'bearing') for a given intersection.
      street_count_mapping: dict mapping intersection_id to its street_count.
      
    Process:
      1. Get the unique bearings (the "true" angles).
      2. Use the provided street_count (if available) to confirm the number of unique angles.
      3. Sort the unique angles and compute the circular differences.
      4. For 3‑way intersections (street_count==3): select the smallest difference.
         For 4‑way intersections (street_count==4): select the two smallest differences and average them.
      5. Compute the absolute difference between the selected value(s) and 90°.
    """
    inter_id = group.name  # group name is the intersection_id
    # Get expected street count for this intersection
    expected_sc = street_count_mapping.get(inter_id, None)
    
    # Deduplicate angles
    unique_angles = np.unique(group['bearing'].values)
    sc = len(unique_angles)
    
    # If we have an expected street count, use that if possible.
    # (It might differ if data are noisy.)
    if expected_sc is not None and expected_sc in (3, 4):
        sc = expected_sc
    else:
        # Only process intersections with 3 or 4 unique angles
        if sc not in (3, 4):
            return np.nan

    # Ensure we have exactly sc unique angles
    angles = np.sort(unique_angles)
    
    # Compute circular differences
    diffs = compute_sequential_differences(angles)
    
    if sc == 3:
        # For a 3-way, select the smallest difference
        selected_diff = np.min(diffs)
        metric = abs(90 - selected_diff)
    elif sc == 4:
        # For a 4-way, select the two smallest differences and average the absolute differences from 90°
        selected_diffs = np.sort(diffs)[:2]
        metric = np.mean(np.abs(90 - selected_diffs))
    else:
        metric = np.nan
    
    return metric


def compute_intersection_mapping(intersection_angles, street_count_mapping):
    """
    Given a DataFrame 'intersection_angles' with columns ['intersection_id', 'bearing'],
    group by intersection_id and compute the metric using the known street counts.
    Returns a Series mapping intersection_id to the computed metric.
    """
    # Group by intersection_id and apply the metric function.
    mapping = intersection_angles.groupby('intersection_id').apply(
        lambda grp: compute_intersection_metric(grp, street_count_mapping),
        meta=('average_angle', 'float64')
    )
    return mapping

"""
FOR TORTUOSITY INDEX
"""

def calculate_tortuosity(roads_df, intersections_df):

    # Merge roads with intersections for start point (u)
    roads_with_start = roads_df.merge(
        intersections_df[['osmid', 'geometry']],
        left_on='u',
        right_on='osmid',
        how='left',
        suffixes=('', '_start')
    )

    # Merge with intersections again for end point (v)
    roads_with_both = roads_with_start.merge(
        intersections_df[['osmid', 'geometry']],
        left_on='v',
        right_on='osmid',
        how='left',
        suffixes=('', '_end')
    )

    # Rename the merged geometry columns for clarity
    roads_with_both = roads_with_both.rename(
        columns={'geometry': 'road_geometry',
                'geometry_start': 'start_geom',
                'geometry_end': 'end_geom'}
    )

    # --- Step 2: Compute Straight-Line Distance between Intersections ---

    # Use the .distance method from GeoPandas (this assumes a projected CRS)
    roads_with_both['straight_distance'] = roads_with_both.apply(
        lambda row: row['start_geom'].distance(row['end_geom']), axis=1
    )

    # --- Step 3: Compute Road Length (if not already present) ---

    if 'length' not in roads_with_both.columns:
        roads_with_both['length'] = roads_with_both['road_geometry'].length

    # --- Step 4: Compute Tortuosity ---
    # Tortuosity is defined as road_length divided by the straight-line distance.
    # To avoid division by zero, we use np.where to set tortuosity to NaN when straight_distance is 0.

    roads_with_both['tortuosity'] = np.where(
        roads_with_both['straight_distance'] > 0,
        roads_with_both['straight_distance']/ roads_with_both['length'],
        np.nan
    )
    
    return roads_with_both

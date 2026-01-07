import dask_geopandas as dgpd
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dask import delayed, compute, visualize
import geopandas as gpd
from dask.diagnostics import ProgressBar
#from metrics_calculation import calculate_minimum_distance_to_roads_option_B
from shapely.geometry import (MultiLineString, LineString, Point,Polygon, MultiPolygon, MultiPoint)
from shapely.algorithms.polylabel import polylabel
from shapely.ops import (polygonize, nearest_points,voronoi_diagram, linemerge, unary_union)
from shapely.validation import make_valid
from shapely import wkb


#from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from scipy.optimize import fminbound, minimize
#from metrics_groupby import metrics
from scipy.stats import entropy
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, mapping, Point
#from polylabel import polylabel 
from pyproj import CRS, Geod


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

def compute_largest_inscribed_circle(geom, tolerance=1.0):
    """
    Returns (optimal_point: Point, max_radius: float) in the geometry's units.
    """
    if geom is None or geom.is_empty:
        return None, None

    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                return None, None
    except Exception:
        return None, None

    def _solve(poly: Polygon):
        p = polylabel(poly, tolerance=tolerance)  # p is a shapely Point
        r = poly.boundary.distance(p)
        return p, float(r)

    if isinstance(geom, Polygon):
        return _solve(geom)

    if isinstance(geom, MultiPolygon):
        best = (None, 0.0)
        for poly in geom.geoms:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            try:
                p, r = _solve(poly)
                if r > best[1]:
                    best = (p, r)
            except Exception:
                continue
        return best

    # ignore other geometry types
    return None, None



def add_inscribed_circle_info(blocks_gdf, tolerance=1.0):
    """
    Adds two new columns to a blocks GeoDataFrame: 'optimal_point' and 'max_radius'
    which indicate the center and radius of the largest inscribed circle for each block.
    Converts the optimal_point geometries to WKT strings for Parquet compatibility.
    
    Parameters:
      blocks_gdf (GeoDataFrame): A GeoDataFrame with block polygons.
      
    Returns:
      GeoDataFrame: The input GeoDataFrame with two new columns.
    """
    # Make sure CRS is projected (meters) to interpret radius meaningfully
    if blocks_gdf.crs is None or not blocks_gdf.crs.is_projected:
        raise ValueError("blocks_gdf must be in a projected CRS (meters) before computing radii.")

    # Apply the computation for each geometry
    results = blocks_gdf.geometry.apply(lambda g: compute_largest_inscribed_circle(g, tolerance))

    # Unpack the tuple results into two new columns
    blocks_gdf["optimal_point"] = results.apply(lambda x: x[0])
    blocks_gdf["max_radius"]    = results.apply(lambda x: x[1])

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


def calculate_azimuths(city_name, YOUR_NAME):
    paths = {
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
    For each road‐intersection pair (u or v), compute the bearing from the node
    to the *first* segment endpoint rather than the centroid.
    Returns a Dask DataFrame with columns ['intersection_id','bearing'].
    """
    def first_segment_pt(line, at_start):
        # shapely LineString
        coords = list(line.coords)
        return coords[1] if at_start else coords[-2]

    # ——— START intersections ———
    roads_u = roads_df.merge(
        intersections_df[['osmid','geometry']],
        left_on='u', right_on='osmid',
        how='inner', suffixes=('','_u')
    )
    # node coords
    roads_u['x_u'] = roads_u.geometry_u.x
    roads_u['y_u'] = roads_u.geometry_u.y

    # first‐segment endpoint coords, split into two Series
    roads_u['px'] = roads_u.geometry.map_partitions(
        lambda geoms: geoms.apply(lambda g: first_segment_pt(g, True)[0]),
        meta=('px','float64')
    )
    roads_u['py'] = roads_u.geometry.map_partitions(
        lambda geoms: geoms.apply(lambda g: first_segment_pt(g, True)[1]),
        meta=('py','float64')
    )

    # bearing
    roads_u['bearing'] = compute_bearing_vectorized(
        roads_u['x_u'], roads_u['y_u'], roads_u['px'], roads_u['py']
    )
    roads_u = roads_u[['u','bearing']].rename(columns={'u':'intersection_id'})

    # ——— END intersections ———
    roads_v = roads_df.merge(
        intersections_df[['osmid','geometry']],
        left_on='v', right_on='osmid',
        how='inner', suffixes=('','_v')
    )
    roads_v['x_v'] = roads_v.geometry_v.x
    roads_v['y_v'] = roads_v.geometry_v.y

    roads_v['px'] = roads_v.geometry.map_partitions(
        lambda geoms: geoms.apply(lambda g: first_segment_pt(g, False)[0]),
        meta=('px','float64')
    )
    roads_v['py'] = roads_v.geometry.map_partitions(
        lambda geoms: geoms.apply(lambda g: first_segment_pt(g, False)[1]),
        meta=('py','float64')
    )
    roads_v['bearing'] = compute_bearing_vectorized(
        roads_v['x_v'], roads_v['y_v'], roads_v['px'], roads_v['py']
    )
    roads_v = roads_v[['v','bearing']].rename(columns={'v':'intersection_id'})

    # concatenate the two halves
    intersection_angles = dd.concat([roads_u, roads_v], interleave_partitions=True)
    return intersection_angles


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

def partition_tortuosity_clipped(df):
    # df.geometry is the clipped segment
    seg = df.geometry
    L = seg.length.values            # true path length
    b = seg.bounds                   # DataFrame with minx,miny,maxx,maxy
    Dx = (b["maxx"] - b["minx"]).values
    Dy = (b["maxy"] - b["miny"]).values
    D  = np.hypot(Dx, Dy)            # straight‐line distance
    S  = np.clip(D / L, 0, 1)        # straightness in [0,1]
    W  = L * S                       # length‐weighted straightness

    return pd.DataFrame({
        "index_right": df["index_right"].values,
        "wt":           W,
        "length":      L
    }, index=df.index)

def overlay_partition(roads_part, grid_small):
    # roads_part: a pandas GeoDataFrame partition of full roads GDF
    # grid_small: a pandas GeoDataFrame of all grid cells with an index_right column
    return gpd.overlay(roads_part, grid_small, how="intersection")


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


# ---------------------------------------------------------------------
# K-blocks helpers (single-block K-complexity)
# ---------------------------------------------------------------------

def _k_ensure_poly(geom):
    """Ensure a clean single Polygon (largest part if MultiPolygon)."""
    if geom is None:
        return None
    geom = make_valid(geom)
    if isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
        if not geoms:
            return None
        geom = max(geoms, key=lambda p: p.area)
    return geom


def _k_prep_buildings(geoms):
    """Convert building geometries to centroid points."""
    pts = []
    for g in geoms:
        if g is None:
            continue
        g = make_valid(g)
        if isinstance(g, (Polygon, MultiPolygon)):
            c = g.centroid
            if not c.is_empty:
                pts.append(c)
        elif isinstance(g, Point):
            if not g.is_empty:
                pts.append(g)
    if not pts:
        return None, 0
    mp = MultiPoint(pts)
    return mp, len(pts)


def _k_prep_streets(geoms):
    """Flatten LineString / MultiLineString geoms into a MultiLineString."""
    lines = []
    for g in geoms:
        if g is None:
            continue
        g = make_valid(g)
        if isinstance(g, LineString):
            if not g.is_empty:
                lines.append(g)
        elif isinstance(g, MultiLineString):
            for seg in g.geoms:
                if not seg.is_empty:
                    lines.append(seg)
    if not lines:
        return None
    return MultiLineString(lines)


def compute_k_single_block(block_id,
                           block_geom,
                           buildings_geoms,
                           streets_geoms,
                           buffer_radius=100.0):
    """
    Shapely-based version of the original compute_k:

    - builds a Voronoi tessellation from building centroids
    - clips by the block polygon
    - peels parcels into layers from the boundary / street-connected side
    - returns k_complexity = number of parcel layers, plus some network info.
    """
    # ---------------- Block / buildings / streets prep ----------------
    block = _k_ensure_poly(block_geom)
    if block is None or block.is_empty:
        return {
            "block_id": block_id,
            "on_network_street_length": 0.0,
            "off_network_street_length": 0.0,
            "nearest_external_street": float("nan"),
            "building_count": 0,
            "building_layers": "0",
            "k_complexity": 1,
        }

    bldg_points, bldg_count = _k_prep_buildings(buildings_geoms)
    if bldg_points is None or bldg_count == 0:
        return {
            "block_id": block_id,
            "on_network_street_length": 0.0,
            "off_network_street_length": 0.0,
            "nearest_external_street": float("nan"),
            "building_count": 0,
            "building_layers": "0",
            "k_complexity": 1,
        }

    streets = _k_prep_streets(streets_geoms)
    on_len = 0.0
    off_len = 0.0
    nearest = float("nan")
    is_conn = False

    # ---------------- Network access metrics ----------------
    from shapely.geometry import LineString, MultiLineString

    # ---------------- Network access metrics ----------------
    if streets is not None and not streets.is_empty:
        clipped = streets.intersection(block)
        if not clipped.is_empty:
            # extract only valid line segments with ≥ 2 coords
            line_parts = []

            def _collect_lines(geom):
                if isinstance(geom, LineString):
                    # guard against 1-point “lines”
                    if len(geom.coords) > 1:
                        line_parts.append(geom)
                elif isinstance(geom, MultiLineString):
                    for seg in geom.geoms:
                        if not seg.is_empty and len(seg.coords) > 1:
                            line_parts.append(seg)
                elif hasattr(geom, "geoms"):
                    for g in geom.geoms:
                        _collect_lines(g)

            _collect_lines(clipped)

            if not line_parts:
                # effectively “no usable streets in block”
                on_len = 0.0
                off_len = 0.0
                nearest = float("nan")
                is_conn = False
            else:
                merged = linemerge(MultiLineString(line_parts))
                internal_buf = merged.buffer(buffer_radius / 2.0)
                external_buf = block.exterior.buffer(buffer_radius)

                # streets that touch the external buffer (i.e., connected outwards)
                ext_touch = [s for s in streets.geoms if s.intersects(external_buf)]
                if ext_touch:
                    ext_net = unary_union(ext_touch)
                    full_net = unary_union([merged, ext_net])
                    on_geom = full_net.intersection(internal_buf)
                    off_geom = full_net.difference(internal_buf)

                    on_len = 0.0 if on_geom.is_empty else on_geom.length
                    off_len = 0.0 if off_geom.is_empty else off_geom.length

                    centroid_all = unary_union([p for p in bldg_points.geoms])
                    nearest = centroid_all.distance(ext_net)
                    is_conn = on_len > 0.0
                else:
                    on_len = 0.0
                    off_len = 0.0 if clipped.is_empty else clipped.length
                    nearest = float("nan")
                    is_conn = False
        else:
            # no intersection with block at all
            on_len = 0.0
            off_len = 0.0
            nearest = float("nan")
            is_conn = False
    else:
        # no streets near this block
        on_len = 0.0
        off_len = 0.0
        nearest = float("nan")
        is_conn = False


    # ---------------- Voronoi parcels inside the block ----------------
    vor = voronoi_diagram(bldg_points, envelope=block)
    parcels = []
    for cell in vor.geoms:
        inter = make_valid(cell.intersection(block))
        if inter.is_empty:
            continue
        if isinstance(inter, Polygon):
            parcels.append(inter)
        elif isinstance(inter, MultiPolygon):
            parcels.extend([p for p in inter.geoms if not p.is_empty])

    if not parcels:
        return {
            "block_id": block_id,
            "on_network_street_length": on_len,
            "off_network_street_length": off_len,
            "nearest_external_street": nearest,
            "building_count": bldg_count,
            "building_layers": "1",
            "k_complexity": 1,
        }

    remaining = parcels.copy()
    layers = []

    # ---- First ring: parcels touching access buffer or boundary ----
    if is_conn and streets is not None and not streets.is_empty:
        access_buf = merged.buffer(1.0)
        ring = [p for p in remaining if p.intersects(access_buf)]
        if not ring:
            # fallback: parcels touching block boundary
            ring = [p for p in remaining if p.touches(block.boundary)]
    else:
        ring = [p for p in remaining if p.touches(block.boundary)]
        if not ring:
            ring = remaining.copy()

    layers.append(len(ring))
    remaining = [p for p in remaining if p not in ring]

    # ---- Interior rings: peel by adjacency ----
    while remaining:
        union_prev = unary_union(ring)
        next_ring = [p for p in remaining if p.touches(union_prev)]
        if not next_ring:
            # last interior bulk
            layers.append(len(remaining))
            break
        layers.append(len(next_ring))
        remaining = [p for p in remaining if p not in next_ring]
        ring = next_ring

    k_val = len(layers)

    return {
        "block_id": block_id,
        "on_network_street_length": on_len,
        "off_network_street_length": off_len,
        "nearest_external_street": nearest,
        "building_count": bldg_count,
        "building_layers": ",".join(str(x) for x in layers),
        "k_complexity": k_val,
    }

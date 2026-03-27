import dask_geopandas as dgpd
import geopandas as gpd
from shapely.geometry import shape, Point, LineString, Polygon, MultiPolygon, MultiLineString, box
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
from cloudpathlib import S3Path


MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
URBAN_EXTENT_PATH = f"{INPUT_PATH}/urban_extent"
URBAN_EXTENT_200M_BUFFER_PATH = f"{INPUT_PATH}/urban_extent_200m_buffer"
CITY_INFO_PATH = f'{INPUT_PATH}/city_info'
EXTENTS_PATH = f'{CITY_INFO_PATH}/extents'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
BUILDINGS_DISTANCES_PATH = f'{INPUT_PATH}/buildings_with_distances'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
NATURAL_FEATURES_PATH = os.path.join(INPUT_PATH, "natural_features_and_railroads")
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
SEARCH_BUFFER_PATH = f'{INPUT_PATH}/city_info/search_buffers'
BLOCKS_PATH = f'{INPUT_PATH}/blocks'
LAND_POLYGONS_PATH = f"{INPUT_PATH}/land_polygons/land_polygons.geoparquet"
INLAND_WATER_PATH = f"{INPUT_PATH}/inland_water"
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


def load_land_polygon_for_city(ue200_geom, epsg, land_polygons_path):
    """
    Load the subset of global land polygons that intersects the city's
    UE200 buffer. Returns a single geometry in the city's CRS.
 
    The land polygons file covers ocean/sea only (derived from
    natural=coastline). It does NOT include inland water bodies.
 
    Uses bbox filtering for fast reads from the global file.
    """
    try:
        # Get bounding box in WGS84 for spatial filtering
        ue200_gdf = gpd.GeoDataFrame(
            geometry=[ue200_geom], crs=f"EPSG:{epsg}"
        ).to_crs(epsg=4326)
        bbox = ue200_gdf.total_bounds  # [minx, miny, maxx, maxy]
 
        # Small buffer to avoid edge effects
        buffer_deg = 0.01
       
        bbox_tuple = (
            bbox[0] - buffer_deg, bbox[1] - buffer_deg,
            bbox[2] + buffer_deg, bbox[3] + buffer_deg
        )

        land = gpd.read_parquet(
            land_polygons_path,
            bbox=bbox_tuple
        )
 
        if land.empty:
            return None
 
        land = land.to_crs(epsg=epsg)
        return unary_union(land.geometry).buffer(0)
 
    except Exception as e:
        print(f"  ⚠ Could not load land polygons: {e}")
        return None
 
 
def load_inland_water_for_city(city_name, epsg, inland_water_path=None):
    """
    Load per-city inland water polygons (lakes, lagoons, reservoirs).
    These were downloaded by gather_data_cities.py.
 
    Returns a single geometry in the city's CRS, or None.
    """
    if inland_water_path is None:
        inland_water_path = f"s3://wri-cities-sandbox/identifyingLandSubdivisions/data/input/inland_water"
 
    water_file = f"{inland_water_path}/{city_name}/{city_name}_OSM_inland_water.geoparquet"
 
    try:
        if not S3Path(water_file).exists():
            return None
 
        water = gpd.read_parquet(water_file)
        if water.empty:
            return None
 
        water = water.to_crs(epsg=epsg)
 
        # Keep only valid polygon geometries
        water = water[water.geometry.notnull()]
        water['geometry'] = water.geometry.make_valid()
        water = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
 
        if water.empty:
            return None
 
        return unary_union(water.geometry).buffer(0)
 
    except Exception as e:
        print(f"  ⚠ Could not load inland water for {city_name}: {e}")
        return None
 
 
def clip_fill_to_land_and_water(fill_geom, land_geom=None, water_geom=None, city_name=""):
    """
    Clip fill geometry using the two-step approach:
      1. Intersect with land polygons (removes ocean/sea)
      2. Difference with inland water (removes lakes/lagoons/reservoirs)
 
    Small waterways (streams, ditches) are NOT handled here — they
    act as block splitters via the linework in polygonization.
    """
    clipped = fill_geom
 
    # Step 1: Clip to land (ocean/sea removal)
    if land_geom is not None:
        try:
            clipped = clipped.intersection(land_geom)
            clipped = clipped.buffer(0)
            print(f"  🌊 Clipped to land for {city_name}")
        except Exception as e:
            print(f"  ⚠ Land clipping failed for {city_name}: {e}")
 
    # Step 2: Remove inland water bodies
    if water_geom is not None:
        try:
            clipped = clipped.difference(water_geom)
            clipped = clipped.buffer(0)
            print(f"  💧 Removed inland water for {city_name}")
        except Exception as e:
            print(f"  ⚠ Inland water removal failed for {city_name}: {e}")
 
    return clipped

@delayed
def produce_blocks(city_name, YOUR_NAME):
    # Construct file paths for the city
    paths = {
        'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
        'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet',
        'intersections': f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet',
        'natural_features': f'{NATURAL_FEATURES_PATH}/{city_name}/{city_name}_OSM_natural_features_and_railroads.geoparquet',
        'city_dir' : f"{INPUT_PATH}/city_info",
        'ue_path' : f"{INPUT_PATH}/urban_extent/{city_name}/{city_name}_urban_extent.geoparquet",
        'ue200_path' : f"{INPUT_PATH}/urban_extent_200m_buffer/{city_name}/{city_name}_urban_extent_200m_buffer.geoparquet"
    }

    MIN_BLOCK_AREA_M2 = 700

    epsg = get_epsg(city_name).compute()

    roads = load_dataset(paths['roads'], epsg=epsg).compute()
    natural_features = load_dataset(paths['natural_features'], epsg=epsg).compute()

    if not S3Path(paths['ue_path']).exists():
        raise FileNotFoundError(f"Missing urban extent for {city_name}")

    if not S3Path(paths['ue200_path']).exists():
        raise FileNotFoundError(f"Missing 200m buffer for {city_name}")

    urban_extent_200m = gpd.read_parquet(paths['ue200_path']).to_crs(epsg)

    ue200_geom = urban_extent_200m.geometry.iloc[0]

    # -----------------------------
    # A) BASELINE BLOCKS 
    # -----------------------------
    linework_all = pd.concat([roads, natural_features], ignore_index=True)

    base_blocks = get_blocks(linework_all)

    base_blocks = add_inscribed_circle_info(base_blocks)
    base_blocks = base_blocks.set_crs(epsg, allow_override=True)

    base_blocks["is_fill_block"] = False
    base_blocks["fill_method"] = None

    # ── Water boundary clipping ───────────────────────────────────
    # Three-tier water handling:
    #   - Ocean/sea: clipped via pre-built land polygons (global file)
    #   - Inland water (lakes, lagoons): subtracted via per-city OSM polygons
    #   - Small waterways (streams, ditches): already in linework as block splitters
    land_geom = load_land_polygon_for_city(ue200_geom, epsg, LAND_POLYGONS_PATH)
    water_geom = load_inland_water_for_city(city_name, epsg, INLAND_WATER_PATH)

    # 2b) Cut BASE blocks with water bodies (coastline + inland water)
    if land_geom is not None or water_geom is not None:
        clipped_geoms = []
        for geom in base_blocks.geometry:
            g = geom
            try:
                if land_geom is not None:
                    g = g.intersection(land_geom).buffer(0)
                if water_geom is not None:
                    g = g.difference(water_geom).buffer(0)
            except Exception:
                pass
            clipped_geoms.append(g)
        base_blocks = base_blocks.copy()
        base_blocks['geometry'] = clipped_geoms
        # Explode multipolygons and drop empties
        base_blocks = base_blocks[~base_blocks.geometry.is_empty].copy()
        base_blocks = base_blocks.explode(index_parts=False).reset_index(drop=True)
        base_blocks = base_blocks[base_blocks.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        # Recompute inscribed circles after clipping changed geometry
        base_blocks = add_inscribed_circle_info(base_blocks)
        print(f"  🌊💧 Clipped {len(base_blocks)} base blocks with water for {city_name}")

    # 3) Compute fill geometry = UE200 minus union(base_blocks)
    base_union = unary_union(base_blocks.geometry).buffer(0.)
    try:
        fill_geom = ue200_geom.difference(base_union)
    except Exception:
        fill_geom = ue200_geom

    # Clean
    try:
        fill_geom = fill_geom.buffer(0)
    except Exception:
        pass

    # 3b) Clip fill geometry to land and remove inland water
    fill_geom = clip_fill_to_land_and_water(fill_geom, land_geom, water_geom, city_name)

    # 4) Generate fill blocks from existing linework + fill boundary; fallback to fill polygons
    fill_blocks = make_fill_blocks_from_linework(
        linework_gdf=linework_all,
        fill_geom=fill_geom,
        crs=epsg,
        min_area_m2=1.0
    )

    # Add inscribed circle info for fill blocks too (consistent schema)
    if fill_blocks is not None and len(fill_blocks) > 0:
        try:
            fill_blocks = add_inscribed_circle_info(fill_blocks)
        except Exception:
            pass
        fill_blocks = fill_blocks.set_crs(epsg, allow_override=True)
    else:
        fill_blocks = gpd.GeoDataFrame(columns=base_blocks.columns, geometry="geometry", crs=epsg)

    # 5) Combine
    blocks = pd.concat([base_blocks, fill_blocks], ignore_index=True)
    blocks = gpd.GeoDataFrame(blocks, geometry="geometry", crs=epsg)

    # 6) Compute block area safely in m² and discard tiny blocks
    if blocks.crs is None:
        raise ValueError(f"Blocks for {city_name} have no CRS; cannot compute area safely.")

    if blocks.crs.is_geographic:
        blocks["block_area"] = blocks.to_crs(6933).geometry.area
    else:
        blocks["block_area"] = blocks.geometry.area

    blocks = blocks[blocks["block_area"] >= MIN_BLOCK_AREA_M2].copy()

    # -----------------------------
    # SAVE
    # -----------------------------
    path_blocks = f'{BLOCKS_PATH}/{city_name}/{city_name}_blocks_{YOUR_NAME}.geoparquet'
    blocks.to_parquet(path_blocks, index=False)
    return path_blocks



@delayed
def produce_azimuths(city_name, YOUR_NAME):
    calculate_azimuths(city_name, YOUR_NAME)
import subprocess
import geopandas as gpd
import osmnx as ox
from dask import delayed, compute
import pandas as pd
import dask.dataframe as dd
import os
import pyproj
from shapely.ops import transform
from cloudpathlib import S3Path
import s3fs
import fsspec
import traceback
import neatnet
from osmnx._errors import InsufficientResponseError


ox.settings.timeout = 300              # requests timeout for Overpass calls
ox.settings.overpass_rate_limit = True # let osmnx pause when it detects limits


fs = s3fs.S3FileSystem(anon=False)

# Paths Configuration
MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data" #
INPUT_PATH = os.path.join(MAIN_PATH, "input")
BUILDINGS_PATH = os.path.join(INPUT_PATH, "buildings")
ROADS_PATH = os.path.join(INPUT_PATH, "roads")
INTERSECTIONS_PATH = os.path.join(INPUT_PATH, "intersections")
NATURAL_FEATURES_PATH = os.path.join(INPUT_PATH, "natural_features_and_railroads")
URBAN_EXTENTS_PATH = os.path.join(INPUT_PATH, "urban_extents")
OUTPUT_PATH = os.path.join(MAIN_PATH, "output")
OUTPUT_PATH_CSV = os.path.join(OUTPUT_PATH, "csv")
SEARCH_BUFFER_PATH = os.path.join(INPUT_PATH, "city_info", "search_buffers")

# Utility Functions

def remove_list_columns(gdf):
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))
        elif gdf[col].dtype == 'object':  # Convert all object columns to string
            gdf[col] = gdf[col].astype(str)
    return gdf

def get_utm_proj(lon, lat):
    utm_zone = int((lon + 180) // 6) + 1
    is_northern = lat >= 0
    return f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

def s3_save(file, output_file, output_temp_path, remote_path):

    os.makedirs(output_temp_path, exist_ok=True)

    local_temporary_file = f"{output_temp_path}/{output_file}"

    # Ensure osmid is a string before saving
    if "osmid" in file.columns:
        file["osmid"] = file["osmid"].astype(str)

    # Save the file based on its extension
    if output_file.endswith(".gpkg"):
        file.to_file(local_temporary_file, driver="GPKG")
    elif output_file.endswith(".csv"):
        file.to_csv(local_temporary_file, index=False)
    elif output_file.endswith(".geoparquet"):
        file.to_parquet(local_temporary_file, index=False)
    else:
        raise ValueError("Unsupported file format. Only .gpkg, .geoparquet and .csv are supported.")

    # Upload to S3
    output_path = S3Path(remote_path)
    output_path.upload_from(local_temporary_file)

    # Delete the local file after upload
    if os.path.exists(local_temporary_file):
        os.remove(local_temporary_file)

def filter_osm_network(
    osm_roads: gpd.GeoDataFrame,
    osm_intersections: gpd.GeoDataFrame,
    included_road_types: list[str]
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    1) Keep only roads whose 'highway' tag intersects included_road_types.
    2) Count each node’s actual degree in the filtered roads.
    3) Drop any intersection whose osmid is missing from the filtered roads OR
       whose original street_count != actual degree.
    4) Return (roads, intersections) ready to write out.
    """
    # 1) roads filter
    def _hw_filter(val):
        if pd.isna(val): return False
        parts = [p.strip() for p in val.split(',')]
        return any(p in included_road_types for p in parts)
    roads = osm_roads[osm_roads['highway'].apply(_hw_filter)].copy()

    # 2) compute actual degree of each node
    deg_u = roads['u'].value_counts()
    deg_v = roads['v'].value_counts()
    degree = (deg_u.add(deg_v, fill_value=0)
                    .astype(int)
                    .rename("actual_degree"))

    # 3) clean intersections
    inter = osm_intersections.reset_index(drop=True).copy()
    # ensure osmid is int for matching
    inter['osmid'] = inter['osmid'].astype(int)

    # keep only those present in our filtered roads
    inter = inter[inter['osmid'].isin(degree.index)]

    # compare street_count vs actual_degree
    inter = inter.merge(degree, left_on='osmid', right_index=True, how='left')
    mask_good = inter['street_count'] == inter['actual_degree']
    # drop bad ones
    inter = inter[mask_good].drop(columns=['actual_degree'])

    return roads, inter



class NoRoadsError(RuntimeError):
    """Raised when the OSM road graph is empty for a city polygon."""
    pass


def osm_command(city_name, search_area, utm_proj_city):
    """
    Run OSMnx queries for a city.
    Roads/intersections are REQUIRED.
    Natural features are OPTIONAL and must never fail the city.
    """

    if len(search_area) == 0:
        raise ValueError(f"Search area for {city_name} is empty.")

    polygon = search_area.geometry.iloc[0]

    # ---------------------------
    # 1) ROADS + INTERSECTIONS (REQUIRED)
    # ---------------------------
    G = ox.graph_from_polygon(
        polygon=polygon,
        custom_filter=(
            '["highway"~"motorway|trunk|primary|secondary|'
            'tertiary|none|unclassified|residential|living_street|'
            'primary_link|secondary_link|tertiary_link|'
            'trunk_link|motorway_link|road"]'
        ),
        retain_all=True
    )

    osm_intersections, osm_roads = ox.graph_to_gdfs(G)

    # REQUIRED: fail early if no roads came back
    if osm_roads is None or len(osm_roads) == 0:
        raise ValueError(f"[{city_name}] NO_ROADS: OSMnx returned 0 road edges for this polygon.")
    if osm_intersections is None or len(osm_intersections) == 0:
        raise ValueError(f"[{city_name}] NO_INTERSECTIONS: OSMnx returned 0 nodes for this polygon.")

    # Clean + neatify roads
    osm_roads_to_neatify = osm_roads.to_crs(utm_proj_city).reset_index(drop=False)
    osm_roads_geom = osm_roads_to_neatify[["osmid", "u", "v", "geometry"]]
    osm_roads = neatnet.neatify(osm_roads_geom).to_crs("EPSG:4326")

    # REQUIRED: fail if neatify wiped everything
    if osm_roads is None or len(osm_roads) == 0:
        raise ValueError(f"[{city_name}] NO_ROADS_AFTER_NEATIFY: roads became empty after neatify().")

    osm_intersections = osm_intersections.reset_index()

    osm_roads = remove_list_columns(osm_roads)
    osm_intersections = remove_list_columns(osm_intersections)

    if "osmid" not in osm_intersections:
        osm_intersections["osmid"] = None

    # ---------------------------
    # 2) NATURAL FEATURES (OPTIONAL — NEVER FAIL)
    # ---------------------------
    custom_filter = ['["waterway"~"stream|ditch|river|canal|dam|weir|rapids|waterfall"]']

    try:
        G_nf = ox.graph_from_polygon(
            polygon=polygon,
            custom_filter=custom_filter,
            retain_all=True
        )
        _, osm_natural_features = ox.graph_to_gdfs(G_nf)
        osm_natural_features = remove_list_columns(osm_natural_features)
        print(f"🌿 Natural features found for {city_name}")

    except (InsufficientResponseError, ValueError):
        # ✅ This covers the exact error in your logs:
        # ValueError: Found no graph nodes within the requested polygon. 
        osm_natural_features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        print(f"🌿 No natural features for {city_name} (expected)")

    except Exception as e:
        # optional: also never fail city on any other nf weirdness
        osm_natural_features = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        print(f"🌿 Natural features failed for {city_name} but continuing: {repr(e)}")

    # ---------------------------
    # 3) SAVE OUTPUTS (ALWAYS)
    # ---------------------------
    s3_save(
        file=osm_roads,
        output_file=f"{city_name}_OSM_roads.geoparquet",
        output_temp_path=".",
        remote_path=f"{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet"
    )

    s3_save(
        file=osm_intersections,
        output_file=f"{city_name}_OSM_intersections.geoparquet",
        output_temp_path=".",
        remote_path=f"{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet"
    )

    s3_save(
        file=osm_natural_features,
        output_file=f"{city_name}_OSM_natural_features_and_railroads.geoparquet",
        output_temp_path=".",
        remote_path=f"{NATURAL_FEATURES_PATH}/{city_name}/{city_name}_OSM_natural_features_and_railroads.geoparquet"
    )

    print(f"✅ OSM completed for {city_name}")


def overturemaps_download_and_save(bbox_str, request_type: str, output_dir, city_name: str):
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_PATH_OVERTURE = os.path.join(output_dir, f'Overture_{request_type}_{city_name}.geoparquet')
    command = [
        "overturemaps", "download",
        "-f", "geoparquet",
        "--bbox=" + bbox_str,
        "--type=" + request_type,
        "--output=" + OUTPUT_PATH_OVERTURE
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return {"city": city_name, "type": request_type, "status": "success"}
    else:
        error_message = result.stderr.split("\n")[0]  # Extract only the first line of the error
        print(f"Error for {city_name}: {result.stderr}")  # Print full error to terminal
        return {"city": city_name, "type": request_type, "status": f"error: {error_message}"}

def make_requests(partition):
    if partition.empty:
        return pd.DataFrame() 

    results = []
    for city_name in partition.city:
        try:
            search_area_path = f"{SEARCH_BUFFER_PATH}/{city_name}/{city_name}_search_buffer.geoparquet"

            if not fs.exists(search_area_path):
                raise FileNotFoundError(f"❌ ERROR: File does not exist in S3: {search_area_path}")
            else:
                print(f"✅ File exists in S3: {search_area_path}")

            with fs.open(search_area_path, mode="rb", anon=False) as f:
                search_area = gpd.read_parquet(f)

            print(f"✅ Successfully read file for {city_name}: {search_area.shape[0]} rows")

            rep_point = search_area.geometry.representative_point().iloc[0]
            utm_proj_city = get_utm_proj(float(rep_point.x), float(rep_point.y))
            transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True)
            print(f"Entering OSM command for {city_name}")
            osm_command(city_name, search_area, utm_proj_city)

            search_area_bounds = search_area.bounds
            bbox_str = ','.join(search_area_bounds[['minx', 'miny', 'maxx', 'maxy']].values[0].astype(str))
            print(f"Entering overturemaps CLI command for {city_name}")
            results.append(overturemaps_download_and_save(bbox_str, "building", os.path.join(BUILDINGS_PATH, city_name), city_name))
        except Exception as e:
            error_message = "".join(traceback.format_exception(None, e, e.__traceback__))  # Get full traceback

            print(f"Error for {city_name}: {e}")  # Print full error to terminal
            results.append({"city": city_name, "type": "general", "status": f"error: {error_message}"})
    return pd.DataFrame(results)


def gather_data_city(city_name: str) -> dict:
    """
    Run the full data-gathering pipeline for a single city.

    Parameters
    ----------
    city_name : str
        City name like "Accra" or "Belo Horizonte". Spaces will be
        replaced by underscores to match the S3 folder convention.

    Returns
    -------
    dict
        A small log dict with keys: city, type, status.
        - On success, this is the dict returned by overturemaps_download_and_save.
        - On error, type == "general" and status contains the traceback text.
    """
    city_clean = city_name.replace(" ", "_")

    try:
        # --- 1. Find and load search buffer from S3 ---
        search_area_path = f"{SEARCH_BUFFER_PATH}/{city_clean}/{city_clean}_search_buffer.geoparquet"
        # print(f"🔍 Checking S3 Path: {search_area_path}")

        if not fs.exists(search_area_path):
            raise FileNotFoundError(f"❌ ERROR: File does not exist in S3: {search_area_path}")
        else:
            print(f"✅ File exists in S3: {search_area_path}")

        with fs.open(search_area_path, mode="rb", anon=False) as f:
            search_area = gpd.read_parquet(f)

        print(f"✅ Successfully read file for {city_clean}: {search_area.shape[0]} rows")

        if len(search_area) == 0:
            raise ValueError(f"Search area for {city_clean} is empty.")

        # --- 2. Compute UTM projection + run OSM command ---
        rep_point = search_area.geometry.representative_point().iloc[0]
        utm_proj_city = get_utm_proj(float(rep_point.x), float(rep_point.y))

        # (transformer currently not used, but keep for compatibility / future use)
        _ = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"),
            utm_proj_city,
            always_xy=True
        )

        # This does the osmnx + neatnet work, writes outputs, etc.
        osm_command(city_clean, search_area, utm_proj_city)

        # --- 3. Call Overture Maps CLI for buildings ---
        search_area_bounds = search_area.bounds
        bbox_str = ",".join(
            search_area_bounds[["minx", "miny", "maxx", "maxy"]]
            .values[0]
            .astype(str)
        )

        result = overturemaps_download_and_save(
            bbox_str,
            "building",
            os.path.join(BUILDINGS_PATH, city_clean),
            city_clean,
        )

        # result is already a {city, type, status} dict
        return result

    except Exception as e:
        # Full traceback for logs, short message for the dict
        error_message = "".join(traceback.format_exception(None, e, e.__traceback__))
        print(f"Error for {city_clean}: {e}")
        return {
            "city": city_clean,
            "type": "general",
            "status": f"error: {error_message}",
        }



def run_all(cities):
    print("Entering HIIIIIIIIIIZZZZZZZZZ BLAAAAAA run_all")
    cities_set = pd.DataFrame({'city': [city.replace(' ', '_') for city in cities]})
    cities_set_ddf = dd.from_pandas(cities_set, npartitions=12)
    meta = pd.DataFrame({"city": pd.Series(dtype="str"), 
                         "type": pd.Series(dtype="str"), 
                         "status": pd.Series(dtype="str")})
    results = cities_set_ddf.map_partitions(make_requests, meta=meta)
    results_df = results.compute()
    '''
    # Save logs
    data_gathering_logs_file = f"data_gather_logs.csv"
    logs_output_path_remote = f"{OUTPUT_PATH_CSV}"
    output_temp_path = "."
    s3_save(file = results_df, 
        output_file = data_gathering_logs_file, 
        output_temp_path = output_temp_path, 
        remote_path = logs_output_path_remote)
    '''
def main():
    cities = ["Belo_Horizonte"] #, "Campinas", "Bogota", "Nairobi", "Bamako", "Lagos", "Accra", "Abidjan", "Cape Town", "Maputo", "Mogadishu", "Luanda"
    run_all(cities)

if __name__ == "__main__":
    main()
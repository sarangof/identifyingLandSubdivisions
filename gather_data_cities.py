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

fs = s3fs.S3FileSystem(anon=False)

# Paths Configuration
MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data" #
INPUT_PATH = os.path.join(MAIN_PATH, "input")
BUILDINGS_PATH = os.path.join(INPUT_PATH, "buildings")
ROADS_PATH = os.path.join(INPUT_PATH, "roads")
INTERSECTIONS_PATH = os.path.join(INPUT_PATH, "intersections")
URBAN_EXTENTS_PATH = os.path.join(INPUT_PATH, "urban_extents")
OUTPUT_PATH = os.path.join(MAIN_PATH, "output")
OUTPUT_PATH_CSV = os.path.join(OUTPUT_PATH, "csv")
SEARCH_BUFFER_PATH = os.path.join(INPUT_PATH, "city_info", "search_buffers")

# Ensure paths exist
os.makedirs(OUTPUT_PATH_CSV, exist_ok=True)

# Utility Functions
def remove_duplicate_roads(osm_roads):
    osm_roads_reset = osm_roads.reset_index()
    osm_roads_reset['sorted_pair'] = osm_roads_reset.apply(lambda row: tuple(sorted([row['u'], row['v']])), axis=1)
    osm_roads = osm_roads_reset.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
    return osm_roads

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


def osm_command(city_name, search_area):
    # … build G & call graph_to_gdfs …
    if len(search_area) > 0:
        polygon = search_area.geometry.iloc[0]
        G = ox.graph_from_polygon(
            polygon=polygon,
            custom_filter=(
                '["highway"~"motorway|trunk|primary|secondary|'
                'tertiary|unclassified|residential|living_street|'
                'service|track|path|footway|cycleway|bridleway|'
                'steps|pedestrian|corridor|road"]'
            ),
            retain_all=True
        )
    else:
        raise ValueError(f"Search area for {city_name} is empty.")

    osm_intersections, osm_roads = ox.graph_to_gdfs(G)

    osm_roads         = remove_duplicate_roads(osm_roads)
    osm_intersections = osm_intersections.reset_index()
    osm_roads         = remove_list_columns(osm_roads)
    osm_intersections = remove_list_columns(osm_intersections)

    # ensure 'osmid' always exists
    if 'osmid' not in osm_intersections:
        osm_intersections['osmid'] = None

    included = [
      'trunk','motorway','primary','secondary','tertiary',
      'primary_link','secondary_link','tertiary_link',
      'trunk_link','motorway_link','residential',
      'unclassified','road','living_street'
    ]
    roads_filt, inters_filt = filter_osm_network(
        osm_roads, osm_intersections, included
    )

    # now save them
    s3_save(
      file=roads_filt,
      output_file=f"{city_name}_OSM_roads.geoparquet",
      output_temp_path=".",
      remote_path=f"{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet"
    )
    s3_save(
      file=inters_filt,
      output_file=f"{city_name}_OSM_intersections.geoparquet",
      output_temp_path=".",
      remote_path=f"{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet"
    )



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
            #print(f"🔍 Checking S3 Path: {search_area_path}")

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
            osm_command(city_name, search_area)

            search_area_bounds = search_area.bounds
            bbox_str = ','.join(search_area_bounds[['minx', 'miny', 'maxx', 'maxy']].values[0].astype(str))
            results.append(overturemaps_download_and_save(bbox_str, "building", os.path.join(BUILDINGS_PATH, city_name), city_name))
        except Exception as e:
            error_message = "".join(traceback.format_exception(None, e, e.__traceback__))  # Get full traceback

            print(f"Error for {city_name}: {e}")  # Print full error to terminal
            results.append({"city": city_name, "type": "general", "status": f"error: {error_message}"})
    return pd.DataFrame(results)

def run_all(cities):
    print("Entering run_all")
    cities_set = pd.DataFrame({'city': [city.replace(' ', '_') for city in cities]})
    cities_set_ddf = dd.from_pandas(cities_set, npartitions=12)
    meta = pd.DataFrame({"city": pd.Series(dtype="str"), 
                         "type": pd.Series(dtype="str"), 
                         "status": pd.Series(dtype="str")})
    results = cities_set_ddf.map_partitions(make_requests, meta=meta)
    results_df = results.compute()

    # Save logs
    data_gathering_logs_file = f"data_gather_logs.csv"
    data_gathering_logs_tmp_path = "." #if this works, create the right path
    logs_output_path_remote = f"{OUTPUT_PATH_CSV}/{data_gathering_logs_file}"
    s3_save(file = results_df, 
        output_file = data_gathering_logs_file, 
        output_temp_path = data_gathering_logs_tmp_path, 
        remote_path = logs_output_path_remote)

def main():
    cities = ["Belo_Horizonte"] #, "Campinas", "Bogota", "Nairobi", "Bamako", "Lagos", "Accra", "Abidjan", "Cape Town", "Maputo", "Mogadishu", "Luanda"
    run_all(cities)

if __name__ == "__main__":
    main()
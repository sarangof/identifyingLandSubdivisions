import subprocess
import geopandas as gpd
import osmnx as ox
from dask import delayed, compute
import pandas as pd
import dask.dataframe as dd
import os
import pyproj
from shapely.ops import transform

# Paths Configuration
main_path = "/Users/sarangof/Documents/Identifying Land Subdivisions/data"
input_path = os.path.join(main_path, "input")
buildings_path = os.path.join(input_path, "buildings")
roads_path = os.path.join(input_path, "roads")
intersections_path = os.path.join(input_path, "intersections")
urban_extents_path = os.path.join(input_path, "urban_extents")
output_path = os.path.join(main_path, "output")
output_path_csv = os.path.join(output_path, "csv")
search_buffers_path = os.path.join(input_path, "city_info", "search_buffers")

# Ensure paths exist
os.makedirs(output_path_csv, exist_ok=True)

# Utility Functions
def remove_duplicate_roads(osm_roads):
    osm_roads_reset = osm_roads.reset_index()
    osm_roads_reset['sorted_pair'] = osm_roads_reset.apply(lambda row: tuple(sorted([row['u'], row['v']])), axis=1)
    osm_roads = osm_roads_reset.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])
    return osm_roads

def remove_list_columns(gdf):
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    return gdf

def get_utm_proj(lon, lat):
    utm_zone = int((lon + 180) // 6) + 1
    is_northern = lat >= 0
    return f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

def osm_command(city_name, search_area):
    if len(search_area) > 0:
        polygon = search_area.geometry.iloc[0]
        G = ox.graph_from_polygon(polygon=polygon, custom_filter='["highway"~"trunk|motorway|primary|secondary|tertiary"]')
    else:
        raise ValueError(f"Search area for {city_name} is empty.")
    
    osm_intersections, osm_roads = ox.graph_to_gdfs(G)

    # Process and save roads
    osm_roads = remove_duplicate_roads(osm_roads)
    output_dir_roads = os.path.join(roads_path, city_name)
    os.makedirs(output_dir_roads, exist_ok=True)
    road_output_file = os.path.join(output_dir_roads, f"{city_name}_OSM_roads.gpkg")
    osm_roads = remove_list_columns(osm_roads)
    osm_roads.to_file(road_output_file, driver="GPKG")

    # Save intersections
    output_dir_intersections = os.path.join(intersections_path, city_name)
    os.makedirs(output_dir_intersections, exist_ok=True)
    osm_intersections.to_file(os.path.join(output_dir_intersections, f"{city_name}_OSM_intersections.gpkg"), driver="GPKG")

def overturemaps_download_and_save(bbox_str, request_type: str, output_dir, city_name: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'Overture_{request_type}_{city_name}.geoparquet')
    command = [
        "overturemaps", "download",
        "-f", "geoparquet",
        "--bbox=" + bbox_str,
        "--type=" + request_type,
        "--output=" + output_path
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
            search_area = gpd.read_parquet(os.path.join(search_buffers_path, city_name, f"{city_name}_search_buffer.geoparquet"))
            utm_proj_city = get_utm_proj(float(search_area.geometry.centroid.x.iloc[0]), float(search_area.geometry.centroid.y.iloc[0]))
            transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True)

            osm_command(city_name, search_area)

            search_area_bounds = search_area.bounds
            bbox_str = ','.join(search_area_bounds[['minx', 'miny', 'maxx', 'maxy']].values[0].astype(str))
            results.append(overturemaps_download_and_save(bbox_str, "building", os.path.join(buildings_path, city_name), city_name))
        except Exception as e:
            error_message = str(e).split("\n")[0]  # Extract only the first line of the error
            print(f"Error for {city_name}: {e}")  # Print full error to terminal
            results.append({"city": city_name, "type": "general", "status": f"error: {error_message}"})
    return pd.DataFrame(results)

def run_all(cities):
    cities_set = pd.DataFrame({'city': [city.replace(' ', '_') for city in cities]})
    cities_set_ddf = dd.from_pandas(cities_set, npartitions=12)
    meta = pd.DataFrame({"city": pd.Series(dtype="str"), 
                         "type": pd.Series(dtype="str"), 
                         "status": pd.Series(dtype="str")})
    results = cities_set_ddf.map_partitions(make_requests, meta=meta)
    results_df = results.compute()

    # Save logs to a CSV file
    results_df.to_csv(os.path.join(output_path_csv, "data_gather_logs.csv"), index=False)

def main():
    cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", "Lagos", "Accra", "Abidjan", "Cape Town", "Maputo", "Mogadishu", "Luanda"]
    run_all(cities)

if __name__ == "__main__":
    main()

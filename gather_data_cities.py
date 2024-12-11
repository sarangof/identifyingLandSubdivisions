import subprocess
from io import StringIO
import geopandas as gpd
import osmnx as ox
from dask import delayed, compute
import pandas as pd
import dask.dataframe as dd
import os
import pyproj
from shapely.ops import transform
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This script collects and saves data on buildings, roads and intersections information 
for all the cities provided in analysis_buffers, search_buffers and city_boundaries
"""

# Define paths
#main_path = '../data'
main_path = "/Users/sarangof/Documents/Identifying Land Subdivisions"
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
urban_extents_path = f'{input_path}/urban_extents'
output_path = f'{main_path}/output'

city_info_path = f'{input_path}/city_info'
extents_path = f'{city_info_path}/extents'
analysis_buffers_path = f'{city_info_path}/analysis_buffers'
search_buffers_path = f'{city_info_path}/search_buffers'
grids_path = f'{city_info_path}/grids'
output_path = f'{main_path}/output'


# Useful auxiliary functions
def remove_duplicate_roads(osm_roads):
    osm_roads_reset = osm_roads.reset_index()
    osm_roads_reset['sorted_pair'] = osm_roads_reset.apply(lambda row: tuple(sorted([row['u'], row['v']])), axis=1)
    osm_roads = osm_roads_reset.drop_duplicates(subset=['sorted_pair'])
    osm_roads = osm_roads.drop(columns=['sorted_pair'])
    return osm_roads

def remove_list_columns(gdf):
    """
    This function removes any columns that contain lists or converts them to a valid format.
    """
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    return gdf

def get_utm_zone(lon):
    return int((lon + 180) // 6) + 1

def get_utm_proj(lon, lat):
    utm_zone = get_utm_zone(lon)
    is_northern = lat >= 0  # Determine if the zone is in the northern hemisphere
    return f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

def osm_command(city_name, search_area):
    custom_filter = '["highway"~"trunk|motorway|primary|secondary|tertiary|primary_link|secondary_link|tertiary_link|trunk_link|motorway_link|residential|unclassified|road"]'
    #G = ox.graph_from_polygon(polygon=transformed_road_search_area, custom_filter=custom_filter)
    if len(search_area) > 0:
        polygon = search_area.geometry.iloc[0]  # Extract the first geometry
        G = ox.graph_from_polygon(polygon=polygon, custom_filter=custom_filter)
    else:
        print("The search area is empty.")
    
    osm_intersections, osm_roads = ox.graph_to_gdfs(G)

    # Process and save roads
    osm_roads = remove_duplicate_roads(osm_roads)
    output_dir_roads = f"{roads_path}/{city_name}"
    os.makedirs(output_dir_roads, exist_ok=True)
    road_output_file = f"{output_dir_roads}/{city_name}_OSM_roads.gpkg"
    osm_roads = osm_roads.drop(columns=['osmid','reversed'])
    osm_roads = remove_list_columns(osm_roads)
    osm_roads.to_file(road_output_file, driver="GPKG")  

    # Save intersections file
    output_dir_intersections = f"{intersections_path}/{city_name}"
    os.makedirs(output_dir_intersections, exist_ok=True)
    osm_intersections.to_file(f"{output_dir_intersections}/{city_name}_OSM_intersections.gpkg", driver="GPKG")


#@delayed
def overturemaps_download_and_save(bbox_str, request_type: str, output_dir, city_name: str):
    print(f"Running Overture command with bbox_str={bbox_str} and request_type={request_type}")
    output_path = os.path.join(output_dir, f'Overture_{request_type}_{city_name}.parquet')
    command = [
        "overturemaps", "download",
        "-f", "geoparquet",
        "--bbox=" + bbox_str,
        "--type=" + request_type,
        "--output=" + output_path  # Specify the output file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            # Load the file into a GeoDataFrame if further processing is needed
            overture_file = gpd.read_parquet(output_path)
            print(f"Saved Overture data to {output_path}")
            return overture_file, output_path
        except Exception as e:
            print(f"Error reading Overture data: {e}")
            return None, None
    else:
        print(f"Error occurred while running Overture command: {result.stderr}")
        return None, None

def make_requests(partition):
    print(f"Processing partition with {len(partition)} rows")
    results = []
    # OJO: THIS NEEDS TO BE CHANGED
    for city_name in partition.city:
        print(city_name)

        # NEED TO GET THE CITY URBAN EXTENT
        #urban_extent = gpd.read_parquet(f'{extents_path}/{city_name}/{city_name}_urban_extent.parquet'
        search_area = gpd.read_parquet(f'{search_buffers_path}/{city_name}/{city_name}_search_buffer.parquet') 

        # Get city geometries in localized UTMs
        utm_proj_city = get_utm_proj(float(search_area.geometry.centroid.x), float(search_area.geometry.centroid.y))
        transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True)
        def transform_geom(geom):
            return transform(transformer.transform, geom)
        
        #transformed_sarch_area_geom = transform_geom(search_area.geometry[0]) 
        #road_search_area = gpd.GeoSeries(transformed_sarch_area_geom,crs=utm_proj_city).geometry[0].buffer(2000)        #transformed_urban_border = transform(pyproj.Transformer.from_crs(utm_proj_city, pyproj.CRS('EPSG:4326'), always_xy=True).transform, urban_border)
        #transformed_road_search_area = transform(pyproj.Transformer.from_crs(utm_proj_city, pyproj.CRS('EPSG:4326'), always_xy=True).transform, road_search_area)
        #transformed_road_search_area = transform(pyproj.Transformer.from_crs(utm_proj_city, pyproj.CRS('EPSG:4326'), always_xy=True).transform, transformed_sarch_area_geom)
        osm_command(city_name,search_area)#transformed_road_search_area)

        try:
            # Overture maps commands
            search_area_bounds = search_area.bounds
            bbox_str = ','.join(search_area_bounds[['minx', 'miny', 'maxx', 'maxy']].values[0].astype(str))
            request_type = 'building'
            print("About to trigger overturemaps command")
            output_dir_buildings = f"{buildings_path}/{city_name}"
            overturemaps_download_and_save(bbox_str, request_type, output_dir_buildings, city_name)
        except Exception as e:
            print(f"Overture error: {e}")
    return results

# Paralellize data gathering for all cities in the list.
def run_all(cities):
    cities = [city.replace(' ', '_') for city in cities]
    logger.info(f'Cities to be processed: {cities}')
    
    cities_set = pd.DataFrame({'city':cities})

    # Create Dask DataFrame
    cities_set_ddf = dd.from_pandas(cities_set, npartitions=12)
    
    # Print the Dask DataFrame columns
    print(f"Columns in the Dask DataFrame: {cities_set_ddf.columns.tolist()}")

    # Apply function to each partition
    results = cities_set_ddf.map_partitions(make_requests, meta=cities_set)

    # Trigger computation
    computed_results = results.compute()
    print(f"Results: {computed_results}")


def main():
    #cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
    #          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
    #          "Maputo", "Luanda"]
    cities = ["Belo Horizonte"]
    run_all(cities)

if __name__ == "__main__":
    main()

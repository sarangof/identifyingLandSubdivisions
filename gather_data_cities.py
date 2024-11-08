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

"""
This script collects and saves data on buildings, roads and intersections information 
for all the cities provided in analysis_buffers, search_buffers and city_boundaries
"""

# Define paths
main_path = '../data'
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
urban_extents_path = f'{input_path}/urban_extents'
output_path = f'{main_path}/output'

# List of cities for which to gather data
cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
          "Maputo", "Luanda"]

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

def osm_command(city_name,transformed_road_search_area):
    custom_filter = '["highway"~"trunk|motorway|primary|secondary|tertiary|primary_link|secondary_link|tertiary_link|trunk_link|motorway_link|residential|unclassified|road"]'
    G = ox.graph_from_polygon(polygon=transformed_road_search_area, custom_filter=custom_filter)
    osm_intersections, osm_roads = ox.graph_to_gdfs(G)
        
    # Ensure the directory exists
    output_dir = f"{roads_path}/{city_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Process and save roads
    osm_roads = remove_duplicate_roads(osm_roads)
    road_output_file = f"{output_dir}/{city_name}_OSM_roads.gpkg"
    osm_roads = osm_roads.drop(columns=['osmid','reversed'])
    osm_roads = remove_list_columns(osm_roads)
    osm_roads.to_file(road_output_file, driver="GPKG")  

    # Save intersections file
    osm_intersections.to_file(f"{output_dir}/{city_name}_OSM_intersections.gpkg", driver="GPKG")

def overture_maps_command(city_name,transformed_road_search_area):
    request_type='building'
    bbox_str = ','.join([str(transformed_road_search_area.bounds[x]) for x in range(4)])
    print(f"Running Overture command with bbox_str={bbox_str} and request_type={request_type}")
    command = [
        "overturemaps", "download",
        "-f", "geojson",
        "--bbox=" + bbox_str,
        "--type=" + request_type
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Ensure the directory exists
    output_dir = f"{roads_path}/{city_name}"
    os.makedirs(output_dir, exist_ok=True)

    if result.returncode == 0:
        geoparquet_data = result.stdout
        try:
            overture_file = gpd.read_file(StringIO(geoparquet_data))
        except Exception as e:
            print(f"Error reading Overture data: {e}")
    else:
        print(f"Error occurred while running Overture command: {result.stderr}")
    if overture_file is not None:
        output_path = f'{output_dir}/Overture_{request_type}_{city_name}.geojson'
        print(f"Saving Overture file to {output_path}")
        try:
            overture_file.to_file(output_path)
        except Exception as e:
            print(f"Error saving Overture file: {e}")
    else:
        print(f"Skipping save for Overture in city: {city_name}")

# Need to get city urban extent
# 

#@delayed
def overturemaps_command(bbox_str, request_type: str):
    print(f"Running Overture command with bbox_str={bbox_str} and request_type={request_type}")
    command = [
        "overturemaps", "download",
        "-f", "geojson",
        "--bbox=" + bbox_str,
        "--type=" + request_type
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        geoparquet_data = result.stdout
        try:
            overture_file = gpd.read_file(StringIO(geoparquet_data))
            return overture_file
        except Exception as e:
            print(f"Error reading Overture data: {e}")
            return None
    else:
        print(f"Error occurred while running Overture command: {result.stderr}")
        return None

#@delayed
def overturemaps_save(overture_file, request_type: str, id: int):
    if overture_file is not None:
        output_path = f'./output_data/Overture_{request_type}_{id}.geojson'
        print(f"Saving Overture file to {output_path}")
        try:
            overture_file.to_file(output_path)
            return output_path
        except Exception as e:
            print(f"Error saving Overture file: {e}")
    else:
        print(f"Skipping save for Overture ID: {id}")
    return None



def make_requests(partition):
    print(f"Processing partitioxn with {len(partition)} rows")
    results = []
    for city_name in partition.iterrows():
        print(city_name)

        # NEED TO GET THE CITY URBAN EXTENT

        # NEED TO GET THE SEARCH AREA

        # Get city geometries in localized UTMs
        utm_proj_city = get_utm_proj(float(city.geometry.centroid.x), float(city.geometry.centroid.y))
        transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True)
        def transform_geom(geom):
            return transform(transformer.transform, geom)
        transformed_city_geom = transform_geom(city.geometry)

        custom_filter = '["highway"~"trunk|motorway|primary|secondary|tertiary|primary_link|secondary_link|tertiary_link|trunk_link|motorway_link|residential|unclassified|road"]'
        road_search_area = gpd.GeoSeries(transformed_city_geom,crs=utm_proj_city).geometry[0].buffer(2000)
        #transformed_urban_border = transform(pyproj.Transformer.from_crs(utm_proj_city, pyproj.CRS('EPSG:4326'), always_xy=True).transform, urban_border)
        transformed_road_search_area = transform(pyproj.Transformer.from_crs(utm_proj_city, pyproj.CRS('EPSG:4326'), always_xy=True).transform, road_search_area)
        osm_command(city_name,transformed_road_search_area)

        # NEED TO GET SEARCH AREA BOUNDARIES
        try:
            # Overture maps commands
            bbox_str = ','.join(rectangle[['minx', 'miny', 'maxx', 'maxy']].astype(str))
            request_type = 'building'
            print("About to trigger overturemaps command")
            overture_file = overturemaps_command(bbox_str, request_type)
            
            if overture_file is not None:
                print("About to trigger overturemaps save")
                save_result = overturemaps_save(overture_file,request_type, index + 1)
                results.append(save_result)
        except Exception as e:
            print(f"Overture error: {e}")


    return results

# Paralellize data gathering for all cities in the list.
def run_all():
    cities_set = pd.DataFrame(cities)

    # Create Dask DataFrame
    cities_set_ddf = dd.from_pandas(cities_set, npartitions=4)
    
    # Print the Dask DataFrame columns
    print(f"Columns in the Dask DataFrame: {cities_set_ddf.columns.tolist()}")

    # Apply function to each partition
    results = cities_set_ddf.map_partitions(make_requests, meta=cities_set)

    # Trigger computation
    computed_results = results.compute()
    print(f"Results: {computed_results}")

# Execute the function
run_all()
print("All tasks completed.")
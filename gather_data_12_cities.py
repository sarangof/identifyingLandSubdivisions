import subprocess
from io import StringIO
import geopandas as gpd
import osmnx as ox
from dask import delayed, compute
import pandas as pd
import dask.dataframe as dd
import os

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

analysis_buffers = gpd.read_file('12 city analysis buffers.geojson')
search_buffers = gpd.read_file('12 city search buffers.geojson')

# Gather OSM roads and intersections data for the borders.
for _, city in search_buffers.iterrows():
    city_name = city['city_name']
    print(city_name)
    custom_filter = '["highway"~"trunk|motorway|primary|secondary|tertiary|primary_link|secondary_link|tertiary_link|trunk_link|motorway_link|residential|unclassified|road"]'
    G = ox.graph_from_polygon(polygon=search_buffers[search_buffers.city_name==city_name]['geometry'].iloc[0], custom_filter=custom_filter)
    osm_intersections, osm_roads = ox.graph_to_gdfs(G)
    osm_roads = remove_duplicate_roads(osm_roads)
    road_output_file = f"./output_data/{city_name}_osm_roads.gpkg"
    osm_roads = osm_roads.drop(columns=['osmid','reversed'])
    osm_roads = remove_list_columns(osm_roads)
    osm_roads.to_file(road_output_file, driver="GPKG")    
    osm_intersections.to_file(f"./output_data/{city_name}_osm_intersections.gpkg", driver="GPKG")  

# Create output directory if it does not exist
if not os.path.exists('./output_data'):
    os.makedirs('./output_data')

# DEFINE LOADING FUNCTIONS

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
    print(f"Processing partition with {len(partition)} rows")
    results = []
    for _, rectangle in partition.iterrows():
        print(f"Processing row {rectangle['n']}")
        index = int(rectangle['n'])  # Ensure index is an integer

        try:
            # Overture maps commands
            bbox_str = ','.join(rectangle[['minx', 'miny', 'maxx', 'maxy']].astype(str))
            request_type = 'building'
            print("About to trigger overturemaps command")
            overture_file = overturemaps_command(bbox_str, request_type)
            
            if overture_file is not None:
                print("About to trigger overturemaps save")
                save_result = overturemaps_save(overture_file, request_type, index + 1)
                results.append(save_result)
        except Exception as e:
            print(f"Overture error: {e}")

        try:
            # Open street maps commands
            bbox = [rectangle['maxy_expanded'], rectangle['miny_expanded'], rectangle['maxx_expanded'], rectangle['minx_expanded']]
            print("About to trigger osm command")
            osm_buildings, osm_intersections, osm_roads = osmnx_command(bbox)
            osm_files = {'OSM_buildings': osm_buildings, 
                         'OSM_intersections': osm_intersections, 
                         'OSM_roads': osm_roads}
            # NEED TO MAKE SURE THIS COMMAND REALLY WORKS.
            if osm_files is not None: 
                print("About to trigger osm save")
                save_result = osmnx_save(osm_files, index + 1)
                results.append(save_result)
        except Exception as e:
            print(f"OSM error: {e}")

    return results

def run_all():
    # Load rectangles file
    rectangles = gpd.read_file('data/rectangles.geojson')
    
    # Prepare DataFrame for Dask
    rectangles[['minx', 'miny', 'maxx', 'maxy']] = rectangles.bounds
    bounds_list = rectangles[['minx', 'miny', 'maxx', 'maxy']].copy()
    bounds_list[['minx_expanded','miny_expanded','maxx_expanded','maxy_expanded']] = rectangles[['minx_expanded','miny_expanded','maxx_expanded','maxy_expanded']]
    bounds_list['n'] = bounds_list.index
    
    # Create Dask DataFrame
    bounds_ddf = dd.from_pandas(bounds_list, npartitions=4)
    
    # Print the Dask DataFrame columns
    print(f"Columns in the Dask DataFrame: {bounds_ddf.columns.tolist()}")

    # Apply function to each partition
    results = bounds_ddf.map_partitions(make_requests, meta=bounds_list)

    # Trigger computation
    computed_results = results.compute()
    print(f"Results: {computed_results}")

# Execute the function
run_all()
print("All tasks completed.")
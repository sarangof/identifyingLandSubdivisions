
import ee
from geemap import ee_to_gdf
from math import sqrt, pi
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
import tempfile
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, get_client
import logging


# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("START")

# Paths Configuration
MAIN_PATH = "../data"
#MAIN_PATH = "/mount/wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = os.path.join(MAIN_PATH, "input")
CITY_INFO_PATH = os.path.join(INPUT_PATH, "city_info")
EXTENTS_PATH = os.path.join(CITY_INFO_PATH, "extents")
ANALYSIS_BUFFERS_PATH = os.path.join(CITY_INFO_PATH, "analysis_buffers")
SEARCH_BUFFERS_PATH = os.path.join(CITY_INFO_PATH, "search_buffers")
GRIDS_PATH = os.path.join(CITY_INFO_PATH, "grids")
OUTPUT_PATH = os.path.join(MAIN_PATH, "output")
OUTPUT_PATH_CSV = os.path.join(OUTPUT_PATH, "csv")


def ensure_paths_exist(paths):
    """Ensure all necessary directories exist."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def compute_buffer_radius(city_area):
    """Compute buffer radius based on city area."""
    return sqrt(city_area) / pi / 4

def create_grid(geometry, grid_size):
    """Creates a grid of full squares over the bounding box of the given geometry."""
    bounds = geometry.bounds
    xmin, ymin, xmax, ymax = bounds

    # Generate columns and rows for grid cells
    cols = np.arange(xmin, xmax, grid_size)
    rows = np.arange(ymin, ymax, grid_size)

    # Create full rectangular polygons for each grid cell
    polygons = [box(x, y, x + grid_size, y + grid_size) for x in cols for y in rows]

    # Convert to GeoDataFrame
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

    return grid

def filter_grid_by_geometry(grid_gdf, target_geometry):
    """Filters grid cells, keeping only those that intersect with the target geometry, retaining full grid cells."""
    # Filter cells that intersect with the target geometry
    filtered_grid = grid_gdf[grid_gdf.intersects(target_geometry)].copy()

    return filtered_grid

# Function to process each city
@delayed
def process_city(city_name, urban_extent, search_buffer_distance=500, grid_sizes=[100, 200]):
    logger.info(f"Processing {city_name} with grid sizes: {grid_sizes}")
    #ee.Initialize()
    # Filter urban extent for the city
    city_extent = urban_extent.filter(ee.Filter.eq('city_name_large', city_name.replace('_', ' ')))
    city_geometry = city_extent.geometry()
    city_area = city_geometry.area().getInfo()
    logger.info(f"City area: {city_area} m2")
    city_resolution_data = []  # Store city cell count data

    # Create analysis buffer and subtract the urban extent to make it hollow
    analysis_buffer_radius = compute_buffer_radius(city_area)
    analysis_buffer = city_geometry.buffer(analysis_buffer_radius)
    analysis_expansion_area = analysis_buffer.difference(city_geometry)

    # Create a full search buffer that extends outward from the analysis area
    search_buffer = analysis_buffer.buffer(search_buffer_distance)

    # Convert the buffers to FeatureCollections with properties
    city_fc = ee.FeatureCollection([ee.Feature(city_geometry, {'name': city_name})])
    analysis_fc = ee.FeatureCollection([ee.Feature(analysis_expansion_area, {'name': f'{city_name}_analysis_buffer'})])
    search_fc = ee.FeatureCollection([ee.Feature(search_buffer, {'name': f'{city_name}_search_buffer'})])

    # Convert EE FeatureCollections to GeoDataFrames using temporary files
    city_gdf = ee_to_gdf(city_fc)
    analysis_gdf = ee_to_gdf(analysis_fc)
    search_gdf = ee_to_gdf(search_fc)

    ensure_paths_exist([f'{EXTENTS_PATH}/{city_name}',f'{ANALYSIS_BUFFERS_PATH}/{city_name}',f'{SEARCH_BUFFERS_PATH}/{city_name}'])

    city_gdf.to_parquet(f'{EXTENTS_PATH}/{city_name}/{city_name}_urban_extent.geoparquet')
    analysis_gdf.to_parquet(f'{ANALYSIS_BUFFERS_PATH}/{city_name}/{city_name}_analysis_buffer.geoparquet')
    search_gdf.to_parquet(f'{SEARCH_BUFFERS_PATH}/{city_name}/{city_name}_search_buffer.geoparquet')

    # Create grids for the search area using GeoPandas
    for grid_size in grid_sizes:
        # Convert meters to degrees approximately (grid_size / 111139 to approximate degrees from meters)
        grid_gdf = create_grid(search_gdf.geometry.iloc[0], grid_size / 111139.0)

        # Filter the grid to include only cells that intersect with the search geometry
        filtered_grid_gdf = filter_grid_by_geometry(grid_gdf, search_gdf.geometry.iloc[0])

        # Add a boolean attribute to indicate if the cell intersects with the analysis area
        filtered_grid_gdf['intersects_analysis_area'] = filtered_grid_gdf.intersects(analysis_gdf.geometry.iloc[0])

        ensure_paths_exist([f"{GRIDS_PATH}/{city_name}"])
        filtered_grid_gdf.to_parquet(f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet')
        logger.info(f"{grid_size}m grid cells: {len(filtered_grid_gdf)}")

        # Log the cell count for this grid size
        cell_count = len(filtered_grid_gdf)
        logger.info(f"{city_name} - {grid_size}m grid cells: {cell_count}")

        # Append data for CSV
        city_resolution_data.append({"city": city_name, "grid_size": grid_size, "cell_count": cell_count})
    
    # Save the resolution data for this city to a CSV
    ensure_paths_exist([f"{OUTPUT_PATH_CSV}/{city_name}"])
    resolution_csv_path = os.path.join(f"{OUTPUT_PATH_CSV}/{city_name}", f"{city_name}_grid_cell_counts.csv")
    pd.DataFrame(city_resolution_data).to_csv(resolution_csv_path, index=False)
    logger.info(f"Saved grid cell counts for {city_name} to {resolution_csv_path}")

    return city_resolution_data



def main():
    cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
              "Lagos", "Accra", "Abidjan", "Cape Town", "Mogadishu", 
              "Maputo", "Luanda"] #, 
    #cities = ["Belo Horizonte"]

    # List to store resolution data for all cities
    all_cities_resolution_data = []

    # Authenticate and initialize Earth Engine  
    logger.info('Authenticate and initialize Earth Engine')
    ee.Authenticate()
    ee.Initialize(project='city-extent', opt_url="https://earthengine-highvolume.googleapis.com")
    
    # Urban extent dataset
    urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

    cities = [city.replace(' ', '_') for city in cities]
    logger.info(f'Cities to be processed: {cities}')
 
    # Use Dask to parallelize the processing of cities
    tasks = [process_city(city, urban_extent) for city in cities]
    logger.info(f'Dask tasks: {tasks}')

    #dask.visualize(*tasks, filename='data/dask.svg')

    # Use ProgressBar to monitor progress
    logger.info('Computing Dask tasks')

    with ProgressBar():
        results = dask.compute(*tasks)
        logger.info('FINISH')

    # Combine all city resolution data into a single CSV
    combined_data = [entry for result in results for entry in result]
    combined_csv_path = os.path.join(OUTPUT_PATH_CSV, "all_cities_grid_cell_counts.csv")
    pd.DataFrame(combined_data).to_csv(combined_csv_path, index=False)
    logger.info(f"Saved combined grid cell counts to {combined_csv_path}")

if __name__ == "__main__":
    main()


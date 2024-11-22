import ee
import geemap
from math import sqrt, pi
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import tempfile
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# Define paths
main_path = '../data'
input_path = f'{main_path}/input'
city_info_path = f'{input_path}/city_info'
extents_path = f'{city_info_path}/extents'
analysis_buffers_path = f'{city_info_path}/analysis_buffers'
search_buffers_path = f'{city_info_path}/search_buffers'
grids_path = f'{city_info_path}/grids'
output_path = f'{main_path}/output'

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='city-extent')

# List of cities to process
cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
          "Maputo", "Luanda"]
cities = [city.replace(' ', '_') for city in cities]

# Urban extent dataset
urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

# Function to compute the buffer radius based on city area
def compute_buffer_radius(city_area):
    return sqrt(city_area) / pi / 4

# Function to create a grid of squares over the given bounding box using geopandas
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

# Function to filter grid cells by intersection with the target geometry, retaining full grid cells
def filter_grid_by_geometry(grid_gdf, target_geometry):
    """Filters grid cells, keeping only those that intersect with the target geometry, retaining full grid cells."""
    # Filter cells that intersect with the target geometry
    filtered_grid = grid_gdf[grid_gdf.intersects(target_geometry)]

    return filtered_grid

# Function to convert Earth Engine FeatureCollection to GeoDataFrame using tempfile
def ee_to_gdf(ee_fc):
    """Exports an Earth Engine FeatureCollection to a GeoDataFrame using a temporary file."""
    with tempfile.NamedTemporaryFile(suffix='.geojson') as tmpfile:
        # Export to a temporary GeoJSON file using geemap
        geemap.ee_export_vector(ee_fc, filename=tmpfile.name)
        
        # Load the GeoJSON into a GeoDataFrame
        gdf = gpd.read_file(tmpfile.name)
    return gdf

# Function to process each city
@delayed
def process_city(city_name, search_buffer_distance=500, grid_sizes=[100, 200]):
    print(city_name)
    # Filter urban extent for the city
    city_extent = urban_extent.filter(ee.Filter.eq('city_name_large', city_name.replace('_', ' ')))
    city_geometry = city_extent.geometry()
    city_area = city_geometry.area().getInfo()

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

    if not os.path.exists(f'{extents_path}/{city_name}'):
        os.makedirs(f'{extents_path}/{city_name}')
    city_gdf.to_parquet(f'{extents_path}/{city_name}/{city_name}_urban_extent.parquet')

    if not os.path.exists(f'{analysis_buffers_path}/{city_name}'):
        os.makedirs(f'{analysis_buffers_path}/{city_name}')
    analysis_gdf.to_parquet(f'{analysis_buffers_path}/{city_name}/{city_name}_analysis_buffer.parquet')

    if not os.path.exists(f'{search_buffers_path}/{city_name}'):
        os.makedirs(f'{search_buffers_path}/{city_name}')
    search_gdf.to_parquet(f'{search_buffers_path}/{city_name}/{city_name}_search_buffer.parquet')

    # Create grids for the search area using GeoPandas
    for grid_size in grid_sizes:
        # Convert meters to degrees approximately (grid_size / 111139 to approximate degrees from meters)
        grid_gdf = create_grid(search_gdf.geometry.iloc[0], grid_size / 111139.0)

        # Filter the grid to include only cells that intersect with the search geometry
        filtered_grid_gdf = filter_grid_by_geometry(grid_gdf, search_gdf.geometry.iloc[0])

        # Add a boolean attribute to indicate if the cell intersects with the analysis area
        filtered_grid_gdf['intersects_analysis_area'] = filtered_grid_gdf.intersects(analysis_gdf.geometry.iloc[0])

        if not os.path.exists(f'{grids_path}/{city_name}'):
            os.makedirs(f'{grids_path}/{city_name}')
        filtered_grid_gdf.to_parquet(f'{grids_path}/{city_name}/{city_name}_{grid_size}m_grid.parquet')

# Use Dask to parallelize the processing of cities
tasks = [process_city(city) for city in cities]

# Use ProgressBar to monitor progress
with ProgressBar():
    dask.compute(*tasks)

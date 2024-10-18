
import ee
import geemap
from math import sqrt, pi
import os

# Define paths
main_path = '../data'
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
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
#cities = [city.replace(" ", "_") for city in cities]

# Urban extent dataset
urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

# Function to compute the buffer radius based on city area
def compute_buffer_radius(city_area):
    return sqrt(city_area) / pi / 4


# Function to create a grid image over a given geometry
def create_grid_image(geometry, grid_size):
    """Creates a raster-based grid over the given geometry using Earth Engine."""
    # Create an image with lon/lat bands
    lon_lat = ee.Image.pixelLonLat()

    # Create a grid using floor division to create unique integer values for each grid cell
    lon_grid = lon_lat.select('longitude').divide(grid_size).floor()
    lat_grid = lon_lat.select('latitude').divide(grid_size).floor()

    # Combine the longitude and latitude grids into one image
    grid = lon_grid.addBands(lat_grid).reduce(ee.Reducer.sum()).rename('grid')

    # Clip the grid to the geometry area
    grid_masked = grid.clip(geometry)

    return grid_masked


# Function to process each city
def process_city(city_name, search_buffer_distance=500):
    # Filter urban extent for the city
    city_extent = urban_extent.filter(ee.Filter.eq('city_name_large', city_name))

    # Calculate city area (assuming the geometry is in square meters)
    city_geometry = city_extent.geometry()
    city_area = city_geometry.area().getInfo()

    # Create analysis buffer
    analysis_buffer_radius = compute_buffer_radius(city_area)
    analysis_buffer = city_geometry.buffer(analysis_buffer_radius)#ee.Geometry(city_extent.geometry()).buffer(analysis_buffer_radius)

    # Create search buffer by buffering inward from urban extent and outward from analysis buffer
    search_buffer_inward = city_geometry.buffer(-search_buffer_distance)
    search_buffer_outward = analysis_buffer.buffer(search_buffer_distance)
    search_buffer = search_buffer_inward.union(search_buffer_outward)

    # Create grids (200m x 200m and 100m x 100m) inside the analysis area using Earth Engine
    grid_200m = create_grid_image(analysis_buffer, 200)
    grid_100m = create_grid_image(analysis_buffer, 100)

    # Convert the buffers to FeatureCollections
    analysis_fc = ee.FeatureCollection([ee.Feature(analysis_buffer)])
    search_fc = ee.FeatureCollection([ee.Feature(search_buffer)])

    # Export the buffers and grids as GeoJSON or TIFF
    # Create output directory if it does not exist
    city_file_name = city_name.replace(" ", "_")

    if not os.path.exists(f'{extents_path}/{city_file_name}'):
        os.makedirs(f'{extents_path}/{city_file_name}')
    geemap.ee_export_vector(city_extent, filename=f'{extents_path}/{city_file_name}/{city_name}_city_extent.geojson')

    if not os.path.exists(f'{analysis_buffers_path}/{city_file_name}'):
        os.makedirs(f'{analysis_buffers_path}/{city_file_name}')
    geemap.ee_export_vector(analysis_fc, filename=f'{analysis_buffers_path}/{city_file_name}/{city_name}_analysis_buffer.geojson')

    if not os.path.exists(f'{search_buffers_path}/{city_file_name}'):
        os.makedirs(f'{search_buffers_path}/{city_file_name}')
    geemap.ee_export_vector(search_fc, filename=f'{search_buffers_path}/{city_file_name}/{city_name}_search_buffer.geojson')
    
    # Export grids as images (TIFF format) for easier processing
    if not os.path.exists(f'{grids_path}/{city_file_name}'):
        os.makedirs(f'{grids_path}/{city_file_name}')
    geemap.ee_export_image(grid_200m, filename=f'{grids_path}/{city_file_name}/{city_name}_200m_grid.tif', scale=200)
    geemap.ee_export_image(grid_100m, filename=f'{grids_path}/{city_file_name}/{city_name}_100m_grid.tif', scale=100)

# Process each city
for city in cities:
    print(f'{city}')
    process_city(city)

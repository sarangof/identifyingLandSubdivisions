import ee
import geemap
from math import sqrt, pi

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='city-extent')

# List of cities to process
cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
          "Maputo", "Luanda"]

# Urban extent dataset
urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

# Function to compute the buffer radius based on city area
def compute_buffer_radius(city_area):
    return sqrt(city_area) / pi / 4

# Function to create grids using Earth Engine
def create_grid_in_ee(analysis_buffer, grid_size):
    # Create a grid over the analysis area using pixelLonLat
    grid = ee.Image.pixelLonLat()

    # Create a mask for the grid using ee.Image.constant(1) and clip it using ee.Geometry.clip()
    mask = ee.Image.constant(1).clip(analysis_buffer)
    
    # Mask the grid to include only points within the analysis buffer
    grid_in_analysis = grid.updateMask(mask)

    return grid_in_analysis

# Function to process each city
def process_city(city_name, search_buffer_distance=500):
    # Filter urban extent for the city
    city_extent = urban_extent.filter(ee.Filter.eq('city_name_large', city_name))

    # Calculate city area (assuming the geometry is in square meters)
    city_area = ee.Geometry(city_extent.geometry()).area().getInfo()

    # Create analysis buffer
    analysis_buffer_radius = compute_buffer_radius(city_area)
    analysis_buffer = ee.Geometry(city_extent.geometry()).buffer(analysis_buffer_radius)

    # Create search buffer by buffering inward and outward by 500m
    search_buffer_inward = ee.Geometry(city_extent.geometry()).buffer(-search_buffer_distance)
    search_buffer_outward = ee.Geometry(city_extent.geometry()).buffer(search_buffer_distance)
    search_buffer = search_buffer_inward.union(search_buffer_outward)

    # Create grids (200m x 200m and 100m x 100m) inside the analysis area using Earth Engine
    grid_200m = create_grid_in_ee(analysis_buffer, 200)
    grid_100m = create_grid_in_ee(analysis_buffer, 100)

    # Convert the buffers to FeatureCollections
    analysis_fc = ee.FeatureCollection([ee.Feature(analysis_buffer)])
    search_fc = ee.FeatureCollection([ee.Feature(search_buffer)])

    # Export the buffers and grids as GeoJSON or TIFF
    geemap.ee_export_vector(analysis_fc, filename=f'{city_name}_analysis_buffer.geojson')
    geemap.ee_export_vector(search_fc, filename=f'{city_name}_search_buffer.geojson')
    
    # Export grids as images (TIFF format) for easier processing
    geemap.ee_export_image(grid_200m, filename=f'{city_name}_200m_grid.tif', scale=200)
    geemap.ee_export_image(grid_100m, filename=f'{city_name}_100m_grid.tif', scale=100)

# Process each city
for city in cities:
    print(f'{city}')
    process_city(city)

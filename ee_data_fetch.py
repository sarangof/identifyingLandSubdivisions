import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from math import sqrt, pi, floor
from pyproj import CRS, Transformer

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='city-extent')

# List of cities to process
cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
          "Maputo", "Luanda"]

# Urban extent dataset
urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

# GeoDataFrames to hold results
analysis_buffers = gpd.GeoDataFrame(columns=['city_name', 'geometry'], crs='EPSG:4326')
search_buffers = gpd.GeoDataFrame(columns=['city_name', 'geometry'], crs='EPSG:4326')

# Function to compute the buffer radius based on city area
def compute_buffer_radius(city_area):
    return sqrt(city_area / pi) / 4

# Function to determine UTM zone for a given longitude
def get_utm_crs(longitude):
    zone_number = floor((longitude + 180) / 6) + 1
    hemisphere = '326' if longitude >= 0 else '327'
    return f'EPSG:{hemisphere}{zone_number:02d}'

# Function to create grids within the analysis area
def create_grid_within_analysis_area(analysis_geom, city_geom, square_width, local_crs):
    bounds = analysis_geom.bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Create local projection transformer
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    inv_transformer = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    
    min_x_proj, min_y_proj = transformer.transform(min_x, min_y)
    max_x_proj, max_y_proj = transformer.transform(max_x, max_y)
    
    grid_cells = []
    x = min_x_proj
    while x < max_x_proj:
        y = min_y_proj
        while y < max_y_proj:
            # Create a square grid cell in the projected space
            cell = box(x, y, x + square_width, y + square_width)
            # Transform back to lat/lon
            cell_latlon = gpd.GeoSeries([cell], crs=local_crs).to_crs('EPSG:4326').geometry[0]
            if cell_latlon.intersects(analysis_geom):  # Only add cells intersecting with the analysis buffer
                grid_cells.append(cell_latlon)
            y += square_width
        x += square_width

    # Create grid GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:4326')
    # Add a column to indicate if each grid cell is within the city boundary
    grid_gdf['in_city_boundary'] = grid_gdf.geometry.apply(lambda x: x.within(city_geom))
    
    return grid_gdf

# Function to process each city
def process_city(city_name, search_buffer_distance=500):
    # Filter urban extent for the city
    city_extent = urban_extent.filter(ee.Filter.eq('city_name_large', city_name))

    # Convert the Earth Engine object to a GeoJSON and then to GeoDataFrame
    city_extent_geojson = geemap.ee_to_geojson(city_extent)
    city_extent_gdf = gpd.GeoDataFrame.from_features(city_extent_geojson, crs='EPSG:4326')

    # Get centroid longitude and determine UTM zone for local projection
    centroid = city_extent_gdf.geometry.centroid.iloc[0]
    local_crs = get_utm_crs(centroid.x)  # CRS for the city's UTM zone

    # Reproject city extents to local UTM zone
    city_extent_gdf = city_extent_gdf.to_crs(local_crs)

    # Calculate city area (in square meters)
    city_area = city_extent_gdf.geometry.area.sum()

    # Create analysis buffer
    analysis_buffer_radius = compute_buffer_radius(city_area)
    analysis_buffer = city_extent_gdf.geometry.buffer(analysis_buffer_radius)

    # Create search buffer by buffering both outward and inward
    search_buffer_outward = analysis_buffer.buffer(search_buffer_distance)
    search_buffer_inward = analysis_buffer.buffer(-search_buffer_distance)
    
    # Union the inward and outward buffers to get the complete search buffer
    search_buffer = search_buffer_outward.union(search_buffer_inward)
    
    # Reproject buffers back to EPSG:4326 for consistent outputs
    analysis_buffer = analysis_buffer.to_crs('EPSG:4326')
    search_buffer = search_buffer.to_crs('EPSG:4326')

    # Append buffers to respective GeoDataFrames
    global analysis_buffers, search_buffers
    analysis_buffers = pd.concat([analysis_buffers, gpd.GeoDataFrame({'city_name': [city_name], 'geometry': analysis_buffer}, crs='EPSG:4326')], ignore_index=True)
    search_buffers = pd.concat([search_buffers, gpd.GeoDataFrame({'city_name': [city_name], 'geometry': search_buffer}, crs='EPSG:4326')], ignore_index=True)

    # Create grids only within the analysis buffer and check city boundaries
    grid_200m = create_grid_within_analysis_area(analysis_buffer.iloc[0], city_extent_gdf.iloc[0].geometry, 200, local_crs)
    grid_100m = create_grid_within_analysis_area(analysis_buffer.iloc[0], city_extent_gdf.iloc[0].geometry, 100, local_crs)

    return grid_200m, grid_100m

# Process each city and gather grids
for city in cities:
    grid_200m, grid_100m = process_city(city)
    # Save or plot grid as necessary
    grid_200m.to_file(f'{city}_200m_grid.geojson', driver='GeoJSON')
    grid_100m.to_file(f'{city}_100m_grid.geojson', driver='GeoJSON')

# Save the buffers to GeoJSON files
analysis_buffers.to_file('./12_city_analysis_buffers.geojson', driver='GeoJSON')
search_buffers.to_file('./12_city_search_buffers.geojson', driver='GeoJSON')

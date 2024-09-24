import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='city-extent')


# List of cities to process
cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
          "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
          "Maputo", "Luanda"]

# Urban extent dataset
urban_extent = ee.FeatureCollection("projects/wri-datalab/cities/urban_land_use/data/global_cities_Aug2024/urbanextents_unions_2020")

# GHSL Population dataset for the projection
GHSLpop = ee.ImageCollection("users/emackres/GHS_POP_GLOBE_R2023A_4326_3ss")
GHSLpop_proj = GHSLpop.first().select('b1').projection()
GHSLpop_scale = GHSLpop_proj.nominalScale()

# Buffer and grid creation function
def buffer_and_grid(city_name, buffer_distance, grid_size):
    # Filter the urban extent for the specific city
    city_fc = urban_extent.filter(ee.Filter.eq('city_name_large', city_name))
    
    # Debug: Print city feature collection info
    #print(f"City Feature Collection for {city_name}: ", city_fc.getInfo())
    
    # Get the city's geometry and apply buffer
    city_geom = city_fc.geometry()
    
    # Debug: Check if city_geom is None
    if city_geom is None:
        print(f"No geometry found for city: {city_name}")
        return None, None
    
    buffered_city = city_geom.buffer(buffer_distance)
    
    # Create a grid around the buffered city
    grid = buffered_city.coveringGrid(ee.Projection(GHSLpop_proj), grid_size)
    
    # Extract the left corners of grid cells
    left_corners = grid.map(lambda f: ee.Feature(ee.Geometry.Point(f.geometry().coordinates().get(0))))
    
    # Get the bounding box of the buffer
    bounding_box = buffered_city.bounds()
    
    return left_corners, bounding_box

# Process each city
buffer_distance = 2000  # 2 km buffer
grid_size = GHSLpop_scale.multiply(2)  # Adjust grid size

# Dictionary to store grids and bounding boxes
city_grids = {}
city_bounding_boxes = {}

for city in cities:
    city_grid, city_bbox = buffer_and_grid(city, buffer_distance, grid_size)
    city_grids[city] = city_grid
    city_bounding_boxes[city] = city_bbox

# Example: Get the grid and bounding box for 'Bogotá'
bogota_grid = city_grids["Bogota"]
bogota_bbox = city_bounding_boxes["Bogota"]

# Check if the bounding box is still None
if bogota_bbox is None:
    print("Failed to generate bounding box for Bogotá")
else:
    print("Bounding Box for Bogotá: ", bogota_bbox.getInfo())

# Visualize Bogotá's grid and bounding box using geemap
Map = geemap.Map()

# Add the grid and bounding box to the map
if bogota_grid:
    Map.addLayer(bogota_grid, {'color': 'blue'}, 'Grid for Bogotá')
if bogota_bbox:
    Map.addLayer(bogota_bbox, {'color': 'green'}, 'Bounding Box for Bogotá')

# Safely center map around Bogotá's bounding box if valid
if bogota_bbox:
    Map.centerObject(bogota_bbox)

# Display the map in a browser as an HTML file
Map.to_html('bogota_grid_map.html')


## GET GRID DATA FOR FIRST 12 CITIES


# Sample function to process the features as polygons
def get_polygons_from_feature_collection(feature_collection):
    """
    Function to ensure that only polygon geometries are processed
    """
    # Define a function to map over the features in Earth Engine
    def process_feature(feature):
        # Ensure that the geometry is a polygon
        geom = feature.geometry()
        return ee.Feature(geom)

    # Map over the feature collection to apply the process_feature function
    polygons_fc = feature_collection.map(process_feature)
    
    return polygons_fc

# Example: Process one of the city grids
city_grids_processed = {}

# Loop through each city in the city_grids dictionary
for city_name, feature_collection in city_grids.items():
    # Process the feature collection for polygons
    processed_fc = get_polygons_from_feature_collection(feature_collection)
    
    # Retrieve the processed polygons as a list of features (client-side with getInfo)
    features = processed_fc.getInfo()['features']
    
    # Convert the features to shapely polygons
    for feature in features:
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            # Extract the outer boundary of the polygon
            polygon_coords = [tuple(coord) for coord in geometry['coordinates'][0]]
            # Create a shapely Polygon object
            polygon_geom = Polygon(polygon_coords)
            # Append the city name and polygon geometry to the list
            grid_data.append({
                'city_name': city_name,
                'geometry': polygon_geom
            })

# Convert the processed grid data to a GeoDataFrame
gdf_city_grids = gpd.GeoDataFrame(grid_data, geometry='geometry')
gdf_city_grids.set_crs(epsg=4326, inplace=True)  # Set the CRS to WGS84





## GET LIST OF BOUNDING BOXES FOR FIRST 12 CITIES



bbox_data = []
for city_name, bbox_ee_geometry in city_bounding_boxes.items():
    # Get the bounding box coordinates from Earth Engine
    bbox_info = bbox_ee_geometry.bounds().getInfo()
    
    # Extract the bounding box coordinates from the GeoJSON
    min_lon, min_lat, max_lon, max_lat = bbox_info['coordinates'][0][0][0], bbox_info['coordinates'][0][0][1], bbox_info['coordinates'][0][2][0], bbox_info['coordinates'][0][2][1]
    
    # Create a bounding box geometry using shapely
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    
    # Append city name and bounding box geometry to the list
    bbox_data.append({
        'city_name': city_name,
        'geometry': bbox_geom
    })

# Create a GeoDataFrame from the bounding boxes
gdf_city_bounding_boxes = gpd.GeoDataFrame(bbox_data, geometry='geometry')
gdf_city_bounding_boxes.set_crs(epsg=4326, inplace=True)
gdf_city_bounding_boxes.to_file('gdf_12_city_bounding_boxes.shp')
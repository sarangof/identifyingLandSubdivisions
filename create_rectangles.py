import geopandas as gpd
import pandas as pd
import re
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

# Convert DMS to float.
def dms_to_dd(dms_str):
    dms_str = str(dms_str).strip()
    parts = re.split(r'[^\w\d.]+', dms_str)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    direction = parts[3]
    
    dd = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Create a rectangle using geodesic distances and an expanded rectangle
def create_rectangle_wgs84_geodesic(point, target_width, target_height):
    lon, lat = point.x, point.y
    
    # Define the four corners of the rectangle using geodesic distances
    lower_left = (lat, lon)
    lower_right = geodesic(meters=target_width).destination(lower_left, 90)  
    upper_left = geodesic(meters=target_height).destination(lower_left, 0) 
    upper_right = geodesic(meters=target_height).destination(lower_right, 0)
    
    # Create the polygon using the four corners
    rectangle = Polygon([
        (lower_left[1], lower_left[0]), 
        (lower_right[1], lower_right[0]), 
        (upper_right[1], upper_right[0]), 
        (upper_left[1], upper_left[0]), 
        (lower_left[1], lower_left[0])
    ])
    
    return rectangle

def create_expanded_rectangle_wgs84_geodesic(point, target_width, target_height, expansion_factor: float):
    lon, lat = point.x, point.y

    lower_left_expanded = (lat, lon)
    lower_right_expanded = geodesic(meters=expansion_factor*target_width).destination(lower_left_expanded, 90)  
    upper_left_expanded = geodesic(meters=expansion_factor*target_height).destination(lower_left_expanded, 0) 
    upper_right_expanded = geodesic(meters=expansion_factor*target_height).destination(lower_right_expanded, 0)

    rectangle_expanded = Polygon([
        (lower_left_expanded[1], lower_left_expanded[0]), 
        (lower_right_expanded[1], lower_right_expanded[0]), 
        (upper_right_expanded[1], upper_right_expanded[0]), 
        (upper_left_expanded[1], upper_left_expanded[0]), 
        (lower_left_expanded[1], lower_left_expanded[0])
    ])
    xmin, ymin, xmax, ymax = rectangle_expanded.bounds
    return xmin, ymin, xmax, ymax

# Load the CSV file into a pandas DataFrame
source_coordinates = pd.read_csv('data/pilot_coordinates.csv', sep=';', encoding='ISO-8859-1')
source_coordinates['latitude_dd'] = source_coordinates['Lat'].apply(dms_to_dd) # Convert DMS to decimal degrees
source_coordinates['longitude_dd'] = source_coordinates['Lon'].apply(dms_to_dd)
geometry = [Point(xy) for xy in zip(source_coordinates['longitude_dd'], source_coordinates['latitude_dd'])]
rectangles = gpd.GeoDataFrame(source_coordinates, geometry=geometry)  #this is a point vector structure
target_width = 400  # in meters
target_height = 250  # in meters
# Apply the rectangle creation for each point using geodesic distances
rectangles['geometry'] = rectangles['geometry'].apply(lambda point: create_rectangle_wgs84_geodesic(point, target_width, target_height))
rectangles[['minx_expanded','miny_expanded','maxx_expanded','maxy_expanded']] = rectangles['geometry'].apply(lambda point: create_expanded_rectangle_wgs84_geodesic(point.centroid, target_width, target_height, expansion_factor=3)).apply(pd.Series)
rectangles[['minx', 'miny', 'maxx', 'maxy']] = rectangles.bounds 
# Save the result to a GeoJSON file
rectangles.to_file('./data/rectangles.geojson', driver='GeoJSON')

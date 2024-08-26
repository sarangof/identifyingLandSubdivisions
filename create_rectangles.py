import geopandas as gpd
import pandas as pd
import re
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

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

# Function to create a rectangle using geodesic distances
def create_rectangle_wgs84_geodesic(point, target_width, target_height):
    lon, lat = point.x, point.y
    
    # Define the four corners of the rectangle using geodesic distances
    lower_left = (lat, lon)
    lower_right = geodesic(meters=target_width).destination(lower_left, 90)  # 90 degrees -> East
    upper_left = geodesic(meters=target_height).destination(lower_left, 0)  # 0 degrees -> North
    upper_right = geodesic(meters=target_height).destination(lower_right, 0)  # 0 degrees -> North
    
    # Create the polygon using the four corners
    rectangle = Polygon([
        (lower_left[1], lower_left[0]), 
        (lower_right[1], lower_right[0]), 
        (upper_right[1], upper_right[0]), 
        (upper_left[1], upper_left[0]), 
        (lower_left[1], lower_left[0])
    ])
    
    return rectangle

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/pilot_coordinates.csv', sep=';', encoding='ISO-8859-1')

# Convert DMS to decimal degrees
df['latitude_dd'] = df['Lat'].apply(dms_to_dd)
df['longitude_dd'] = df['Lon'].apply(dms_to_dd)

# Create the GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude_dd'], df['latitude_dd'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Define target dimensions in meters
target_width = 400  # in meters
target_height = 250  # in meters

# Apply the rectangle creation for each point using geodesic distances
gdf['geometry'] = gdf['geometry'].apply(lambda point: create_rectangle_wgs84_geodesic(point, target_width, target_height))

# Save the result to a GeoJSON file
gdf.to_file('./data/rectangles.geojson', driver='GeoJSON')

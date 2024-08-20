import geopandas as gpd
import pandas as pd
import re
from shapely.geometry import Point, box
import pyproj


def dms_to_dd(dms_str):
    dms_str = str(dms_str).strip()
    parts = re.split(r'[^\w\d.]+', dms_str) #re.split('[^\d\w]+', dms_str)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    direction = parts[3]
    
    dd = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

# Function to determine the UTM zone for a given longitude
def utm_zone(longitude):
    return int(1 + (longitude + 180.0) / 6.0)

# Function to create a 400m x 250m rectangle centered on a point
def create_rectangle_wgs84(point, width, height):
    lon, lat = point.x, point.y
    zone_number = utm_zone(lon)
    is_northern = lat >= 0  # Determine if the zone is in the northern hemisphere

    # Define the UTM projection string
    utm_proj = f"+proj=utm +zone={zone_number} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Transform the point to UTM coordinates
    project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_proj, always_xy=True)
    project_to_wgs84 = pyproj.Transformer.from_crs(utm_proj, "EPSG:4326", always_xy=True)
    
    x, y = project_to_utm.transform(lon, lat)

    # Create the rectangle in UTM coordinates
    rectangle = box(x, y, x + width, y + height)

    # Transform the rectangle back to WGS84 coordinates
    minx, miny = project_to_wgs84.transform(rectangle.bounds[0], rectangle.bounds[1])
    maxx, maxy = project_to_wgs84.transform(rectangle.bounds[2], rectangle.bounds[3])

    return box(minx, miny, maxx, maxy)


# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/pilot_coordinates.csv')#,sep=";", encoding = "ISO-8859-1")

# Assuming your CSV has columns 'latitude' and 'longitude' in DMS format
df['latitude_dd'] = df['Lat'].apply(dms_to_dd)
df['longitude_dd'] = df['Lon'].apply(dms_to_dd)

# Create the GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude_dd'], df['latitude_dd'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Apply the rectangle creation for each point
gdf['geometry'] = gdf['geometry'].apply(lambda point: create_rectangle_wgs84(point, 400, 250))

# gdf now contains 400m x 250m rectangles across the area of interest in WGS84 coordinates


gdf.to_file('./data/rectangles.geojson', driver='GeoJSON')

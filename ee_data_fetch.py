import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='city-extent')

lc = ee.ImageCollection('MODIS/006/MCD12Q1')





import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='city-extent')

# Set the year and other parameters
year = 2020
count = [17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
toyear = [1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020,2025,2030]

# Load the GHSL dataset
GHSL2023release = ee.Image("users/emackres/GHS_BUILT_S_MT_2023_100_BUTOT_MEDIAN")

# Process the dataset: filter built-up areas and remap years
GHSL2023releaseYear = (GHSL2023release.gte(1000)
    .selfMask()
    .reduce(ee.Reducer.count())
    .remap(count, toyear)
    .selfMask()
    .rename(['bu'])
)

GHSLyear = GHSL2023releaseYear.updateMask(GHSL2023releaseYear.lte(year)).gt(0)

# Define the reduce function for built-up areas
def reducebd(IC):
    IC = IC.select("builtup_class").reduce(ee.Reducer.firstNonNull())
    IC = IC.updateMask(IC.select("builtup").reduce(ee.Reducer.firstNonNull()).gte(1))
    return IC

# Cities of interest
cities = [
    "Belo Horizonte", "Campinas", "Bogotá", "Nairobi", "Bamako", 
    "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
    "Maputo", "Luanda"
]

# Load a feature collection containing the cities
city_feature_collection = ee.FeatureCollection('users/some_dataset_with_city_name_large')

# Buffer and grid creation function
def buffer_and_grid(city_geometry, buffer_size, grid_size):
    # Apply buffer to city geometry
    buffered_city = city_geometry.buffer(buffer_size)
    
    # Create a grid over the buffered geometry
    grid = buffered_city.coveringGrid(ee.Projection('EPSG:4326').scale(grid_size, grid_size))
    
    return grid

# Function to process the city by applying buffer and grid
def process_city(city_name, grid_size, buffer_size):
    # Filter the city geometry by name
    city_geometry = city_feature_collection.filter(ee.Filter.eq('city_name_large', city_name)).geometry()
    
    # Apply buffer and create grid
    city_grid = buffer_and_grid(city_geometry, buffer_size, grid_size)
    
    return city_grid

# Example usage for a 100m grid with 400m buffer
buffer_size_100m = 400  # 400 meter buffer
grid_size_100m = 100    # 100 meter grid size

# Process each city to get the grid
city_grids = {city: process_city(city, grid_size_100m, buffer_size_100m) for city in cities}

# Example: retrieving the grid for 'Bogotá'
bogota_grid = city_grids["Bogotá"]

# Visualize the grid in the Earth Engine UI if using Google Earth Engine Code Editor
# Map.addLayer(bogota_grid, {color: 'blue'}, 'Buffered Grid for Bogotá')

# Returning the grids for further analysis
print(city_grids)



# Process Bogotá grid (assuming you already have the city_grids dictionary)
bogota_grid = city_grids["Bogotá"]

# Create a geemap map instance
Map = geemap.Map()

# Add the grid layer to the map for visualization
Map.addLayer(bogota_grid, {'color': 'blue'}, 'Buffered Grid for Bogotá')

# Optionally, display the map in a Jupyter Notebook
# Map.show() 

# Convert to GeoJSON and then to GeoDataFrame
geojson = geemap.ee_to_geojson(bogota_grid)
gdf = gpd.GeoDataFrame.from_features(geojson["features"])

# Plot the GeoDataFrame using matplotlib
gdf.plot(color='blue', edgecolor='black')
plt.title('Buffered Grid for Bogotá')
plt.show()

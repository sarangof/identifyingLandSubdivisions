

import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely import wkb
from difflib import get_close_matches
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
from shapely.ops import unary_union

import pyarrow.parquet as pq
import pandas as pd
from shapely import wkb
from shapely.ops import unary_union

from cloudpathlib import S3Path
import s3fs
import fsspec
import traceback

import os

# Paths Configuration
MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data" #
INPUT_PATH = os.path.join(MAIN_PATH, "input")
BUILDINGS_PATH = os.path.join(INPUT_PATH, "buildings")
ROADS_PATH = os.path.join(INPUT_PATH, "roads")
INTERSECTIONS_PATH = os.path.join(INPUT_PATH, "intersections")
URBAN_EXTENTS_PATH = os.path.join(INPUT_PATH, "urban_extents")
OUTPUT_PATH = os.path.join(MAIN_PATH, "output")
OUTPUT_PATH_CSV = os.path.join(OUTPUT_PATH, "csv")
SEARCH_BUFFER_PATH = os.path.join(INPUT_PATH, "city_info", "search_buffers")

fs = s3fs.S3FileSystem(anon=False)


import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import MultiPolygon

import s3fs

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=False)

# Read Parquet in batches
parquet_file = pq.ParquetFile("../combined_cities.parquet")

unique_cities = set()

# Read in batches to handle large files efficiently
batch_size = 100000  # Adjust as needed

for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["city_name"]):
    df_batch = batch.to_pandas()  # Convert batch to Pandas DataFrame
    unique_cities.update(df_batch["city_name"].unique())  # Update the set with unique values

# Convert set to sorted list (optional)
all_cities_list = list(sorted(unique_cities))

# Print summary
print(f"Found {len(all_cities_list)} unique city names.")



city_names_500 = pd.read_csv('../500cities.csv',sep=';', encoding="ISO-8859-1")

city_names_500["city_name_no_country"] = city_names_500["City Name"].apply(lambda x: x.split(',')[0].replace(" ", "_"))
city_names_500["city_name_no_country"] = city_names_500["city_name_no_country"].apply(lambda x: x.replace(" ", "_"))


city_set = set(city_names_500["city_name_no_country"])



from difflib import get_close_matches
import re

# Function to clean and normalize city names
def normalize_city_name(city):
    """Removes underscores intelligently based on context."""
    return re.sub(r'_(?=[a-z])', '', city)  # Remove underscore only if followed by lowercase letter

# Function to match city names based on refined rules

# Function to match city names based on refined rules
def refined_fuzzy_match(city, city_set):
    """Matches city names by allowing underscore corrections but avoiding over-relaxed matches."""
    if city in city_set:  # Exact match
        return (city, city)

    # Normalize city name: remove underscores only where appropriate
    cleaned_city = normalize_city_name(city)

    # Find close matches allowing a max difference of 1 edit
    possible_matches = get_close_matches(cleaned_city, city_set, n=1, cutoff=0.9)  # More restrictive cutoff

    # Return the closest match and the original city, or (None, city) if no match found
    return (possible_matches[0], city) if possible_matches else (None, None)


# Apply refined fuzzy matching
matching_list_500_cities = [refined_fuzzy_match(x, city_set)[0] for x in all_cities_list]
matched_list_all_cities = [refined_fuzzy_match(x, city_set)[1] for x in all_cities_list]

# Count how many cities were successfully matched
matched_cities = [x for x in matched_list_all_cities if x is not None]
num_matched = len(matched_cities)




# Configuration
grid_size = 200  # Define grid size for naming
output_base_dir = "../separate_city_geoparquets"

# Convert matched city names to sets for filtering & naming
city_match_dict = dict(zip(matched_list_all_cities, matching_list_500_cities))  # Maps filtered names to saved names
matched_cities_set = set(matched_list_all_cities)  # For efficient filtering

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Process Parquet in batches
batch_size = 100000  # Adjust as needed

for batch in parquet_file.iter_batches(batch_size=batch_size):
    df_batch = batch.to_pandas()[["city_name", "geom"]]  # Read necessary columns

    # Filter only the matched cities
    df_filtered = df_batch[df_batch["city_name"].isin(matched_cities_set)].copy()

    if df_filtered.empty:
        continue  # Skip if no relevant rows in this batch

    # Convert WKB geometries to Shapely
    df_filtered["geometry"] = df_filtered["geom"].apply(wkb.loads)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df_filtered, geometry="geometry", crs="EPSG:4326")

    # Save each city in its corresponding folder
    for original_city_name, city_gdf in gdf.groupby("city_name"):
        matched_city_name = city_match_dict.get(original_city_name, original_city_name)  # Get the correct save name
        
        city_folder = os.path.join(output_base_dir, matched_city_name)
        os.makedirs(city_folder, exist_ok=True)  # Create city folder if it doesn't exist

        output_path = os.path.join(city_folder, f"{matched_city_name}_{grid_size}m_grid.geoparquet")
        city_gdf.to_parquet(output_path)

print(f"Saved individual city grids in {output_base_dir}")

import os
import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union, polygonize
from shapely import wkt

# Paths
INPUT_DIR = "../grids/"
OUTPUT_DIR = "../city_search_buffers/"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through city folders
for city_name in os.listdir(INPUT_DIR):
    city_folder = os.path.join(INPUT_DIR, city_name)

    # Check if it's a directory
    if not os.path.isdir(city_folder):
        print(f'{city_name} not a folder')
        continue

    # Find the .geoparquet file inside the city's folder
    grid_files = [f for f in os.listdir(city_folder) if f.endswith("_200m_grid.geoparquet")]
    
    if not grid_files:
        print(f"No grid file found for {city_name}, skipping.")
        continue

    grid_path = os.path.join(city_folder, grid_files[0])  # Assuming one file per city
    #print(f"Processing: {grid_path}")

    # Read the grid file
    gdf = gpd.read_parquet(grid_path)

    # Merge all grid cell geometries into a single multipolygon
    merged_geom = unary_union(gdf["geometry"])  # Dissolve

    # Ensure it remains a MultiPolygon
    if not isinstance(merged_geom, MultiPolygon):
        merged_geom = MultiPolygon([merged_geom])

    # Instead of convex hull, create a more accurate outline
    dissolved_boundary = merged_geom.buffer(0.001).simplify(0.0005, preserve_topology=True)

    # Create a new GeoDataFrame
    urban_gdf = gpd.GeoDataFrame({"city_name": [city_name], "geometry": [dissolved_boundary]}, geometry="geometry", crs=gdf.crs)

    # Define output folder and filename
    city_output_folder = os.path.join(OUTPUT_DIR, city_name)
    os.makedirs(city_output_folder, exist_ok=True)

    output_file = os.path.join(city_output_folder, f"{city_name}_search_buffer.geoparquet")

    # Save to GeoParquet
    urban_gdf.to_parquet(output_file)
    #print(f"Saved: {output_file}")

print("Processing complete! 🚀")



#------------

gdf_all_cities[gdf_all_cities['matched_bool']][['city_name_matched','merged_geometry']].to_file('../421_cities.shp')
print(num_matched, "cities matched")


selected_cities = gdf_all_cities[gdf_all_cities['matched_bool']][['city_name_matched','merged_geometry']].set_index('city_name_matched')


import matplotlib.pyplot as plt

def s3_save(file, output_file, output_temp_path, remote_path):

    os.makedirs(output_temp_path, exist_ok=True)

    local_temporary_file = f"{output_temp_path}/{output_file}"

    # Ensure osmid is a string before saving
    if "osmid" in file.columns:
        file["osmid"] = file["osmid"].astype(str)

    # Save the file based on its extension
    if output_file.endswith(".gpkg"):
        file.to_file(local_temporary_file, driver="GPKG")
    elif output_file.endswith(".csv"):
        file.to_csv(local_temporary_file, index=False)
    elif output_file.endswith(".geoparquet"):
        file.to_parquet(local_temporary_file, index=False)
    else:
        raise ValueError("Unsupported file format. Only .gpkg, .geoparquet and .csv are supported.")

    # Upload to S3
    output_path = S3Path(remote_path)
    output_path.upload_from(local_temporary_file)

    # Delete the local file after upload
    if os.path.exists(local_temporary_file):
        os.remove(local_temporary_file)

for city_name in selected_cities.index:
    print(city_name)
    search_area_path = f"{SEARCH_BUFFER_PATH}/{city_name}/{city_name}_search_buffer.geoparquet"
    search_area_file = gpd.GeoDataFrame(geometry=gpd.GeoSeries(selected_cities.loc[city_name]['merged_geometry'])).to_parquet(search_area_path)
    s3_save(file=search_area_file, 
        output_file=f"{city_name}_search_buffer.geoparquet", 
        output_temp_path='.', 
        remote_path=f"{SEARCH_BUFFER_PATH}/{city_name}")




#----------



# Get the set of successfully matched cities
matched_cities = set(x.loc[gdf_all_cities["city_name_matched"].notna(), "city_name"])

# Find cities in city_set that were never matched -- THIS NO LONGER WORKS
unmatched_cities_from_set = city_set - matched_cities

city_set_transformed = {city for city in city_set}

df_filtered = df[df["city_name"].isin(city_set_transformed)]
df_filtered["geom"] = df_filtered["geom"].apply(wkb.loads) 
df_filtered = gpd.GeoDataFrame(df_filtered, geometry="geom", crs="EPSG:4326")  # Adjust CRS if necessary
#df_consolidated = df_filtered.dissolve(by="city_name")

# Save to GeoPackage
output_path = "../consolidated_cities_filter_384.gpkg"
df_consolidated.reset_index().to_file(output_path, driver="GPKG")

print(f"Saved consolidated geometries to {output_path}")

# Convert sets to sorted lists
unmatched_cities_1 = sorted(unmatched_cities_from_set)  # From city_set
unmatched_cities_2 = sorted(set(df["city_name"]) - matched_cities)  # From df

# Create a DataFrame with both lists
max_len = max(len(unmatched_cities_1), len(unmatched_cities_2))
df_unmatched = pd.DataFrame({
    "Unmatched in city_set": unmatched_cities_1 + [""] * (max_len - len(unmatched_cities_1)),
    "Unmatched in df": unmatched_cities_2 + [""] * (max_len - len(unmatched_cities_2))
})

# Save to Excel
output_excel = "../unmatched_cities.xlsx"
df_unmatched.to_excel(output_excel, index=False)

print(f"Saved unmatched cities list to {output_excel}")

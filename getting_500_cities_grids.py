

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

'''
# Read Parquet in batches
parquet_file = pq.ParquetFile("../combined_cities.parquet")

# Dictionary to store merged geometries per city
city_geom_dict = {}

batch_size = 100000  # Adjust based on memory

for batch in parquet_file.iter_batches(batch_size=batch_size):
    df_batch = batch.to_pandas()[["city_name", "geom"]]  # Read necessary columns

    # Convert WKB to Shapely geometries
    df_batch["geometry"] = df_batch["geom"].apply(wkb.loads)

    # Group by city and store geometries
    grouped = df_batch.groupby("city_name")["geometry"].agg(list)

    # Merge with existing data
    for city, geom_list in grouped.items():
        if city in city_geom_dict:
            city_geom_dict[city].extend(geom_list)
        else:
            city_geom_dict[city] = geom_list

# Now merge geometries for each city using unary_union
final_geometries = {city: unary_union(geom_list) for city, geom_list in city_geom_dict.items()}

# Convert to DataFrame
df_result = pd.DataFrame(list(final_geometries.items()), columns=["city_name", "merged_geometry"])
gdf_all_cities = gpd.GeoDataFrame(df_result,geometry='merged_geometry')

gdf_all_cities.to_parquet('../all_cities_combined.geoparquet')
'''

gdf_all_cities = gpd.read_parquet('../all_cities_combined.geoparquet')

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
def refined_fuzzy_match(city, city_set):
    """Matches city names by allowing underscore corrections but avoiding over-relaxed matches."""
    if city in city_set:  # Exact match
        return city
    
    # Normalize city name: remove underscores only where appropriate
    cleaned_city = normalize_city_name(city)
    
    # Find close matches allowing a max difference of 1 edit
    possible_matches = get_close_matches(cleaned_city, city_set, n=1, cutoff=0.9)  # More restrictive cutoff
    return possible_matches[0] if possible_matches else None  # Return matched city or None

# Apply refined fuzzy matching
gdf_all_cities["city_name_matched"] = gdf_all_cities["city_name"].apply(lambda x: refined_fuzzy_match(x, city_set))

# Count how many cities were successfully matched
num_matched = gdf_all_cities["city_name_matched"].notna().sum()
gdf_all_cities['matched_bool'] = gdf_all_cities["city_name_matched"].notna()
gdf_all_cities[gdf_all_cities['matched_bool']][['city_name_matched','merged_geometry']].to_file('../421_cities.shp')
print(num_matched, "cities matched")









'''



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

print(f"Saved unmatched cities list to {output_excel}")'
'''
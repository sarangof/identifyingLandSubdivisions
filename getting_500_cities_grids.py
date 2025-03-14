

import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely import wkb
from difflib import get_close_matches
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
from shapely.ops import unary_union

# Read only necessary columns
columns_to_keep = ["city_name", "geom"]

# Read the Parquet file in chunks
parquet_file = pq.ParquetFile("../combined_cities.parquet")

df_list = []
for batch in parquet_file.iter_batches(batch_size=100000):  # Increase batch size for efficiency
    df_batch = batch.to_pandas()  # Convert batch to Pandas
    df_batch = df_batch[columns_to_keep]  # Select only necessary columns
    df_list.append(df_batch)

# Concatenate batches
df = pd.concat(df_list, ignore_index=True)

# Convert WKB to geometry
df["geom"] = df["geom"].apply(wkb.loads)

# Convert to GeoDataFrame
df_filtered = gpd.GeoDataFrame(df, geometry="geom", crs="EPSG:4326")
df_filtered = df_filtered.set_geometry("geom")

# Convert Pandas GeoDataFrame to Dask GeoDataFrame
dask_gdf = dgpd.from_geopandas(df, npartitions=100)

df_dask = df_dask.set_crs("EPSG:4326", allow_override=True)

# Perform parallel dissolve
df_dissolved = df_dask.dissolve(by="city_name").compute()  # Runs in parallel

# Save the result
df_dissolved.to_parquet("../combined_500_cities_dissolved.parquet", compression="zstd")
df_dissolved["geometry"] = df_dissolved["geometry"].simplify(tolerance=0.0001, preserve_topology=True)


print(df.shape)  # Check final row count

city_names = pd.read_csv('../500cities.csv',sep=';', encoding="ISO-8859-1")

city_names["city_name_no_country"] = city_names["City Name"].apply(lambda x: x.split(',')[0].replace(" ", "_"))
city_names["city_name_no_country"] = city_names["city_name_no_country"].apply(lambda x: x.replace(" ", "_"))


city_set = set(city_names["city_name_no_country"])



# Fuzzy Matching Function: Handle "_"
def fuzzy_match(city, city_set, threshold=0.85):
    if city in city_set:  # Exact match
        return True
    cleaned_city = city.replace("_", "")  # Remove underscores
    possible_matches = get_close_matches(cleaned_city, city_set, n=1, cutoff=threshold)
    return bool(possible_matches)  # Returns True if a close match is found

# Apply fuzzy matching
df["city_name_matched"] = df["city_name"].apply(lambda x: fuzzy_match(x, city_set))

# Print the number of matches
print(df["city_name_matched"].sum(), "cities matched")

#df["city_name_matched"] = df["city_name"].map(city_set.__contains__)

# Get all matched cities from df
matched_cities = set(df.loc[df["city_name_matched"], "city_name"])

# Find cities in city_set that were never matched
unmatched_cities_from_set = city_set - matched_cities

print(len(unmatched_cities_from_set))
print(len(matched_cities))

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


import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from shapely import wkb

parquet_file = pq.ParquetFile("../combined_cities.parquet")

# Collect all chunks in a list
df_list = []

for batch in parquet_file.iter_batches(batch_size=10000):
    df_list.append(batch.to_pandas())  # Convert batch to DataFrame

# Concatenate all batches into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

print(df.shape)  # Check final row count

city_names = pd.read_csv('../500cities.csv',sep=';', encoding="ISO-8859-1")

city_names["city_name_no_country"] = city_names["City Name"].apply(lambda x:x.split(',')[0])
city_names["city_name_no_country"] = city_names["city_name_no_country"].apply(lambda x: x.replace(" ", "_"))

city_set = set(city_names["city_name_no_country"])

df["city_name_matched"] = df["city_name"].map(city_set.__contains__)

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
df_consolidated = df_filtered.dissolve(by="city_name")

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
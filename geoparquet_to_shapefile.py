import os
import geopandas as gpd

def convert_geoparquet_to_shapefile(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.geoparquet'):
                geoparquet_path = os.path.join(dirpath, filename)
                shapefile_name = filename.replace('.geoparquet', '.shp')
                shapefile_path = os.path.join(dirpath, shapefile_name)

                try:
                    gdf = gpd.read_parquet(geoparquet_path)
                    gdf.to_file(shapefile_path)
                    print(f"Converted: {geoparquet_path} -> {shapefile_path}")
                except Exception as e:
                    print(f"Failed to convert {geoparquet_path}: {e}")

# Example usage
convert_geoparquet_to_shapefile("/Users/sarangof/Documents/Identifying Land Subdivisions")

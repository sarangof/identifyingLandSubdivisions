import dask_geopandas as dgpd
import dask.dataframe as dd
import geopandas as gpd
import pyarrow.parquet as pq
import s3fs
import os
from pathlib import Path
from dask.delayed import delayed
import dask
from shapely import wkb  #  Convert geometry to WKB for S3-safe saving

fs = s3fs.S3FileSystem(anon=False)

#  Local temp folder for S3 uploads
LOCAL_TEMP = "/tmp/dask_temp/"
os.makedirs(LOCAL_TEMP, exist_ok=True)

#  Function to **read each city's `.geoparquet`** file lazily
@delayed
def read_city_geoparquet(folder_path):
    """Finds and reads the .geoparquet file inside a city folder, adding city_name."""
    city_name = Path(folder_path).name
    try:
        files = fs.ls(folder_path)
        geo_file = next(f"s3://{f}" for f in files if f.endswith(".geoparquet"))  #  Get correct S3 path
        gdf = dgpd.read_parquet(geo_file)  #  Read as Dask GeoDataFrame

        #  Ensure the geometry column is properly reconstructed
        if "geometry" in gdf.columns:
            gdf["geometry"] = gdf["geometry"].map_partitions(lambda x: x.apply(wkb.loads if isinstance(x.iloc[0], bytes) else lambda y: y))

        gdf["city_name"] = city_name
        return gdf
    except StopIteration:
        print(f"⚠️ No .geoparquet found in {folder_path}")
    except Exception as e:
        print(f"❌ Error reading {folder_path}: {e}")
    return None


from shapely import wkb  #  CORRECT
import os
from pathlib import Path

#  Function to save **Dask GeoDataFrame** as **partitioned S3 Parquet**
def s3_save_dask(gdf, output_folder):
    """Saves a Dask GeoDataFrame as Parquet to S3, converting geometry to WKB."""

    #  Debugging: Check if geometry column exists
    print(f"🔍 Checking geometry column in {output_folder}: {gdf.dtypes}")
    
    if "geometry" not in gdf.columns:
        raise ValueError("❌ Geometry column is missing before saving!")

    #  Convert `geometry` column to WKB (fixes PyArrow errors)
    gdf["geometry"] = gdf["geometry"].map_partitions(lambda x: x.apply(lambda g: wkb.dumps(g) if g else None))

    #  Save to S3 (ensuring metadata is written)
    gdf.to_parquet(
        f"{output_folder}/data.parquet",
        engine="pyarrow",
        storage_options={"anon": False},
        write_metadata_file=True  #  Ensure metadata is saved!
    )

    print(f" Successfully saved {output_folder} to S3!")

    #  Debugging: Check if files are actually in S3
    uploaded_files = fs.glob(f"{output_folder}/*")
    print(f"🔍 S3 Files in {output_folder}: {uploaded_files}")

    if not uploaded_files:
        print("⚠️ WARNING: No files detected in S3. Something went wrong.")


#  Optimized function: Process **ALL** cities in a category and save in **S3**
def process_category_dask(input_path, output_path):
    """
    Reads and merges all city geoparquet files in a category, then saves as a single combined file.

    Parameters:
        input_path (str): S3 path where the original files are stored.
        output_path (str): S3 path where the combined file should be saved.
    """
    city_folders = [f for f in fs.ls(input_path) if not f.endswith(".geoparquet")]
    all_gdfs = [read_city_geoparquet(folder) for folder in city_folders]
    
    all_gdfs = dask.compute(*all_gdfs)  #  Compute only necessary parts
    all_gdfs = [gdf for gdf in all_gdfs if gdf is not None]  # Remove failed reads

    if all_gdfs:
        #  Keep everything as a **Dask GeoDataFrame** (memory efficient)
        combined_dgdf = dd.concat(all_gdfs, ignore_index=True)

        #  Save efficiently to S3
        s3_save_dask(combined_dgdf, output_path)

    else:
        print(f"⚠️ No valid data found in {input_path}")

#  Processing raster using the same optimized approach
def process_raster_dask(input_path, output_path):
    """
    Reads and merges all raster data per city and saves a single combined file.

    Parameters:
        input_path (str): S3 path where raster files are stored.
        output_path (str): S3 path where the combined raster file should be saved.
    """
    city_folders = [f for f in fs.ls(input_path) if not f.endswith(".geoparquet")]

    @delayed
    def read_raster_geoparquet(folder):
        city_name = Path(folder).name
        grid_folder = f"s3://{folder}/{city_name}_200m_grid_sara.geoparquet/"

        try:
            raster_files = fs.ls(grid_folder)
            part_file = next(f"s3://{f}" for f in raster_files if f.endswith(".parquet"))  #  Ensure S3 path
            gdf = dgpd.read_parquet(part_file)
            gdf["city_name"] = city_name
            return gdf
        except StopIteration:
            print(f"⚠️ No raster .parquet found in {grid_folder}")
        except Exception as e:
            print(f"❌ Error processing raster for {city_name}: {e}")
        return None

    #  Read all raster files lazily
    raster_gdfs = [read_raster_geoparquet(folder) for folder in city_folders]
    raster_gdfs = dask.compute(*raster_gdfs)  #  Compute lazily
    raster_gdfs = [gdf for gdf in raster_gdfs if gdf is not None]  # Remove failed reads

    if raster_gdfs:
        #  Keep everything as a **Dask GeoDataFrame** (memory efficient)
        combined_dgdf = dd.concat(raster_gdfs, ignore_index=True)

        #  Save efficiently to S3
        s3_save_dask(combined_dgdf, output_path)
        gdf = dgpd.read_parquet(output_path)
        gdf.to_parquet(f'{output_path}/combined.geoparquet', 
                       engine="pyarrow", 
                       storage_options={"anon": False})
    else:
        print(f"⚠️ No valid raster data found in {input_path}")
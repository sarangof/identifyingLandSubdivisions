import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from metrics_calculation import *
#from create_rectangles import *
from standardize_metrics import *
import matplotlib.pyplot as plt
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from datetime import datetime
#import pyproj
from pyproj import CRS, Geod
import json
from dask import delayed, compute
import dask_geopandas as dgpd  
import dask.dataframe as dd
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler
from shapely.errors import TopologicalError
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from functools import partial
from cloudpathlib import S3Path
import s3fs


MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
OUTPUT_PATH_CSV = f'{MAIN_PATH}/output/csv'
OUTPUT_PATH_RASTER = f'{MAIN_PATH}/output/raster'
OUTPUT_PATH_PNG = f'{MAIN_PATH}/output/png'

fs = s3fs.S3FileSystem(anon=False)

# Define important parameters for this run
grid_size = 200
row_epsilon = 0.01

# Helper function to get UTM CRS based on city geometry centroid
def get_utm_crs(geometry):
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_crs = CRS.from_user_input(f"+proj=utm +zone={(int((lon + 180) // 6) + 1)} +{'south' if lat < 0 else 'north'} +datum=WGS84")
    authority_code = utm_crs.to_authority()
    if authority_code is not None:
        epsg_code = int(authority_code[1])
    else:
        epsg_code = None
    return epsg_code
    

def process_metrics(final_geo_df):
    all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']

    # Save original values before transformations
    metrics_original_names = [col+'_original' for col in all_metrics_columns]
    final_geo_df[metrics_original_names] = final_geo_df[all_metrics_columns].copy()

    metrics_standardized_names = {col:col+'_standardized' for col in all_metrics_columns}

    # Apply the standardization functions
    for metric, func in standardization_functions.items():
        final_geo_df[metrics_standardized_names[metric]] = func(final_geo_df[metric])

    zero_centered_names_list = [col+'_zero-centered' for col in all_metrics_columns]
    final_geo_df[zero_centered_names_list] = final_geo_df[list(metrics_standardized_names.values())].copy()

    # Center at zero and maximize information
    final_geo_df.loc[:, zero_centered_names_list] = (
        final_geo_df.loc[:,zero_centered_names_list]
        .apply(lambda x: (x - x.mean()) / (x.std()))
    )

    final_geo_df[all_metrics_columns] = final_geo_df[zero_centered_names_list].copy()

    # Center at zero and maximize information
    final_geo_df.loc[:, all_metrics_columns] = (
        final_geo_df[all_metrics_columns]
        .apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    )

    # Calculate equal-weights irregularity index
    final_geo_df['regularity_index'] = final_geo_df[all_metrics_columns].mean(axis=0)

    return final_geo_df

def s3_save(file, output_file, output_temp_path, remote_path):

    os.makedirs(output_temp_path, exist_ok=True)

    local_temporary_file = f"{output_temp_path}/{output_file}"
    # Save the file based on its extension
    if output_file.endswith(".gpkg"):
        file.to_file(local_temporary_file, driver="GPKG")
    elif output_file.endswith(".csv"):
        file.to_csv(local_temporary_file, index=False)
    elif output_file.endswith(".geoparquet"):
        file.to_parquet(local_temporary_file, engine="pyarrow", index=False)
    else:
        raise ValueError(f"Unsupported file format. Only .gpkg and .csv are supported but we got {file}.")

    # Upload to S3
    output_path = S3Path(remote_path)
    output_path.upload_from(local_temporary_file)

    # Delete the local file after upload
    if os.path.exists(local_temporary_file):
        os.remove(local_temporary_file)

def save_city_grid_results(city_grid, sampled_grid, output_dir_csv):
    """
    Saves city_grid to results.csv in output_dir_csv.
    If results.csv exists, it updates 'processed' values for rows present in sampled_grid.
    Adds a timestamp to track when updates happen.
    """
    results_path = os.path.join(output_dir_csv, "results.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure sampled_grid has the same index (id) as city_grid
    sampled_ids = sampled_grid["grid_id"].unique()

    # Add timestamp column to track updates
    city_grid["timestamp"] = pd.NA  # Set as missing initially

    # Set 'processed' to True for rows in sampled_grid
    city_grid.loc[city_grid["grid_id"].isin(sampled_ids), "processed"] = True

    # Set timestamp only for rows that were newly processed
    city_grid.loc[city_grid["grid_id"].isin(sampled_ids), "timestamp"] = timestamp

    if fs.exists(results_path):
        # Load existing results
        existing_results = pd.read_csv(results_path)

        # Ensure necessary columns exist in existing results
        if "processed" not in existing_results.columns:
            existing_results["processed"] = False
        if "timestamp" not in existing_results.columns:
            existing_results["timestamp"] = pd.NA

        # Merge new results, updating only 'processed' and 'timestamp' where needed
        updated_results = existing_results.set_index("grid_id").combine_first(city_grid.set_index("grid_id")).reset_index()

        # Explicitly update 'processed' and 'timestamp' based on sampled_grid
        updated_results.loc[updated_results["grid_id"].isin(sampled_ids), "processed"] = True
        updated_results.loc[updated_results["grid_id"].isin(sampled_ids), "timestamp"] = timestamp

    else:
        # No existing results, just save city_grid as new results
        updated_results = city_grid

    # Save the updated results
    updated_results.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")

def save_metric_maps(city_grid, output_dir_png):
    """
    Generates a matrix of geographic maps for all metrics and the regularity index,
    and saves each map separately.
    """
    os.makedirs(output_dir_png, exist_ok=True)  # Ensure output directory exists

    # Define the metrics and regularity index
    all_metrics_columns = [
        'metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5',
        'metric_6', 'metric_7', 'metric_8', 'metric_9', 'metric_10',
        'metric_11', 'metric_12', 'metric_13'
    ]
    plot_columns = all_metrics_columns + ['regularity_index']

    # Ensure city_grid is a valid GeoDataFrame and has a geometry column
    if not isinstance(city_grid, gpd.GeoDataFrame) or 'geometry' not in city_grid.columns:
        raise ValueError("city_grid must be a GeoDataFrame with a 'geometry' column.")

    # Ensure city_grid has a valid CRS (coordinate reference system)
    if city_grid.crs is None:
        raise ValueError("GeoDataFrame must have a valid CRS. Use city_grid.set_crs('EPSG:XXXX', inplace=True) to set it.")

    # Determine consistent color scale across all maps
    vmin = city_grid[plot_columns].min().min()
    vmax = city_grid[plot_columns].max().max()

    # Create a grid of plots (adjust rows & cols for number of metrics)
    num_cols = 4  # Define how many columns in the figure matrix
    num_rows = (len(plot_columns) + num_cols - 1) // num_cols  # Compute necessary rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()  # Convert axes to a 1D list for easy iteration

    # **1. Generate matrix of spatial maps**
    for i, metric in enumerate(plot_columns):
        ax = axes[i]
        city_grid.plot(column=metric, cmap='Reds', linewidth=0.5, ax=ax, edgecolor='black',
                       legend=True, vmin=vmin, vmax=vmax)
        ax.set_title(metric)
        ax.axis("off")  # Hide axis labels

    # Hide any extra subplots if the grid is larger than necessary
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Save the matrix of maps
    matrix_plot_path = os.path.join(output_dir_png, "metrics_map_matrix.png")
    plt.tight_layout()
    plt.savefig(matrix_plot_path, dpi=300)
    plt.close()
    print(f"Saved matrix plot to {matrix_plot_path}")

    # **2. Generate and save individual maps**
    for metric in plot_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        city_grid.plot(column=metric, cmap='viridis', linewidth=0.5, ax=ax, edgecolor='black',
                       legend=True, vmin=vmin, vmax=vmax)
        ax.set_title(metric)
        ax.axis("off")

        # Save individual map
        metric_plot_path = os.path.join(output_dir_png, f"{metric}_map.png")
        plt.savefig(metric_plot_path, dpi=300)
        plt.close()
        print(f"Saved {metric} map to {metric_plot_path}")

    print("All spatial maps have been saved.")

def output_results(city_grid, sampled_grid, city_name, grid_size, sample_prop, OUTPUT_PATH_RASTER, output_dir_csv, output_dir_png):
        #Save raste results to geoparquet
        os.makedirs(f'{OUTPUT_PATH_RASTER}/{city_name}', exist_ok=True)

        output_file = f'{city_name}_{str(grid_size)}m_results.geoparquet'
        remote_path = f'{OUTPUT_PATH_RASTER}/{city_name}/'
        output_temp_path = '.'
        s3_save(city_grid, output_file, output_temp_path, remote_path)

        #city_grid.to_parquet(f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_results.geoparquet', engine="pyarrow", index=False)

        # Save summaries
        all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']
        metrics_standardized_names = [col+'_standardized' for col in all_metrics_columns]
        zero_centered_names_list = [col+'_zero-centered' for col in all_metrics_columns]
        city_grid[all_metrics_columns+metrics_standardized_names+zero_centered_names_list].describe().transpose().to_excel(f'{output_dir_csv}/summary_prop={str(sample_prop)}.xlsx')

        # Save raw data
        save_city_grid_results(city_grid, sampled_grid, output_dir_csv)

        # Save PNG files
        save_metric_maps(city_grid, output_dir_png)

        output_dir_csv

        



@delayed
def process_cell(cell_id, geod, rectangle, rectangle_projected, buildings, blocks_all, 
                 OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city):

    """
    Processes a single cell using Dask Delayed.
    """

    try:
        # Ensure result is always initialized
        result = None

        bounding_box = rectangle_projected.bounds
        bounding_box_geom = box(*bounding_box)
        rectangle_area, _ = geod.geometry_area_perimeter(rectangle)

        if rectangle_area > 0: 
            # Preparatory calculations
            if not buildings.empty and buildings.sindex:
                possible_matches_index = list(buildings.sindex.intersection(bounding_box_geom.bounds))
                possible_matches = buildings.iloc[possible_matches_index]
                buildings_clipped = gpd.clip(possible_matches, bounding_box_geom)
                buildings_clipped = buildings_clipped[
                    (buildings_clipped['confidence'] > 0.75) | buildings_clipped['confidence'].isna()
                ].reset_index(drop=True)

                building_area = buildings_clipped.area.sum()
                n_buildings = len(buildings_clipped)
                building_density = (1000.0 * 1000 * n_buildings) / rectangle_area if rectangle_area > 0 else np.nan

            else:
                buildings_clipped = gpd.GeoDataFrame([])
                building_area, building_density, n_buildings = np.nan, np.nan, np.nan

            # Clip roads
            try:
                roads_clipped = OSM_roads_all_projected[
                    OSM_roads_all_projected.geometry.intersects(bounding_box_geom)
                ]
                OSM_roads_bool = not roads_clipped.empty
            except (fiona.errors.DriverError, TopologicalError) as e:
                print(f"Error clipping roads for cell {cell_id}: {e}")
                roads_clipped = gpd.GeoDataFrame([])
                OSM_roads_bool = False

            # Clip intersections
            try:
                OSM_intersections = OSM_intersections_all_projected[
                    OSM_intersections_all_projected.geometry.intersects(bounding_box_geom)
                ]
                OSM_intersections_bool = not OSM_intersections.empty
                n_intersections = len(OSM_intersections.drop_duplicates('osmid'))
            except fiona.errors.DriverError:
                OSM_intersections = gpd.GeoDataFrame([])
                OSM_intersections_bool = False
                n_intersections = np.nan

            # If NO roads and NO intersections → return NaNs
            if not OSM_roads_bool and not OSM_intersections_bool:
                return {
                    'index': cell_id, 'metric_1': np.nan, 'metric_2': np.nan, 'metric_3': np.nan, 'metric_4': np.nan,
                    'metric_5': np.nan, 'metric_6': np.nan, 'metric_7': np.nan, 'metric_8': np.nan,
                    'metric_9': np.nan, 'metric_10': np.nan, 'metric_11': np.nan, 'metric_12': np.nan, 'metric_13': np.nan,
                    'OSM_buildings_available': np.nan, 'OSM_intersections_available': np.nan,
                    'OSM_roads_available': np.nan, 'rectangle_area': rectangle_area,
                    'building_area': np.nan, 'share_tiled_by_blocks': np.nan,
                    'road_length': np.nan, 'n_intersections': np.nan, 'n_buildings': np.nan,
                    'building_density': np.nan
                }

            # Otherwise, proceed with normal metric calculations
            if not buildings_clipped.empty and not roads_clipped.empty:
                m1, buildings_clipped = metric_1_distance_less_than_20m(buildings_clipped, road_union, utm_proj_city)
                m2 = metric_2_average_distance_to_roads(buildings_clipped)
            else:
                m1, m2 = np.nan, np.nan

            m3 = metric_3_road_density(rectangle_area, roads_clipped) if not roads_clipped.empty else 0

            if not OSM_intersections.empty:
                m4 = metric_4_share_4way_intersections(OSM_intersections)
                m5 = metric_5_intersection_density(OSM_intersections, rectangle_area)
            else:
                m4, m5 = (np.nan if not roads_clipped.empty else 0), 0

            m6 = (
                metric_6_entropy_of_building_azimuth(buildings_clipped, rectangle_id=1, bin_width_degrees=5, plot=False)[0]
                if not buildings_clipped.empty else np.nan
            )

            if not blocks_all.empty:
                minx, miny, maxx, maxy = rectangle_projected.bounds
                rectangle_box = box(minx, miny, maxx, maxy)
                blocks_clipped_within_rectangle = blocks_all.clip(rectangle_box)

                area_tiled_by_blocks = blocks_clipped_within_rectangle.area.sum()
                share_tiled_by_blocks = area_tiled_by_blocks / rectangle_area

                m7, blocks_clipped = metric_7_average_block_width(blocks_all, blocks_clipped_within_rectangle, rectangle_projected, rectangle_area)
                m8, _, _ = metric_8_two_row_blocks(blocks_all, buildings_clipped, utm_proj_city, row_epsilon=row_epsilon)
            else:
                m7, m8, share_tiled_by_blocks = np.nan, np.nan, 0

            m9 = metric_9_tortuosity_index(roads_clipped) if not roads_clipped.empty else np.nan
            m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads_clipped) if not roads_clipped.empty and not OSM_intersections.empty else np.nan

            road_length = roads_clipped.length.sum() if not roads_clipped.empty else np.nan

            if not buildings_clipped.empty:
                m11 = metric_11_building_density(n_buildings, rectangle_area)
                m12 = metric_12_built_area_share(building_area, rectangle_area)
                m13 = metric_13_average_building_area(building_area, n_buildings)
            else:
                m11, m12, m13 = 0, 0, np.nan

            # Final result
            result = {
                'index': cell_id, 'metric_1': m1, 'metric_2': m2, 'metric_3': m3, 'metric_4': m4,
                'metric_5': m5, 'metric_6': m6, 'metric_7': m7, 'metric_8': m8,
                'metric_9': m9, 'metric_10': m10, 'metric_11': m11, 'metric_12': m12, 'metric_13': m13,
                'OSM_buildings_available': not buildings_clipped.empty,
                'OSM_intersections_available': OSM_intersections_bool,
                'OSM_roads_available': OSM_roads_bool,
                'rectangle_area': rectangle_area, 'building_area': building_area,
                'share_tiled_by_blocks': share_tiled_by_blocks,
                'road_length': road_length, 'n_intersections': n_intersections,
                'n_buildings': n_buildings, 'building_density': building_density
            }
            result_df = pd.DataFrame([result])

        return result_df

    except Exception as e:
        print(f"Error processing cell {cell_id}: {e}")
        return None  # Avoid returning invalid rows on errors
    

def extract_confidence_and_dataset(df):
    """Extract confidence and dataset from the sources column."""
    
    def extract_first_value(x, key):
        """Extract the key from the first dictionary inside an array."""
        if isinstance(x, np.ndarray) and len(x) > 0:  # Check if it's a NumPy array
            first_entry = x[0]  # Get the first dictionary
            if isinstance(first_entry, dict) and key in first_entry:
                return first_entry[key]
        return np.nan  # Return NaN if missing or incorrect format

    df["confidence"] = df["sources"].apply(lambda x: extract_first_value(x, "confidence"))
    df["dataset"] = df["sources"].apply(lambda x: extract_first_value(x, "dataset"))
    
    return df

def load_buildings(city_name):
    path = f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet'

    if not fs.exists(path):
        print(f"Missing buildings data for city {city_name}. Skipping.")
        return None  

    # Load as Dask GeoDataFrame
    Overture_data_all = dd.read_parquet(path)

    # Get existing metadata to prevent mismatches
    meta = Overture_data_all._meta.copy()

    # **Ensure metadata includes confidence and dataset**
    meta["confidence"] = "float64"
    meta["dataset"] = "object"

    # Ensure 'sources' exists before processing
    if "sources" not in Overture_data_all.columns:
        print(f"⚠️ Warning: 'sources' column missing in {city_name}, skipping extraction.")
        return Overture_data_all  # Return without transformation

    # Apply transformation using map_partitions while preserving all original columns
    def safe_extract(df):
        df = extract_confidence_and_dataset(df)
        return df  # Ensure function does not remove existing columns

    Overture_data_all = Overture_data_all.map_partitions(safe_extract, meta=meta)

    # **Check if 'dataset' column exists**
    print(f"🔍 Columns after extraction in {city_name}: {Overture_data_all.columns}")

    if "dataset" not in Overture_data_all.columns:
        print(f"⚠️ Warning: 'dataset' column missing in {city_name}! Skipping filter.")
    else:
        # Filter out OpenStreetMap entries if the column exists
        Overture_data_all = Overture_data_all[Overture_data_all["dataset"] != "OpenStreetMap"]

    # Persist the modified DataFrame
    Overture_data_all = Overture_data_all.persist()

    print(f"✅ {city_name}: Successfully loaded Overture buildings.")
    return Overture_data_all

def load_intersections(city_name):
    path = f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.geoparquet'

    if not fs.exists(path):
        print(f"Missing intersections data for city {city_name}. Skipping.")
        return None

    try:
        # Check file extension before reading
        if path.endswith(".gpkg"):
            return gpd.read_file(path)  # Use geopandas for GPKG
        elif path.endswith(".geoparquet"):
            return dgpd.read_parquet(path).persist()  # Use Dask for Parquet
        else:
            raise ValueError(f"Unsupported file format for {path}")

    except Exception as e:
        print(f"Error loading intersections data for {city_name}: {e}")
        return None
 

def load_roads(city_name):
    path_parquet = f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet'
    file_exists_parquet = fs.exists(path_parquet)

    print(f"🛠️ Checking roads file for {city_name}:")
    print(f"   - Parquet file exists: {file_exists_parquet}")

    if file_exists_parquet:
        try:
            print(f"📂 Found Parquet roads data for {city_name}. Loading...")
            roads = dgpd.read_parquet(path_parquet).persist()
        except Exception as e:
            print(f"❌ Error loading Parquet roads data for {city_name}: {e}")
            roads = None
    else:
        print(f"⚠️ No roads data found for {city_name}. Skipping.")
        return None

    # **✅ Compute a small part of the DataFrame to check if it's empty**
    if roads is not None:
        sample = roads.head(1)  # Load only the first row
        if sample.empty:
            print(f"⚠️ Roads data for {city_name} is empty after loading.")
            return None

        print(f"✅ Successfully loaded roads data for {city_name}")
        print(f"   - Columns: {list(roads.columns)}")
        print(f"   - Number of rows (approximate): {len(roads)}")
        print(f"   - CRS: {roads.crs}")

    return roads



@delayed
def project_and_process(buildings, roads, intersections):    
    # **Check if any data is missing**
    if buildings is None:
        print("⚠️ No buildings data available. Skipping buildings projection.")
        Overture_data_all_projected = None
    else:
        print(f"✅ Buildings data loaded with {len(buildings)} records.")
    
    if roads is None:
        print("⚠️ No roads data available. Skipping roads projection.")
        return None  # No point in continuing without roads

    if intersections is None:
        print("⚠️ No intersections data available. Skipping intersections projection.")
        OSM_intersections_all_projected = gpd.GeoDataFrame(columns=["geometry"])  # Empty GeoDataFrame

    # Get UTM projection for the city
    first_row = roads.head(1).iloc[0]  # Convert to Pandas
    utm_proj_city = get_utm_crs(first_row.geometry)  # Now access correctly

    if utm_proj_city is None:
        print("❌ Error: Unable to determine EPSG code for city. Skipping projection.")
        return None  # Fail early if we can't determine the UTM CRS

    try:
        OSM_roads_all_projected = roads.to_crs(epsg=utm_proj_city)
        OSM_intersections_all_projected = intersections.to_crs(epsg=utm_proj_city) if intersections is not None else None
        Overture_data_all_projected = buildings.to_crs(epsg=utm_proj_city) if buildings is not None else None
    except Exception as e:
        print(f"❌ Error reprojecting data for city: {e}")
        return None

    road_union = OSM_roads_all_projected.unary_union

    if not OSM_roads_all_projected.empty:
        blocks = get_blocks(road_union, OSM_roads_all_projected)
    else:
        blocks = gpd.GeoDataFrame([])

    # **Debugging: Print What is Being Returned**
    print(f"📦 Returning from project_and_process() for city:")
    print(f"   - Overture: {type(Overture_data_all_projected)}")
    print(f"   - Blocks: {type(blocks)}")
    print(f"   - Roads: {type(OSM_roads_all_projected)}")
    print(f"   - Intersections: {type(OSM_intersections_all_projected)}")
    print(f"   - Road union: {type(road_union)}")
    print(f"   - UTM Projection: {utm_proj_city}")

    # **Ensure all dictionary keys exist**
    result = {
        "overture": Overture_data_all_projected if Overture_data_all_projected is not None else gpd.GeoDataFrame([]),
        "blocks": blocks,
        "roads": OSM_roads_all_projected,
        "intersections": OSM_intersections_all_projected if OSM_intersections_all_projected is not None else gpd.GeoDataFrame([]),
        "road_union": road_union,
        "utm_proj": utm_proj_city
    }

    # **Check if "overture" exists before returning**
    if "overture" not in result:
        raise KeyError("🚨 'overture' key is missing from project_and_process() return value!")

    return result


@delayed
def process_city(city_name, city_data, sample_prop=1.0, override_processed=False, grid_size=200):
    try:
        # Define metric column names
        all_metrics_columns = [
            'metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5',
            'metric_6', 'metric_7', 'metric_8', 'metric_9', 'metric_10',
            'metric_11', 'metric_12', 'metric_13'
        ]
        
        # Read city grid
        city_grid = gpd.read_parquet(f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet').reset_index()
        city_grid.rename(columns={'index': 'grid_id'}, inplace=True)

        if city_grid.empty or 'geometry' not in city_grid.columns:
            print(f"No grid cells available for {city_name}. Skipping.")
            return

        # Initialize 'processed' column
        city_grid['processed'] = city_grid.get('processed', False)

        # **Filter Unprocessed Cells & Apply Sampling**
        unprocessed_grid = city_grid[~city_grid['processed']]
        sampled_grid = unprocessed_grid.sample(frac=sample_prop, random_state=42) if sample_prop < 1.0 else unprocessed_grid

        if sampled_grid.empty:
            print(f"Skipping {city_name}: No unprocessed cells left after sampling.")
            return
        
        print(f"🧐 Debug city_data before persisting: {list(city_data.keys())}")

        # Compute the delayed dictionary so we get actual objects
        city_data_computed = compute(city_data)[0]  # Extract the actual dictionary
        print(f"✅ Computed city_data: {list(city_data_computed.keys())}")

        # Only persist if the value is a Dask GeoDataFrame
        city_data_persisted = {
            key: val.persist() if isinstance(val, dgpd.GeoDataFrame) else val
            for key, val in city_data_computed.items()
        }
        print(f"🔍 Debug city_data_persisted: {list(city_data_persisted.keys())}")

        
        # Extract processed city data
        Overture_data_all_projected = city_data_persisted["overture"]
        blocks = city_data_persisted["blocks"]
        OSM_roads_all_projected = city_data_persisted["roads"]
        OSM_intersections_all_projected = city_data_persisted["intersections"]
        road_union = city_data_persisted["road_union"]
        utm_proj_city = city_data_persisted["utm_proj"]

        # **Check if any object is still a Dask GeoDataFrame and needs computation**
        if isinstance(Overture_data_all_projected, dgpd.GeoDataFrame):
            Overture_data_all_projected = Overture_data_all_projected.compute()
        if isinstance(blocks, dgpd.GeoDataFrame):
            blocks = blocks.compute()
        if isinstance(OSM_roads_all_projected, dgpd.GeoDataFrame):
            OSM_roads_all_projected = OSM_roads_all_projected.compute()
        if isinstance(OSM_intersections_all_projected, dgpd.GeoDataFrame):
            OSM_intersections_all_projected = OSM_intersections_all_projected.compute()

        print(f"✅ Final extracted city data: {type(Overture_data_all_projected)}, {type(blocks)}, {type(OSM_roads_all_projected)}, {type(OSM_intersections_all_projected)}")


        # **Transform grid geometries into UTM CRS**
        geod = Geod(ellps="WGS84")
        sampled_grid["geometry_projected"] = sampled_grid["geometry"].to_crs(epsg=utm_proj_city)

        # **Parallelize Cell Processing**
        delayed_results = []
        for cell_id, rectangle_projected in zip(sampled_grid['grid_id'], sampled_grid['geometry_projected']):
            rectangle = sampled_grid.loc[sampled_grid['grid_id'] == cell_id, 'geometry'].iloc[0]  # FIXED

            delayed_results.append(
                delayed(process_cell)(
                    cell_id, geod, rectangle, rectangle_projected,
                    Overture_data_all_projected, blocks,
                    OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city
                )
            )

        # Compute results
        #final_geo_df = compute(*delayed_results)
        # Compute results
        print(f"Number of delayed process_cell tasks: {len(delayed_results)}")
        delayed_results[0].visualize(filename="process_cell_graph_delayed_results.svg", format="svg")

        sample_task = delayed_results[0].compute()
        print("sample_task")
        print(sample_task)
        # Do not eagerly persist this DataFrame yet
        final_geo_df = dd.from_delayed(delayed_results, meta=pd.DataFrame(columns=['index'] + all_metrics_columns))

        # Only persist *after* the dataframe has been created
        final_geo_df = final_geo_df.persist()

        # Convert to Pandas (but this might still be large)
        final_geo_df = final_geo_df.compute()

        # Process the metrics
        final_geo_df = process_metrics(final_geo_df)

        print("Final DataFrame columns:", final_geo_df.columns)
        print(final_geo_df.head())

        # Merge results back into city_grid
        final_geo_df['index'] = sampled_grid['grid_id'].values
        city_grid = city_grid.merge(final_geo_df, how='left', left_on='grid_id', right_on='index')

        # Save outputs
        output_dir_csv = f'{OUTPUT_PATH_CSV}/{city_name}'
        output_dir_png = f'{OUTPUT_PATH_PNG}/{city_name}'
        output_results(city_grid, sampled_grid, city_name, grid_size, sample_prop, OUTPUT_PATH_RASTER, output_dir_csv, output_dir_png)

        # Save updated grid status to CSV
        s3_save(city_grid[['grid_id', 'processed']], f'{city_name}_grid_status_{grid_size}m.csv', '.', output_dir_csv)

        print(f"{city_name}: Processing complete.")

    except Exception as e:
        print(f"Error processing {city_name}: {e}")
        raise


def run_all_citywide_calculation(cities, sample_prop=0.05, grid_size=200):
    tasks = []
    for city in cities:
        buildings = load_buildings(city)
        roads = load_roads(city)
        intersections = load_intersections(city)
        city_data = delayed(project_and_process)(buildings, roads, intersections)

        # Process city in parallel
        task = process_city(city, city_data, sample_prop=sample_prop, grid_size=grid_size)
        tasks.append(task)
    tasks[0].visualize(filename="dask_graph.svg", format="svg")
    with Profiler() as prof, ResourceProfiler(dt=1) as rprof, ProgressBar():
        #compute(*tasks)
        compute(*tasks, timeout=1200)
    prof.visualize(filename="profiler_graph.svg")


# This function allows external cluster calls like Code C.
if __name__ == "__main__":
    cities = ["Belo_Horizonte", "Campinas", "Bogota"]
    run_all_citywide_calculation(cities)
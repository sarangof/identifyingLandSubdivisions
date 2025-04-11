import os
import s3fs
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from cloudpathlib import S3Path
from standardize_metrics import *
import dask.dataframe as dd

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f'{MAIN_PATH}/input'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
OUTPUT_PATH = f'{MAIN_PATH}/output'
OUTPUT_PATH_CSV = f'{MAIN_PATH}/output/csv'
OUTPUT_PATH_RASTER = f'{MAIN_PATH}/output/raster'
OUTPUT_PATH_PNG = f'{MAIN_PATH}/output/png'
OUTPUT_PATH_RAW = f'{OUTPUT_PATH}/raw_results'

fs = s3fs.S3FileSystem(anon=False)

def s3_save(file, output_file, output_temp_path, remote_path):
    """
    Saves files to the S3 filesystem.
    """

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
        raise ValueError(f"Unsupported file format. Only .gpkg, .geoparquet and .csv are supported but we got {file}.")

    # Upload to S3
    output_path = S3Path(remote_path)
    output_path.upload_from(local_temporary_file)
    print(f"✅ Saved to {output_path}")

    # Delete the local file after upload
    if os.path.exists(local_temporary_file):
        os.remove(local_temporary_file)

def merge_parquet_parts(city_name, grid_size=200):
    raw_output_path = f"{OUTPUT_PATH}/{city_name}/raw_results_{grid_size}"
    output_file = f"{OUTPUT_PATH_RASTER}/{city_name}_final.parquet"

    # List all Parquet part files
    part_files = sorted([f for f in fs.ls(raw_output_path) if f.endswith(".parquet")])


    if not part_files:
        print(f"No part files found for {city_name}. Skipping...")
        return None

    print(f"Merging {len(part_files)} files for {city_name}...")

    # Read with Dask for memory efficiency
    city_raster = dd.read_parquet(part_files, engine="pyarrow")

    output_file = f'{city_name}_{str(grid_size)}m_results.geoparquet'
    remote_path = f'{OUTPUT_PATH_RASTER}/{city_name}/'
    output_temp_path = '.'
    s3_save(city_raster, output_file, output_temp_path, remote_path)

    print(f"Unified file saved: {output_file}")

    # Optional: Remove part files after merging
    for file in part_files:
        os.remove(file)
    
    print(f"Removed {len(part_files)} part files for {city_name}")

    return city_raster

def save_summary(post_processed_results, city_name, grid_size):
    """
    Saves summaries of each city for all metrics.
    """
    # Save summaries
    all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']
    metrics_standardized_names = [col+'_standardized' for col in all_metrics_columns]
    zero_centered_names_list = [col+'_zero-centered' for col in all_metrics_columns]
    summary_file = post_processed_results[all_metrics_columns+metrics_standardized_names+zero_centered_names_list].describe().transpose()

    output_file = f'summary_prop_{str(grid_size)}.xlsx'
    remote_path = f'{OUTPUT_PATH_CSV}/{city_name}/{output_file}'
    output_temp_path = '.'

    s3_save(summary_file, output_file, output_temp_path, remote_path)


    print(f"✅ Successfully processed and saved {city_name} to {OUTPUT_PATH_CSV}")

def save_metric_maps(post_processed_results, city_name, grid_size):
    """
    Generates a matrix of geographic maps for all metrics and the regularity index,
    and saves each map separately.
    """
    # Define the metrics and regularity index
    all_metrics_columns = [
        'metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5',
        'metric_6', 'metric_7', 'metric_8', 'metric_9', 'metric_10',
        'metric_11', 'metric_12', 'metric_13'
    ]
    plot_columns = all_metrics_columns + ['regularity_index']

    # Ensure city_grid is a valid GeoDataFrame and has a geometry column
    if not isinstance(post_processed_results, gpd.GeoDataFrame) or 'geometry' not in post_processed_results.columns:
        raise ValueError("city_grid must be a GeoDataFrame with a 'geometry' column.")

    # Ensure city_grid has a valid CRS (coordinate reference system)
    if post_processed_results.crs is None:
        raise ValueError("GeoDataFrame must have a valid CRS. Use city_grid.set_crs('EPSG:XXXX', inplace=True) to set it.")

    # Determine consistent color scale across all maps
    vmin = post_processed_results[plot_columns].min().min()
    vmax = post_processed_results[plot_columns].max().max()

    # Create a grid of plots (adjust rows & cols for number of metrics)
    num_cols = 4  # Define how many columns in the figure matrix
    num_rows = (len(plot_columns) + num_cols - 1) // num_cols  # Compute necessary rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()  # Convert axes to a 1D list for easy iteration

    # **1. Generate matrix of spatial maps**
    for i, metric in enumerate(plot_columns):
        ax = axes[i]
        post_processed_results.plot(column=metric, cmap='Reds', linewidth=0.5, ax=ax, edgecolor='black',
                    legend=True, vmin=vmin, vmax=vmax)
        ax.set_title(metric)
        ax.axis("off")  # Hide axis labels

    # Hide any extra subplots if the grid is larger than necessary
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Save the matrix of maps
    output_dir_png = f'{OUTPUT_PATH_PNG}/{city_name}'
    matrix_plot_path = matrix_plot_path = f"{output_dir_png}/metrics_map_matrix.png"
    plt.tight_layout()
    plt.savefig(matrix_plot_path, dpi=300)
    plt.close()
    print(f"Saved matrix plot to {matrix_plot_path}")

    # **2. Generate and save individual maps**
    for metric in plot_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        post_processed_results.plot(column=metric, cmap='viridis', linewidth=0.5, ax=ax, edgecolor='black',
                    legend=True, vmin=vmin, vmax=vmax)
        ax.set_title(metric)
        ax.axis("off")

        # Save individual map
        metric_plot_path = os.path.join(output_dir_png, f"{metric}_map.png")
        plt.savefig(metric_plot_path, dpi=300)
        plt.close()
        print(f"Saved {metric} map to {metric_plot_path}")

    print("All spatial maps have been saved.")

def post_process_metrics(final_geo_df):
    """
    Standardize, zero-center, normalize and produce regularity index.
    """
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

def post_process_cities(city_name, grid_size):

    merged_raw_output = merge_parquet_parts(city_name, grid_size)

    if merged_raw_output is not None:
        # Standardize and normalize metrics
        post_processed_results = post_process_metrics(merged_raw_output)
        # Save summary
        save_summary(post_processed_results, city_name, grid_size)  
        # Save PNG files
        save_metric_maps(post_processed_results, city_name, grid_size)
        return True
    else: 
        print("❌ No output from merge_parquet_parts, skipping further processing.")
        return False

if __name__ == "__main__":
    cities = ["Belo_Horizonte", "Campinas", "Bogota"]
    grid_size = 200
    for city_name in cities:
        post_process_cities(city_name, grid_size)

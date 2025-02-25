import os
import s3fs
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from cloudpathlib import S3Path
from standardize_metrics import *

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

fs = s3fs.S3FileSystem(anon=False)

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
        raise ValueError(f"Unsupported file format. Only .gpkg and .csv are supported but we got {file}.")

    # Upload to S3
    output_path = S3Path(remote_path)
    output_path.upload_from(local_temporary_file)
    print(f"✅ Saved to {output_path}")

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

def save_summary(city_grid, sample_prop):
    """
    Saves summaries of each city for all metrics.
    """
    # Save summaries
    all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']
    metrics_standardized_names = [col+'_standardized' for col in all_metrics_columns]
    zero_centered_names_list = [col+'_zero-centered' for col in all_metrics_columns]
    city_grid[all_metrics_columns+metrics_standardized_names+zero_centered_names_list].describe().transpose().to_excel(f'{output_dir_csv}/summary_prop={str(sample_prop)}.xlsx')

def output_results(city_grid, sampled_grid, city_name, grid_size, sample_prop, OUTPUT_PATH_RASTER, output_dir_csv, output_dir_png):
    """
    Saves the raster results file, saves the summary results
    """
    '''
            city_grid = 
        sampled_grid = 
        grid_size = 
        sample_prop = 
    '''
    
    #Save raste results to geoparquet
    os.makedirs(f'{OUTPUT_PATH_RASTER}/{city_name}', exist_ok=True)

    output_file = f'{city_name}_{str(grid_size)}m_results.geoparquet'
    remote_path = f'{OUTPUT_PATH_RASTER}/{city_name}/'
    output_temp_path = '.'
    s3_save(city_grid, output_file, output_temp_path, remote_path)

    #city_grid.to_parquet(f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_results.geoparquet', engine="pyarrow", index=False)  

    # Save raster data from raw data
    output_dir_csv = f'{OUTPUT_PATH_CSV}/{city_name}'
    save_city_grid_results(city_grid, sampled_grid, output_dir_csv)

    # Save summary
    save_summary(city_grid, sample_prop)  

    # Save PNG files
    save_metric_maps(city_grid, output_dir_png)

    output_dir_csv

def post_process_cities(cities):
    for city_name in cities:
        remote_path = f"{OUTPUT_PATH}/{city_name}/raw_results"
        output_results(city_name, grid_size, sample_prop)

if __name__ == "__main__":
    cities = ["Belo_Horizonte", "Campinas", "Bogota"]
    post_process_cities(cities)

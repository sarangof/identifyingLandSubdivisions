import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from metrics_calculation import *
#from create_rectangles import *
from standardize_metrics import *
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import box
#import pyproj
from pyproj import CRS, Geod
import json
from dask import delayed, compute
from shapely.errors import TopologicalError
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from functools import partial


MAIN_PATH = '../data'
INPUT_PATH = f'{MAIN_PATH}/input'
BUILDINGS_PATH = f'{INPUT_PATH}/buildings'
ROADS_PATH = f'{INPUT_PATH}/roads'
INTERSECTIONS_PATH = f'{INPUT_PATH}/intersections'
GRIDS_PATH = f'{INPUT_PATH}/city_info/grids'
OUTPUT_PATH_CSV = f'{MAIN_PATH}/output/csv'
OUTPUT_PATH_RASTER = f'{MAIN_PATH}/output/raster'

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



def process_cell(cell_id, geod, rectangle, rectangle_projected, buildings, blocks_all, OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city):
    print(f"cell_id: {cell_id}")

    bounding_box = rectangle_projected.bounds
    bounding_box_geom = box(*bounding_box)
    rectangle_area, _ = geod.geometry_area_perimeter(rectangle)

    if rectangle_area > 0: 

        # Preparatory calculations
        if not buildings.empty and buildings.sindex:
            possible_matches_index = list(buildings.sindex.intersection(bounding_box_geom.bounds))
            possible_matches = buildings.iloc[possible_matches_index]
            buildings_clipped = gpd.clip(possible_matches, bounding_box_geom)
            buildings_clipped = buildings_clipped[(buildings_clipped['confidence'] > 0.75) | buildings_clipped['confidence'].isna()].reset_index(drop=True)
            building_area = buildings_clipped.area.sum()
            n_buildings = len(buildings_clipped)
            building_density = (1000.*1000*n_buildings)/rectangle_area
        else:
            buildings_clipped = gpd.GeoDataFrame([])
            building_area = np.nan
            buildings_clipped = gpd.GeoDataFrame([])
            building_density = np.nan
            n_buildings = np.nan

        print(building_density)
        # Only execute calculations above a certain building density
        if building_density > 64:
            blocks_clipped = blocks_all[blocks_all.geometry.intersects(bounding_box_geom)]
            OSM_buildings_bool = False
            
            # Roads
            try:
                roads_clipped = OSM_roads_all_projected[OSM_roads_all_projected.geometry.intersects(bounding_box_geom)]
                roads_intersection = OSM_roads_all_projected[OSM_roads_all_projected.geometry.intersects(bounding_box_geom)]
                OSM_roads_bool = True
            except (fiona.errors.DriverError, TopologicalError) as e:
                print(f"Error clipping roads for cell {cell_id}: {e}")
                roads_clipped = gpd.GeoDataFrame([])
                OSM_roads_all = gpd.GeoDataFrame([])
                roads_intersection = gpd.GeoDataFrame([])
                OSM_roads_bool = False

            # Intersections
            try:
                OSM_intersections = OSM_intersections_all_projected[OSM_intersections_all_projected.geometry.intersects(bounding_box_geom)]#OSM_intersections_all_projected.clip(list(rectangle_projected.geometry.bounds.values[0]))
                OSM_intersections_bool = True
                n_intersections = len(OSM_intersections.drop_duplicates('osmid'))
            except fiona.errors.DriverError:
                OSM_intersections = gpd.GeoDataFrame([])
                OSM_intersections_bool = False
                n_intersections = np.nan

            print(buildings.head(), buildings.crs)
            print(OSM_roads_all_projected.head(), OSM_roads_all_projected.crs)
            print(OSM_intersections_all_projected.head(), OSM_intersections_all_projected.crs)

            print(f"Buildings intersecting cell {cell_id}: {len(buildings_clipped)}")
            print(f"Roads intersecting cell {cell_id}: {len(roads_clipped)}")
            print(f"Intersections intersecting cell {cell_id}: {len(OSM_intersections)}")

            #Overture_data = Overture_data_all_projected[Overture_data_all_projected.geometry.intersects(rectangle_projected[0])]
            #if not Overture_data.empty:
            #    Overture_buildings_bool = True
            #else:
            #    Overture_buildings_bool = False


            if (not buildings_clipped.empty):
                # Metric 1 -- share of buildings closer than 10 ms from the road
                m1, buildings_clipped = metric_1_distance_less_than_20m(buildings_clipped, road_union, utm_proj_city)
                
                # Metric 2 -- average distance to roads
                m2 = metric_2_average_distance_to_roads(buildings_clipped)
                #plot_distance_to_roads(buildings_clipped, roads_clipped, rectangle_projected, cell_id)
            else:
                m1, m2 = np.nan, np.nan

            if (not roads_clipped.empty):
                # Metric 3 -- road density
                m3 = metric_3_road_density(rectangle_area, roads_clipped)
            else:
                m3 = np.nan

            # Metrics 4 and 5 -- share of 3 and 4-way intersections
            if not OSM_intersections.empty:
                if ((4 in OSM_intersections['street_count'].values) or (3 in OSM_intersections['street_count'].values)):
                    m4 = metric_4_share_4way_intersections(OSM_intersections)
                else:
                    m4 = np.nan
                m5 = metric_5_intersection_density(OSM_intersections, rectangle_area)    
            else:
                m4, m5 = np.nan, np.nan

            # Metric 6 -- building azimuth
            if (not buildings_clipped.empty) and (len(buildings_clipped)>5):
                n_orientation_groups = 4
                m6, buildings_clipped = metric_6_entropy_of_building_azimuth(buildings_clipped, rectangle_id=1, bin_width_degrees=5, plot=False)
            else:
                m6 = np.nan

            # Metric 7 -- average block width
            # Metric 8 -- two-row blocks
            if not blocks_clipped.empty:
                m7, blocks_clipped = metric_7_average_block_width(blocks_clipped, rectangle_projected, rectangle_area)

                minx, miny, maxx, maxy = rectangle_projected.bounds
                rectangle_box = box(minx, miny, maxx, maxy)
                blocks_clipped_within_rectangle = blocks_clipped.clip(rectangle_box)

                area_tiled_by_blocks = blocks_clipped_within_rectangle.area.sum()
                share_tiled_by_blocks = area_tiled_by_blocks/rectangle_area

                if not buildings_clipped.empty:
                    m8, internal_buffers = metric_8_two_row_blocks(blocks_clipped, buildings_clipped, utm_proj_city, row_epsilon=row_epsilon)
                else:
                    m8 = np.nan
            else:
                m7 = np.nan
                m8 = np.nan
                share_tiled_by_blocks = np.nan

            
            if (not roads_clipped.empty) and (not OSM_intersections.empty):
                # Metric 9 -- tortuosity index
                m9 = metric_9_tortuosity_index(roads_clipped)
                                                                
                # Metric 10 -- average angle between road segments
                m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads_clipped) #OJO, ROADS EXPANDED
                #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads_clipped)
            else:
                m9, m10 = np.nan, np.nan

            if not roads_clipped.empty:
                road_length = roads_clipped.length.sum()
            else:
                road_length = np.nan
            
            # Metrics 11, 12 and 13
            if not buildings_clipped.empty:
                n_buildings = len(buildings_clipped)
                building_area = buildings_clipped.area.sum()
                m11 = metric_11_building_density(n_buildings,rectangle_area)
                m12 = metric_12_built_area_share(building_area,rectangle_area)
                m13 = metric_13_average_building_area(building_area,n_buildings)
            else:
                n_buildings = np.nan
                building_area = np.nan
                average_building_area = np.nan
                m11, m12, m13 = np.nan, np.nan, np.nan

        else:
            m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            OSM_buildings_bool, OSM_roads_bool, OSM_intersections_bool, building_area, building_density, share_tiled_by_blocks, road_length, n_buildings, n_intersections = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        print(f"One round of metrics done for cell_id: {cell_id}")

        result = {'index':cell_id,
                'metric_1':m1,
                'metric_2':m2,
                'metric_3':m3,
                'metric_4':m4,
                'metric_5':m5,
                'metric_6':m6,
                'metric_7':m7,
                'metric_8':m8,
                'metric_9':m9,
                'metric_10':m10,
                'metric_11':m11,
                'metric_12':m12,
                'metric_13':m13,
                'OSM_buildings_available':OSM_buildings_bool,
                'OSM_intersections_available':OSM_intersections_bool,
                'OSM_roads_available':OSM_roads_bool,
                #'Overture_buildings_available':Overture_buildings_bool,
                'rectangle_area': rectangle_area,
                'building_area':building_area,
                'share_tiled_by_blocks': share_tiled_by_blocks,
                'road_length':road_length,
                'n_intersections':n_intersections,
                'n_buildings':n_buildings,
                'building_density':building_density
                }
        cell_id += 1
    else:
        result = {'index':np.nan,
                'metric_1':np.nan,
                'metric_2':np.nan,
                'metric_3':np.nan,
                'metric_4':np.nan,
                'metric_5':np.nan,
                'metric_6':np.nan,
                'metric_7':np.nan,
                'metric_8':np.nan,
                'metric_9':np.nan,
                'metric_10':np.nan,
                'metric_11':np.nan,
                'metric_12':np.nan,
                'metric_13':np.nan,
                'OSM_buildings_available':np.nan,
                'OSM_intersections_available':np.nan,
                'OSM_roads_available':np.nan,
                #'Overture_buildings_available':Overture_buildings_bool,
                'rectangle_area': rectangle_area,
                'building_area':np.nan,
                'share_tiled_by_blocks': np.nan,
                'road_length':np.nan,
                'n_intersections':np.nan,
                'n_buildings':np.nan,
                'building_density':building_density
                }
    return result

def process_city(city_name, sample_prop=1.0, override_processed=False, grid_size=200):
    """
    Processes a city grid, calculates metrics for unprocessed rows, and updates the grid status CSV.
    """
    try:
        # Read city grid
        city_grid = gpd.read_parquet(f'{GRIDS_PATH}/{city_name}/{city_name}_{str(grid_size)}m_grid.geoparquet').reset_index()
        city_grid.rename(columns={'index': 'grid_id'}, inplace=True)  # Ensure we have a grid ID column

        if city_grid.empty or not 'geometry' in city_grid.columns:
            print(f"No grid cells available for {city_name}. Skipping.")
            return

        # Initialize 'processed' column
        if 'grid_id' not in city_grid.columns:
            city_grid['grid_id'] = city_grid.index
        city_grid['processed'] = False

        # Check if a grid status CSV already exists
        output_dir_csv = f'{OUTPUT_PATH_CSV}/{city_name}'
        os.makedirs(output_dir_csv, exist_ok=True)
        STATUS_CSV_PATH = f'{output_dir_csv}/{city_name}_grid_status_{grid_size}m.csv'

        if os.path.exists(STATUS_CSV_PATH) and not override_processed:
            # Load existing status CSV and merge with city grid
            status_df = pd.read_csv(STATUS_CSV_PATH)
            city_grid = city_grid.merge(status_df, on='grid_id', how='left', suffixes=('', '_existing'))
            city_grid['processed'] = city_grid['processed_existing'].fillna(False)
            city_grid.drop(columns=['processed_existing'], inplace=True)

        # Filter for unprocessed rows
        unprocessed_grid = city_grid[~city_grid['processed']]

        # Apply sampling to unprocessed rows
        if sample_prop < 1.0:
            sampled_grid = unprocessed_grid.sample(frac=sample_prop, random_state=42)
        else:
            sampled_grid = unprocessed_grid

        # Update 'processed' column for sampled rows
        city_grid.loc[sampled_grid.index, 'processed'] = True

        rectangles = sampled_grid['geometry']

        # Read and process required datasets (buildings, roads, intersections)
        if not os.path.exists(f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet'):
            print(f"Missing buildings data for city {city_name}. Skipping.")
            return
        Overture_data_all = gpd.read_parquet(f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet')
        print(f"{city_name}: Overture file read")
        Overture_data_all['confidence'] = Overture_data_all.sources.apply(lambda x: x[0]['confidence'])
        Overture_data_all['dataset'] = Overture_data_all.sources.apply(lambda x: x[0]['dataset'])
        Overture_data_all = Overture_data_all.set_geometry('geometry')[Overture_data_all.dataset != 'OpenStreetMap']

        # Read intersections
        if not os.path.exists(f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.gpkg'):
            print(f"Missing intersections data for city {city_name}. Skipping.")
            return
        OSM_intersections_all = gpd.read_file(f'{INTERSECTIONS_PATH}/{city_name}/{city_name}_OSM_intersections.gpkg')

        # Read roads
        if not os.path.exists(f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.gpkg'):
            print(f"Missing roads data for city {city_name}. Skipping.")
            return
        OSM_roads_all = gpd.read_file(f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.gpkg')

        print(f"{city_name}: OSM files read")

        # Get UTM projection for the city
        utm_proj_city = get_utm_crs(OSM_roads_all.iloc[0].geometry)
        if utm_proj_city is not None:
            try:
                OSM_roads_all_projected = OSM_roads_all.to_crs(epsg=utm_proj_city)
                OSM_intersections_all_projected = OSM_intersections_all.to_crs(epsg=utm_proj_city)
                Overture_data_all_projected = Overture_data_all.to_crs(epsg=utm_proj_city)
            except Exception as e:
                print(f"Error reprojecting data for city {city_name}: {e}")
                return
        else:
            raise ValueError(f"Unable to determine EPSG code for city {city_name}.")

        # Prepare delayed processing
        geod = Geod(ellps="WGS84")
        rectangles_projected = sampled_grid['geometry'].to_crs(epsg=utm_proj_city)

        road_union = OSM_roads_all_projected.unary_union

        if not OSM_roads_all_projected.empty:
            blocks = get_blocks(road_union, OSM_roads_all_projected)
        else:
            blocks = gpd.GeoDataFrame([])


        delayed_results = [
            delayed(process_cell)(
                cell_id, geod, rectangle, rectangle_projected, Overture_data_all_projected, blocks,
                OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city
            )
            for cell_id, (rectangle, rectangle_projected) in enumerate(zip(rectangles, rectangles_projected))
        ]

        # Compute metrics for sampled rows
        final_geo_df = compute(*delayed_results)
        final_geo_df = pd.DataFrame(final_geo_df)
        final_geo_df = process_metrics(final_geo_df)

        final_geo_df['index'] = sampled_grid['grid_id'].values
        if final_geo_df['index'].isna().any():
            print("Warning: Some rows in the results could not be matched with the grid.")

        # Merge results back into city_grid
        city_grid = city_grid.merge(final_geo_df, how='left', left_on='grid_id', right_on='index')

        #Save results to geoparquet
        os.makedirs(f'{OUTPUT_PATH_RASTER}/{city_name}', exist_ok=True)
        city_grid.to_parquet(f'{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_{str(grid_size)}m_results.geoparquet', engine="pyarrow", index=False)

        # Save updated grid status to CSV
        city_grid[['grid_id', 'processed']].to_csv(STATUS_CSV_PATH, index=False)

        print(f"{city_name}: Processing complete. Grid status updated in {STATUS_CSV_PATH}.")

    except Exception as e:
        print(f"Error processing {city_name}: {e}")
        raise 

def main():
    cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
              "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
              "Maputo", "Luanda"]
    #cities = ["Belo Horizonte"]
    cities = [city.replace(' ', '_') for city in cities]
    sample_prop = 0.1  # Sample 10% of the grid cells

    # City-Level Parallelization
    with ProcessPoolExecutor() as executor:
        try:
            executor.map(partial(process_city, sample_prop=sample_prop, override_processed=False), cities)
        except Exception as e:
            print(f"Error in ProcessPoolExecutor: {e}")


if __name__ == "__main__":
    main()

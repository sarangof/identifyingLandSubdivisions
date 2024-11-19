import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from metrics_calculation import *
from create_rectangles import *
from standardize_metrics import *
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import pyproj
import json
from dask import delayed, compute


main_path = '../data'
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
grids_path = f'{input_path}/city_info/grids'
output_path_csv = f'{main_path}/output'

# Define important parameters for this run
grid_size = 200
row_epsilon = 0.01

# Helper function to get UTM CRS based on city geometry centroid
def get_utm_crs(geometry):
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_zone = int((lon + 180) // 6) + 1
    hemisphere = '6' if lat < 0 else '3'  # Southern hemisphere gets '6', northern gets '3'
    return CRS(f"EPSG:32{hemisphere}{utm_zone:02d}")

def process_cell(cell_id, city_name, rectangle_projected, buildings, blocks_all, OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city):
    print(f"cell_id: {cell_id}")

    # Preparatory calculations
    if not buildings.empty:
        buildings_clipped = buildings[buildings.geometry.intersects(bounding_box_geom)]#buildings[buildings.geometry.intersects(rectangle_projected[0])]#buildings[buildings.geometry.intersects(rectangle_projected['geometry'])]
        buildings_clipped = buildings_clipped[(buildings_clipped['confidence']>0.75)|buildings_clipped['confidence'].isna()].reset_index()
        building_density = (1000.*1000*n_buildings)/rectangle_area
        n_buildings = len(buildings_clipped)
        building_area = buildings_clipped.area.sum()
    else:
        buildings_clipped = gpd.GeoDataFrame([])
        building_density = np.nan
        n_buildings = np.nan
        building_area = np.nan

    rectangle_area, _ = geod.geometry_area_perimeter(rectangles.iloc[cell_id])

    # Only execute calculations above a certain building density
    if building_density > 64:

        blocks_clipped = blocks_all[blocks_all.geometry.intersects(bounding_box_geom)]
        OSM_buildings_bool = False
        bounding_box = rectangle_projected.bounds
        bounding_box_geom = box(*bounding_box)

        # Roads
        try:
            roads_clipped = OSM_roads_all_projected[OSM_roads_all_projected.geometry.intersects(bounding_box_geom)]
            roads_intersection = OSM_roads_all_projected[OSM_roads_all_projected.geometry.intersects(bounding_box_geom)]
            OSM_roads_bool = True
        except fiona.errors.DriverError:
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


        #Overture_data = Overture_data_all_projected[Overture_data_all_projected.geometry.intersects(rectangle_projected[0])]
        #if not Overture_data.empty:
        #    Overture_buildings_bool = True
        #else:
        #    Overture_buildings_bool = False

        geod = Geod(ellps="WGS84")

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
            #plot_azimuth(buildings_clipped, roads_clipped, rectangle_projected, rectangle_id, n_orientation_groups)
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
            m9 = metric_9_tortuosity_index(city_name, roads_intersection, OSM_intersections, rectangle_projected, angular_threshold=30, tortuosity_tolerance=5)
            # WHY DOES THIS WORK?
            #m9 = metric_9_tortuosity_index(roads_clipped)
                                                            
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
            # Calculate relevant building metrics, making use of the if statement.
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
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        OSM_buildings_bool, OSM_intersections_bool, building_area, building_density, share_tiled_by_blocks, road_length, n_buildings = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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
            'n_buildings':n_buildings
            }
    cell_id += 1
    return result


def process_city(city_name):
    try:
        city_grid = gpd.read_parquet(f'{grids_path}/{city_name}/{city_name}_{str(grid_size)}m_grid.parquet')

        metrics_pilot = []
        rectangles = city_grid['geometry']
        Overture_data_all = gpd.read_parquet(f'{buildings_path}/{city_name}/Overture_building_{city_name}.parquet')
        print(f"{city_name}: Overture file read")

        Overture_data_all['confidence'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['confidence'])
        Overture_data_all['dataset'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['dataset'])
        Overture_data = Overture_data_all.set_geometry('geometry')[Overture_data_all.dataset != 'OpenStreetMap']

        OSM_intersections_all = gpd.read_file(f'{intersections_path}/{city_name}/{city_name}_OSM_intersections.gpkg')
        OSM_roads_all = gpd.read_file(f'{roads_path}/{city_name}/{city_name}_OSM_intersections.gpkg')
        print(f"{city_name}: OSM files read")

        utm_proj_city = get_utm_crs(OSM_roads_all.iloc[0].geometry)
        project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True).transform  
        OSM_roads_all_projected = OSM_roads_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
        OSM_intersections_all_projected = OSM_intersections_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())

        blocks_all = get_blocks(OSM_roads_all_projected.unary_union, OSM_roads_all_projected)
        Overture_data_all_projected = Overture_data_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
        buildings = Overture_data_all_projected
        road_union = OSM_roads_all_projected.unary_union

        rectangles_projected = rectangles.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
        print(f"{city_name}: All preparations done. Ready to calculate metrics on grid.")

        # Grid-Level Parallelization using Dask Delayed
        delayed_results = [
            delayed(process_cell)(
                cell_id, city_name, rectangle, buildings, blocks_all, OSM_roads_all_projected,
                OSM_intersections_all_projected, road_union, utm_proj_city
            )
            for cell_id, rectangle in enumerate(rectangles_projected)
        ]

        # Compute the results using Dask
        batch_results = compute(*delayed_results)
        batch_df = pd.DataFrame(batch_results)

        # Save each batch as a CSV to avoid memory overflow
        output_dir_csv = f'{output_path_csv}/{city_name}'
        os.makedirs(output_dir_csv, exist_ok=True)
        batch_df.to_csv(f'{output_dir_csv}/{city_name}_results.csv', index=False)

    except Exception as e:
        print(f"Error processing {city_name}: {e}")

def main():
    cities = ["Belo Horizonte", "Campinas", "Bogota", "Nairobi", "Bamako", 
              "Lagos", "Accra", "Abidjan", "Mogadishu", "Cape Town", 
              "Maputo", "Luanda"]

    # City-Level Parallelization using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(process_city, cities)

if __name__ == "__main__":
    main()

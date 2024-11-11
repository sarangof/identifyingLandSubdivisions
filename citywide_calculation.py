from metrics_calculation import *
from create_rectangles import *
#from gather_data_pilot import *
from standardize_metrics import *
from metric_plots import *
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, box
from geopy.distance import geodesic
from pyproj import CRS
import dask
import pyproj
import json
from dask import delayed, compute


main_path = '../data'
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
urban_extents_path = f'{input_path}/urban_extents'
output_path = f'{main_path}/output'
output_path_csv = f'{main_path}/output'
output_path_raster = f'{main_path}/output'

city_info_path = f'{input_path}/city_info'
extents_path = f'{city_info_path}/extents'
analysis_buffers_path = f'{city_info_path}/analysis_buffers'
search_buffers_path = f'{city_info_path}/search_buffers'
grids_path = f'{city_info_path}/grids'

def create_square_from_coords(coord_pair, grid_size):
    x, y = coord_pair
    return box(x, y, x + grid_size, y + grid_size)


# Helper function to get UTM CRS based on city geometry centroid
def get_utm_crs(geometry):
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_zone = int((lon + 180) // 6) + 1
    hemisphere = '6' if lat < 0 else '3'  # Southern hemisphere gets '6', northern gets '3'
    return CRS(f"EPSG:32{hemisphere}{utm_zone:02d}")

# Define important parameters for this run
grid_size = 200
row_epsilon = 0.01


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

    city_grid = gpd.read_parquet(f'{grids_path}/{city_name}/{city_name}_{str(grid_size)}m_grid.parquet')

    metrics_pilot = []
    #rectangles = city_grids[city_grids.city_name==city_name]['lower_left_coordinates'][0]
    rectangles = city_grid['geometry']
    Overture_data_all = gpd.read_parquet(f'{buildings_path}/{city_name}/Overture_building_{city_name}.parquet')
    print("Overture file read")

    Overture_data_all['confidence'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['confidence'])
    Overture_data_all['dataset'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['dataset'])
    Overture_data = Overture_data_all.set_geometry('geometry')[Overture_data_all.dataset!='OpenStreetMap']

    OSM_intersections_all = gpd.read_file(f'{intersections_path}/{city_name}/{city_name}_OSM_intersections.gpkg')
    OSM_roads_all = gpd.read_file(f'{roads_path}/{city_name}/{city_name}_OSM_intersections.gpkg')
    print("OSM files read")
    

    utm_proj_city = get_utm_proj(float(OSM_roads_all.iloc[0].geometry.centroid.x), float(OSM_roads_all.iloc[0].geometry.centroid.y))
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True).transform  
    OSM_roads_all_projected = OSM_roads_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
    OSM_intersections_all_projected = OSM_intersections_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())

    blocks_all  = get_blocks(OSM_roads_all_projected.unary_union, OSM_roads_all_projected)
    Overture_data_all_projected = Overture_data_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
    buildings = Overture_data_all_projected
    road_union = OSM_roads_all_projected.unary_union

    rectangles_projected = rectangles.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())
    print("All preparations done. Ready to calculate metrics on grid.")

    cell_id = 0
    # Use delayed to parallelize the processing
    delayed_results = [delayed(process_cell)(cell_id, city_name, rectangle_projected, buildings, blocks_all, OSM_roads_all_projected, OSM_intersections_all_projected, road_union, utm_proj_city) for cell_id in enumerate(rectangles_projected)]

    # Process in batches
    batch_size = 5000
    n_batches = len(delayed_results) // batch_size + 1
    all_results = []

    for batch_num in range(n_batches):
        print(batch_num)
        batch_results = delayed_results[batch_num * batch_size: (batch_num + 1) * batch_size]
        computed_batch = compute(*batch_results)  # This will execute the batch in parallel
        batch_df = pd.DataFrame(computed_batch)
        
        # Save each batch as a CSV to avoid memory overflow
        batch_df.to_csv(f'metrics_batch_{batch_num}_{city_name}_{str(grid_size)}m.csv', index=False)
        
        all_results.append(batch_df)

    # Combine all batch results after processing
    final_metrics = pd.concat(all_results, ignore_index=True)
    metrics = pd.DataFrame(final_metrics)
    result = pd.merge(rectangles, metrics, how='left', left_index=True, right_index=True)
    all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']

    # Save original values before transformations
    metrics_original_names = [col+'_original' for col in all_metrics_columns]
    result[metrics_original_names] = result[all_metrics_columns].copy()

    # Apply the standardization functions
    for metric, func in standardization_functions.items():
        result[metric+'_standardized'] = func(result[metric])

    metrics_standardized_names = [col+'_standardized' for col in all_metrics_columns]

    # Center at zero and maximize information
    result.loc[:, all_metrics_columns] = (
        result[metrics_standardized_names]
        .apply(lambda x: (x - x.mean()) / (x.std()), axis=0)
    )

    # Convert metrics to a range between 0 and 1
    result.loc[:,all_metrics_columns] = (
        result[all_metrics_columns]
        .apply(lambda x: (x - x.min()) / (x.max()-x.min()), axis=0)
    )

    # Calculate equal-weights irregularity index
    result['regularity_index'] = result[all_metrics_columns].mean(axis=1)

    # Save output file
    cols_to_save = [col for col in result.columns if col!='geometry']
    output_dir_csv = f'{output_path_csv}/{city_name}'
    os.makedirs(output_dir_csv, exist_ok=True)
    result[cols_to_save].to_csv(f'{output_dir_csv}/{city_name}_results.csv')
from metrics_calculation import *
from standardize_metrics import *
from create_rectangles import *
#from gather_data_pilot import *
from metric_plots import *
import fiona
from pyproj import CRS
from shapely.geometry import box
import json
import cProfile
import pstats
from io import StringIO

"""
This script iterates through all study areas in the initial pilot, reads data
on roads, intersections and buildings, and performs metric calculations on each 
study area, returning an excel file with metrics values and a column with 
the irrregularity index.
"""

main_path = '../data/pilot'
input_path = f'{main_path}/input'
buildings_path = f'{input_path}/buildings'
roads_path = f'{input_path}/roads'
intersections_path = f'{input_path}/intersections'
urban_extents_path = f'{input_path}/urban_extents'
output_path = f'{main_path}/output'

rectangles = gpd.read_file(f'{output_path}/rectangles/rectangles.geojson')
blocks_clipped_empty = 0 
metrics_pilot = []
row_epsilon = 0.01

for rectangle_id, rectangle in rectangles.iterrows():

    rectangle_id += 1 
    print(f"rectangle_id: {rectangle_id}")

    # Gather information about the geometry of the study area

    rectangle_centroid = rectangle.geometry.centroid
    utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))
    geometry = gpd.GeoSeries(rectangle['geometry'])
    rectangle_projected = gpd.GeoDataFrame({'geometry': geometry}, crs="EPSG:4326").to_crs(epsg=CRS.from_proj4(utm_proj_rectangle).to_epsg())
    rectangle_area = calculate_area_geodesic(rectangle)
    bounding_box = rectangle_projected.bounds.values[0]
    bounding_box_geom = box(*bounding_box)

    # Read buildings, roads and intersections data

    # OSM buildings
    try:
        OSM_buildings = gpd.read_file(f"{buildings_path}/OSM_buildings_{rectangle_id}.gpkg")
        buildings_OSM = OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
        buildings_OSM = buildings_OSM.set_geometry('geometry')
        OSM_buildings_bool = True
    except fiona.errors.DriverError:
        OSM_buildings = gpd.GeoDataFrame([])
        buildings_OSM = gpd.GeoDataFrame([])
        OSM_buildings_bool = False

    # OSM roads
    try:
        OSM_roads = gpd.read_file(f"{roads_path}/OSM_roads_{rectangle_id}.gpkg").drop_duplicates(subset='geometry')
        roads = OSM_roads.to_crs(utm_proj_rectangle)
        roads_clipped = roads.clip(list(rectangle_projected.geometry.bounds.values[0]))
        roads_intersection = roads[roads.geometry.intersects(bounding_box_geom)]
        road_union = roads.unary_union 
        OSM_roads_bool = True
    except fiona.errors.DriverError:
        OSM_roads = gpd.GeoDataFrame([])
        roads = gpd.GeoDataFrame([])
        roads_clipped = gpd.GeoDataFrame([])
        roads_intersection = gpd.GeoDataFrame([]) 
        road_union = gpd.GeoDataFrame([])
        OSM_roads_bool = False

    # OSM intersections
    try:
        OSM_intersections = gpd.read_file(f"{intersections_path}/OSM_intersections_{rectangle_id}.gpkg").to_crs(utm_proj_rectangle)
        OSM_intersections_clipped = OSM_intersections.clip(list(rectangle_projected.geometry.bounds.values[0]))
        OSM_intersections_bool = True
    except fiona.errors.DriverError:
        OSM_intersections = gpd.GeoDataFrame([])
        OSM_intersections_clipped = gpd.GeoDataFrame([])
        OSM_intersections_bool = False

    # Overture buildings
    try:
        Overture_data = gpd.read_file(f"{buildings_path}/Overture_building_{rectangle_id}.geojson").to_crs(utm_proj_rectangle)#.clip(rectangle['geometry'])
        if not Overture_data.empty:
            Overture_data['confidence'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['confidence'])
            Overture_data['dataset'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['dataset'])
            Overture_data = Overture_data.set_geometry('geometry')[Overture_data.dataset!='OpenStreetMap']
        Overture_buildings_bool = True
    except fiona.errors.DriverError: 
        Overture_data = gpd.GeoDataFrame([])
        Overture_buildings_bool = False

    buildings = gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle).dropna(how='all')
    try:
        buildings_clipped = buildings[buildings.geometry.intersects(bounding_box_geom)]
        #buildings_clipped = buildings_clipped.reset_index()
        buildings_clipped = buildings_clipped[(buildings_clipped['confidence']>0.75)|buildings_clipped['confidence'].isna()].reset_index()
    except KeyError:
        continue

    # Create blocks wherever possible and crop the blocks structure as needed.
    if not roads.empty:
        blocks = get_blocks(road_union, roads)
    else:
        blocks = gpd.GeoDataFrame([])

    if not blocks.empty:
        blocks_clipped = blocks[blocks.geometry.intersects(bounding_box_geom)]#blocks[blocks.geometry.intersects(rectangle_projected['geometry'])]
        blocks_clipped_within_rectangle = blocks_clipped.clip(bounding_box_geom)
        area_tiled_by_blocks = blocks_clipped_within_rectangle.area.sum()
        share_tiled_by_blocks = area_tiled_by_blocks/rectangle_area
    else:
        blocks_clipped = gpd.GeoDataFrame([])
        blocks_clipped_empty += 1

    # Metrics calculation

    # Metrics 1, 2 and 3
    if (not roads_clipped.empty) and (not buildings_clipped.empty):
        m1, buildings_clipped = metric_1_distance_less_than_20m(buildings_clipped, road_union, utm_proj_rectangle)
        m2 = metric_2_average_distance_to_roads(buildings_clipped)
        #plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id)
        m3 = metric_3_road_density(rectangle_area,roads_clipped)
    else:
        m1, m2, m3 = np.nan, np.nan, np.nan

    # Metrics 4 and 5 
    if not OSM_intersections_clipped.empty:
        if ((4 in OSM_intersections_clipped['street_count'].values) or (3 in OSM_intersections_clipped['street_count'].values)):
            m4 = metric_4_share_4way_intersections(OSM_intersections_clipped)
        else:
            m4 = np.nan
        m5 = metric_5_intersection_density(OSM_intersections_clipped, rectangle_area)    
    else:
        m4, m5 = np.nan, np.nan

    # Metric 6 -- building azimuth
    if not buildings_clipped.empty:
        n_orientation_groups = 4
        m6, buildings_clipped = metric_6_entropy_of_building_azimuth(buildings_clipped, rectangle_id, bin_width_degrees=5, plot=False)
        #plot_azimuth(buildings_clipped, roads, rectangle_projected, rectangle_id, n_orientation_groups)
    else:
        m6 = np.nan


    # Metrics 7 and 8
    if not blocks_clipped.empty:
        minx, miny, maxx, maxy = list(rectangle_projected.bounds.values[0])
        rectangle_box = box(minx, miny, maxx, maxy)
        rectangle_projected_arg = rectangle_projected.geometry
        blocks_clipped_within_rectangle = blocks_clipped.clip(rectangle_box)
        m7, blocks_clipped = metric_7_average_block_width(blocks_clipped, blocks_clipped_within_rectangle, rectangle_projected_arg, rectangle_area)
        #m7=np.nan
        #plot_largest_inscribed_circle(rectangle_id, rectangle_projected,  blocks_clipped, roads)
        m8, epsilon_buffers, width_buffers = metric_8_two_row_blocks(blocks_clipped, buildings, utm_proj_rectangle, row_epsilon=row_epsilon)
        
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon)
    else:
        print("Blocks_clipped is empty")
        m7, m8 = np.nan, np.nan

    # Metric 9 -- tortuosity index
    if (not roads_clipped.empty):
        rectangle_projected_arg = rectangle_projected.geometry
        #m9, all_road_vertices = metric_9_tortuosity_index(rectangle_id, roads_clipped, OSM_intersections_clipped, rectangle_projected_arg, angular_threshold=30, tortuosity_tolerance=5)
        m9 = metric_9_tortuosity_index(roads_intersection)
        road_length = roads_clipped.length.sum()
    else:
        m9 = np.nan
        road_length = np.nan

    # Metric 10 -- average angle between road segments
    if (not OSM_intersections_clipped.empty) and ((not roads.empty)):                                               
        m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads)
        #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads)
    else:
        m10 = np.nan
    
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

    metrics_pilot.append({'index':rectangle_id,
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
                        'Overture_buildings_available':Overture_buildings_bool,
                        'rectangle_area': rectangle_area,
                        'building_area':building_area,
                        'share_tiled_by_blocks': share_tiled_by_blocks,
                        'road_length':road_length,
                        'n_intersections':len(OSM_intersections_clipped.drop_duplicates('osmid')),
                        'n_buildings':n_buildings
                        })

metrics_pilot = pd.DataFrame(metrics_pilot)
final_geo_df = gpd.GeoDataFrame(pd.merge(rectangles, metrics_pilot, how='left', left_on='n', right_on='index'), geometry=rectangles.geometry)
all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10','metric_11','metric_12','metric_13']

# Save original values before transformations
metrics_original_names = [col+'_original' for col in all_metrics_columns]
final_geo_df[metrics_original_names] = final_geo_df[all_metrics_columns].copy()


metrics_standardized_names = {col:col+'_standardized' for col in all_metrics_columns}

# Apply the standardization functions
for metric, func in standardization_functions.items():
    final_geo_df[metrics_standardized_names[metric]] = func(final_geo_df[metric])

# # Center at zero and maximize information
# final_geo_df.loc[:, all_metrics_columns] = (
#     final_geo_df[list(metrics_standardized_names.values())]
#     .apply(lambda x: (x - x.mean()) / (x.std()))
# )

# # Convert metrics to a range between 0 and 1
# final_geo_df.loc[:,all_metrics_columns] = (
#     final_geo_df[list(metrics_standardized_names.values())]
#     .apply(lambda x: (x - x.min()) / (x.max()-x.min()), axis=0)
# )

# Calculate equal-weights irregularity index
final_geo_df['regularity_index'] = final_geo_df[all_metrics_columns].mean(axis=1)

# Save output file
cols_to_save = [col for col in final_geo_df.columns if col!='geometry']
final_geo_df[cols_to_save].to_excel(f"{output_path}/excel/pilot_test_results.xlsx")
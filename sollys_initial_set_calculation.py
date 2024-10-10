from metrics_calculation import *
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


rectangles = gpd.read_file('./data/rectangles.geojson')
blocks_clipped_empty = 0 
metrics_pilot = []
row_epsilon = 0.01

for rectangle_id, rectangle in rectangles.iterrows():
    # Read needed files: rectangle, buildings, streets
    rectangle_id += 1 
    print(rectangle_id)

    print(f"rectangle_id: {rectangle_id}")
    rectangle_centroid = rectangle.geometry.centroid
    utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))

    geometry = gpd.GeoSeries(rectangle['geometry'])

    rectangle_projected = gpd.GeoDataFrame({'geometry': geometry}, crs="EPSG:4326").to_crs(epsg=CRS.from_proj4(utm_proj_rectangle).to_epsg())
    #rectangle_projected = rectangle.to_crs(epsg=CRS.from_proj4(utm_proj_rectangle).to_epsg())#rectangle.apply(project_geometry) 

    try:
        OSM_buildings = gpd.read_file(f"./output_data/OSM_buildings_{rectangle_id}.gpkg")
        buildings_OSM = OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
        buildings_OSM = buildings_OSM.set_geometry('geometry')
        OSM_buildings_bool = True
    except fiona.errors.DriverError:
        OSM_buildings = gpd.GeoDataFrame([])
        buildings_OSM = gpd.GeoDataFrame([])
        OSM_buildings_bool = False

    try:
        OSM_roads = gpd.read_file(f"./output_data/OSM_roads_{rectangle_id}.gpkg")
        roads = OSM_roads.to_crs(utm_proj_rectangle)
        roads_clipped = roads.clip(list(rectangle_projected.geometry.bounds.values[0]))
        road_union = roads.unary_union # Create a unary union of the road geometries to simplify distance calculation
        OSM_roads_bool = True
    except fiona.errors.DriverError:
        OSM_roads = gpd.GeoDataFrame([])
        roads = gpd.GeoDataFrame([])
        roads_clipped = gpd.GeoDataFrame([])
        road_union = gpd.GeoDataFrame([])
        OSM_roads_bool = False

    try:
        OSM_intersections = gpd.read_file(f"./output_data/OSM_intersections_{rectangle_id}.gpkg").to_crs(utm_proj_rectangle)
        OSM_intersections_clipped = OSM_intersections.clip(list(rectangle_projected.geometry.bounds.values[0]))
        OSM_intersections_bool = True
    except fiona.errors.DriverError:
        OSM_intersections = gpd.GeoDataFrame([])
        OSM_intersections_clipped = gpd.GeoDataFrame([])
        OSM_intersections_bool = False

    try:
        Overture_data = gpd.read_file(f"./output_data/Overture_building_{rectangle_id}.geojson").to_crs(utm_proj_rectangle)#.clip(rectangle['geometry'])
        if not Overture_data.empty:
            Overture_data['confidence'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['confidence'])
            Overture_data['dataset'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['dataset'])
            Overture_data = Overture_data.set_geometry('geometry')[Overture_data.dataset!='OpenStreetMap']
        Overture_buildings_bool = True
    except fiona.errors.DriverError: 
        Overture_data = gpd.GeoDataFrame([])
        Overture_buildings_bool = False

    buildings = gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle).dropna(how='all')
    #buildings_clipped = buildings[buildings.geometry.intersects(rectangle_projected['geometry'])]

    bounding_box = rectangle_projected.bounds.values[0]
    bounding_box_geom = box(*bounding_box)
    try:
        buildings_clipped = buildings[buildings.geometry.intersects(bounding_box_geom)]
        buildings_clipped = buildings_clipped[(buildings_clipped['confidence']>0.75)|buildings_clipped['confidence'].isna()].reset_index()
    except KeyError:
        continue

    rectangle_area = calculate_area_geodesic(rectangle)

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
    #blocks_clipped = blocks.clip(list(rectangle_projected.geometry.bounds))


    if (not roads_clipped.empty) and (not buildings_clipped.empty):
        # Metric 1 -- share of buildings closer than 10 ms from the road
        m1, buildings_clipped = metric_1_distance_less_than_10m(buildings_clipped, road_union, utm_proj_rectangle)
        
        # Metric 2 -- average distance to roads
        m2 = metric_2_average_distance_to_roads(buildings_clipped)
        #plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id)

        # Metric 3 -- road density
        m3 = metric_3_road_density(rectangle_area,roads_clipped)
    else:
        m1, m2, m3 = np.nan, np.nan, np.nan

    # Metrics 4 and 5 -- share of 3 and 4-way intersections
    if not OSM_intersections_clipped.empty:
        if ((4 in OSM_intersections_clipped['street_count'].values) or (3 in OSM_intersections_clipped['street_count'].values)):
            m4 = metric_4_share_3_and_4way_intersections(OSM_intersections_clipped)
        else:
            m4 = np.nan
        m5 = metric_5_4way_intersections(OSM_intersections_clipped, rectangle_area)    
    else:
        m4, m5 = np.nan, np.nan

    # Metric 6 -- building azimuth
    if not buildings_clipped.empty:
        n_orientation_groups = 4
        m6, buildings_clipped = metric_6_deviation_of_building_azimuth(buildings_clipped, n_orientation_groups, rectangle_id)
        #plot_azimuth(buildings_clipped, roads, rectangle_projected, rectangle_id, n_orientation_groups)
    else:
        m6 = np.nan

    # Metric 7 -- average block width
    # Metric 8 -- two-row blocks
    if not blocks_clipped.empty:
        rectangle_projected_arg = rectangle_projected.geometry
        m7, blocks_clipped = metric_7_average_block_width(blocks_clipped, rectangle_projected_arg, rectangle_area)
        #m7=np.nan
        m8, internal_buffers = metric_8_two_row_blocks(blocks_clipped, buildings_clipped, utm_proj_rectangle, row_epsilon=row_epsilon)
        #plot_largest_inscribed_circle(rectangle_id, rectangle_projected,  blocks_clipped, roads)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.01)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.1)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.001)
    else:
        print("Blocks_clipped is empty")
        m7, m8 = np.nan, np.nan

    
    if (not roads_clipped.empty):
        # Metric 9 -- tortuosity index
        rectangle_projected_arg = rectangle_projected.geometry
        #m9, all_road_vertices = metric_9_tortuosity_index(rectangle_id, roads_clipped, OSM_intersections_clipped, rectangle_projected_arg, angular_threshold=30, tortuosity_tolerance=5)
        m9 = metric_9_tortuosity_index_option_B(roads_clipped)
    else:
        m9 = np.nan

    if (not OSM_intersections_clipped.empty) and ((not roads.empty)):                                               
        # Metric 20 -- average angle between road segments
        m10 = metric_10_average_angle_between_road_segments(OSM_intersections_clipped, roads)
        #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads)
    else:
        m10 = np.nan

    if not roads_clipped.empty:
        road_length = roads_clipped.length.sum()
    else:
        road_length = np.nan

    if not buildings_clipped.empty:
        n_buildings = len(buildings_clipped)
        building_area = buildings_clipped.area.sum()
        building_density = n_buildings/rectangle_area
        built_share = building_area/rectangle_area
    else:
        n_buildings = np.nan

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
                        'OSM_buildings_available':OSM_buildings_bool,
                        'OSM_intersections_available':OSM_intersections_bool,
                        'OSM_roads_available':OSM_roads_bool,
                        'Overture_buildings_available':Overture_buildings_bool,
                        'rectangle_area': rectangle_area,
                        'building_area':building_area,
                        'building_density':building_density,
                        'share_tiled_by_blocks': share_tiled_by_blocks,
                        'built_share':built_share,
                        'road_length':road_length,
                        'n_intersections':len(OSM_intersections_clipped.drop_duplicates('osmid')),
                        'n_buildings':n_buildings
                        })

metrics_pilot = pd.DataFrame(metrics_pilot)
final_geo_df = gpd.GeoDataFrame(pd.merge(rectangles, metrics_pilot, how='left', left_on='n', right_on='index'), geometry=rectangles.geometry)


all_metrics_columns = ['metric_1','metric_2','metric_3','metric_4','metric_5','metric_6','metric_7','metric_8','metric_9','metric_10']
final_geo_df[all_metrics_columns]
metrics_with_magnitude = ['metric_2','metric_3','metric_5','metric_6','metric_7','metric_10']
#scalar_metrics = ['metric_2','metric_3','metric_5','metric_6','metric_7','metric_10']
not_inverted_metrics = ['metric_2','metric_6','metric_7','metric_8']

final_geo_df[['metric_2_original', 'metric_3_original', 'metric_5_original', 'metric_6_original', 'metric_7_original', 'metric_10_original']] = final_geo_df[['metric_2', 'metric_3', 'metric_5', 'metric_6', 'metric_7', 'metric_10']].copy()

final_geo_df.loc[:,all_metrics_columns] = (
    final_geo_df[all_metrics_columns]
    .apply(lambda x: (x - x.mean()) / (x.std()), axis=0)
)

final_geo_df.loc[:,all_metrics_columns] = (
    final_geo_df[all_metrics_columns]
    .apply(lambda x: (x - x.min()) / (x.max()-x.min()), axis=0)
)

metrics_to_invert = [col for col in all_metrics_columns if col not in not_inverted_metrics]
metrics_to_invert_names = [col+'_invert' for col in all_metrics_columns if col not in not_inverted_metrics]


final_geo_df[metrics_to_invert_names] = final_geo_df[metrics_to_invert].apply(lambda x: 1-x, axis=0)

final_geo_df['irregularity_index'] = final_geo_df[all_metrics_columns].mean(axis=1)
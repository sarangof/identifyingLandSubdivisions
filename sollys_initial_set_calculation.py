from metrics_calculation import *
from create_rectangles import *
#from gather_data_pilot import *
from metric_plots import *
import fiona

rectangles = gpd.read_file('./data/rectangles.geojson')

metrics_pilot = []
row_epsilon = 0.01

for rectangle_id, rectangle in rectangles.iterrows():
    try:
        # Read needed files: rectangle, buildings, streets
        rectangle_id += 1 
        print(rectangle_id)
        
        print(f"rectangle_id: {rectangle_id}")
        rectangle_centroid = rectangle.geometry.centroid
        utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))
        project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_rectangle, always_xy=True).transform  

        def project_geometry(geometry):
            if isinstance(geometry, Polygon):  
                return transform(project, geometry)
            else:
                return geometry  
            
        rectangle_projected = rectangle.apply(project_geometry) 

        OSM_buildings = gpd.read_file(f"./output_data/OSM_buildings_{rectangle_id}.gpkg")
        OSM_roads = gpd.read_file(f"./output_data/OSM_roads_{rectangle_id}.gpkg")
        OSM_intersections = gpd.read_file(f"./output_data/OSM_intersections_{rectangle_id}.gpkg").to_crs(utm_proj_rectangle)
        OSM_intersections_clipped = OSM_intersections.clip(list(rectangle_projected.geometry.bounds))

        Overture_data = gpd.read_file(f"./output_data/Overture_building_{rectangle_id}.geojson").to_crs(utm_proj_rectangle)#.clip(rectangle['geometry'])
        Overture_data['confidence'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['confidence'])
        Overture_data['dataset'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['dataset'])

        buildings_OSM = OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
        Overture_data = Overture_data.set_geometry('geometry')[Overture_data.dataset!='OpenStreetMap']
        buildings_OSM = buildings_OSM.set_geometry('geometry')
        buildings = gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle)
        buildings_clipped = buildings[buildings.geometry.intersects(rectangle_projected['geometry'])]

        roads = OSM_roads.to_crs(utm_proj_rectangle)
        roads_clipped = roads.clip(list(rectangle_projected.geometry.bounds))
        road_union = roads.unary_union # Create a unary union of the road geometries to simplify distance calculation

        blocks = get_blocks(road_union, roads)
        blocks_clipped = blocks[blocks.geometry.intersects(rectangle_projected['geometry'])]
        #blocks_clipped = blocks.clip(list(rectangle_projected.geometry.bounds))

        # Metric 1 -- share of buildings closer than 10 ms from the road
        m1, buildings_clipped = metric_1_distance_less_than_10m(buildings_clipped, road_union, utm_proj_rectangle)
        
        # Metric 2 -- average distance to roads
        m2 = metric_2_average_distance_to_roads(buildings_clipped)
        #plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id)

        # Metric 3 -- road density
        rectangle_area = calculate_area_geodesic(rectangle)
        m3  = metric_3_road_density(rectangle_area,roads_clipped)

        # Metrics 4 and 5 -- share of 3 and 4-way intersections
        if not OSM_intersections_clipped.empty:
            if ((4 in OSM_intersections_clipped['street_count'].values) or (3 in OSM_intersections_clipped['street_count'].values)):
                m4 = metric_4_share_3_and_4way_intersections(OSM_intersections_clipped)
            else:
                m4 = np.nan
            m5 = metric_5_4way_intersections(OSM_intersections_clipped)    
        else:
            m4, m5 = np.nan, np.nan
        
        # Metric 6 -- building azimuth
        m6, buildings_clipped = metric_6_deviation_of_building_azimuth(buildings_clipped)
        #plot_azimuth(buildings_clipped, roads, rectangle_projected, rectangle_id)

        # Metric 7 -- average block width
        # Metric 8 -- two-row blocks
        if not blocks_clipped.empty:
            m7, blocks_clipped = metric_7_average_block_width(blocks_clipped)
            m8, internal_buffers = metric_8_two_row_blocks(blocks_clipped, buildings_clipped, utm_proj_rectangle, row_epsilon=row_epsilon)
            #plot_largest_inscribed_circle(rectangle_id, rectangle_projected,  blocks_clipped)
            plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, row_epsilon)
        else:
            m7, m8 = np.nan, np.nan
        

        # Metric 9 -- tortuosity index
        m9, all_road_vertices = metric_9_tortuosity_index(roads_clipped, OSM_intersections_clipped, angular_threshold=30, tortuosity_tolerance=5)
        #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads)

        m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads)

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
                            'rectangle_area': rectangle_area,
                            'building_area':buildings.area.sum(),
                            'road_area':road_union.area,
                            'n_intersections':len(OSM_intersections),
                            'n_buildings':len(buildings)
                            })
    except fiona.errors.DriverError:
        print("File not available most likely.")
        continue

metrics_pilot = pd.DataFrame(metrics_pilot)

full_index = pd.Index(range(1, 50))
missing_indices = full_index.difference(metrics_pilot.index)
missing_df = pd.DataFrame(index=missing_indices, columns=metrics_pilot.columns)

# Concatenate the original DataFrame with the missing rows, and sort by index
metrics_pilot = pd.concat([metrics_pilot, missing_df]).sort_index()
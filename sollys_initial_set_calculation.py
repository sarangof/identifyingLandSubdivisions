from metrics_calculation import *
from create_rectangles import *
from gather_data_pilot import *

rectangles = gpd.read_file('./data/rectangles.geojson')

metrics_pilot = pd.DataFrame([])

for rectangle_id, rectangle in rectangles.iterrows():
    # Read needed files: rectangle, buildings, streets
    rectangle_id += 1 
    
    print(f"rectangle_id: {rectangle_id}")
    rectangle_centroid = rectangle.geometry.centroid
    utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_rectangle, always_xy=True).transform
    
    # I would like to replace these by the data query commands. 

    # Overture maps commands
    
    print("About to trigger overturemaps command")
    Overture_data = overturemaps_command(bbox_str = ','.join(rectangle[['minx', 'miny', 'maxx', 'maxy']].astype(str)), request_type = 'building')
    #if overture_file is not None:
    #    pass

    print("About to trigger osm command")
    OSM_buildings, OSM_intersections, OSM_roads = osmnx_command(bbox = [rectangle['xmin_expanded'],rectangle['ymin_expanded'],rectangle['xmax_expanded'],rectangle['ymax_expanded']])
    
    print("All data gathered")
    #OSM_buildings = gpd.read_file(f"./output_data/OSM_buildings_{rectangle_id}.gpkg")
    #OSM_roads = gpd.read_file(f"./output_data/OSM_roads_{rectangle_id}.gpkg")
    #OSM_intersections = gpd.read_file(f"./output_data/OSM_intersections_{rectangle_id}.gpkg").to_crs(utm_proj_rectangle)

    #Overture_data = gpd.read_file(f"./output_data/Overture_building_{rectangle_id}.geojson").to_crs(utm_proj_rectangle)#.clip(rectangle['geometry'])
    Overture_data['confidence'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['confidence'])
    Overture_data['dataset'] = Overture_data.sources.apply(lambda x: json.loads(x)[0]['dataset'])

    buildings_OSM = OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
    Overture_data = Overture_data.set_geometry('geometry')[Overture_data.dataset!='OpenStreetMap']
    buildings_OSM = buildings_OSM.set_geometry('geometry')
    buildings = gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle)

    roads = OSM_roads.to_crs(utm_proj_rectangle)
    road_union = roads.to_crs(utm_proj_rectangle).unary_union # Create a unary union of the road geometries to simplify distance calculation


    blocks = get_blocks(road_union, roads)
    #roads.plot()
    buildings.plot()

    m1 = metric_1_distance_less_than_10m(buildings)
    m2 = metric_2_average_distance_to_roads(buildings)
    #rectangle_area = calculate_area_geodesic(rectangle)
    #metric_3_road_density(roads,rectangle)
    m3=0
    m4 = metric_4_share_3_and_4way_intersections(OSM_intersections)
    m5 = metric_5_4way_intersections(OSM_intersections)    
    blocks = get_blocks(road_union,roads)
    blocks, m7 = metric_7_average_block_width(blocks)
    #blocks.to_file(f"blocks_tile_{rectangle_id}_with_weighted_width.gpkg", driver="GPKG")
    #print(metric_8_two_row_blocks(blocks,buildings,row_epsilon=0.1))
    blocks.plot()
    with open(f'buildings_17', 'w') as f:
                json.dump(mapping(buildings.geometry), f)
    roads.plot()
    m9, combined_points = metric_9_tortuosity_index(roads, OSM_intersections, angular_threshold=30, tortuosity_tolerance=5)
    #print(m9)
    combined_points.plot()
    metric_10_average_angle_between_road_segments(OSM_intersections, roads)

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
                          'metric_10':m10})


"""     # SOME DRAWINGS
    inflection_points_gdf = get_inflection_points(roads, threshold=30)
    inflection_points_gdf = inflection_points_gdf.set_crs(utm_proj_rectangle)

    # Plotting
    ax = roads.plot(figsize=(10, 10), color='blue', linewidth=1)
    OSM_intersections.to_crs(utm_proj_rectangle).plot(ax=ax, color='black', markersize=20)
    inflection_points_gdf.plot(ax=ax, color='red', markersize=15)

    # Optional: Show plot
    plt.show()
    optimal_point,max_radius = get_largest_inscribed_circle(blocks.iloc[0])
    draw_optimal_circle(blocks.iloc[0],optimal_point,max_radius) """
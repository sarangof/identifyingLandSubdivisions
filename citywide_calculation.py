from metrics_calculation import *
from create_rectangles import *
#from gather_data_pilot import *
from metric_plots import *
import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, box
from geopy.distance import geodesic
from pyproj import CRS


def create_square_from_coords(coord_pair, grid_size):
    x, y = coord_pair
    return box(x, y, x + grid_size, y + grid_size)


# Helper function to get UTM CRS based on city geometry centroid
def get_utm_crs(geometry):
    lon, lat = geometry.centroid.x, geometry.centroid.y
    utm_zone = int((lon + 180) // 6) + 1
    hemisphere = '6' if lat < 0 else '3'  # Southern hemisphere gets '6', northern gets '3'
    return CRS(f"EPSG:32{hemisphere}{utm_zone:02d}")

# Create grid function
def create_grid(geometry, grid_size):
    bounds = geometry.bounds  
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    grid_polygons = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid_polygons = [poly for poly in grid_polygons if geometry.intersects(poly)]
    lower_left_coords = [(p.bounds[0], p.bounds[1]) for p in grid_polygons]
    return lower_left_coords

# Main function to grid cities
def grid_cities(analysis_buffers, grid_size=100):
    city_grids = []

    for _, city in analysis_buffers.iterrows():
        city_name = city['city_name']
        geometry = city['geometry']
        city_gdf = gpd.GeoDataFrame(geometry=[geometry], crs='EPSG:4326')
        utm_crs = get_utm_crs(geometry)
        city_projected = city_gdf.to_crs(utm_crs)  
        grid = create_grid(city_projected.geometry[0], grid_size)
        city_grids.append({
            'city_name': city_name,
            'lower_left_coordinates': grid, 
            'grid_crs': utm_crs
        })
    result_df = gpd.GeoDataFrame(city_grids)
    
    return result_df


analysis_buffers = gpd.read_file('12 city analysis buffers.geojson')
search_buffers = gpd.read_file('12 city search buffers.geojson')


# Define important parameters for this run
grid_size = 200
row_epsilon = 0.01
city_name = 'Belo Horizonte'

#city_grids = grid_cities(analysis_buffers, grid_size=grid_size)
#city_grids = pd.read_csv('12 city grids.csv')



city_grid = gpd.read_file('Belo Horizonte_200m_grid.geojson')

metrics_pilot = []
#rectangles = city_grids[city_grids.city_name==city_name]['lower_left_coordinates'][0]
rectangles = city_grid['geometry']
Overture_data_all = gpd.read_file('Overture_buildings_Belo_Horizonte.gpkg')
#Overture_data_all = gpd.read_file(f"./output_data/12 cities/Overture_building_{city_name}.geojson")

Overture_data_all['confidence'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['confidence'])
Overture_data_all['dataset'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['dataset'])
Overture_data = Overture_data_all.set_geometry('geometry')[Overture_data_all.dataset!='OpenStreetMap']

OSM_intersections_all = gpd.read_file(f"./output_data/12 cities/{city_name}_osm_intersections.gpkg")
OSM_roads_all = gpd.read_file(f"./output_data/12 cities/{city_name}_osm_roads.gpkg")
cell_id = 0

rectangles = rectangles.sample(100)
for cell in rectangles[:10]:#iter(rectangles['lower_left_coordinates'][0])
    # Read needed files: rectangle, buildings, street
    cell_id += 1
    print(f"cell_id: {cell_id}")
    #coord_pair = cell
    #rectangle = gpd.GeoDataFrame(geometry=[create_square_from_coords(coord_pair, grid_size)],crs='EPSG:4326')
    rectangle = cell
    utm_proj_rectangle = rectangle.crs#rectangle.crs
    rectangle_centroid = rectangle.centroid#rectangle.geometry.centroid
    #utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))
    #project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_rectangle, always_xy=True).transform  

    def project_geometry(geometry):
        if isinstance(geometry, Polygon):  
            return transform(project, geometry)
        else:
            return geometry  
        
    #rectangle_projected = rectangle.apply(project_geometry) 

    #try:
        #OSM_buildings = OSM_buildings_all.clip(rectangle)
        #buildings_OSM = OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
        #buildings_OSM = buildings_OSM.set_geometry('geometry')
        #OSM_buildings_bool = True
    #except fiona.errors.DriverError:
    #    OSM_buildings = gpd.GeoDataFrame([])
    #    buildings_OSM = gpd.GeoDataFrame([])
    #    OSM_buildings_bool = False

    try:
        roads_window = OSM_roads_all.clip(rectangle).to_crs(utm_proj_rectangle) #need to expand 
        roads = roads.clip(list(rectangle_projected.geometry.bounds)) 
        road_union = roads.unary_union # Create a unary union of the road geometries to simplify distance calculation
        OSM_roads_bool = True
    except fiona.errors.DriverError:
        roads_window = gpd.GeoDataFrame([])
        OSM_roads_all = gpd.GeoDataFrame([])
        road_union = gpd.GeoDataFrame([])
        OSM_roads_bool = False

    try:
        OSM_intersections = OSM_intersections.clip(list(rectangle_projected.geometry.bounds))
        OSM_intersections_window = OSM_intersections.clip(list(rectangle_projected.geometry.bounds))
        OSM_intersections_bool = True
    except fiona.errors.DriverError:
        OSM_intersections = gpd.GeoDataFrame([])
        OSM_intersections_window = gpd.GeoDataFrame([])
        OSM_intersections_bool = False

    
    Overture_data = Overture_data_all.clip(list(rectangle_projected.geometry.bounds))
    if not Overture_data.empty:
        Overture_buildings_bool = True

    buildings = gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle).dropna(how='all')
    buildings_clipped = buildings[buildings.geometry.intersects(rectangle_projected['geometry'])]

    if not roads.empty:
        blocks = get_blocks(road_union, roads)
    else:
        blocks = gpd.GeoDataFrame([])
    if not blocks.empty:
        blocks_clipped = blocks[blocks.geometry.intersects(rectangle_projected['geometry'])]
    else:
        blocks_clipped = gpd.GeoDataFrame([])
    #blocks_clipped = blocks.clip(list(rectangle_projected.geometry.bounds))

    rectangle_area = calculate_area_geodesic(rectangle)
    if not calculate_minimum_distance_to_roads.empty:
        # Metric 1 -- share of buildings closer than 10 ms from the road
        m1, buildings = metric_1_distance_less_than_10m(buildings, road_union, utm_proj_rectangle)
        
        # Metric 2 -- average distance to roads
        m2 = metric_2_average_distance_to_roads(buildings)
        #plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id)

        # Metric 3 -- road density
        m3 = metric_3_road_density(rectangle_area,roads)
    else:
        m1, m2, m3 = np.nan, np.nan, np.nan

    # Metrics 4 and 5 -- share of 3 and 4-way intersections
    if not OSM_intersections.empty:
        if ((4 in OSM_intersections['street_count'].values) or (3 in OSM_intersections['street_count'].values)):
            m4 = metric_4_share_3_and_4way_intersections(OSM_intersections)
        else:
            m4 = np.nan
        m5 = metric_5_4way_intersections(OSM_intersections, rectangle_area)    
    else:
        m4, m5 = np.nan, np.nan

    # Metric 6 -- building azimuth
    if not buildings_clipped.empty:
        n_orientation_groups = 4
        m6, buildings_clipped = metric_6_deviation_of_building_azimuth(buildings, n_orientation_groups)
        #plot_azimuth(buildings_clipped, roads, rectangle_projected, rectangle_id, n_orientation_groups)
    else:
        m6 = np.nan

    # Metric 7 -- average block width
    # Metric 8 -- two-row blocks
    if not blocks_clipped.empty:
        m7, blocks_clipped = metric_7_average_block_width(blocks, rectangle_projected, rectangle_area)
        #m7=np.nan
        m8, internal_buffers = metric_8_two_row_blocks(blocks, buildings, utm_proj_rectangle, row_epsilon=row_epsilon)
        #plot_largest_inscribed_circle(rectangle_id, rectangle_projected,  blocks_clipped, roads)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.01)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.1)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon=0.001)
    else:
        m7, m8 = np.nan, np.nan

    
    if (not roads.empty) and (not OSM_intersections.empty):
        # Metric 9 -- tortuosity index
        m9, all_road_vertices = metric_9_tortuosity_index(city_name, roads, OSM_intersections, rectangle_projected, angular_threshold=30, tortuosity_tolerance=5)
                                                        
        # Metric 20 -- average angle between road segments
        m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads) #OJO, ROADS EXPANDED
        #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads)
    else:
        m9, m10 = np.nan, np.nan

    if not roads.empty:
        road_length = roads.length.sum()
    else:
        road_length = np.nan

    if not buildings.empty:
        n_buildings = len(buildings)
        building_area = buildings.area.sum()
    else:
        n_buildings = np.nan

    metrics_pilot.append({'index':city_name,
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
                        'road_length':road_length,
                        'n_intersections':len(OSM_intersections_clipped.drop_duplicates('osmid')),
                        'n_buildings':n_buildings
                        })

metrics_pilot = pd.DataFrame(metrics_pilot)

full_index = pd.Index(range(1, 50))
missing_indices = full_index.difference(metrics_pilot.index)
missing_df = pd.DataFrame(index=missing_indices, columns=metrics_pilot.columns)

# Concatenate the original DataFrame with the missing rows, and sort by index
metrics_pilot = pd.concat([metrics_pilot, missing_df]).sort_index()
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



#analysis_buffers = gpd.read_file('12 city analysis buffers.geojson')
#search_buffers = gpd.read_file('12 city search buffers.geojson')


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
Overture_data_all = gpd.read_file('./output_data/12 cities/Overture_buildings_{city_name}.geojson')
#Overture_data_all = gpd.read_file(f"./output_data/12 cities/Overture_building_{city_name}.geojson")

Overture_data_all['confidence'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['confidence'])
Overture_data_all['dataset'] = Overture_data_all.sources.apply(lambda x: json.loads(x)[0]['dataset'])
Overture_data = Overture_data_all.set_geometry('geometry')[Overture_data_all.dataset!='OpenStreetMap']

OSM_intersections_all = gpd.read_file(f"./output_data/12 cities/{city_name}_osm_intersections.gpkg")
OSM_roads_all = gpd.read_file(f"./output_data/12 cities/{city_name}_osm_roads.gpkg")
cell_id = 0


utm_proj_city = get_utm_proj(float(OSM_roads_all.iloc[0].geometry.centroid.x), float(OSM_roads_all.iloc[0].geometry.centroid.y))
project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_city, always_xy=True).transform  
OSM_roads_all_projected = gpd.GeoSeries(OSM_roads_all.geometry.apply(lambda x: transform(project,x))).set_crs(utm_proj_city,allow_override=True)

blocks_all  = get_blocks(OSM_roads_all_projected.unary_union, OSM_roads_all_projected)
Overture_data_all_projected = Overture_data_all.to_crs(epsg=CRS.from_proj4(utm_proj_city).to_epsg())

#rectangles_sample = rectangles.sample(10)
for cell in rectangles:
    #iter(rectangles['lower_left_coordinates'][0])
    # Read needed files: rectangle, buildings, street

    print(f"cell_id: {cell_id}")
    #coord_pair = cell
    #rectangle = gpd.GeoDataFrame(geometry=[create_square_from_coords(coord_pair, grid_size)],crs='EPSG:4326')
    rectangle = cell
    utm_proj_rectangle = rectangles.crs#rectangle.crs
    rectangle_centroid = rectangle.centroid#rectangle.geometry.centroid
    utm_proj_rectangle = get_utm_proj(float(rectangle_centroid.x), float(rectangle_centroid.y))
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), utm_proj_rectangle, always_xy=True).transform  

    def project_geometry(geometry):
        if isinstance(geometry, Polygon):  
            return transform(project, geometry)
        else:
            return geometry  

    rectangle_projected = gpd.GeoSeries(rectangle).apply(project_geometry).set_crs(utm_proj_rectangle)

    #try:
        #OSM_buildings = OSM_buildings_all.clip(rectangle)
    buildings_OSM = []#OSM_buildings[(OSM_buildings.building=='yes')].to_crs(utm_proj_rectangle)
    OSM_buildings_bool = False
        #buildings_OSM = buildings_OSM.set_geometry('geometry')
        #OSM_buildings_bool = True
    #except fiona.errors.DriverError:
    #    OSM_buildings = gpd.GeoDataFrame([])
    #    buildings_OSM = gpd.GeoDataFrame([])
    #    OSM_buildings_bool = False

    try:
        #roads_window = OSM_roads_all.clip(rectangle).to_crs(utm_proj_rectangle) #need to expand 
        roads_clipped = OSM_roads_all.to_crs(utm_proj_rectangle).clip(list(rectangle_projected.geometry.bounds.values[0])) 
        if roads_clipped.geometry.notnull().any():
            road_union = roads_clipped.unary_union
        else:
            road_union = None  # Or handle this case differently if needed

        OSM_roads_bool = True
    except fiona.errors.DriverError:
        roads_clipped = gpd.GeoDataFrame([])
        OSM_roads_all = gpd.GeoDataFrame([])
        road_union = gpd.GeoDataFrame([])
        OSM_roads_bool = False


    try:
        OSM_intersections = OSM_intersections_all.to_crs(utm_proj_rectangle).clip(list(rectangle_projected.geometry.bounds.values[0]))
        #OSM_intersections_window = OSM_intersections.clip(list(rectangle_projected.geometry.bounds.values[0]))
        OSM_intersections_bool = True
    except fiona.errors.DriverError:
        OSM_intersections = gpd.GeoDataFrame([])
        #OSM_intersections_window = gpd.GeoDataFrame([])
        OSM_intersections_bool = False

    
    Overture_data = Overture_data_all_projected[Overture_data_all_projected.geometry.intersects(rectangle_projected[0])]
    if not Overture_data.empty:
        Overture_buildings_bool = True
    else:
        Overture_buildings_bool = False

    buildings = Overture_data.to_crs(utm_proj_rectangle)#gpd.GeoDataFrame(pd.concat([buildings_OSM, Overture_data], axis=0, ignore_index=True, join='outer')).drop_duplicates('geometry').to_crs(utm_proj_rectangle).dropna(how='all')
    if not buildings.empty:
        buildings_clipped = buildings[buildings.geometry.intersects(rectangle_projected[0])]#buildings[buildings.geometry.intersects(rectangle_projected['geometry'])]
    else:
        buildings_clipped = gpd.GeoDataFrame([])

    blocks_clipped = blocks_all[blocks_all.geometry.intersects(rectangle_projected[0])]
    
    if not blocks_clipped.empty:
        blocks_bool = True
    else:
        blocks_bool = False
    #blocks_clipped = blocks.clip(list(rectangle_projected.geometry.bounds))

    geod = Geod(ellps="WGS84")
    rectangle_area, _ = geod.geometry_area_perimeter(rectangle)
    #rectangle_area = calculate_area_geodesic(gpd.GeoSeries(rectangle))
    if (not roads_clipped.empty) and (not buildings.empty):
        # Metric 1 -- share of buildings closer than 10 ms from the road
        m1, buildings = metric_1_distance_less_than_10m(buildings, road_union, utm_proj_rectangle)
        
        # Metric 2 -- average distance to roads
        m2 = metric_2_average_distance_to_roads(buildings)
        #plot_distance_to_roads(buildings_clipped, roads_clipped, rectangle_projected, cell_id)
    else:
        m1, m2 = np.nan, np.nan


    if (not roads_clipped.empty):
        # Metric 3 -- road density
        m3 = metric_3_road_density(rectangle_area,roads_clipped)
    else:
        m3 = np.nan

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
        #plot_azimuth(buildings_clipped, roads_clipped, rectangle_projected, rectangle_id, n_orientation_groups)
    else:
        m6 = np.nan

    # Metric 7 -- average block width
    # Metric 8 -- two-row blocks
    if not blocks_clipped.empty:
        m7, blocks_clipped = metric_7_average_block_width(blocks_clipped, rectangle_projected, rectangle_area)

        minx, miny, maxx, maxy = rectangle_projected.geometry.bounds.iloc[0]
        rectangle_box = box(minx, miny, maxx, maxy)
        blocks_clipped_within_rectangle = blocks_clipped.clip(rectangle_box)

        area_tiled_by_blocks = blocks_clipped_within_rectangle.area.sum()
        share_tiled_by_blocks = area_tiled_by_blocks/rectangle_area

        #m7=np.nan
        m8, internal_buffers = metric_8_two_row_blocks(blocks_clipped, buildings, utm_proj_rectangle, row_epsilon=row_epsilon)
        #plot_largest_inscribed_circle(rectangle_id, rectangle_projected,  blocks_clipped, roads_clipped)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads_clipped, row_epsilon=0.01)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads_clipped, row_epsilon=0.1)
        #plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads_clipped, row_epsilon=0.001)
    else:
        m7, m8 = np.nan, np.nan
        share_tiled_by_blocks = np.nan

    
    if (not roads_clipped.empty) and (not OSM_intersections.empty):
        # Metric 9 -- tortuosity index
        m9, all_road_vertices = metric_9_tortuosity_index(city_name, roads_clipped, OSM_intersections, rectangle_projected, angular_threshold=30, tortuosity_tolerance=5)
                                                        
        # Metric 20 -- average angle between road segments
        m10 = metric_10_average_angle_between_road_segments(OSM_intersections, roads_clipped) #OJO, ROADS EXPANDED
        #plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads_clipped)
    else:
        m9, m10 = np.nan, np.nan

    if not roads_clipped.empty:
        road_length = roads_clipped.length.sum()
    else:
        road_length = np.nan

    if not buildings.empty:
        n_buildings = len(buildings)
        building_area = buildings.area.sum()
        building_density = n_buildings/rectangle_area
        built_share = building_area/rectangle_area
    else:
        n_buildings = np.nan
        building_area = np.nan
        building_density = np.nan
        built_share = np.nan

    metrics_pilot.append({'index':cell_id,
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
                        'built_share':built_share,
                        'share_tiled_by_blocks':share_tiled_by_blocks,
                        'road_length':road_length,
                        'n_intersections':len(OSM_intersections.drop_duplicates('osmid')),
                        'n_buildings':n_buildings
                        })
    cell_id += 1

metrics_pilot = pd.DataFrame(metrics_pilot)
result = pd.merge(rectangles,metrics_pilot,how='left',left_index=True,right_index=True)
gpd.GeoDataFrame(result,geometry=result.geometry)
#save
# Imports
from shapely.ops import nearest_points
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from shapely.ops import transform, polygonize, unary_union
from scipy.optimize import fminbound, minimize
import matplotlib.pyplot as plt
from pyproj import Geod
import math
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import json
from shapely.geometry import mapping, Polygon

# DEFINE USEFUL FUNCTIONS

def get_utm_zone(lon):
    return int((lon + 180) // 6) + 1

def get_utm_proj(lon, lat):
    utm_zone = get_utm_zone(lon)
    is_northern = lat >= 0  # Determine if the zone is in the northern hemisphere
    return f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

# Simplified function to convert a distance in degrees to meters
def degrees_to_meters(distance_in_degrees, utm_proj_rectangle):
    
    # Create transformers for a small unit distance in degrees (1 degree) to meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_proj_rectangle, always_xy=True)
    
    # Convert 1 degree distance to meters (latitude = 0 -- assume small distance near the UTM zone)
    lon1, lat1 = 0, 0
    lon2, lat2 = 1, 0
    
    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)
    
    # Calculate the distance in meters for 1 degree of longitude
    meters_per_degree = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    # Convert the input distance in degrees to meters
    distance_in_meters = distance_in_degrees * meters_per_degree
    
    return distance_in_meters

def calculate_area_geodesic(rectangle):
    # Create a Geod object for geodesic calculations
    geod = Geod(ellps="WGS84")
    # Calculate the area using the geod object. The area is returned in square meters.
    area, _ = geod.geometry_area_perimeter(rectangle.geometry)
    return abs(area)

# Function to calculate the shortest distance to roads
def calculate_minimum_distance_to_roads(building, road_union):
    if isinstance(building, Polygon):
        # If it's a single Polygon, calculate distance directly
        nearest_geom = nearest_points(building.exterior, road_union)[1]
        return building.exterior.distance(nearest_geom)
    
    elif isinstance(building, MultiPolygon):
        # If it's a MultiPolygon, iterate over each polygon and calculate the minimum distance
        min_distance = float('inf')
        for poly in building.geoms:  # Use .geoms to access the individual Polygons
            nearest_geom = nearest_points(poly.exterior, road_union)[1]
            distance = poly.exterior.distance(nearest_geom)
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    else:
        raise TypeError("Unsupported geometry type: {}".format(type(building)))


# Function to calculate the angle between two vectors
def calculate_angle(vector1, vector2):
    angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
    angle = np.degrees(angle)
    if angle < 0:
        angle += 360
    return angle


def extract_coords(geometry):
    if isinstance(geometry, (LineString, Polygon)):
        # If it's a single geometry, return its coordinates
        return list(geometry.coords)
    elif isinstance(geometry, (MultiLineString, MultiPolygon)):
        # If it's a multi-part geometry, iterate through the sub-geometries
        coords = []
        for part in geometry.geoms:  # Use .geoms to access each part of the MultiLineString/MultiPolygon
            coords.extend(list(part.coords))
        return coords
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")



def calculate_sequential_angles(intersections, roads):
    records = []  # List to store angle records

    # Iterate through each intersection
    for _, intersection in intersections.iterrows():
        intersection_id = intersection['osmid']
        intersection_point = intersection.geometry
        
        # Get all roads connected to the intersection
        connected_roads = roads[(roads['u'] == intersection_id) | (roads['v'] == intersection_id)]
        vectors = []
        
        for _, road in connected_roads.iterrows():
            #coords = list(road.geometry.coords)
            coords = extract_coords(road.geometry)
            
            # Determine the vector for the road segment away from the intersection
            if road['u'] == intersection_id:
                vector = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
            else:
                vector = (coords[-2][0] - coords[-1][0], coords[-2][1] - coords[-1][1])
            
            vectors.append((vector, road['u'], road['v']))

        # Sort vectors based on the angle relative to a fixed axis (e.g., x-axis)
        vectors.sort(key=lambda v: np.arctan2(v[0][1], v[0][0]))

        # Calculate the sequential angles between each pair of vectors
        for i in range(len(vectors)):
            vector1 = vectors[i][0]
            vector2 = vectors[(i + 1) % len(vectors)][0]  # Next vector, looping back to the start
            angle = calculate_angle(vector1, vector2)
            
            record = {
                'Intersection ID': intersection_id,
                'Segment 1': (vectors[i][1], vectors[i][2]),
                'Segment 2': (vectors[(i + 1) % len(vectors)][1], vectors[(i + 1) % len(vectors)][2]),
                'Angle': angle
            }
            records.append(record)

    # Create a DataFrame from the records
    df_angles = pd.DataFrame(records)
    
    return df_angles

# Block polygons

def get_blocks(road_union, roads):
    # Use polygonize to create polygons from the merged lines
    blocks = list(polygonize(road_union))
    blocks_gdf = gpd.GeoDataFrame(geometry=blocks, crs=roads.crs)
    # Filter out very small polygons if necessary
    blocks_gdf = blocks_gdf[blocks_gdf.area > 9]
    blocks_gdf.loc[:,'area'] = blocks_gdf.area
    #blocks_gdf.to_file("blocks_tile1.gpkg", driver="GPKG")
    return blocks_gdf.sort_values('area')

def longest_segment(geometry):
    # Flatten geometry by extracting all Polygon components from MultiPolygon or Polygon
    if isinstance(geometry, Polygon):
        polygons = [geometry]  # Treat single Polygon as a list with one element
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)  # Extract Polygons from MultiPolygon
    else:
        raise TypeError("The input must be a shapely Polygon or MultiPolygon object.")
    
    # Initialize variables to track the longest segment
    max_length = 0
    longest_segment = None
    
    # Iterate over each Polygon in the flattened geometry
    for polygon in polygons:
        exterior_coords = polygon.exterior.coords
        
        # Iterate through the exterior coordinates to find the longest segment
        for i in range(len(exterior_coords) - 1):
            # Create a line segment from consecutive coordinates
            segment = LineString([exterior_coords[i], exterior_coords[i+1]])
            
            # Calculate the length of the segment
            segment_length = segment.length
            
            # Update the longest segment if this one is longer
            if segment_length > max_length:
                max_length = segment_length
                longest_segment = segment
    
    return longest_segment

def calculate_azimuth(segment):
    # Extract start and end points of the segment
    start_point = segment.coords[0]
    end_point = segment.coords[1]
    
    # Calculate the difference in coordinates
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    
    # Calculate the azimuth in radians
    azimuth_rad = math.atan2(delta_x, delta_y)
    
    # Convert the azimuth to degrees
    azimuth_deg = math.degrees(azimuth_rad)
    
    # Normalize the azimuth to be within 0 to 360 degrees
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return np.abs(azimuth_deg)

# Get largest circle inscribed in block
def get_largest_inscribed_circle(block):

    polygon = block
    # Initial guess: the centroid of the polygon
    centroid = polygon.geometry.centroid

    if not polygon.geometry.contains(centroid):
        # If centroid is outside, find an interior point as an alternative starting point
        interior_point = polygon.geometry.representative_point()  # A guaranteed point inside the polygon
        initial_guess = [interior_point.x, interior_point.y]
    else:
        initial_guess = [centroid.x, centroid.y]

    # Calculate negative radius to maximize
    def negative_radius(point_coords):
        point = Point(point_coords)
        if polygon.geometry.contains(point):
            return -polygon.geometry.boundary.distance(point)
        else:
            return np.inf  # Outside the polygon, so invalid
    # Optimization to find the largest inscribed circle
    result = minimize(negative_radius, initial_guess, method='Nelder-Mead')
    # Get the maximum inscribed circle
    optimal_point = Point(result.x)
    max_radius = -result.fun  # Negative of the minimized value
    #print(f"Largest circle center: {optimal_point}")
    #print(f"Maximum radius: {max_radius}")
    return optimal_point, max_radius

def objective_function(radius, block, target_area):
    internal_buffer = block.geometry.difference(block.geometry.buffer(-radius))
    # Handle cases where the result is empty or not a valid polygon
    if internal_buffer.is_empty or not isinstance(internal_buffer, (Polygon, MultiPolygon)):
        return float('inf')
    buffer_area = internal_buffer.area
    return abs(buffer_area - target_area)

def get_internal_buffer_with_target_area(block, target_area, tolerance=1e-6):
    # Get the largest inscribed circle's radius for the upper bound
    _, max_radius = get_largest_inscribed_circle(block)
    # Use fminbound to find the radius that minimizes the area difference
    optimal_radius = fminbound(objective_function, 0, max_radius, args=(block, target_area), xtol=tolerance)
    # Compute the internal buffer with the optimal radius
    internal_buffer = block.geometry.difference(block.geometry.buffer(-optimal_radius))
    return internal_buffer

# Inflection points
def get_inflection_points(roads,threshold):
    inflection_gdf = gpd.GeoDataFrame({'geometry':[],'angle':[]})
    for row in roads.iterfeatures():
        line = row['geometry']
        inflection_points = []
        angles = []
        coords = list(line['coordinates'])
        for i in range(1, len(coords) - 1):
            p1 = np.array(coords[i - 1])
            p2 = np.array(coords[i])
            p3 = np.array(coords[i + 1])

            # Calculate angle between the segments
            v1 = p2 - p1
            v2 = p3 - p2

            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])

            # Calculate the difference and convert to degrees
            angle_diff = np.degrees(np.abs(angle2 - angle1))
            angle_diff = np.mod(angle_diff, 360)

            # Normalize the angle difference to [0, 180] degrees
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # Store significant changes (e.g., greater than 10 degrees)
            if angle_diff > threshold:  
                inflection_points.append(Point(p2))
                angles.append(angle_diff)
        inflection_dict = gpd.GeoDataFrame({
            'geometry': inflection_points,
            'angle': angles
        })
        if len(inflection_points)>0:
            inflection_gdf = pd.concat([inflection_gdf,inflection_dict])
    return inflection_gdf

#1 Share of building footprints that are less than 10-meters away from the nearest road
def metric_1_distance_less_than_10m(buildings, road_union, utm_proj_rectangle):
    # Apply the distance calculation to each building
    #buildings.loc[:,'distance_to_road'] = buildings['geometry'].apply(lambda x: x.centroid).apply(calculate_minimum_distance_to_roads, 
    #                                                                                              road_union = road_union)
    buildings.loc[:,'distance_to_road'] = buildings['geometry'].apply(lambda x: calculate_minimum_distance_to_roads(x, road_union))
    m1 = 1.*((sum(buildings['distance_to_road']<=10))/len(buildings))
    return m1, gpd.GeoDataFrame(buildings)

#2 Average distance of building footprint centroids to roads
# THIS FUNCTION CAN ONLY BE CALLED AFTER metric_1
def metric_2_average_distance_to_roads(buildings):
    m2 = buildings['distance_to_road'].mean()
    return m2

#3 Density of roads
def metric_3_road_density(rectangle_area,roads_clipped):
    km_length = roads_clipped.length.sum()/1000.
    rectangle_area_km2 = rectangle_area/1000000.
    m3 = km_length/rectangle_area_km2
    return m3

#4 Share of 3-way and 4-way intersections 
def metric_4_share_3_and_4way_intersections(intersections):
    n_intersections_3_and_4 = 1.*len(intersections[(intersections.street_count == 4)|(intersections.street_count == 3)])
    n_4_way = 1.*len(intersections[(intersections.street_count == 4)])
    m4 = (n_4_way / n_intersections_3_and_4)
    return m4

#5 Density of intersections
def metric_5_4way_intersections(intersections, rectangle_area):
    m5 = (1000.**2)*(len(intersections[(intersections.street_count >= 3)])/rectangle_area)
    return m5

#6 Average building footprint orientation of the tile
def metric_6_deviation_of_building_azimuth(buildings_clipped, n_orientation_groups):
    buildings_clipped.loc[:,'azimuth'] = buildings_clipped['geometry'].apply(lambda x: calculate_azimuth(longest_segment(x)) % 90.)
    cutoff_angles = [(x*90./(2*n_orientation_groups)) for x in range(0,(2*n_orientation_groups)+1)]
    labels = [1] + [i for i in range(2, n_orientation_groups+1) for _ in (0, 1)] + [1]
    buildings_clipped.loc[:,'azimuth_group'] = pd.DataFrame(pd.cut(buildings_clipped.azimuth.values.reshape(-1,),bins=cutoff_angles,labels=labels,ordered=False))[0].values
    mean_azimuth = buildings_clipped['azimuth'].mean()
    m6 = np.std(np.abs(buildings_clipped['azimuth'] - mean_azimuth))
    return m6, buildings_clipped

#7 Average block width
def metric_7_average_block_width(blocks_clipped, rectangle_projected, rectangle_area):

    blocks_within_rectangle = []
    radius_avg = []

    for block_id, block in blocks_clipped.iterrows():
        optimal_point, max_radius = get_largest_inscribed_circle(block)
        block_copy = gpd.GeoSeries(block.copy().geometry).set_crs(blocks_clipped.crs)
        rectangle_geom = gpd.GeoSeries(rectangle_projected.geometry)
        block_within_rectangle = (block_copy.intersection(rectangle_geom))
        block_area_within_rectangle = block_within_rectangle.area.sum()
        block_weight = block_area_within_rectangle / rectangle_area
        weighted_width = block_weight*max_radius
        blocks_clipped.loc[block_id,'weighted_width'] = weighted_width
        blocks_within_rectangle.append(block_within_rectangle)
        radius_avg.append(max_radius)

    rectangle_projected_gdf = gpd.GeoSeries(rectangle_projected.geometry)
    rectangle_projected_gdf = rectangle_projected_gdf.set_crs(blocks_clipped.crs, allow_override=True)
    rectangle_projected_gdf = rectangle_projected_gdf.reset_index(drop=True)
    blocks_clipped = blocks_clipped.reset_index(drop=True)
    blocks_union = blocks_clipped.unary_union
    left_over_blocks = gpd.GeoDataFrame(geometry=gpd.GeoDataFrame(geometry=rectangle_projected_gdf).difference(blocks_union).geometry)
    left_over_blocks = left_over_blocks.explode(index_parts=False).reset_index(drop=True)
    left_over_blocks = left_over_blocks[left_over_blocks.is_valid & ~left_over_blocks.is_empty]


    if left_over_blocks.area.sum() > 0.0:
        for block_id, leftover_block in left_over_blocks.iterrows():
            optimal_point, max_radius = get_largest_inscribed_circle(leftover_block)
            block_within_rectangle = (gpd.GeoSeries(leftover_block.geometry).intersection(gpd.GeoSeries(rectangle_projected.geometry)))
            block_area_within_rectangle = block_within_rectangle.area.sum()
            block_weight = block_area_within_rectangle / rectangle_area
            weighted_width = block_weight*max_radius
            left_over_blocks.loc[block_id,'weighted_width'] = weighted_width
        m7 = blocks_clipped['weighted_width'].sum() + left_over_blocks['weighted_width'].sum()
        blocks_clipped = pd.concat([blocks_clipped,left_over_blocks])
    else:
        m7 = blocks_clipped['weighted_width'].sum()

    return m7, blocks_clipped

#8 Two row blocks
def metric_8_two_row_blocks(blocks_clipped, buildings, utm_proj_rectangle, row_epsilon):

    internal_buffers = []
    for block_id, block in blocks_clipped.iterrows():
        block_area = block.area
        #optimal_point, max_radius = get_largest_inscribed_circle(block)
        target_area = block_area*(1.-row_epsilon)  # Set your target area
        internal_buffer = get_internal_buffer_with_target_area(block, target_area)
        #print(type(internal_buffer))
        buildings_in_block = buildings.clip(block.geometry)
        buildings_for_union = unary_union(buildings_in_block.geometry)
        result_union = internal_buffer.union(buildings_for_union)
        union_area = result_union.area
        internal_buffer_area = internal_buffer.area  

        internal_buffer = block.geometry.difference(internal_buffer)
        buildings_intersecting_buffer = buildings_in_block[buildings_in_block.geometry.intersects(internal_buffer)]
        buildings_intersecting_buffer_area = buildings_intersecting_buffer.area.sum()
        buildings_area_all = buildings_in_block.area.sum()
        buildings_inside_buffer_area = buildings.intersection(internal_buffer).area.sum()
        internal_buffers.append(internal_buffer)
        if buildings_intersecting_buffer_area > 0:
            blocks_clipped.loc[block_id,'share_of_buildings_inside_buffer_intersection'] = buildings_inside_buffer_area/buildings_intersecting_buffer_area
        else:
            blocks_clipped.loc[block_id,'share_of_buildings_inside_buffer_intersection'] = np.nan
        
        if buildings_area_all > 0:
            blocks_clipped.loc[block_id,'share_of_buildings_inside_buffer_all'] = buildings_inside_buffer_area/buildings_area_all
        else:
            blocks_clipped.loc[block_id,'share_of_buildings_inside_buffer_all'] = np.nan
        #if union_area > internal_buffer_area:
        #    blocks_clipped.loc[block_id,'buildings_outside_buffer'] = True
        #elif union_area <= internal_buffer_area:
        #    blocks_clipped.loc[block_id,'buildings_outside_buffer'] = False
        #m8 = blocks_clipped['buildings_outside_buffer'].mean()
    internal_buffers = gpd.GeoDataFrame(geometry=internal_buffers).set_crs(utm_proj_rectangle)
    #m8 = blocks_clipped.share_of_buildings_inside_buffer_intersection.mean()
    m8 = blocks_clipped.share_of_buildings_inside_buffer_all.mean()

    return m8, internal_buffers


def visualize_tortuosity(rectangle_id, angular_threshold, tortuosity_tolerance, roads_clipped, all_road_vertices, mst, distance_comparison):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Step 1: Plot the roads
    roads_clipped.plot(ax=ax, color='blue', linewidth=0.8, label='Clipped Roads')

    # Step 2: Plot all the road vertices (intersections, inflection points, boundary intersections)
    all_road_vertices.plot(ax=ax, marker='o', color='green', markersize=5, label='All Road Vertices')

    # Step 3: Visualize the MST edges
    for edge in mst.edges(data=True):
        i, j, data = edge
        if i < len(all_road_vertices) and j < len(all_road_vertices):  # Ensure valid indices
            point_A = all_road_vertices.iloc[i].geometry
            point_B = all_road_vertices.iloc[j].geometry
            # Plot the MST edge in black dashed lines
            plt.plot([point_A.x, point_B.x], [point_A.y, point_B.y], 'k--', linewidth=1, label='MST Connection' if i == 0 else "")

    # Step 4: Plot the Euclidean and Road Network distances from the distance_comparison
    for i in range(len(distance_comparison)):
        point_A = distance_comparison['Point_A'].iloc[i]
        point_B = distance_comparison['Point_B'].iloc[i]

        # Plot Euclidean Distance (Red Line)
        plt.plot([point_A.x, point_B.x], [point_A.y, point_B.y], color='red', linestyle='-', linewidth=1, label='Euclidean Distance' if i == 0 else "")

        # Plot Road Network Distance (Black Line)
        nearest_A = nearest_points(roads_clipped.unary_union, point_A)[0]
        nearest_B = nearest_points(roads_clipped.unary_union, point_B)[0]
        plt.plot([nearest_A.x, nearest_B.x], [nearest_A.y, nearest_B.y], color='black', linestyle='-', linewidth=1, label='Road Network Distance' if i == 0 else "")

    # Add legend and titles
    plt.legend()
    plt.title('Tortuosity Visualization: Roads, MST, and Distances')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # Show the plot
    plt.savefig(f'pilot_plots/plot_two_row_blocks_{rectangle_id}_angular_threshold_{str(angular_threshold)}_tolerance_{str(tortuosity_tolerance)}.png', dpi=500, bbox_inches='tight')


 
#9 Tortuosity index
def metric_9_tortuosity_index(rectangle_id, roads_clipped, intersections, rectangle_projected, angular_threshold, tortuosity_tolerance):

    # Calculate hypotenuse of the rectangle
    rectangle_projected_gdf = gpd.GeoSeries(rectangle_projected.geometry)
    rectangle_bounds = rectangle_projected_gdf.bounds
    width = rectangle_bounds['maxx'] - rectangle_bounds['minx']
    height = rectangle_bounds['maxy'] - rectangle_bounds['miny']
    rectangle_hypotenuse = math.sqrt(width.iloc[0]**2 + height.iloc[0]**2)
    distance_threshold = 0.75 * rectangle_hypotenuse  # 75% of the hypotenuse

    inflection_points_gdf = get_inflection_points(roads_clipped, angular_threshold)

    ### This part is about boundary vertices

    rectangle_boundary = rectangle_projected_gdf.geometry.unary_union.boundary
    roads_union = roads_clipped.geometry.unary_union
    intersection_result = roads_union.intersection(rectangle_boundary)
    boundary_intersection_points = []

    if intersection_result.is_empty:
        print("No intersections found.")
    else:
        # Handle MultiPoint
        if intersection_result.geom_type == 'MultiPoint':
            boundary_intersection_points.extend([point for point in intersection_result.geoms])
        # Handle LineString or MultiLineString
        elif intersection_result.geom_type in ['LineString', 'MultiLineString']:
            for line in intersection_result.geoms:
                boundary_intersection_points.extend([Point(coord) for coord in line.coords])

    # Convert to GeoDataFrame
    boundary_intersection_points_gdf = gpd.GeoDataFrame(geometry=boundary_intersection_points)
    boundary_intersection_points_gdf['point_type'] = 'boundary intersection'

    all_road_vertices = pd.concat([inflection_points_gdf, intersections, boundary_intersection_points_gdf], ignore_index=True)
    all_road_vertices.point_type = all_road_vertices.point_type.fillna('inflection point')

    # Handle intersections
    all_road_vertices.loc[all_road_vertices.geometry.isin(intersections.geometry), 'point_type'] = 'intersection'
    all_road_vertices = all_road_vertices.drop_duplicates(subset='geometry', keep='last')

    if not all_road_vertices.empty:
        all_road_vertices = all_road_vertices.set_geometry('geometry')
        all_road_vertices.set_crs(roads_clipped.crs, inplace=True)

        # Create a graph
        G = nx.Graph()

        # Add nodes
        for idx, point in all_road_vertices.iterrows():
            G.add_node(idx, geometry=point.geometry)

        # Add edges based on proximity
        for i in range(len(all_road_vertices)):
            for j in range(i + 1, len(all_road_vertices)):
                point_A = all_road_vertices.iloc[i].geometry
                point_B = all_road_vertices.iloc[j].geometry
                
                # Find the nearest points on the road network for A and B
                nearest_A = nearest_points(roads_clipped.unary_union, point_A)[0]
                nearest_B = nearest_points(roads_clipped.unary_union, point_B)[0]

                # Calculate road network distance
                road_network_distance = roads_clipped.geometry.length.loc[
                    roads_clipped.intersects(nearest_A.buffer(1e-6)) & roads_clipped.intersects(nearest_B.buffer(1e-6))
                ].min()

                # Handle NaN values
                if np.isnan(road_network_distance):
                    road_network_distance = 1e6  # Arbitrary large number

                # Calculate Euclidean distance
                euclidean_distance = point_A.distance(point_B)

                # Apply the distance threshold for the road network
                if euclidean_distance < distance_threshold:
                    # Add edge to the graph
                    G.add_edge(i, j, weight=road_network_distance)

        # Find the minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        ordered_indices = list(nx.dfs_preorder_nodes(mst, source=0))  # Starting from the first point
        ordered_vertices = all_road_vertices.iloc[ordered_indices].reset_index(drop=True)

        # Initialize lists to store distances
        euclidean_distances = []
        road_network_distances = []
        indices_to_keep = []

        # Iterate through contiguous points
        for i in range(len(ordered_vertices) - 1):
            point_A = ordered_vertices.iloc[i].geometry
            point_B = ordered_vertices.iloc[i + 1].geometry

            # Calculate the straight-line Euclidean distance
            euclidean_distance = point_A.distance(point_B)

            # Find the nearest points on the road network for A and B
            nearest_A = nearest_points(roads_clipped.unary_union, point_A)[0]
            nearest_B = nearest_points(roads_clipped.unary_union, point_B)[0]

            # Calculate road network distance
            road_network_distance = roads_clipped.geometry.length.loc[
                roads_clipped.intersects(nearest_A.buffer(1e-6)) & roads_clipped.intersects(nearest_B.buffer(1e-6))
            ].min()

            if road_network_distance > tortuosity_tolerance and euclidean_distance < distance_threshold:
                euclidean_distances.append(euclidean_distance)
                road_network_distances.append(road_network_distance)
                indices_to_keep.append(i)

        # Combine the results into a DataFrame
        distance_comparison = pd.DataFrame({
            'Point_A': ordered_vertices.geometry[:-1].reset_index(drop=True).iloc[indices_to_keep],
            'Point_B': ordered_vertices.geometry[1:].reset_index(drop=True).iloc[indices_to_keep],
            'Euclidean_Distance': euclidean_distances,
            'Road_Network_Distance': road_network_distances
        })

        # Calculate tortuosity index (mean ratio)
        m9 = (distance_comparison['Euclidean_Distance'] / distance_comparison['Road_Network_Distance']).mean()
        #visualize_tortuosity(rectangle_id, angular_threshold, tortuosity_tolerance, roads_clipped, all_road_vertices, mst, distance_comparison)
        print(m9)
    else:
        m9 = np.nan

    
    return m9, all_road_vertices

#10 Average angle between road segments
def metric_10_average_angle_between_road_segments(intersections, roads):
    df_angles = calculate_sequential_angles(intersections, roads)
    intersection_angles_df = intersections[['osmid','street_count']].set_index('osmid').merge(df_angles.set_index('Intersection ID'),left_index=True,right_index=True,how='outer')
    # In 3-way intersections, include only the smallest angle in the tile average. 
    df_3_way = intersection_angles_df[(intersection_angles_df.street_count==3)]
    if not df_3_way.empty:
        to_keep_3 = df_3_way.reset_index().loc[(df_3_way.reset_index().groupby(df_3_way.index)['Angle'].idxmin())]#.set_index('index')
    else:
        to_keep_3 = pd.DataFrame([])
    # In 4-way intersections, include only the two smallest angles in the tile average.
    df_4_way = intersection_angles_df[intersection_angles_df.street_count==4]
    if not df_4_way.empty:
        to_keep_4 = df_4_way.groupby(df_4_way.index).apply(lambda x: x.nsmallest(2, 'Angle')).reset_index(drop=True)   
    else:
        to_keep_4 = pd.DataFrame([])
    to_keep_4.index.names = ['index']
    to_keep_df = pd.concat([to_keep_3,to_keep_4])
    if not to_keep_df.empty:
        m10 = np.std(np.abs(90. - to_keep_df['Angle']))
    else:
        m10 = np.nan
    return m10
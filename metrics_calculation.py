# Imports
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod, Transformer
import matplotlib.pyplot as plt
from math import atan2, degrees
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString, GeometryCollection
from shapely.ops import polygonize, nearest_points
from shapely.strtree import STRtree
from scipy.optimize import fminbound, minimize
from scipy.spatial import cKDTree
from scipy.stats import t, sem, entropy

# DEFINE USEFUL FUNCTIONS

plots_path = '../plots/pilot'

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

def get_utm_zone(lon):
    return int((lon + 180) // 6) + 1

def get_utm_proj(lon, lat):
    utm_zone = get_utm_zone(lon)
    is_northern = lat >= 0  # Determine if the zone is in the northern hemisphere
    return f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

# Simplified function to convert a distance in degrees to meters
def degrees_to_meters(distance_in_degrees, utm_proj_rectangle):
    
    # Create transformers for a small unit distance in degrees (1 degree) to meters
    transformer = Transformer.from_crs("EPSG:4326", utm_proj_rectangle, always_xy=True)
    
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


def calculate_minimum_distance_to_roads_option_B(building, road_union):
    # Handle Polygon and MultiPolygon cases more efficiently
    def get_minimum_distance(polygon_exterior, road_union):
        nearest_geom = nearest_points(polygon_exterior, road_union)[1]
        return polygon_exterior.distance(nearest_geom)

    if isinstance(building, Polygon):
        # For a single Polygon, calculate distance directly
        return get_minimum_distance(building.exterior, road_union)

    elif isinstance(building, MultiPolygon):
        # For MultiPolygon, iterate over polygons but track the minimum in one loop
        return min(get_minimum_distance(poly.exterior, road_union) for poly in building.geoms)

    else:
        raise TypeError(f"Unsupported geometry type: {type(building)}")

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

def calculate_sequential_angles_option_A(intersections, roads):
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

def calculate_sequential_angles(intersections, roads):
    records = []  # List to store angle records

    # Convert to numpy arrays for faster processing
    intersection_ids = intersections['osmid'].values
    intersection_geometries = intersections.geometry.values
    road_u = roads['u'].values
    road_v = roads['v'].values
    road_geometries = roads.geometry.values

    # Iterate over each intersection
    for idx, intersection_id in enumerate(intersection_ids):
        intersection_point = intersection_geometries[idx]
        
        # Get all roads connected to the intersection
        mask_connected_roads = (road_u == intersection_id) | (road_v == intersection_id)
        connected_roads = road_geometries[mask_connected_roads]
        
        vectors = []

        # Create vectors for each road segment
        for road_geometry, u, v in zip(connected_roads, road_u[mask_connected_roads], road_v[mask_connected_roads]):
            coords = extract_coords(road_geometry)
            
            # Determine the vector for the road segment away from the intersection
            if u == intersection_id:
                vector = np.array([coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]])
            else:
                vector = np.array([coords[-2][0] - coords[-1][0], coords[-2][1] - coords[-1][1]])

            vectors.append((vector, u, v))

        # Convert to numpy array for faster angle calculations
        vectors = np.array(vectors, dtype=object)
        
        # Sort vectors based on their angle relative to the x-axis using arctan2
        vectors = sorted(vectors, key=lambda v: np.arctan2(v[0][1], v[0][0]))

        # Calculate the sequential angles between each pair of vectors
        num_vectors = len(vectors)
        for i in range(num_vectors):
            vector1 = vectors[i][0]
            vector2 = vectors[(i + 1) % num_vectors][0]  # Next vector, looping back to the start

            # Directly calculate the angle between the two vectors
            angle = calculate_angle(vector1, vector2)
            
            record = {
                'Intersection ID': intersection_id,
                'Segment 1': (vectors[i][1], vectors[i][2]),
                'Segment 2': (vectors[(i + 1) % num_vectors][1], vectors[(i + 1) % num_vectors][2]),
                'Angle': angle
            }
            records.append(record)

    # Convert the records into a DataFrame
    df_angles = pd.DataFrame(records)
    
    return df_angles

def longest_segment_option_A(geometry):
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

def longest_segment(geometry):
    # Handle both Polygon and MultiPolygon
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]

    max_length = 0
    longest_segment = None
    
    # Iterate over each Polygon in the geometry
    for polygon in polygons:
        exterior_coords = polygon.exterior.coords
        
        # Use a more efficient calculation to avoid creating LineString for every segment
        for i in range(len(exterior_coords) - 1):
            # Directly calculate the segment length using coordinates
            x1, y1 = exterior_coords[i]
            x2, y2 = exterior_coords[i+1]
            segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance
            
            # Update the longest segment if the current one is longer
            if segment_length > max_length:
                max_length = segment_length
                longest_segment = ((x1, y1), (x2, y2))  # Store as a tuple of coordinates

    # Return the longest segment as a LineString (shapely object) for consistency
    return LineString(longest_segment)

def calculate_azimuth(segment):
    # Extract start and end points of the segment
    start_point = segment.coords[0]
    end_point = segment.coords[1]
    
    # Calculate the difference in coordinates
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    
    # Calculate the azimuth in radians
    azimuth_rad = atan2(delta_x, delta_y)
    
    # Convert the azimuth to degrees
    azimuth_deg = degrees(azimuth_rad)
    
    # Normalize the azimuth to be within 0 to 360 degrees
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return np.abs(azimuth_deg)

# Get largest circle inscribed in block
def get_largest_inscribed_circle(block):
    polygon = block
    # Initial guess: the centroid of the polygon
    centroid = polygon.geometry.centroid
    if not polygon.geometry.contains(centroid):
    #if not polygon.geometry.iloc[0].contains(centroid).values[0]:
        # If centroid is outside, find an interior point as an alternative starting point
        interior_point = polygon.geometry.representative_point()  # A guaranteed point inside the polygon
        initial_guess = [interior_point.x, interior_point.y]
    else:
        initial_guess = [centroid.x, centroid.y]
        #initial_guess = [centroid.x[0], centroid.y[0]]
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

def get_largest_inscribed_circle_option_B(polygon):
    # Precompute the polygon's boundary as a MultiPoint for fast spatial indexing
    boundary_points = np.array(polygon.geometry.boundary.coords)
    boundary_tree = STRtree([Point(p) for p in boundary_points])  # Spatial index using STRtree
    
    # Initial guess: the centroid of the polygon
    centroid = polygon.geometry.centroid

    if not polygon.geometry.contains(centroid):
        # If centroid is outside, find an interior point as an alternative starting point
        interior_point = polygon.geometry.representative_point()  # A guaranteed point inside the polygon
        initial_guess = [interior_point.x, interior_point.y]
    else:
        initial_guess = [centroid.x, centroid.y]

    # Calculate negative radius to maximize, using spatial index for fast distance calculation
    def negative_radius(point_coords):
        point = Point(point_coords)
        if polygon.geometry.contains(point):
            # Find the closest boundary point using the spatial index
            nearest_boundary_point = boundary_tree.nearest(point)
            return -point.distance(nearest_boundary_point)
        else:
            return np.inf  # Outside the polygon, so invalid
    
    # Optimization to find the largest inscribed circle
    result = minimize(negative_radius, initial_guess, method='Nelder-Mead')
    
    # Get the maximum inscribed circle
    optimal_point = Point(result.x)
    max_radius = -result.fun  # Negative of the minimized value
    
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
    internal_buffer = block.geometry.buffer(-optimal_radius)#block.geometry.difference(block.geometry.buffer(-optimal_radius))
    return internal_buffer


def plot_building_azimuths(buildings_clipped, save_path):
    fig, ax = plt.subplots(figsize=(20, 16))

    # Plot each building colored by its azimuth
    buildings_clipped.plot(ax=ax, column='azimuth', cmap='viridis', legend=True, legend_kwds={'label': "Azimuth group"})

    # Plot building ID, closest building ID, azimuth, and closest building azimuth over each building
    for idx, row in buildings_clipped.iterrows():
        centroid = row['centroid']
        building_id = idx
        closest_id = row['closest_building_id']
        azimuth = row['azimuth']
        closest_azimuth = row['closest_azimuth']
        azimuth_diff = row['azimuth_diff']

        # Display building ID and closest building ID (in red) with a background box
        ax.text(centroid.x, centroid.y, f"ID: {int(building_id)}\nClosest: {int(closest_id)}\nDiff: {azimuth_diff:.2f}", color='red', fontsize=3,
                ha='center') #, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)

        # Display azimuth and closest building azimuth (in blue) with a background box
        ax.text(centroid.x, centroid.y - 5, f"A: {azimuth:.2f}\nCA: {closest_azimuth:.2f}",
                color='blue', fontsize=3, ha='center') #, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)

    plt.title('Building IDs, Closest IDs, and Azimuths')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save the figure in high resolution if a path is provided
    plt.savefig(save_path, dpi=400, bbox_inches='tight')  # Save with high DPI (300) and tight layout

    #plt.show()

def save_azimuth_histograms(buildings_clipped, file_name ,output_dir='histogram_plots'):
    """
    Creates and saves two histograms: one for azimuths and another for the difference in azimuths.
    
    Parameters:
    buildings_clipped (GeoDataFrame): A GeoDataFrame with a column 'azimuth' for building orientations.
    output_dir (str): Directory to save the plots. Default is 'plots'.
    """
    # Extract azimuth values from the buildings_clipped GeoDataFrame
    azimuths = buildings_clipped['azimuth'].dropna().values
    
    # Compute azimuth differences between consecutive buildings
    azimuth_diffs = buildings_clipped['azimuth_diff'].dropna().values
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot styles for aesthetics
    #plt.style.use('seaborn-darkgrid')

    # Plot 1: Histogram of Azimuths
    plt.figure(figsize=(10, 6))
    plt.hist(azimuths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Building Azimuths', fontsize=16)
    plt.xlabel('Azimuth (degrees)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(0, 90) 
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, file_name), dpi=300)
    plt.close()

    # Plot 2: Histogram of Azimuth Differences
    plt.figure(figsize=(10, 6))
    plt.hist(azimuth_diffs, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Azimuth Differences', fontsize=16)
    plt.xlabel('Azimuth Difference (degrees)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(0, 45)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'diff_'+file_name), dpi=300)
    plt.close()

    #print(f'Histograms saved in {output_dir}/')

def calculate_azimuth_diff(x,y):
    comp_list = [np.abs(y-x)]
    if ((x > 45) & (y > 45)):
        comp_list.append(np.abs((y-90) - (x - 90)))
    elif ((x > 45) & (y <= 45)):
        comp_list.append(np.abs(y - (x - 90)))
    elif ((x <= 45) & (y > 45)):
        comp_list.append(np.abs((y-90) - x))
    elif ((x <= 45) & (y <= 45)):
        comp_list.append(np.abs(y - x))
    return min(comp_list)

def apply_internal_buffer(blocks_clipped, row_epsilon, tolerance=1e-6):
    def process_block(row):
        block_area = row.geometry.area
        
        # Epsilon buffer: very small internal buffer based on row_epsilon
        epsilon_target_area = block_area * (1.0 - row_epsilon)
        epsilon_buffer = get_internal_buffer_with_target_area(row, epsilon_target_area, tolerance)
        
        # 50%-width buffer: buffer based on 50% of the block width
        half_width_buffer = row['radius'] * 0.5
        width_buffer = row.geometry.buffer(-half_width_buffer)
        
        return epsilon_buffer, width_buffer

    # Apply the function to each row and unpack results into two separate GeoDataFrames
    results = blocks_clipped.apply(process_block, axis=1)
    blocks_clipped['epsilon_buffer'] = results.apply(lambda x: x[0])
    blocks_clipped['width_buffer'] = results.apply(lambda x: x[1])

    return blocks_clipped


def visualize_tortuosity_comparison(roads_clipped, n_samples=5):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Sample a few roads for visualization (you can adjust this)
    sampled_roads = roads_clipped.sample(n=n_samples)
    
    for idx, road in sampled_roads.iterrows():
        geometry = road.geometry
        
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            start_point = Point(coords[0])
            end_point = Point(coords[-1])

            # Plot actual road
            xs, ys = geometry.xy
            ax.plot(xs, ys, color='blue', label='Road Segment' if idx == 0 else "")

            # Plot straight line (Euclidean distance)
            straight_line = LineString([start_point, end_point])
            xs_straight, ys_straight = straight_line.xy
            ax.plot(xs_straight, ys_straight, color='red', linestyle='--', label='Straight Line' if idx == 0 else "")

            # Add annotations for distances
            ax.annotate(f"Road: {geometry.length:.2f}", (xs[0], ys[0]), fontsize=10, color='blue')
            ax.annotate(f"Straight: {start_point.distance(end_point):.2f}", (xs_straight[0], ys_straight[0]), fontsize=10, color='red')

        elif isinstance(geometry, MultiLineString):
            # Handle each line separately
            for line in geometry.geoms:
                coords = list(line.coords)
                start_point = Point(coords[0])
                end_point = Point(coords[-1])

                # Plot actual road
                xs, ys = line.xy
                ax.plot(xs, ys, color='blue')

                # Plot straight line (Euclidean distance)
                straight_line = LineString([start_point, end_point])
                xs_straight, ys_straight = straight_line.xy
                ax.plot(xs_straight, ys_straight, color='red', linestyle='--')

                # Add annotations for distances
                ax.annotate(f"Road: {line.length:.2f}", (xs[0], ys[0]), fontsize=10, color='blue')
                ax.annotate(f"Straight: {start_point.distance(end_point):.2f}", (xs_straight[0], ys_straight[0]), fontsize=10, color='red')

    # Set the axis limits and labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Road Segment vs Straight Line Comparison")
    ax.legend()
    
    plt.show()

def calculate_tortuosity(geometry):
    road_lengths = []
    euclidean_distances = []

    # If the geometry is a MultiLineString, handle each LineString separately
    if isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            coords = list(line.coords)
            road_length = line.length
            euclidean_distance = Point(coords[0]).distance(Point(coords[-1]))
            if euclidean_distance > 0:
                road_lengths.append(road_length)
                euclidean_distances.append(euclidean_distance)
    elif isinstance(geometry, LineString):
        coords = list(geometry.coords)
        road_length = geometry.length
        euclidean_distance = Point(coords[0]).distance(Point(coords[-1]))
        if euclidean_distance > 0:
            road_lengths.append(road_length)
            euclidean_distances.append(euclidean_distance)

    # Calculate tortuosity index for each road segment (length/Euclidean)
    tortuosity_index = np.array(euclidean_distances) / np.array(road_lengths)
    if len(tortuosity_index) > 0:
        return euclidean_distances,road_lengths
    elif len(tortuosity_index)==0:
        return [], []


#1 Share of building footprints that are less than 10-meters away from the nearest road
def metric_1_distance_less_than_20m(buildings, road_union, utm_proj_rectangle):
    # Apply the distance calculation to each building
    #buildings.loc[:,'distance_to_road'] = buildings['geometry'].apply(lambda x: x.centroid).apply(calculate_minimum_distance_to_roads, 
    
    buildings_geometry_copy = buildings['geometry'].copy()
    buildings.loc[:,'distance_to_road'] = buildings_geometry_copy.apply(lambda x: calculate_minimum_distance_to_roads_option_B(x, road_union))

    m1 = 1.*((sum(buildings['distance_to_road']<=20))/len(buildings))
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
def metric_4_share_4way_intersections(intersections):
    n_intersections_3_or_higher = 1.*len(intersections[(intersections.street_count >= 3)])
    n_4_way = 1.*len(intersections[(intersections.street_count == 4)])
    m4 = (n_4_way / n_intersections_3_or_higher)
    return m4

#5 Density of intersections
def metric_5_intersection_density(intersections, rectangle_area):
    m5 = (1000.**2)*(len(intersections)/rectangle_area) #(1000.**2)*(len(intersections[(intersections.street_count >= 3)])/rectangle_area)
    return m5

#6 Entropy of building Azimuth
def metric_6_entropy_of_building_azimuth(buildings_clipped, blocks, rectangle_id, bin_width_degrees, plot=True):
    """
    Calculate the weighted average KL divergence between the azimuth distribution of buildings and a uniform distribution,
    where the weights are based on the number of buildings per block.

    Parameters:
    - buildings_clipped: GeoDataFrame with building geometries.
                         It should contain azimuths (in degrees) for each building.
    - blocks: GeoDataFrame containing block geometries with an index (block_id).
    - rectangle_id: Identifier for the rectangle being analyzed.
    - bin_width_degrees: Width of the histogram bins in degrees.
    - plot: Whether to plot the distributions for each block.

    Returns:
    - weighted_avg_kl_divergence: Weighted average KL divergence where the weight is the number of buildings per block.
    - buildings_clipped: Updated GeoDataFrame with an assigned block_id.
    """

    # Assign block_id to each building using a spatial join
    buildings_clipped = buildings_clipped.sjoin(blocks[['geometry']], how='left', predicate='within')
    buildings_clipped = buildings_clipped.rename(columns={'index_right': 'block_id'})  # Ensure block_id is assigned

    # Assign a temporary bogus block_id for buildings not matched to a block
    buildings_clipped['block_id'] = buildings_clipped['block_id'].fillna(-1).astype(int)

    # Compute azimuths for each building
    buildings_clipped.loc[:, 'azimuth'] = buildings_clipped['geometry'].apply(lambda x: calculate_azimuth(longest_segment(x)) % 90.)

    # Define the number of bins for the histogram
    num_bins = int(90 / bin_width_degrees)
    
    block_results = []  # Store per-block KL divergences and counts

    for block_id, group in buildings_clipped.groupby('block_id'):
        # Extract azimuths for the current block
        azimuths = group['azimuth'].values

        if len(azimuths) == 0:
            continue  # Skip empty blocks

        # Create histogram of observed azimuths
        histogram, bin_edges = np.histogram(azimuths, bins=num_bins, range=(0, 90))
        P = histogram / histogram.sum()  # Normalize to make it a probability distribution

        # Create uniform distribution Q
        Q = np.ones(num_bins) / num_bins

        # Compute KL divergence
        kl_divergence = entropy(P, Q)

        # Standardize KL divergence
        max_kl_divergence = np.log(num_bins)
        standardized_kl_divergence = kl_divergence / max_kl_divergence if max_kl_divergence > 0 else 0

        # Store results
        block_results.append((block_id, standardized_kl_divergence, len(group)))

        # Optional: Plot the distributions for this block
        if plot:
            plt.figure(figsize=(10, 6))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.bar(bin_centers, P, width=bin_width_degrees, alpha=0.6, label=f'Observed Azimuths (P) - Block {block_id}')
            plt.plot(bin_centers, Q, 'r--', linewidth=2, label='Uniform Distribution (Q)')

            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Probability')
            plt.title(f'Azimuth Distribution for Block {block_id}')
            plt.legend(loc='upper right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(f'{plots_path}/entropy_{str(bin_width_degrees)}-wide_bins_{rectangle_id}_block_{block_id}.png')

    # Convert results to DataFrame
    block_df = pd.DataFrame(block_results, columns=['block_id', 'kl_divergence', 'num_buildings'])

    # Compute weighted average KL divergence
    if not block_df.empty:
        weighted_avg_kl_divergence = np.average(block_df['kl_divergence'], weights=block_df['num_buildings'])
    else:
        weighted_avg_kl_divergence = np.nan  # No valid blocks

    return weighted_avg_kl_divergence, buildings_clipped


#7 Average block width
def metric_7_average_block_width(blocks_clipped, blocks_clipped_within_rectangle, rectangle_projected, rectangle_area):
    #radius_avg = []
    # Loop through blocks within the rectangle and calculate 
    for block_id, block in blocks_clipped.iterrows():
        optimal_point, max_radius = get_largest_inscribed_circle(block)
        if block_id in list(blocks_clipped_within_rectangle.index):
            block_area_within_rectangle = blocks_clipped_within_rectangle.loc[block_id].geometry.area
        else:
            block_area_within_rectangle = 0.
        block_weight = 1.*block_area_within_rectangle / rectangle_area
        weighted_width = block_weight*max_radius
        blocks_clipped.loc[block_id,'weighted_width'] = weighted_width
        blocks_clipped.loc[block_id,'radius'] = max_radius
        #radius_avg.append(max_radius)

    rectangle_projected_gdf = gpd.GeoSeries(rectangle_projected)
    blocks_clipped = blocks_clipped.reset_index(drop=True)
    blocks_union = blocks_clipped.unary_union
    left_over_blocks = gpd.GeoDataFrame(geometry=gpd.GeoDataFrame(geometry=rectangle_projected_gdf).difference(blocks_union).geometry)
    left_over_blocks = left_over_blocks.explode(index_parts=False).reset_index(drop=True)
    left_over_blocks = left_over_blocks[left_over_blocks.is_valid & ~left_over_blocks.is_empty]


    if left_over_blocks.area.sum() > 0.0:

        # OJO -- CAN THIS LOOP BE REPLACED?
        # left_over_blocks['optimal_point'], left_over_blocks['max_radius'] = zip(*left_over_blocks['geometry'].apply(lambda geom: get_largest_inscribed_circle(geom)))
        # left_over_blocks_within_rectangle = left_over_blocks.intersection(rectangle_projected)
        # left_over_blocks['block_weight'] = left_over_blocks_within_rectangle.area / rectangle_area
        # left_over_blocks['weighted_width'] = left_over_blocks['block_weight'] * left_over_blocks['max_radius']

        for block_id, leftover_block in left_over_blocks.iterrows():
            #print(block_id)
            #leftover_block = gpd.GeoDataFrame([{'geometry': leftover_block}], crs=blocks_clipped.crs.to_epsg())
            optimal_point, max_radius = get_largest_inscribed_circle(leftover_block)
            block_within_rectangle = (gpd.GeoSeries(leftover_block.geometry).intersection(gpd.GeoSeries(rectangle_projected)))
            block_area_within_rectangle = block_within_rectangle.area.sum()
            block_weight = block_area_within_rectangle / rectangle_area
            weighted_width = block_weight*max_radius
            left_over_blocks.loc[block_id,'weighted_width'] = weighted_width 
            left_over_blocks.loc[block_id,'radius'] = max_radius 
            left_over_blocks.loc[block_id,'area'] = leftover_block.geometry.area
        m7 = blocks_clipped['weighted_width'].sum() + left_over_blocks['weighted_width'].sum()
        blocks_clipped = pd.concat([blocks_clipped,left_over_blocks]).reset_index()
    else:
        m7 = blocks_clipped['weighted_width'].sum()
    return m7, blocks_clipped

#8 Two row blocks
def metric_8_two_row_blocks(blocks_clipped, buildings, utm_proj_rectangle, row_epsilon):
    """
    Computes the two-row blocks metric, only including blocks where the internal buffer is ≤ 100 meters.
    """
    
    # Filter blocks to include only those where `radius` ≤ 100 meters
    if 'radius' not in blocks_clipped.columns:
        raise ValueError("🚨 Blocks must have a 'radius' column from metric_7 before running metric_8.")
    
    blocks_clipped = blocks_clipped[blocks_clipped['radius'] <= 100].copy()  # Keep only valid blocks

    if blocks_clipped.empty:
        #print("🚨 No blocks with internal buffer ≤ 100m. Assigning NaN to m8.")
        return np.nan, None, None  # Return NaN and empty buffers

    # Apply internal buffers (both epsilon and width-based buffers)
    blocks_clipped = apply_internal_buffer(blocks_clipped, row_epsilon)

    # Calculate share of buildings in the 50%-width buffer (building_to_block_share)
    width_buffers = gpd.GeoDataFrame(blocks_clipped, geometry='width_buffer', crs=blocks_clipped.crs)
    buildings_in_width_buffers = buildings.clip(width_buffers.geometry.unary_union)
    buildings_in_width_buffers_area = buildings_in_width_buffers.area.sum()
    width_buffer_area = width_buffers.geometry.area.sum()

    building_to_block_share = buildings_in_width_buffers_area / width_buffer_area if width_buffer_area > 0 else 0

    # Calculate share of buildings in epsilon buffer (building_to_buffer_share)
    epsilon_buffers = gpd.GeoDataFrame(blocks_clipped, geometry='epsilon_buffer', crs=blocks_clipped.crs)
    buildings_in_epsilon_buffers = buildings.clip(epsilon_buffers.geometry.unary_union)
    buildings_in_epsilon_buffers_area = buildings_in_epsilon_buffers.area.sum()
    epsilon_buffer_area = epsilon_buffers.geometry.area.sum()

    building_to_buffer_share = buildings_in_epsilon_buffers_area / epsilon_buffer_area if epsilon_buffer_area > 0 else 0

    # Compute metric safely
    if building_to_block_share != 0:
        m8 = building_to_buffer_share / building_to_block_share
    else:
        #print("🚨 m8 assigned to 1 because building_to_block_share was zero.")
        m8 = 1.0  # Assign 1 or 0 depending on logic

    return m8, epsilon_buffers, width_buffers

#9 Tortuosity index
def metric_9_tortuosity_index(roads_clipped):
 
    # Apply the function to the entire geometry column in roads_clipped
    euclidean_distances,road_lengths = zip(*roads_clipped.geometry.apply(calculate_tortuosity))
    
    #roads_clipped['tortuosity_index'] = zip(*roads_clipped.geometry.apply(calculate_tortuosity))

    # Calculate the overall mean tortuosity index for the study section
    m9 = np.sum(np.hstack(euclidean_distances)) / np.sum(np.hstack(road_lengths))#roads_clipped['tortuosity_index'].mean()
    #visualize_tortuosity_comparison(roads_clipped,len(roads_clipped))
    #print(f'Metric 9: {str(m9)}')
    return m9

#10 Average angle between road segments
def metric_10_average_angle_between_road_segments(intersections, roads):
    """
    Computes the average absolute deviation of intersection angles from 90 degrees.
    """
    
    df_angles = calculate_sequential_angles_option_A(intersections, roads)

    # Safeguard: Ensure df_angles is not empty and contains 'Intersection ID'
    if df_angles.empty or 'Intersection ID' not in df_angles.columns:
        #print("Metric 10 Warning: No valid intersection angles found. Returning NaN.")
        return np.nan

    # Merge intersection data
    intersection_angles_df = intersections[['osmid', 'street_count']].set_index('osmid') \
        .merge(df_angles.set_index('Intersection ID'), left_index=True, right_index=True, how='outer')

    # Drop NaN angles early to avoid errors later
    intersection_angles_df = intersection_angles_df.dropna(subset=['Angle'])

    # Process 3-way intersections
    df_3_way = intersection_angles_df[intersection_angles_df.street_count == 3]
    
    if not df_3_way.empty:
        idx_min = df_3_way.groupby(df_3_way.index)['Angle'].idxmin().dropna()
        to_keep_3 = df_3_way.loc[idx_min]  # Keep original index
    else:
        to_keep_3 = pd.DataFrame([])

    # Process 4-way+ intersections (keep two smallest angles)
    df_4_way_or_more = intersection_angles_df[intersection_angles_df.street_count >= 4]
    
    if not df_4_way_or_more.empty:
        to_keep_4_or_more = df_4_way_or_more.groupby(df_4_way_or_more.index) \
            .apply(lambda x: x.nsmallest(2, 'Angle')).reset_index(drop=True)
    else:
        to_keep_4_or_more = pd.DataFrame([])

    # Ensure index consistency
    to_keep_4_or_more.index.names = ['index']

    # Concatenate results
    to_keep_df = pd.concat([to_keep_3, to_keep_4_or_more])

    # Compute metric
    if not to_keep_df.empty:
        m10 = np.mean(np.abs(90. - to_keep_df['Angle']))
    else:
        m10 = np.nan  # No valid data

    return m10


#11 Building density
def metric_11_building_density(n_buildings,rectangle_area):
    """Number of buildings per hectare"""
    return (10000*n_buildings)/rectangle_area

#12 Built area share
def metric_12_built_area_share(building_area,rectangle_area):
    return building_area/rectangle_area

# 13 Average building area
def metric_13_average_building_area(building_area,n_buildings):
    return building_area/n_buildings
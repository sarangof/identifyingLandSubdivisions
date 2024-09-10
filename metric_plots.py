
import matplotlib.pyplot as plt
from metrics_calculation import get_largest_inscribed_circle
from shapely.geometry import Polygon, MultiPolygon

def plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id):
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='lightcoral', ec='blue', label='Study area')

    roads.plot(ax=ax, color='grey', linewidth=1, label='Roads')
    buildings_clipped.plot(column='distance_to_road', ax=ax, cmap='Blues', legend=True, label='Buildings')

    ax.set_title('Distance to nearest road', fontsize=15)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    plt.legend(loc='upper right')

    plt.tight_layout()

    plt.savefig(f'pilot_plots/distance_to_roads_plot_{rectangle_id}.png', dpi=300, bbox_inches='tight')

def plot_azimuth(buildings, roads, rectangle_projected, rectangle_id):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='lightcoral', ec='blue', label='Study area')

    roads.plot(ax=ax, color='grey', linewidth=1, label='Roads')
    buildings.plot(column='azimuth', ax=ax, cmap='Greens', legend=True, label='Buildings')

    ax.set_title('Building orientation', fontsize=15)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'pilot_plots/azimuth_plot_{rectangle_id}.png', dpi=300, bbox_inches='tight')
 
def plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads):
    inflection_points = all_road_vertices[all_road_vertices['point_type'] == 'inflection point']
    intersections = all_road_vertices[all_road_vertices['point_type'] == 'intersection']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='lightcoral', ec='blue', label='Study area')

    roads.plot(ax=ax, color='gray', linewidth=1)

    ax.scatter(inflection_points.geometry.x, inflection_points.geometry.y, color='lightblue', s=10, label='Inflection Points')
    ax.scatter(intersections.geometry.x, intersections.geometry.y, color='blue', s=10, label='Intersections')

    ax.set_title('Intersections and Inflection Points', fontsize=15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'pilot_plots/intersections_and_inflection_points_plot_{rectangle_id}.png', dpi=300, bbox_inches='tight')

def plot_largest_inscribed_circle(rectangle_id, rectangle_projected, blocks_clipped):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    blocks_clipped.plot(column='weighted_width', ax=ax, alpha=0.5, cmap='Greys', ec='gray', linewidth=1, legend=True, label='Block width')
    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, linewidth=3, fc='none', ec='black', label='Study area')

    for block_id, block in blocks_clipped.iterrows():
            optimal_point, max_radius = get_largest_inscribed_circle(block)
            circle = optimal_point.buffer(max_radius)

            x, y = circle.exterior.xy
            ax.plot(x, y, color='red', linewidth=2)  # Draw the circle

            ax.plot(optimal_point.x, optimal_point.y, 'ro')

            # Set limits and show the plot
            #ax.set_xlim(-1, 5)
            #ax.set_ylim(-1, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'pilot_plots/plot_block_width_{rectangle_id}.png', dpi=300, bbox_inches='tight')

def plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, row_epsilon):
    fig, ax = plt.subplots()
    blocks_clipped.plot(ax=ax, ec='black', fc='royalblue', linewidth=0.2, legend=True, label='Blocks')
    
    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, linewidth=1.5, fc='none', ec='lightseagreen', label='Study area')
    buildings_clipped.plot(ax=ax, ec='black', fc='firebrick', linewidth=0.5, legend=True, label='Buildings')
    internal_buffers.plot(ax=ax, ec='black', fc='gold', alpha=0.8,  linewidth=0.2, legend=True, label='Internal buffer')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(f'pilot_plots/plot_two_row_blocks_{rectangle_id}_epsilon_{str(row_epsilon)}.png', dpi=300, bbox_inches='tight')

   
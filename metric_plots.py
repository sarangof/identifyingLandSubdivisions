
import matplotlib.pyplot as plt
from metrics_calculation import get_largest_inscribed_circle
from shapely.geometry import Polygon, MultiPolygon
import matplotlib as mpl
import matplotlib.patches as mpatches
import geopandas as gpd

def plot_distance_to_roads(buildings_clipped, roads, rectangle_projected, rectangle_id):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='lightcoral', ec='blue', label='Study area')

    roads.plot(ax=ax, zorder=1, color='grey', linewidth=0.5, label='Roads')
    buildings_clipped_plot = buildings_clipped.plot(column='distance_to_road', zorder=2, ax=ax, cmap='Blues', label='Buildings')

    norm = mpl.colors.Normalize(vmin=buildings_clipped['distance_to_road'].min(), vmax=buildings_clipped['distance_to_road'].max())
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm._A = []  
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)  

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'pilot_plots/distance_to_roads_plot_{rectangle_id}.png', dpi=500, bbox_inches='tight')


def plot_azimuth(buildings_clipped, roads, rectangle_projected, rectangle_id, n_orientation_groups):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    try:
        roads.plot(ax=ax, zorder=1, color='grey', linewidth=0.5, label='Roads')
    except AttributeError:
        pass
    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, alpha=0.5, zorder=2, fc='none', ec='blue', label='Study area')
    buildings_clipped_plot = buildings_clipped.plot(column='azimuth_group', zorder=3, ax=ax, cmap='Accent', label='Buildings')

    norm = mpl.colors.Normalize(vmin=1, vmax=n_orientation_groups)
    colors = plt.cm.Accent(norm(range(1, n_orientation_groups+1)))
    patches = [mpatches.Patch(color=colors[i], label=f'Group {i+1}') for i in range(n_orientation_groups)]
    ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'pilot_plots/azimuth_group_plot_{rectangle_id}_{n_orientation_groups}_orientation_groups.png', dpi=500, bbox_inches='tight')

def plot_inflection_points(rectangle_id, rectangle_projected, all_road_vertices, roads):
    inflection_points = all_road_vertices[all_road_vertices['point_type'] == 'inflection point']
    intersections = all_road_vertices[all_road_vertices['point_type'] == 'intersection']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    roads.plot(ax=ax, zorder=1,color='gray', linewidth=0.5)
    x, y = rectangle_projected.geometry.exterior.xy
    ax.fill(x, y, zorder=2, alpha=0.5, fc='none', ec='royalblue', label='Study area')
    ax.scatter(inflection_points.geometry.x, inflection_points.geometry.y, color='lightblue', s=10, label='Inflection Points')
    ax.scatter(intersections.geometry.x, intersections.geometry.y, color='blue', s=10, label='Intersections')

    #ax.set_title('Intersections and Inflection Points', fontsize=15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'pilot_plots/intersections_and_inflection_points_plot_{rectangle_id}.png', dpi=500, bbox_inches='tight')


def plot_largest_inscribed_circle(rectangle_id, rectangle_projected, blocks_clipped, roads):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot roads and blocks, without automatic colorbar
    roads.plot(ax=ax, color='gray', linewidth=0.5, zorder=1)
    blocks_clipped_plot = blocks_clipped.plot(column='weighted_width', zorder=2, ax=ax, alpha=0.5, cmap='Greys', ec='gray', linewidth=1, label='Block width')

    # Plot the study area rectangle
    x, y = rectangle_projected.geometry.exterior.iloc[0].xy
    ax.fill(x, y, linewidth=1.5, zorder=3, fc='none', ec='black', label='Study area')

    # Plot circles and their centers
    for block_id, block in blocks_clipped.iterrows():
        optimal_point, max_radius = get_largest_inscribed_circle(block)
        circle = optimal_point.buffer(max_radius)
        x, y = circle.exterior.xy
        ax.plot(x, y, color='red', zorder=4, linewidth=2)  # Draw the circle
        ax.plot(optimal_point.x, optimal_point.y, 'ro', linewidth=2, zorder=5)

        # Annotate the radius next to the circle center
        ax.annotate(f'{max_radius:.2f}', 
                    (optimal_point.x, optimal_point.y), 
                    textcoords="offset points", 
                    xytext=(5, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='blue', 
                    zorder=6)

    # Add manual colorbar
    norm = mpl.colors.Normalize(vmin=blocks_clipped['weighted_width'].min(), vmax=blocks_clipped['weighted_width'].max())
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
    sm._A = []  # Dummy mappable for the colorbar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)  # Adjust shrink parameter to control colorbar size

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'pilot_plots/plot_block_width_{rectangle_id}.png', dpi=500, bbox_inches='tight')


def plot_two_row_blocks(rectangle_id, rectangle_projected, blocks_clipped, internal_buffers, buildings_clipped, roads, row_epsilon):
    fig, ax = plt.subplots()
    roads.plot(ax=ax, color='gray', linewidth=0.5, zorder=1)
    blocks_clipped.plot(ax=ax, ec='black', fc='lightsteelblue', zorder=2, linewidth=0.2, legend=True, label='Blocks')
    x, y = rectangle_projected.geometry.exterior.iloc[0].xy
    ax.fill(x, y, linewidth=1, fc='none', ec='lightseagreen', zorder=3, label='Study area')
    buildings_clipped.plot(ax=ax, ec='black', fc='pink', zorder=4, linewidth=0.3, legend=True, label='Buildings')
    internal_buffers.plot(ax=ax, ec='black', fc='yellow', zorder=5, alpha=0.7,  linewidth=0.2, legend=True, label='Internal buffer')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(f'pilot_plots/plot_two_row_blocks_{rectangle_id}_epsilon_{str(row_epsilon)}.png', dpi=500, bbox_inches='tight')

   
max_distance = 200.
default_distance = 500.

def compute_distance_partition(buildings_df, roads_geom_list, max_distance, default_distance):
    tree = STRtree(roads_geom_list)

    def distance_fn(bgeom):
        try:
            bgeom = shape(bgeom) if not isinstance(bgeom, BaseGeometry) else bgeom
            nearby_indices = tree.query(bgeom.buffer(max_distance))
            if nearby_indices is None or len(nearby_indices) == 0:
                return default_distance
            nearby_geoms = [roads_geom_list[i] for i in nearby_indices]
            return min(bgeom.distance(road) for road in nearby_geoms)
        except Exception:
            return default_distance

    buildings_df = buildings_df.copy()
    buildings_df['geometry'] = buildings_df['geometry'].apply(shape)  # extra safe
    buildings_df["distance_to_nearest_road"] = buildings_df.geometry.apply(distance_fn)
    return buildings_df


@delayed
def calculate_building_distances_to_roads(city_name):
    paths = {
    'grid': f'{GRIDS_PATH}/{city_name}/{city_name}_{grid_size}m_grid.geoparquet',
    'buildings': f'{BUILDINGS_PATH}/{city_name}/Overture_building_{city_name}.geoparquet',
    'roads': f'{ROADS_PATH}/{city_name}/{city_name}_OSM_roads.geoparquet'
    }
    epsg = get_epsg(city_name).compute()  
    # Load and prepare roads for spatial index
    roads = load_dataset(paths['roads'], epsg=epsg).compute()
    roads_geom_list = [geom for geom in roads.geometry]

    # Load buildings lazily
    buildings = load_dataset(paths['buildings'], epsg=epsg)

    meta = buildings._meta.assign(distance_to_nearest_road='f8')

    # Apply distance computation per partition
    buildings_with_dist = buildings.map_partitions(
        compute_distance_partition,
        roads_geom_list,
        max_distance,
        default_distance,
        meta=meta
    )

    # Write output
    columns_to_keep = ['id', 'geometry','distance_to_nearest_road']
    buildings_with_dist = buildings_with_dist[columns_to_keep].set_index('id')
    out_path = paths['buildings'].replace(".geoparquet", "_with_distances.geoparquet")
    buildings_with_dist.to_parquet(out_path)
    return out_path


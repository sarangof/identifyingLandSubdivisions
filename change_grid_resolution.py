import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split

# --- 1) Load your 200 m grid (update paths/layer names) ---
in_gpkg = "../Nairobi_200m_grid.gpkg"
#in_layer = "grid_200"         # or omit layer= if single-layer gpkg
g200 = gpd.read_file(in_gpkg)#, layer=in_layer)

'''
# Optional safety: ensure projected CRS in meters
if g200.crs is None or g200.crs.is_geographic:
    raise ValueError("Grid must be in a projected CRS (meters). Reproject to UTM first.")
'''
    
# --- 2) Split each 200 m cell into 4 x 100 m cells ---
rows = []
attr_cols = [c for c in g200.columns if c != "geometry"]  # keep existing attributes

for idx, row in g200.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue

    minx, miny, maxx, maxy = geom.bounds
    mx = 0.5 * (minx + maxx)
    my = 0.5 * (miny + maxy)

    # vertical + horizontal split lines (slightly extended)
    splitter = MultiLineString([
        LineString([(mx, miny - 1), (mx, maxy + 1)]),
        LineString([(minx - 1, my), (maxx + 1, my)])
    ])

    parts = split(geom, splitter)  # returns 4 polygons for a square cell
    for k, part in enumerate(parts.geoms):
        rec = {c: row[c] for c in attr_cols}
        rec["parent_id"] = row.get("grid_id", idx)  # keep link to original cell
        rec["child"] = k                            # 0..3
        rec["geometry"] = part
        rows.append(rec)

g100 = gpd.GeoDataFrame(rows, crs=g200.crs)

# --- 3) Save as a new GeoPackage layer ---
out_geoparquet = "../Nairobi_100m_grid.geoparquet"     # or same file with a different layer name
#out_layer = "grid_100"
g100.to_parquet(out_geoparquet)#, layer=out_layer, driver="GPKG")
print(f"Wrote {len(g100)} cells to {out_geoparquet}")

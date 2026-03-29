"""
==========================================================================
Score city blocks and generate interactive regularity maps
==========================================================================

Usage (in notebook, after running cascade_models.py):

    from score_and_map import score_cities, make_interactive_map

    # Score Nairobi and Medellín
    scored = score_cities(
        city_names=['Nairobi__Kenya', 'Medellin__Colombia'],
        cascade_results=results,  # from run_full_cascade()
        your_name='sara'
    )

    # Generate interactive map
    make_interactive_map(scored, output_path='regularity_map.html')
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import os

# ======================================================================
# PATHS 
# ======================================================================

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = f"{MAIN_PATH}/input"
OUTPUT_PATH = f"{MAIN_PATH}/output"
OUTPUT_PATH_RASTER = f"{OUTPUT_PATH}/raster"

# Metric column definitions
ALL_13 = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std',
    'm10_std', 'm11_std', 'm12_std'
]

REGULARITY_10 = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std'
]

# Stage 1 uses a subset
STAGE1_FEATURES = ['m3_std', 'm4_std', 'm5_std', 'm7_std', 'm10_std', 'm11_std', 'k_complexity_std']


# ======================================================================
# SCORING
# ======================================================================

def score_single_city(city_df, cascade_results):
    """
    Apply the trained cascade to a single city's block metrics.

    Parameters
    ----------
    city_df : GeoDataFrame
        Block-level metrics for one city, with _std columns.
    cascade_results : dict
        Output from run_full_cascade(), containing stage1..stage4 dicts
        each with 'scaler' and trained calibrated 'model' (from evaluate_stage).

    Returns
    -------
    GeoDataFrame with added columns:
        p_built, p_residential, p_subdivision, p_formal,
        regularity_index, classification
    """
    df = city_df.copy()

    # ── Stage definitions: (result_key, output_col, feature_cols) ──
    stages = [
        ('stage1', 'p_built', STAGE1_FEATURES),
        ('stage2', 'p_residential', ALL_13),
        ('stage3_reg', 'p_subdivision', REGULARITY_10),
        ('stage4', 'p_formal', REGULARITY_10),
    ]

    for stage_key, col_name, feat_cols in stages:
        stage = cascade_results.get(stage_key)
        if stage is None:
            df[col_name] = np.nan
            continue

        # The evaluate_stage function stores 'scaler' but we need to
        # retrain on the full data for scoring. Use the scaler from
        # the stage results.
        scaler = stage['scaler']

        # For the calibrated model, we need to train one on the full
        # training data. The stage results contain the CV-evaluated
        # model but also a scaler fit on the full training data.
        # We'll refit a calibrated model here for scoring.
        # Actually — the scaler in evaluate_stage is fit on the full
        # stage data, and we can use it to transform new data.
        # But we need a trained model. Let's check if the stage has one.

        # The cascade_models.py evaluate_stage doesn't save a trained
        # model object (it only does CV evaluation). We need to train
        # one. Let's do it here using the same parameters.
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        # We don't have the training data here, so we need to pass it
        # or pre-train. For now, we'll add model training to the stage.
        # BUT — the simpler approach: score using the scaler + a model
        # that we train once and pass in.

        # Check if a 'model' key exists
        if 'model' not in stage:
            df[col_name] = np.nan
            print(f"  ⚠ No trained model for {stage_key}, skipping")
            continue

        model = stage['model']

        X = df[feat_cols].copy()
        valid_mask = X.notna().all(axis=1)
        X_valid = X.loc[valid_mask]

        if len(X_valid) > 0:
            X_scaled = scaler.transform(X_valid)
            probs = model.predict_proba(X_scaled)[:, 1]
            df.loc[valid_mask, col_name] = probs
        else:
            df[col_name] = np.nan

    # ── Hierarchical classification ──
    df['classification'] = 'unclassified'

    # Open space: n_buildings == 0 OR P(built) < 0.5
    mask_no_buildings = df['n_buildings'] == 0 if 'n_buildings' in df.columns else pd.Series(False, index=df.index)
    mask_open = mask_no_buildings | (df.get('p_built', pd.Series(1.0, index=df.index)) < 0.5)
    df.loc[mask_open, 'classification'] = 'open_space'

    # Non-residential: built but P(residential) < 0.5
    mask_nonres = (~mask_open) & (df.get('p_residential', pd.Series(1.0, index=df.index)) < 0.5)
    df.loc[mask_nonres, 'classification'] = 'non_residential'

    # Irregular: residential but P(subdivision) < 0.5
    mask_irreg = (~mask_open) & (~mask_nonres) & (df.get('p_subdivision', pd.Series(1.0, index=df.index)) < 0.8)
    df.loc[mask_irreg, 'classification'] = 'irregular_settlement'

    # Subdivision: residential and P(subdivision) >= 0.5
    mask_subdiv = (~mask_open) & (~mask_nonres) & (df.get('p_subdivision', pd.Series(0.0, index=df.index)) >= 0.8)

    # Formal/informal split
    if 'p_formal' in df.columns and df['p_formal'].notna().any():
        mask_formal = mask_subdiv & (df['p_formal'] >= 0.5)
        mask_informal = mask_subdiv & (df['p_formal'] < 0.5)
        df.loc[mask_formal, 'classification'] = 'formal_subdivision'
        df.loc[mask_informal, 'classification'] = 'informal_subdivision'
    else:
        df.loc[mask_subdiv, 'classification'] = 'subdivision'

    # ── Regularity index = P(subdivision | residential) ──
    mask_residential = (~mask_open) & (~mask_nonres)
    df['regularity_index'] = np.nan
    if 'p_subdivision' in df.columns:
        df.loc[mask_residential, 'regularity_index'] = df.loc[mask_residential, 'p_subdivision']

    return df


def score_cities(city_names, cascade_results, your_name='sara'):
    """
    Load metrics for multiple cities, score them, and return a single GeoDataFrame.
    """
    all_scored = []

    for city_name in city_names:
        print(f"Scoring {city_name}...")
        out_file = f"{city_name}_block_metrics_ALL_{your_name}.geoparquet"
        remote = f"{OUTPUT_PATH_RASTER}/{city_name}/{out_file}"

        try:
            city_df = gpd.read_parquet(remote).to_crs(epsg=4326)
            city_df['city_name'] = city_name

            # Filter
            city_df = city_df[city_df['block_area'] >= 700].copy()

            scored = score_single_city(city_df, cascade_results)
            all_scored.append(scored)
            print(f"  ✅ {city_name}: {len(scored):,} blocks scored")
            print(f"     {scored['classification'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  ⚠ {city_name}: {e}")

    if all_scored:
        combined = pd.concat(all_scored, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        return combined
    else:
        return gpd.GeoDataFrame()


# ======================================================================
# INTERACTIVE MAP
# ======================================================================

def make_interactive_map(scored_gdf, output_path='regularity_map.html',
                         save_to_s3=None):
    """
    Generate a Folium interactive map with colored blocks.

    Colors:
        Blue   - formal subdivisions
        Purple - informal subdivisions
        Red    - irregular settlements
        Green  - open space
        Gray   - non-residential / unclassified
    """
    import folium
    from folium.plugins import GroupedLayerControl

    COLOR_MAP = {
        'formal_subdivision': '#2166ac',     # blue
        'informal_subdivision': '#7b3294',   # purple
        'subdivision': '#5e4fa2',            # blue-purple (if no formal/informal split)
        'irregular_settlement': '#d6604d',   # red
        'open_space': '#a6d96a',             # light green
        'non_residential': '#bdbdbd',        # gray
        'unclassified': '#f0f0f0',           # light gray
    }

    OPACITY_MAP = {
        'formal_subdivision': 0.6,
        'informal_subdivision': 0.6,
        'subdivision': 0.6,
        'irregular_settlement': 0.7,
        'open_space': 0.4,
        'non_residential': 0.3,
        'unclassified': 0.2,
    }

    # Center map on the data
    bounds = scored_gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # Add city layers
    cities = scored_gdf['city_name'].unique() if 'city_name' in scored_gdf.columns else ['all']

    for city in cities:
        if city == 'all':
            city_data = scored_gdf
        else:
            city_data = scored_gdf[scored_gdf['city_name'] == city]

        fg = folium.FeatureGroup(name=city.replace('__', ', ').replace('_', ' '))

        for _, row in city_data.iterrows():
            cls = row.get('classification', 'unclassified')
            color = COLOR_MAP.get(cls, '#f0f0f0')
            opacity = OPACITY_MAP.get(cls, 0.3)

            # Build tooltip
            ri = row.get('regularity_index', np.nan)
            ri_str = f"{ri:.2f}" if pd.notna(ri) else "N/A"
            tooltip = (
                f"<b>{cls.replace('_', ' ').title()}</b><br>"
                f"Regularity index: {ri_str}<br>"
                f"Block area: {row.get('block_area', 0):,.0f} m²"
            )

            geojson = folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color, o=opacity: {
                    'fillColor': c,
                    'color': c,
                    'weight': 0.5,
                    'fillOpacity': o,
                },
                tooltip=tooltip,
            )
            geojson.add_to(fg)

        fg.add_to(m)

    folium.LayerControl().add_to(m)

    # Save
    m.save(output_path)
    print(f"✅ Map saved to {output_path}")

    if save_to_s3:
        import s3fs
        fs = s3fs.S3FileSystem()
        fs.put(output_path, save_to_s3)
        print(f"✅ Map uploaded to {save_to_s3}")

    return m


def make_fast_map(scored_gdf, output_path='regularity_map.html'):
    """
    Faster alternative using GeoJSON overlay instead of per-row iteration.
    Better for cities with many blocks (>10K).
    """
    import folium

    COLOR_MAP = {
        'formal_subdivision': '#2166ac',
        'informal_subdivision': '#7b3294',
        'subdivision': '#5e4fa2',
        'irregular_settlement': '#d6604d',
        'open_space': '#a6d96a',
        'non_residential': '#bdbdbd',
        'unclassified': '#f0f0f0',
    }

    OPACITY_MAP = {
        'formal_subdivision': 0.6,
        'informal_subdivision': 0.6,
        'subdivision': 0.6,
        'irregular_settlement': 0.7,
        'open_space': 0.4,
        'non_residential': 0.3,
        'unclassified': 0.2,
    }

    bounds = scored_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # Add regularity index as a rounded string for tooltip
    gdf = scored_gdf.copy()
    gdf['ri_str'] = gdf['regularity_index'].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    gdf['area_str'] = gdf['block_area'].apply(lambda x: f"{x:,.0f}")

    # Style function
    def style_fn(feature):
        cls = feature['properties'].get('classification', 'unclassified')
        return {
            'fillColor': COLOR_MAP.get(cls, '#f0f0f0'),
            'color': COLOR_MAP.get(cls, '#f0f0f0'),
            'weight': 0.5,
            'fillOpacity': OPACITY_MAP.get(cls, 0.3),
        }

    # Keep only columns needed for the map to reduce file size
    cols_to_keep = ['geometry', 'classification', 'regularity_index',
                    'block_area', 'ri_str', 'area_str', 'city_name']
    cols_to_keep = [c for c in cols_to_keep if c in gdf.columns]
    gdf_light = gdf[cols_to_keep].copy()

    # Simplify geometry for faster rendering
    gdf_light['geometry'] = gdf_light.geometry.simplify(0.0001, preserve_topology=True)

    folium.GeoJson(
        gdf_light.__geo_interface__,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=['classification', 'ri_str', 'area_str'],
            aliases=['Type', 'Regularity Index', 'Area (m²)'],
            localize=True,
        ),
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;">
        <b>Urban Regularity Index</b><br>
        <span style="color:#2166ac;">■</span> Formal subdivision<br>
        <span style="color:#7b3294;">■</span> Informal subdivision<br>
        <span style="color:#d6604d;">■</span> Irregular settlement<br>
        <span style="color:#a6d96a;">■</span> Open space<br>
        <span style="color:#bdbdbd;">■</span> Non-residential
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"✅ Map saved to {output_path}")

    return m


# ======================================================================
# SAVE SCORED RESULTS
# ======================================================================

def save_scored_results(scored_gdf, your_name='sara'):
    """
    Save scored blocks per city and as a combined file.
    """
    # Per-city files
    if 'city_name' in scored_gdf.columns:
        for city_name, city_df in scored_gdf.groupby('city_name'):
            out_path = f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}_block_scored_{your_name}.geoparquet"
            city_df.to_parquet(out_path, index=False)
            print(f"  ✅ Saved {city_name}: {len(city_df):,} blocks")

    # Combined file
    combined_path = f"{OUTPUT_PATH}/scored_blocks_all_cities.geoparquet"
    scored_gdf.to_parquet(combined_path, index=False)
    print(f"✅ Combined file saved: {combined_path}")


if __name__ == "__main__":
    print("""
    Usage in notebook:

        from score_and_map import score_cities, make_fast_map, save_scored_results

        # 1. Score cities (after running cascade_models.py)
        scored = score_cities(
            city_names=['Nairobi__Kenya', 'Medellin__Colombia'],
            cascade_results=results,
            your_name='sara'
        )

        # 2. Save results
        save_scored_results(scored)

        # 3. Generate interactive map
        make_fast_map(scored, output_path='regularity_map.html')
    """)

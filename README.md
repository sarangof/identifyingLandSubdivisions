# Identifying Land Subdivisions

## Project summary

This project develops a scalable, automated method to measure the physical regularity of urban blocks across cities in Sub-Saharan Africa and Latin America. The goal is to distinguish planned land subdivisions from irregular settlements using only publicly available geospatial data, specifically OpenStreetMap road networks and Overture Maps building footprints.

The pipeline computes 13 morphometric indicators for each urban block (the polygon enclosed by surrounding roads). These indicators capture characteristics like road straightness, intersection geometry, building alignment, and block shape. A calibrated logistic regression model combines these indicators into a single regularity index: a continuous score between 0 (irregular) and 1 (planned subdivision) for every block. The model was validated against 60,000 human-labeled blocks across 102 cities in both regions, achieving an AUC of 0.84 and near-perfect probability calibration (ECE = 0.003).

The resulting dataset covers approximately 1,238 cities.

This work was conducted in partnership between the World Resources Institute (WRI Ross Center for Sustainable Cities), New York University, and the SALURBAL project.

## Metrics

The pipeline computes the following block-level indicators:

| Metric | Description |
|--------|-------------|
| M1 | Share of buildings within 20m of the nearest road |
| M2 | Average distance from building footprints to the nearest road |
| M3 | Road density (total road length per unit block area) |
| M4 | Share of intersections that are 4-way |
| M5 | Intersection density (count per unit block area) |
| M6 | Building orientation coherence (KL divergence from uniform) |
| M7 | Block width (diameter of the largest inscribed circle) |
| M8 | Road tortuosity (length-weighted straightness of road segments) |
| M9 | Intersection angle deviation from 90 degrees |
| M10 | Building density (count per unit block area) |
| M11 | Built-up area fraction (total building footprint area / block area) |
| M12 | Average building footprint size |
| K | Parcel-layer complexity (Voronoi peeling depth) |

Each metric is standardized to a 0-1 scale where 0 indicates irregular and 1 indicates planned subdivision.

## Installation

0. Install the AWS CLI
1. Clone this repository
2. Create the conda environment: `conda env create -f environment.yml`

## Running on Coiled

From the root of the repository:

0. `conda env update -f environment.yml --prune` and `conda update --all`
1. `conda activate subdivisions`
2. `export AWS_PROFILE=cities`
3. `aws sso login`
4. Verify bucket access: `aws s3 ls wri-cities-sandbox`
5. Start a Coiled notebook:
   ```
   coiled notebook start --account wri-cities-data --region us-west-2 --cpu 16 \
     --name identifyingLandSubdivisions --mount-bucket wri-cities-sandbox \
     --sync --sync-ignore data --sync-ignore cache
   ```
   Choose a CPU count above the number of cities you want to run in parallel. Options are 1, 2, 4, 8, 12, 16, 20, 24, etc. See https://instances.vantage.sh/ for available instance types.

## Pipeline

Run the following in order. Each step depends on the outputs of the previous one.

### Step 0: Define city search areas
**Script:** `getting_1000_cities_search_areas.py`

Matches cities from the WRI city list to urban extents generated via Google Earth Engine, and produces search buffer polygons for each city. Outputs are saved to S3.

### Step 1: Gather input data
**Executor:** `gather_data_executor.ipynb`
**Core logic:** `gather_data_cities.py`

Downloads OpenStreetMap roads, intersections, and natural features, Overture building footprints, and inland water polygons for each city. Saves results to S3 as geoparquet files.

### Step 2: Pre-processing
**Executor:** `pre_processing_executor.ipynb`
**Core logic:** `pre_processing.py`

Computes building-to-road distances, building orientations, and constructs block polygons from the road and natural feature network. Applies coastline clipping (using pre-built land polygons) and inland water subtraction. Computes inscribed circle radii for block width.

### Step 3: Block calculation
**Executor:** `calculate_blocks_executor.ipynb`
**Core logic:** `pre_processing.py` (block production functions)

Produces the final block geometries used as the unit of analysis. Blocks are the polygons enclosed by the road network and natural features, clipped to the urban extent and water boundaries.

### Step 4: Metric calculation
**Executor:** `metric_calculation_executor.ipynb`
**Core logic:** `metrics_calculation.py`, `auxiliary_functions.py`

Computes all 13 block-level metrics (M1-M12 and K). Metrics are computed in groups using Dask-delayed functions for parallelism. Results are saved as per-city geoparquet files on S3.

### Step 5: Post-processing and assembly
**Executor:** `post_processing_executor.ipynb`
**Core logic:** `post_processing_auxiliaries.py`, `standardize_metrics.py`

Merges per-metric outputs into a single block-level file per city (`_block_metrics_ALL_` files). Applies metric standardization (0 = irregular, 1 = subdivision) using the functions in `standardize_metrics.py`.

### Step 6: Validation and scoring
**Notebook:** `validation.ipynb`
**Supporting scripts:** `cascade_models.py`, `pca_fa_pipeline.py`, `model_comparison.py`, `score_and_map.py`

Trains the calibrated cascade model on human-labeled blocks and scores all cities. Produces the regularity index and classification labels. Generates interactive maps.

## Main scripts

| File | Purpose |
|------|---------|
| `getting_1000_cities_search_areas.py` | City matching and urban extent generation via GEE |
| `gather_data_cities.py` | OSM and Overture data download per city |
| `pre_processing.py` | Building distances, block construction, water clipping |
| `auxiliary_functions.py` | Block creation, inscribed circles, K-complexity, azimuth, tortuosity, intersection angles |
| `metrics_calculation.py` | All 13 metric computation functions |
| `standardize_metrics.py` | Per-metric standardization (0-1 scale) |
| `post_processing_auxiliaries.py` | Metric assembly and merge |
| `cascade_models.py` | Calibrated cascade model training and evaluation |
| `pca_fa_pipeline.py` | PCA, factor analysis, and dimensionality analysis |
| `model_comparison.py` | Full vs regularity-only model comparison |
| `score_and_map.py` | Apply trained model to new cities and generate maps |

## Data structure

All data is stored on S3 at `s3://wri-cities-sandbox/identifyingLandSubdivisions/data/`.

```
data/
  input/
    urban_extent/{city}/              # Urban extent polygons (from GEE)
    urban_extent_200m_buffer/{city}/   # Buffered search areas
    roads/{city}/                      # OSM road networks (neatified)
    intersections/{city}/              # OSM street intersections
    buildings/{city}/                  # Overture building footprints
    natural_features_and_railroads/{city}/  # OSM water features
    inland_water/{city}/              # Inland water polygons for clipping
    land_polygons/                    # Global coastline land polygons
    blocks/{city}/                    # Block geometries
  output/
    raster/{city}/                    # Per-city metric and scored block files
    scored_blocks_all_cities.geoparquet  # Combined scored output
    logs/                             # Execution logs and summaries
```

## Environment

The primary environment is defined in `subdivisions.yml`. Key dependencies include Python 3.12, GeoPandas, Shapely, OSMnx, scikit-learn, Dask, and Folium. The pipeline uses AWS S3 for storage and Coiled for distributed computation.

## License

This work was produced under contract for the World Resources Institute. See contract terms for intellectual property and licensing details.
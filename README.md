# Identifying Land Subdivisions

## Installation

0. Install aws cli
1. Clone this repository
2. `conda env create -f environment.yml`


# Run on Coiled notebook
0. `conda env update -f environment.yml --prune` and `conda update --all`
1. `conda activate subdivisions`
2. `export AWS_PROFILE=cities`
3. `aws sso login`
4. `coiled notebook start --account wri-cities-data --region us-west-2 --cpu 16 --name identifyingLandSubdivisions --mount-bucket wri-cities-sandbox --sync --sync-ignore data`
    * Note: Choose a number of cpus above the number of cities you want to run, so they will all run in parrellel. Options are 1, 2, 4, 8, 12, 16, 20, 24... (See https://instances.vantage.sh/ for more.)



# Proposed code flow
Create list of cities and loop through them using Dask so that everything happens in parrallel
1. Get Grid from s3, if not there, create and save to s3. Save grid cell counts to csv.
    * Get Urban Extent from s3, if it is not there, get it from GEE and save to s3. Save `city_area` to csv.
2. Get data from s3, if not there, download it. Save file sizes to csv.
3. Run calculation.
    * Set # of grid cell to process.


# Questions
1. Was there a reason you were defining your own ee_to_gdf funtion and saving to geojson file?
2. Reading data takes a long time. Can we store in compressed file formats like parquet instead of geojson?
3. Do we need to compute `batch_results` in process_city or can we let Dask handle that?

# Identifying Land Subdivisions

## Installation

0. Install aws cli
1. Clone this repository
2. `conda env create -f environment.yml`


# Run on Coiled notebook
From the root of the repository:
0. `conda env update -f environment.yml --prune` and `conda update --all`
1. `conda activate subdivisions`
2. `export AWS_PROFILE=cities`
3. `aws sso login`
4. Test that you can access the bucket with `aws s3 ls wri-cities-sandbox`
4. `coiled notebook start --account wri-cities-data --region us-west-2 --cpu 16 --name identifyingLandSubdivisions --mount-bucket wri-cities-sandbox --sync --sync-ignore data --sync-ignore cache`
    * Note: Choose a number of cpus above the number of cities you want to run, so they will all run in parrellel. Options are 1, 2, 4, 8, 12, 16, 20, 24... (See https://instances.vantage.sh/ for more.)


# How to run
0. Set up the environment.
1. Run gather_data_cities.py
2. From pre_processing.py, run the following functions: calculate_building_distances_to_roads(), produce_blocks(), produce_azimuths()
1. pre_process_all_cities.py
2. produce_azimuths


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


# Run the following notebooks in order
0. `gather_1000_cities_search_areas.py`
1. `gather_data_executor.ipynb`
2. `calculate_blocks_executor.ipynb`
3. `pre_processing_executor.ipynb`
4. `metric_calculation_executor.ipynb`
5. `post_processing_executor.ipynb`


# Main files
0. `getting_1000_cities_search_areas.py`
1. `gather_data_cities.py`
2. `auxiliary_functions.py`
3. `pre_processing.py`
4. `metrics_calculation.py`
5. `standardize_metrics.py`
6. `post_processing_auxiliaries.py`

# Analysis files
* `validation_small_scale_test`
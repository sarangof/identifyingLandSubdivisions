# Identifying Land Subdivisions

## Installation

0. Install aws cli
1. Clone this repository
2. `conda env create -f environment.yml`


# Run on Coiled notebook
1. `conda activate subdivisions`
2. `export AWS_PROFILE=cities`
3. `aws sso login`
4. `coiled notebook start --account data-lab --region us-west-2 --name test-s3 --mount-bucket wri-datalab-coiled --sync --vm-type c6g.metal`
5. Set `main_path = '/mount/wri-cities-sandbox/identifyingLandSubdivisions'`
6. Use `run.ipynb` on the notebook server once it opens

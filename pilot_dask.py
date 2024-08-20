import geopandas as gpd
from dask import delayed, compute

@delayed
def dummy_function():
    print("Dummy function executed.")
    return "Done"

result = dummy_function()
computed_result = compute(result)
print(f"Computed Result: {computed_result}")
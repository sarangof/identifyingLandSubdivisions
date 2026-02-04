import os
import s3fs
import pandas as pd
import geopandas as gpd
from cloudpathlib import S3Path
from standardize_metrics import *
import dask.dataframe as dd

# ------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------

MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
OUTPUT_PATH = f"{MAIN_PATH}/output"
OUTPUT_PATH_RASTER = f"{OUTPUT_PATH}/raster"
OUTPUT_PATH_CSV = f"{OUTPUT_PATH}/csv"
OUTPUT_PATH_PNG = f"{OUTPUT_PATH}/png"

fs = s3fs.S3FileSystem(anon=False)

# ------------------------------------------------------------------------------
# S3 SAVE
# ------------------------------------------------------------------------------

def s3_save(df, output_file, tmp_path, remote_path):
    os.makedirs(tmp_path, exist_ok=True)
    tmp_file = f"{tmp_path}/{output_file}"

    if output_file.endswith(".csv"):
        df.to_csv(tmp_file, index=False)
    elif output_file.endswith(".parquet") or output_file.endswith(".geoparquet"):
        df.to_parquet(tmp_file, index=False)
    else:
        raise ValueError("Unsupported file type for s3_save.")

    S3Path(remote_path).upload_from(tmp_file)
    os.remove(tmp_file)
    print(f"✅ Saved {remote_path}")


# ------------------------------------------------------------------------------
# LOAD ONE METRIC DATASET
# ------------------------------------------------------------------------------

def _load_metric_dataset(path, is_file=False):
    """
    Load a metric dataset from S3.

    - if is_file=False: treat `path` as a *directory* containing part*.parquet
    - if is_file=True: treat `path` as a single parquet file
    """
    print(f"📥 Trying to load: {path} (is_file={is_file})")

    if is_file:
        # single .parquet / .geoparquet
        if not fs.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        # geopandas can read both parquet and geoparquet
        df = gpd.read_parquet(path)
    else:
        # directory dataset written by dask.to_parquet
        # check there is at least one .parquet inside
        glob_pattern = path.rstrip("/") + "/*.parquet"
        matches = fs.glob(glob_pattern)
        if not matches:
            raise FileNotFoundError(f"No .parquet files under {path}")
        # dask can read the dataset dir directly
        df = dd.read_parquet(path, engine="pyarrow").compute()

    # make sure we have block_id as index
    if "block_id" in df.columns:
        df = df.set_index("block_id")
    else:
        df.index.name = "block_id"

    return df

def minmax_normalize_std_columns(df):
    """
    Create _final columns from _std columns using min–max normalization.
    """
    std_cols = [c for c in df.columns if c.endswith("_std")]

    for c in std_cols:
        min_val = df[c].min()
        max_val = df[c].max()

        if max_val == min_val:
            # Degenerate case: constant metric
            df[c.replace("_std", "_final")] = 0.0
        else:
            df[c.replace("_std", "_final")] = (
                (df[c] - min_val) / (max_val - min_val)
            )

    return df

# ------------------------------------------------------------------------------
# MERGE ALL METRIC OUTPUTS PER CITY
# ------------------------------------------------------------------------------

def merge_city_metrics(city_name, YOUR_NAME):
    """
    Loads and merges all block-level metric datasets into a single GeoDataFrame.
    """
    base = f"{OUTPUT_PATH_RASTER}/{city_name}"

    paths = {
        "m1_m2":             (f"{base}/{city_name}_block_metrics_1_2_{YOUR_NAME}", False),
        "m3_m4_m5_10_11_12": (f"{base}/{city_name}_block_metrics_3_4_5_10_11_12_{YOUR_NAME}", False),
        "m6_m7":             (f"{base}/{city_name}_block_metrics_6_7_{YOUR_NAME}", False),
        "m8_m9":             (f"{base}/{city_name}_block_metrics_8_9_{YOUR_NAME}", False),
        "k_blocks":          (f"{base}/{city_name}_block_metrics_k_{YOUR_NAME}.geoparquet", True),
    }

    df_main = None

    # pick the first available dataset as base (ideally one with geometry)
    for key in ["m1_m2", "m3_m4_m5_10_11_12", "m6_m7", "m8_m9", "k_blocks"]:
        path, is_file = paths[key]
        try:
            df = _load_metric_dataset(path, is_file=is_file)
            if "geometry" in df.columns:
                df = gpd.GeoDataFrame(df, geometry="geometry")
            df_main = df
            print(f"✅ Loaded base metrics from {key}")
            break
        except Exception as e:
            print(f"⚠️ Could not load {key} from {path}: {e}")

    if df_main is None:
        raise ValueError(f"No metric datasets found for {city_name}.")

    # merge the remaining datasets
    for key in ["m1_m2", "m3_m4_m5_10_11_12", "m6_m7", "m8_m9", "k_blocks"]:
        path, is_file = paths[key]
        try:
            df = _load_metric_dataset(path, is_file=is_file)
        except Exception:
            continue  # already logged above; just skip

        # Drop geometry from the right if left already has one
        if "geometry" in df.columns and "geometry" in df_main.columns:
            df = df.drop(columns=["geometry"])

        # NEW: drop any other overlapping columns so join doesn't explode
        overlap = [c for c in df.columns if c in df_main.columns]
        if overlap:
            print(f"   dropping overlapping columns from {key}: {overlap}")
            df = df.drop(columns=overlap)

        df_main = df_main.join(df, how="left")
        print(f"→ merged {key}")


    df_main.index.name = "block_id"
    return df_main


# ------------------------------------------------------------------------------
# STANDARDIZE + REGULARITY INDEX
# ------------------------------------------------------------------------------

def finalize_metrics(df):
    """
    Standardize raw metrics (m1_raw, ..., m12_raw, etc.),
    then min–max normalize them into _final columns,
    then compute regularity_index from _final columns.
    """

    # --------------------------------------------------
    # 1. Ensure all _std columns exist
    # --------------------------------------------------
    df['k_complexity_raw'] = df['k_complexity'].copy()
    raw_cols = [c for c in df.columns if c.endswith("_raw")]
    print(raw_cols)
    for col in raw_cols:
        std_col = col.replace("_raw", "_std")
        if std_col not in df.columns:
            func = standardization_functions[col.replace("_raw", "")]
            df[std_col] = df[col].apply(func)

    # --------------------------------------------------
    # 2. Min–max normalize _std → _final  
    # --------------------------------------------------
    df = minmax_normalize_std_columns(df)

    # --------------------------------------------------
    # 3. Compute regularity index from _final columns
    # --------------------------------------------------
    final_cols = [c for c in df.columns if c.endswith("_final")]

    df["regularity_index"] = df[final_cols].mean(axis=1)

    return df



# ------------------------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------------------------

def save_summary(df, city_name, YOUR_NAME):
    raw_cols = [c for c in df.columns if c.endswith("_raw")]
    std_cols = [c for c in df.columns if c.endswith("_std")]

    summary = df[raw_cols + std_cols + ["regularity_index"]].describe().T

    output_file = f"{city_name}_summary_{YOUR_NAME}.csv"
    remote_path = f"{OUTPUT_PATH_CSV}/{city_name}/{output_file}"
    s3_save(summary, output_file, ".", remote_path)


# ------------------------------------------------------------------------------
# PIPELINE WRAPPER
# ------------------------------------------------------------------------------

def post_process_city(city_name, YOUR_NAME):
    """
    Full pipeline:
    - load all block metric datasets (M1..M12, M6/7, M8/9, K)
    - merge them
    - standardize + compute regularity index
    - save merged + save summary
    """
    print(f"🔄 Post-processing {city_name}...")

    df = merge_city_metrics(city_name, YOUR_NAME)
    df_final = finalize_metrics(df)

    # Save final merged dataset
    out_file = f"{city_name}_all_block_metrics_{YOUR_NAME}.geoparquet"
    remote = f"{OUTPUT_PATH_RASTER}/{city_name}/{out_file}"
    s3_save(df_final, out_file, ".", remote)

    # Save CSV summary
    save_summary(df_final, city_name, YOUR_NAME)

    print(f"🎉 Done: {city_name}")
    return df_final




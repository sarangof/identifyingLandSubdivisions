import os
import s3fs
import pandas as pd
import geopandas as gpd
from cloudpathlib import S3Path
from standardize_metrics import *
import dask.dataframe as dd
import os, uuid
import dask_geopandas as dgpd
import geopandas as gpd

from standardize_metrics import *


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
    #df = minmax_normalize_std_columns(df) #no longer working

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






def safe_merge_on_index(left, right, right_prefix=None):
    """
    Merge right into left on index without suffix collisions.
    - Drops geometry from right
    - If right_prefix is provided, prefixes ALL right columns before merge
    - Drops any overlaps to avoid MergeError from duplicate suffixes
    """
    if right is None:
        return left

    right2 = right.drop(columns="geometry", errors="ignore")

    if right_prefix:
        right2 = right2.rename(columns={c: f"{right_prefix}{c}" for c in right2.columns})

    overlap = [c for c in right2.columns if c in left.columns]
    if overlap:
        right2 = right2.drop(columns=overlap, errors="ignore")

    return left.merge(right2, left_index=True, right_index=True, how="left")


def write_parquet_to_s3_atomic(gdf, out_s3_uri: str, fs, city_name: str):
    """
    Robust single-file S3 write:
    local -> put(tmp) -> mv(tmp->final)
    Prevents 0-byte final objects.
    """
    local_tmp = f"/tmp/{city_name}_{uuid.uuid4().hex}.geoparquet"
    out_tmp = out_s3_uri + ".tmp"

    # remove 0-byte leftover at final if any
    try:
        if fs.exists(out_s3_uri) and fs.info(out_s3_uri).get("Size", 0) == 0:
            fs.rm(out_s3_uri)
    except Exception:
        pass

    # write local first
    gdf.to_parquet(local_tmp, engine="pyarrow", index=False)

    # upload then swap into place
    fs.put(local_tmp, out_tmp)
    fs.mv(out_tmp, out_s3_uri)

    try:
        os.remove(local_tmp)
    except Exception:
        pass

def apply_metric_standardization(df):
    """
    Adds m{i}_std from m{i}_raw using the per-metric functions in standardize_metrics.py.
    Expects keys like 'metric_1', 'metric_2', ...
    """
    import re

    for key, std_func in standardization_functions.items():
        m = re.match(r"metric_(\d+)$", str(key))
        if not m:
            continue

        i = int(m.group(1))
        raw_col = f"m{i}_raw"
        std_col = f"m{i}_std"

        if raw_col not in df.columns:
            continue

        df[std_col] = df[raw_col].map_partitions(
            std_func,
            meta=(std_col, "float64")
        )

    return df


def consolidate_city_to_all(city_name: str, YOUR_NAME: str):
    """
    PASS 2:
    Read the per-metric-group outputs, merge safely (k_* prefixed),
    add *_std (per-city), and write ONE consolidated file.
    """
    base = f"{OUTPUT_PATH_RASTER}/{city_name}/{city_name}"

    p_12        = f"{base}_block_metrics_1_2_{YOUR_NAME}"
    p_345101112 = f"{base}_block_metrics_3_4_5_10_11_12_{YOUR_NAME}"
    p_67        = f"{base}_block_metrics_6_7_{YOUR_NAME}"
    p_89        = f"{base}_block_metrics_8_9_{YOUR_NAME}"
    p_k         = f"{base}_block_metrics_k_{YOUR_NAME}.geoparquet"

    out = f"{base}_block_metrics_ALL_{YOUR_NAME}.geoparquet"

    # skip if exists and non-empty
    try:
        if fs.exists(out) and fs.info(out).get("Size", 0) > 0:
            return out
    except Exception:
        pass

    # geometry master
    g_base = dgpd.read_parquet(p_345101112)

    g_12 = dgpd.read_parquet(p_12)
    g_67 = dgpd.read_parquet(p_67)
    g_89 = dgpd.read_parquet(p_89)
    g_k  = dgpd.read_parquet(p_k)

    df = g_base
    df = safe_merge_on_index(df, g_12)
    df = safe_merge_on_index(df, g_67)
    df = safe_merge_on_index(df, g_89)

    # prefix ALL k-block cols to avoid collisions (optimal_point, max_radius, etc.)
    df = safe_merge_on_index(df, g_k, right_prefix="k_")

    # per-city standardization (NOT global)
    df = apply_metric_standardization(df)

    pdf = df.compute()
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs=g_base.crs)

    write_parquet_to_s3_atomic(gdf, out, fs, city_name=city_name)
    return out

import os
import re
import sqlite3
import tempfile
import traceback
import unicodedata
from difflib import SequenceMatcher, get_close_matches

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

from cloudpathlib import S3Path
import s3fs
from difflib import get_close_matches

os.environ["AWS_PROFILE"] = "cities"


# ----------------------------
# CONFIG
# ----------------------------
MAIN_PATH = "s3://wri-cities-sandbox/identifyingLandSubdivisions/data"
INPUT_PATH = os.path.join(MAIN_PATH, "input")
SEARCH_BUFFER_PATH = os.path.join(INPUT_PATH, "city_info", "search_buffers")

GPKG_PATH = "../data/Sumans_data/combined_cities.gpkg"
GPKG_LAYER = None  # None = auto-pick first layer

CITIES_CSV = "../data/city_lists/cities_ssa_latam.csv"
CITIES_CSV_SEP = ";"
CITIES_CSV_ENCODING = "utf-8"

CITY_COL_GPKG = "city_name"
GEOM_COL_GPKG = "geometry"

BUFFER_DEG = 0.01
SKIP_IF_EXISTS = True

# If True, rows that are ambiguous in GPKG will be logged+skipped (instead of choosing a guess).
# You can set to False to auto-choose and proceed, while still logging.
SKIP_AMBIGUOUS_GPKG_ROWS = False

# Optional: manual overrides when a specific CSV row should map to a specific GPKG raw name.
# Keyed by (csv_city_key, normalized_country_key) -> gpkg_city_raw
# Example:
# GPKG_OVERRIDES = {
#   ("georgetown", "guyana"): "Georgetown",
#   ("rioverde", "brazil"): "Rio_Verde",
# }
GPKG_OVERRIDES = {}

# Initialize S3 FS (forces creds to be present)
fs = s3fs.S3FileSystem(anon=False)


# ----------------------------
# Encoding + normalization
# ----------------------------
def _normalize_smart_quotes(s: str) -> str:
    if s is None:
        return s
    s = str(s)
    repl = {
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "’": "'", "‘": "'", "‚": "'", "‛": "'",
        "´": "'", "`": "'",
        "": '"', "": '"', "": "'", "": "'",
        "": "-", "": "-",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _fix_mojibake(s: str) -> str:
    if s is None:
        return s
    s = _normalize_smart_quotes(str(s))

    # quick exit: pure ascii usually fine
    if s.isascii():
        return s

    for enc in ("cp1252", "latin1"):
        try:
            repaired = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
            if "�" not in repaired:
                return repaired
        except Exception:
            pass
    return s


def s3_safe_token(s: str) -> str:
    """
    Safe token for S3 folder/file names.
    """
    s = "" if s is None else str(s).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""

    s = _fix_mojibake(s)

    # strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # normalize separators
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
    s = s.replace("-", "_").replace("/", "_").replace(" ", "_")

    # strip quotes
    s = s.replace("'", "_").replace('"', "_")

    # remove anything else unsafe
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def name_key(s: str) -> str:
    """
    Canonical key for MATCHING.
    """
    s = "" if s is None else str(s).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    s = _fix_mojibake(s)

    # strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ").replace("'", " ")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def s3_exists(s3_uri: str) -> bool:
    try:
        return S3Path(s3_uri).exists()
    except Exception:
        return False


# ----------------------------
# GPKG helpers
# ----------------------------
def pick_layer(gpkg_path: str, explicit_layer: str | None) -> str:
    if explicit_layer:
        return explicit_layer
    layers = gpd.list_layers(gpkg_path)
    if layers.empty:
        raise ValueError(f"No layers found in {gpkg_path}")
    return layers.loc[0, "name"]


def get_distinct_cities_sqlite(gpkg_path: str, layer: str, city_col: str) -> list[str]:
    conn = sqlite3.connect(gpkg_path)
    try:
        q = f"SELECT DISTINCT {city_col} AS city_name FROM {layer} WHERE {city_col} IS NOT NULL"
        df = pd.read_sql_query(q, conn)
        return sorted(df["city_name"].astype(str).unique().tolist())
    finally:
        conn.close()


def find_gpkg_city_like_sqlite(gpkg_path: str, layer: str, city_col: str, pattern: str, limit: int = 50) -> list[str]:
    """
    Fallback: search for city names via LIKE (case-insensitive).
    pattern example: '%Mexico%City%'
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        q = f"""
            SELECT DISTINCT {city_col} AS city_name
            FROM {layer}
            WHERE {city_col} IS NOT NULL
              AND LOWER({city_col}) LIKE LOWER(?)
            LIMIT ?
        """
        df = pd.read_sql_query(q, conn, params=(pattern, limit))
        return df["city_name"].astype(str).tolist()
    finally:
        conn.close()


# ----------------------------
# Geometry
# ----------------------------
def build_search_buffer(combined_gdf: gpd.GeoDataFrame, buffer_deg: float = BUFFER_DEG):
    geom = combined_gdf.geometry
    geom = geom[geom.notna() & ~geom.is_empty]
    if geom.empty:
        return None
    city_outline = geom.apply(lambda g: g.convex_hull.buffer(buffer_deg))
    return unary_union(city_outline)


# ----------------------------
# MAIN
# ----------------------------
def main():
    layer = pick_layer(GPKG_PATH, GPKG_LAYER)
    print(f"Using layer: {layer}")

    # ---- Read CSV (one output per ROW) ----
    city_df = pd.read_csv(CITIES_CSV, sep=CITIES_CSV_SEP, encoding=CITIES_CSV_ENCODING).copy()

    # Extract city from "city_name" (before comma)
    city_df["csv_city_raw"] = city_df["city_name"].apply(lambda x: _fix_mojibake(str(x).split(",")[0].strip()))
    city_df["csv_city_key"] = city_df["csv_city_raw"].apply(name_key)

    if "country_name" in city_df.columns:
        city_df["csv_country_raw"] = city_df["country_name"].astype(str).fillna("").apply(_fix_mojibake)
    else:
        city_df["csv_country_raw"] = ""

    city_df["csv_country_key"] = city_df["csv_country_raw"].apply(name_key)

    print("CSV rows:", len(city_df))
    print("Unique CSV city keys:", city_df["csv_city_key"].nunique())
    print("CSV duplicated city keys:", (city_df.groupby("csv_city_key").size().gt(1)).sum())

    # ---- Read distinct GPKG city_name -> key index ----
    gpkg_city_raw = get_distinct_cities_sqlite(GPKG_PATH, layer, CITY_COL_GPKG)
    gpkg_raw_to_key = {raw: name_key(raw) for raw in gpkg_city_raw}

    gpkg_key_to_raws = {}
    for raw, k in gpkg_raw_to_key.items():
        if k:
            gpkg_key_to_raws.setdefault(k, []).append(raw)

    print("Distinct GPKG city_name:", len(gpkg_city_raw))
    print("Distinct GPKG name_keys:", len(gpkg_key_to_raws))

    # ---- Build per-row match plan ----
    plan = []
    ambiguous = []
    missing = []

    for idx, r in city_df.iterrows():
        csv_city_raw = r.get("csv_city_raw", "")
        csv_city_key = r.get("csv_city_key", "")
        csv_country_raw = r.get("csv_country_raw", "")
        csv_country_key = r.get("csv_country_key", "")

        if not csv_city_key:
            missing.append({
                "row_index": idx,
                "csv_city_raw": csv_city_raw,
                "csv_country_raw": csv_country_raw,
                "reason": "EMPTY_CITY_KEY"
            })
            continue

        # Override hook (row-specific)
        ov_key = (csv_city_key, csv_country_key)
        if ov_key in GPKG_OVERRIDES:
            chosen = GPKG_OVERRIDES[ov_key]
            plan.append({
                "row_index": idx,
                "csv_city_raw": csv_city_raw,
                "csv_country_raw": csv_country_raw,
                "csv_city_key": csv_city_key,
                "gpkg_city_raw": chosen,
                "match_method": "OVERRIDE",
            })
            continue

        candidates = gpkg_key_to_raws.get(csv_city_key, [])

        # Fallback if missing by key: try LIKE search using the raw city tokens
        if not candidates:
            # build a forgiving pattern e.g. "%mexico%city%"
            parts = re.split(r"[\s\-_\/]+", str(csv_city_raw).strip())
            parts = [p for p in parts if p]
            if parts:
                pattern = "%" + "%".join([p.lower() for p in parts]) + "%"
                like_hits = find_gpkg_city_like_sqlite(GPKG_PATH, layer, CITY_COL_GPKG, pattern, limit=100)
            else:
                like_hits = []

            # keep only those whose key matches OR that are very close by similarity
            if like_hits:
                like_by_key = [h for h in like_hits if name_key(h) == csv_city_key]
                candidates = like_by_key if like_by_key else like_hits

        

        if not candidates:
            # Fuzzy fallback on gpkg *keys* (handles Medellin vs Medell_n -> medellin vs medelln)
            gpkg_key_list = list(gpkg_key_to_raws.keys())

            # try a fairly permissive cutoff; we’ll validate with scoring next
            guess_keys = get_close_matches(csv_city_key, gpkg_key_list, n=5, cutoff=0.80)

            if guess_keys:
                # pick the best by SequenceMatcher on the keys themselves
                scored_keys = sorted(
                    [(k, _sim(csv_city_key, k)) for k in guess_keys],
                    key=lambda x: x[1],
                    reverse=True
                )
                best_key, best_score = scored_keys[0]

                # only accept if it’s actually close (tune threshold if needed)
                if best_score >= 0.88:
                    print(f"🔎 FUZZY MATCH: '{csv_city_raw}' ({csv_city_key}) -> gpkg_key '{best_key}' score={best_score:.3f} candidates={candidates[:3]}")
                    candidates = gpkg_key_to_raws[best_key]


        if not candidates:
            missing.append({
                "row_index": idx,
                "csv_city_raw": csv_city_raw,
                "csv_country_raw": csv_country_raw,
                "csv_city_key": csv_city_key,
                "reason": "NO_GPKG_MATCH",
            })
            continue

        if len(candidates) == 1:
            plan.append({
                "row_index": idx,
                "csv_city_raw": csv_city_raw,
                "csv_country_raw": csv_country_raw,
                "csv_city_key": csv_city_key,
                "gpkg_city_raw": candidates[0],
                "match_method": "NAME_KEY_UNIQUE" if gpkg_key_to_raws.get(csv_city_key) else "LIKE_UNIQUE",
            })
            continue

        # Multiple GPKG candidates for same key -> choose best by similarity to csv_city_raw
        # Normalize for comparison (underscores/spaces)
        csv_norm = re.sub(r"[\s_]+", " ", str(csv_city_raw).strip().lower())
        scored = []
        for c in candidates:
            c_norm = re.sub(r"[\s_]+", " ", str(c).strip().lower())
            scored.append((c, _sim(csv_norm, c_norm)))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_c, best_s = scored[0]
        second_s = scored[1][1] if len(scored) > 1 else -1

        # If best is not clearly better, log ambiguity (and optionally skip)
        if (best_s - second_s) < 0.08:
            ambiguous.append({
                "row_index": idx,
                "csv_city_raw": csv_city_raw,
                "csv_country_raw": csv_country_raw,
                "csv_city_key": csv_city_key,
                "gpkg_candidates": "; ".join(sorted(candidates)),
                "best_guess": best_c,
                "best_score": best_s,
                "second_score": second_s,
                "reason": "AMBIGUOUS_IN_GPKG",
            })
            if SKIP_AMBIGUOUS_GPKG_ROWS:
                continue

        # Otherwise pick the best guess
        plan.append({
            "row_index": idx,
            "csv_city_raw": csv_city_raw,
            "csv_country_raw": csv_country_raw,
            "csv_city_key": csv_city_key,
            "gpkg_city_raw": best_c,
            "match_method": "NAME_KEY_MULTI_BEST_GUESS",
        })

    df_plan = pd.DataFrame(plan)
    df_amb = pd.DataFrame(ambiguous)
    df_missing = pd.DataFrame(missing)

    print("\n--- Matching Summary (per CSV row) ---")
    print("Will process rows:", len(df_plan))
    print("Ambiguous rows:", len(df_amb))
    print("Missing rows:", len(df_missing))

    os.makedirs("../data", exist_ok=True)
    df_plan.to_csv("../data/match_plan_rows.csv", index=False)
    df_amb.to_csv("../data/match_ambiguous_rows.csv", index=False)
    df_missing.to_csv("../data/match_missing_rows.csv", index=False)

    # ---- Build buffers + upload (one output per CSV row) ----
    n_ok = n_skip = n_fail = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for _, row in df_plan.iterrows():
            gpkg_city_raw = row["gpkg_city_raw"]
            csv_city_raw = row.get("csv_city_raw", "")
            csv_country_raw = row.get("csv_country_raw", "")

            city_token = s3_safe_token(csv_city_raw) or s3_safe_token(gpkg_city_raw)
            country_token = s3_safe_token(csv_country_raw) if csv_country_raw else ""
            folder_token = f"{city_token}__{country_token}" if country_token else city_token

            out_dir_s3 = f"{SEARCH_BUFFER_PATH}/{folder_token}"
            out_file_s3 = f"{out_dir_s3}/{folder_token}_search_buffer.geoparquet"

            if SKIP_IF_EXISTS and s3_exists(out_file_s3):
                print(f"⏭️  Exists, skipping: {out_file_s3}")
                n_skip += 1
                continue


            try:
                gpkg_city_sql = str(gpkg_city_raw).replace("'", "''")
                where = f"{CITY_COL_GPKG} = '{gpkg_city_sql}'"

                gdf_city = gpd.read_file(
                    GPKG_PATH,
                    layer=layer,
                    where=where,
                    columns=[CITY_COL_GPKG, GEOM_COL_GPKG],
                )

                if gdf_city.empty:
                    print(f"⚠️  Empty subset for GPKG city '{gpkg_city_raw}' -> {folder_token}")
                    n_fail += 1
                    continue

                if gdf_city.crs is None:
                    gdf_city = gdf_city.set_crs("EPSG:4326", allow_override=True)

                bounding_geom = build_search_buffer(gdf_city, buffer_deg=BUFFER_DEG)
                if bounding_geom is None:
                    print(f"⚠️  Could not build geometry for {folder_token}")
                    n_fail += 1
                    continue

                out_gdf = gpd.GeoDataFrame(
                    pd.DataFrame({
                        "city_name": [folder_token],
                        "csv_city_raw": [csv_city_raw],
                        "csv_country_raw": [csv_country_raw],
                        "gpkg_city_raw": [gpkg_city_raw],
                        "csv_city_key": [row.get("csv_city_key", "")],
                        "match_method": [row.get("match_method", "")],
                        "row_index": [row.get("row_index", -1)],
                    }),
                    geometry=[bounding_geom],
                    crs=gdf_city.crs
                )

                local_path = os.path.join(tmpdir, f"{folder_token}_search_buffer.geoparquet")
                out_gdf.to_parquet(local_path)

                S3Path(out_dir_s3).mkdir(parents=True, exist_ok=True)
                S3Path(out_file_s3).upload_from(local_path)
                print(f"✅ Wrote: {out_file_s3}   (row_index={row.get('row_index')}, gpkg='{gpkg_city_raw}')")
                n_ok += 1
                if (n_ok + n_skip + n_fail) % 50 == 0:
                    print(f"…progress: OK={n_ok} SKIP={n_skip} FAIL={n_fail}")


            except Exception as e:
                print(f"❌ Failed for '{gpkg_city_raw}' -> '{folder_token}': {e}")
                traceback.print_exc()
                n_fail += 1

    print("\n--- Summary ---")
    print(f"OK:   {n_ok}")
    print(f"SKIP: {n_skip}")
    print(f"FAIL: {n_fail}")
    print("\nReports written:")
    print(" - ../data/match_plan_rows.csv")
    print(" - ../data/match_ambiguous_rows.csv")
    print(" - ../data/match_missing_rows.csv")


if __name__ == "__main__":
    main()

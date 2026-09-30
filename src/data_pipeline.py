"""
data_pipeline.py -- loading real CICIoT2023 CSVs, label mapping, LightGBM
gain-based feature selection (Section III.3), stratified splitting, and
Eq. (1) direction-aware min-max normalization.

HONESTY NOTE ON SPLITTING: the manuscript specifies "device/session-aware
temporal partitions rather than random row-level splitting" (Section VI).
The publicly released CICIoT2023 CSVs do not carry an explicit device or
session identifier column, so a literal device-aware split is not directly
constructible from the standard release. This script falls back to
stratified random splitting with a fixed seed, and prints a warning saying
so every time it runs. If you have the original per-device PCAP-to-CSV
mapping metadata, replace `train_val_test_split` with a real group-aware
split (e.g., sklearn's GroupShuffleSplit keyed on that device ID) instead
of silently trusting this fallback.
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import config as C


def find_csv_files(data_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. Place the CICIoT2023 CSV "
            f"part-files there (or point config.DATA_DIR at wherever you "
            f"extracted them) before running this pipeline."
        )
    return files


def load_ciciot2023(data_dir: str = C.DATA_DIR,
                     sample_frac: float = None,
                     max_rows_per_file: int = None,
                     label_col: str = C.LABEL_COL) -> pd.DataFrame:
    """
    Load and concatenate CICIoT2023 CSV part-files.

    The full release is on the order of tens of GB across ~169 files and
    ~47 million rows. For iteration speed, pass sample_frac (e.g. 0.05 for
    5%) or max_rows_per_file to subsample per file as it's read, rather
    than loading everything and then subsampling in memory. Leave both
    None for a full run once you have the compute/time budget for it.
    """
    files = find_csv_files(data_dir)
    print(f"[data_pipeline] Found {len(files)} CSV file(s) under {data_dir}")

    frames = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        # normalize the label column name if the release uses a different case
        if label_col not in df.columns:
            candidates = [c for c in df.columns if c.lower() == label_col.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: label_col})
            else:
                raise KeyError(
                    f"Could not find a label column (looked for '{label_col}') "
                    f"in {fp}. Columns present: {list(df.columns)}"
                )
        if max_rows_per_file is not None and len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=C.RANDOM_SEED)
        elif sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=C.RANDOM_SEED)
        frames.append(df)
        if (i + 1) % 20 == 0:
            print(f"[data_pipeline]   ...loaded {i+1}/{len(files)} files")

    full = pd.concat(frames, ignore_index=True)
    # downcast numeric dtypes to keep memory sane on large loads
    for col in full.select_dtypes(include=["float64"]).columns:
        full[col] = pd.to_numeric(full[col], downcast="float")
    for col in full.select_dtypes(include=["int64"]).columns:
        full[col] = pd.to_numeric(full[col], downcast="integer")

    print(f"[data_pipeline] Loaded {len(full):,} rows total.")
    return full


def map_labels(df: pd.DataFrame, label_col: str = C.LABEL_COL) -> pd.DataFrame:
    """Adds 'category' (7-class + Benign) and 'binary_label' (0=benign,1=malicious)."""
    df = df.copy()
    raw_labels = df[label_col].astype(str)
    df["category"] = raw_labels.map(C.LABEL_TO_CATEGORY)

    unmatched = sorted(raw_labels[df["category"].isna()].unique().tolist())
    if unmatched:
        warnings.warn(
            f"[data_pipeline] {len(unmatched)} label string(s) not found in "
            f"config.LABEL_TO_CATEGORY and left as NaN category: {unmatched}. "
            f"Extend the mapping in config.py to cover them; rows with an "
            f"unmapped category are dropped from category-level analysis "
            f"below but kept for binary classification (treated as malicious "
            f"unless the string is an exact, case-sensitive match for a known "
            f"benign label)."
        )
        looks_benign = raw_labels.str.lower().str.contains("benign")
        df.loc[df["category"].isna() & looks_benign, "category"] = "Benign"
        df.loc[df["category"].isna() & ~looks_benign, "category"] = "Unmapped"

    df["binary_label"] = (df["category"] != "Benign").astype(int)
    return df


def select_top_features(train_df: pd.DataFrame, raw_features: list,
                         label_col: str = "binary_label",
                         k: int = C.N_TOP_FEATURES,
                         seed: int = C.RANDOM_SEED) -> list:
    """
    Reproduces Section III.3: fit a quick LightGBM on all raw features,
    rank by gain importance, keep the top k. This model is throwaway; the
    real Layer-1 model is retrained on the selected features only, in
    trust_engine.train_layer1().
    """
    X = train_df[raw_features].fillna(0.0)
    y = train_df[label_col]
    quick_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, random_state=seed, verbosity=-1
    )
    quick_model.fit(X, y)
    importances = pd.Series(quick_model.feature_importances_, index=raw_features)
    top_k = importances.sort_values(ascending=False).head(k).index.tolist()
    print(f"[data_pipeline] Top-{k} features by LightGBM gain importance:")
    for feat in top_k:
        print(f"    {feat}: {importances[feat]:.1f}")
    return top_k


def train_val_test_split(df: pd.DataFrame, seed: int = C.RANDOM_SEED,
                          train_frac: float = 0.6, val_frac: float = 0.2):
    """
    Stratified split on binary_label (fallback, see module docstring).
    Returns (train_df, val_df, test_df).
    """
    warnings.warn(
        "[data_pipeline] Using stratified random split, NOT a device/session"
        "-aware split, because the public CICIoT2023 CSVs carry no device ID"
        " column. See this module's docstring."
    )
    train_df, temp_df = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=df["binary_label"]
    )
    rel_val = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, random_state=seed, stratify=temp_df["binary_label"]
    )
    print(f"[data_pipeline] Split sizes: train={len(train_df):,}, "
          f"val={len(val_df):,}, test={len(test_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Eq. (1): direction-aware min-max normalization
# ----------------------------------------------------------------------
# Default direction assumption: for every selected traffic feature, a
# LARGER value is treated as a "cost" criterion (more consistent with
# flooding/scanning behavior => lower r_j => less trust-supporting
# evidence). This is a modeling choice, not a verified fact about your
# specific feature set; override individual entries here after your own
# EDA if a particular feature behaves the other way for your traffic mix.
FEATURE_DIRECTION_OVERRIDE = {
    "Variance": "benefit",
    "Header_Length": "benefit",
    "Tot size": "benefit",
    "Duration": "benefit",
    "flow_duration": "benefit",
    "Tot sum": "benefit",
    "Max": "benefit",
    "Min": "benefit",
    "IAT": "benefit",
    "rst_count": "benefit",
    "urg_count": "benefit",
    "syn_count": "benefit",
    "Rate": "cost",
    "Protocol Type": "cost",
}


def fit_normalization_bounds(train_df: pd.DataFrame, features: list) -> dict:
    """Fit L_j, U_j (Eq. 1) on the training partition ONLY."""
    bounds = {}
    for feat in features:
        col = train_df[feat].fillna(0.0)
        bounds[feat] = {"L": float(col.min()), "U": float(col.max())}
    return bounds


def apply_normalization(df: pd.DataFrame, features: list, bounds: dict) -> pd.DataFrame:
    """Applies Eq. (1)'s clip[(x-L)/(U-L)] (benefit) or clip[(U-x)/(U-L)] (cost)."""
    out = pd.DataFrame(index=df.index)
    for feat in features:
        L, U = bounds[feat]["L"], bounds[feat]["U"]
        x = df[feat].fillna(0.0)
        direction = FEATURE_DIRECTION_OVERRIDE.get(feat, "cost")
        denom = (U - L) if (U - L) != 0 else 1.0
        if direction == "benefit":
            r = (x - L) / denom
        else:
            r = (U - x) / denom
        out[feat] = r.clip(0.0, 1.0)
    return out

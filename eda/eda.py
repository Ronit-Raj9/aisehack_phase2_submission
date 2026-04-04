                                                        
"""
Competition-aware EDA for ANRF AISEHack Phase 2 PM2.5 forecasting.

This script is built around the actual Phase 2 objective:
1. Forecast 16 hours ahead from a 10 hour lookback.
2. Perform well globally and during extreme pollution episodes.
3. Avoid misleading conclusions from random overlapping train/val splits.

Outputs:
- CSV tables with dataset, feature, shift, leakage, and episode summaries
- PNG plots for temporal, spatial, and episode-aware diagnostics
- Markdown report with modeling takeaways grounded in the data

The episode definition follows the helper notebook exactly:
    episode iff STL_residual > 3 * residual_std and PM2.5 > 1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import io
from scipy.stats import (
    boxcox,
    boxcox_normmax,
    iqr as scipy_iqr,
    jarque_bera,
    ks_2samp,
    kurtosis as scipy_kurtosis,
    median_abs_deviation,
    spearmanr,
    skew as scipy_skew,
    wasserstein_distance,
    yeojohnson,
    yeojohnson_normmax,
)
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", category=FutureWarning)

MET_FEATURES = [
    "cpm25",
    "q2",
    "t2",
    "u10",
    "v10",
    "swdown",
    "pblh",
    "psfc",
    "rain",
]

EMISSION_FEATURES = [
    "PM25",
    "NH3",
    "SO2",
    "NOx",
    "NMVOC_e",
    "NMVOC_finn",
    "bio",
]

ALL_FEATURES = MET_FEATURES + EMISSION_FEATURES

LOOKBACK_HOURS = 10
FORECAST_HOURS = 16
WINDOW_HOURS = LOOKBACK_HOURS + FORECAST_HOURS
DEFAULT_LAG_HOURS = 24
DEFAULT_SAMPLE_SIZE = 120_000
DEFAULT_EPISODE_WORKERS = max(1, min(8, (os.cpu_count() or 1)))
DEFAULT_TABLE_IMAGE_ROWS = 36
DEFAULT_TABLE_IMAGE_COLS = 8
DEFAULT_MODE_BINS = 80

_EPISODE_DATA_2D: Optional[np.ndarray] = None
_EPISODE_PERIOD = 24
_EPISODE_SIGMA = 3.0
_EPISODE_MIN_PM25 = 1.0

REFERENCE_LINKS = {
    "stl": "https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/stl_decomposition.html",
    "power_transform": "https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.power_transform.html",
    "robust_scaler": "https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.RobustScaler.html",
    "quantile_transform": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
    "convlstm": "https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html",
    "air_quality_gcn_lstm": "https://link.springer.com/article/10.1007/s11869-025-01713-8",
}


@dataclass
class AnalysisPaths:
    raw_dir: Path
    test_dir: Path
    stats_file: Optional[Path]
    output_dir: Path
    plot_dir: Path
    table_dir: Path
    array_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed, competition-aware EDA for the PM2.5 forecasting dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "raw"),
        help="Path to raw train directory containing monthly folders.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "test_in"),
        help="Path to test_in directory.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=str(REPO_ROOT / "data" / "stats" / "feat_min_max.mat"),
        help="Path to provided min/max .mat file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "Ronit" / "eda" / "artifacts"),
        help="Directory where all EDA artifacts will be written.",
    )
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Specific month folders to analyze, e.g. APRIL_16 JULY_16. Defaults to all discovered months.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Random flattened sample size used for approximate quantiles and shift tests.",
    )
    parser.add_argument(
        "--lag-hours",
        type=int,
        default=DEFAULT_LAG_HOURS,
        help="Max future lead in hours for feature-to-PM2.5 correlation analysis.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride assumed for sample construction analysis.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation fraction used when auditing random window split leakage.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed used when auditing the baseline-like random split.",
    )
    parser.add_argument(
        "--top-hotspots",
        type=int,
        default=15,
        help="Number of top episode hotspot grid cells to save.",
    )
    parser.add_argument(
        "--skip-episode",
        action="store_true",
        help="Skip exact STL-based episode analysis.",
    )
    parser.add_argument(
        "--save-episode-arrays",
        action="store_true",
        help="Save per-month episode masks and residual-std maps as .npy files.",
    )
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=DEFAULT_EPISODE_WORKERS,
        help="Number of worker processes for exact STL episode detection.",
    )
    parser.add_argument(
        "--episode-chunk-size",
        type=int,
        default=128,
        help="Number of grid columns handled per worker task during episode detection.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure dpi.",
    )
    parser.add_argument(
        "--table-image-rows",
        type=int,
        default=DEFAULT_TABLE_IMAGE_ROWS,
        help="Maximum rows per rendered table image page.",
    )
    parser.add_argument(
        "--table-image-cols",
        type=int,
        default=DEFAULT_TABLE_IMAGE_COLS,
        help="Maximum columns per rendered table image page.",
    )
    parser.add_argument(
        "--mode-bins",
        type=int,
        default=DEFAULT_MODE_BINS,
        help="Histogram bins used for approximate continuous-value mode estimation.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_paths(args: argparse.Namespace) -> AnalysisPaths:
    output_dir = ensure_dir(Path(args.output_dir))
    return AnalysisPaths(
        raw_dir=Path(args.raw_dir),
        test_dir=Path(args.test_dir),
        stats_file=Path(args.stats_file) if args.stats_file else None,
        output_dir=output_dir,
        plot_dir=ensure_dir(output_dir / "plots"),
        table_dir=ensure_dir(output_dir / "tables"),
        array_dir=ensure_dir(output_dir / "arrays"),
    )


def discover_months(raw_dir: Path) -> List[str]:
    months = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    priority = {"APRIL_16": 0, "JULY_16": 1, "OCT_16": 2, "DEC_16": 3}
    return sorted(months, key=lambda item: (priority.get(item, 99), item))


def month_label(month: str) -> str:
    mapping = {
        "APRIL_16": "April 2016",
        "JULY_16": "July 2016",
        "OCT_16": "October 2016",
        "DEC_16": "December 2016",
        "TEST_IN": "Test Inputs",
    }
    return mapping.get(month, month)


def feature_category(feature: str) -> str:
    if feature in MET_FEATURES:
        return "meteorology_or_target"
    if feature in EMISSION_FEATURES:
        return "emission"
    return "other"


def load_provided_minmax(stats_file: Optional[Path]) -> Dict[str, float]:
    if stats_file is None or not stats_file.exists():
        return {}

    mat = io.loadmat(stats_file)
    out: Dict[str, float] = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value).squeeze()
        if arr.size == 1:
            out[key] = float(arr)
    return out


def load_timestamps(path: Path) -> pd.DatetimeIndex:
    return pd.to_datetime(np.load(path))


def sample_flat_values(arr: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    flat = arr.reshape(-1)
    total = flat.shape[0]
    if total == 0:
        return np.empty(0, dtype=np.float64)
    if total <= sample_size:
        return np.asarray(flat, dtype=np.float64)
    idx = rng.choice(total, size=sample_size, replace=False)
    return np.asarray(flat[idx], dtype=np.float64)


def sample_masked_values(
    arr: np.ndarray,
    mask: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_mask = mask.reshape(-1)
    idx = np.flatnonzero(flat_mask)
    if idx.size == 0:
        return np.empty(0, dtype=np.float64)
    flat = arr.reshape(-1)
    if idx.size <= sample_size:
        return np.asarray(flat[idx], dtype=np.float64)
    chosen = rng.choice(idx, size=sample_size, replace=False)
    return np.asarray(flat[chosen], dtype=np.float64)


def safe_quantiles(values: np.ndarray, quantiles: Sequence[float]) -> List[float]:
    if values.size == 0:
        return [float("nan")] * len(quantiles)
    return [float(x) for x in np.quantile(values, quantiles)]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size != y.size:
        raise ValueError("safe_corr inputs must have the same length")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def consecutive_run_lengths(binary_values: np.ndarray) -> List[int]:
    run_lengths: List[int] = []
    current = 0
    for value in binary_values.astype(bool):
        if value:
            current += 1
        elif current > 0:
            run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)
    return run_lengths


def compute_window_count(time_len: int, window_hours: int, stride: int) -> int:
    if time_len < window_hours:
        return 0
    return 1 + (time_len - window_hours) // stride


def nearest_distance(sorted_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    if sorted_idx.size == 0:
        return np.full(query_idx.shape, np.inf)

    insert_pos = np.searchsorted(sorted_idx, query_idx)
    left = np.where(insert_pos > 0, sorted_idx[np.clip(insert_pos - 1, 0, sorted_idx.size - 1)], -10**9)
    right = np.where(
        insert_pos < sorted_idx.size,
        sorted_idx[np.clip(insert_pos, 0, sorted_idx.size - 1)],
        10**9,
    )
    return np.minimum(np.abs(query_idx - left), np.abs(query_idx - right))


def audit_random_split_leakage(
    n_windows: int,
    window_hours: int,
    val_frac: float,
    seed: int,
) -> Mapping[str, float]:
    if n_windows <= 1:
        return {
            "n_windows": n_windows,
            "n_train": 0,
            "n_val": 0,
            "share_val_with_any_overlap": float("nan"),
            "share_val_with_gt50_overlap": float("nan"),
            "mean_overlap_hours_with_nearest_train_window": float("nan"),
            "median_overlap_hours_with_nearest_train_window": float("nan"),
        }

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_windows)
    n_val = max(1, int(n_windows * val_frac))
    val_idx = np.sort(shuffled[:n_val])
    train_idx = np.sort(shuffled[n_val:])
    distances = nearest_distance(train_idx, val_idx)
    overlap_hours = np.maximum(window_hours - distances, 0)

    return {
        "n_windows": int(n_windows),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "share_val_with_any_overlap": float(np.mean(overlap_hours > 0)),
        "share_val_with_gt50_overlap": float(np.mean(overlap_hours >= (window_hours / 2))),
        "mean_overlap_hours_with_nearest_train_window": float(np.mean(overlap_hours)),
        "median_overlap_hours_with_nearest_train_window": float(np.median(overlap_hours)),
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_json(payload: Mapping[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def format_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0.0:
            return "0"
        if abs(value) >= 10_000 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def approximate_mode(values: np.ndarray, bins: int = DEFAULT_MODE_BINS) -> Tuple[float, str, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), "none", float("nan")

    rounded = np.round(clean, 6)
    counts = pd.Series(rounded).value_counts(dropna=True)
    if not counts.empty:
        top_value = float(counts.index[0])
        top_frac = float(counts.iloc[0] / rounded.size)
        if abs(top_value) < 1e-12 and top_frac > 0:
            return 0.0, "exact_zero", top_frac
        if top_frac >= 0.01:
            return top_value, "rounded_mode", top_frac

    hist_bins = min(max(20, int(np.sqrt(clean.size) / 2)), max(20, bins))
    hist, edges = np.histogram(clean, bins=hist_bins)
    idx = int(hist.argmax())
    center = float((edges[idx] + edges[idx + 1]) / 2)
    frac = float(hist[idx] / clean.size)
    return center, "hist_bin_center", frac


def estimate_power_lambda(values: np.ndarray, method: str) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    try:
        if method == "box_cox":
            strictly_positive = clean[clean > 0]
            if strictly_positive.size < 10:
                return float("nan")
            return float(boxcox_normmax(strictly_positive, method="mle"))
        if method == "yeo_johnson":
            if clean.size < 10:
                return float("nan")
            return float(yeojohnson_normmax(clean))
    except Exception:
        return float("nan")
    return float("nan")


def recommend_transform(
    feature: str,
    metrics: Mapping[str, float],
) -> Tuple[str, str, float, str]:
    zero_frac = float(metrics["zero_frac"])
    skewness = float(metrics["skewness"])
    abs_skew = abs(skewness)
    outlier_frac = float(metrics["outlier_low_frac"] + metrics["outlier_high_frac"])
    strictly_positive = bool(metrics["strictly_positive"])
    has_negative = bool(metrics["negative_frac"] > 0)

    if feature in EMISSION_FEATURES and zero_frac > 0.90:
        scaler = "max_clip_or_robust"
        return (
            "identity_or_log1p",
            scaler,
            estimate_power_lambda(metrics["sample_values"], "box_cox"),
            "Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.",
        )

    if strictly_positive and abs_skew > 2.5:
        scaler = "robust" if outlier_frac > 0.01 else "standardize"
        transform = "log1p" if zero_frac > 0 else "box_cox"
        lambda_guess = estimate_power_lambda(metrics["sample_values"], "box_cox")
        reason = "Strictly positive and strongly right-skewed."
        if zero_frac > 0:
            reason += " Zeros make log1p safer than Box-Cox."
        return transform, scaler, lambda_guess, reason

    if has_negative and abs_skew > 1.0:
        scaler = "robust" if outlier_frac > 0.01 else "standardize"
        return (
            "yeo_johnson",
            scaler,
            estimate_power_lambda(metrics["sample_values"], "yeo_johnson"),
            "Sign-changing feature with notable skew; Yeo-Johnson is safer than log transforms.",
        )

    if outlier_frac > 0.01 or float(metrics["max_over_p99"]) > 1.5:
        return (
            "identity",
            "robust",
            float("nan"),
            "Outliers dominate more than skew; robust centering/scaling is safer than aggressive warping.",
        )

    if abs_skew < 0.5:
        return "identity", "standardize", float("nan"), "Distribution is close enough to symmetric for standard scaling."

    return "identity", "standardize", float("nan"), "Moderate skew, but not severe enough to force a nonlinear transform."


def describe_distribution_sample(
    sample: np.ndarray,
    feature: str,
    split: str,
    month: str,
    month_label_value: str,
    mode_bins: int,
) -> MutableMapping[str, object]:
    clean = np.asarray(sample, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "split": split,
            "month": month,
            "month_label": month_label_value,
            "feature": feature,
            "category": feature_category(feature),
            "sample_size_used": 0,
        }

    q01, q05, q25, q50, q75, q95, q99 = safe_quantiles(clean, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    mean_val = float(clean.mean())
    std_val = float(clean.std())
    mode_val, mode_kind, mode_freq = approximate_mode(clean, bins=mode_bins)
    mad_val = float(median_abs_deviation(clean, nan_policy="omit", scale="normal"))
    iqr_val = float(scipy_iqr(clean, rng=(25, 75)))
    q1 = float(q25)
    q3 = float(q75)
    lower_fence = q1 - 1.5 * iqr_val
    upper_fence = q3 + 1.5 * iqr_val
    outlier_low_frac = float(np.mean(clean < lower_fence))
    outlier_high_frac = float(np.mean(clean > upper_fence))
    skewness = float(scipy_skew(clean, bias=False)) if std_val > 1e-12 else 0.0
    kurtosis_excess = float(scipy_kurtosis(clean, fisher=True, bias=False)) if std_val > 1e-12 else 0.0
    pearson_skew = safe_div(3 * (mean_val - q50), std_val)
    bowley_skew = safe_div((q3 + q1 - 2 * q50), (q3 - q1))
    zero_frac = float(np.mean(np.isclose(clean, 0.0)))
    negative_frac = float(np.mean(clean < 0))
    positive_frac = float(np.mean(clean > 0))
    unique_ratio = float(np.unique(np.round(clean, 6)).size / clean.size)
    jb_sample = clean[: min(clean.size, 20_000)]
    jb_stat, jb_p = jarque_bera(jb_sample)

    metrics: Dict[str, float] = {
        "zero_frac": zero_frac,
        "skewness": skewness,
        "outlier_low_frac": outlier_low_frac,
        "outlier_high_frac": outlier_high_frac,
        "negative_frac": negative_frac,
        "max_over_p99": safe_div(float(clean.max()), q99),
        "strictly_positive": float(np.all(clean > 0)),
        "sample_values": clean,
    }
    suggested_transform, suggested_scaler, suggested_lambda, transform_reason = recommend_transform(feature, metrics)

    return {
        "split": split,
        "month": month,
        "month_label": month_label_value,
        "feature": feature,
        "category": feature_category(feature),
        "sample_size_used": int(clean.size),
        "mean": mean_val,
        "median": float(q50),
        "mode_approx": mode_val,
        "mode_kind": mode_kind,
        "mode_frequency_frac": mode_freq,
        "std": std_val,
        "mad_normal": mad_val,
        "iqr": iqr_val,
        "min": float(clean.min()),
        "p01": q01,
        "p05": q05,
        "p25": q25,
        "p50": q50,
        "p75": q75,
        "p95": q95,
        "p99": q99,
        "max": float(clean.max()),
        "zero_frac": zero_frac,
        "negative_frac": negative_frac,
        "positive_frac": positive_frac,
        "unique_ratio_rounded_6dp": unique_ratio,
        "coeff_var": safe_div(std_val, abs(mean_val)),
        "skewness": skewness,
        "pearson_skewness": pearson_skew,
        "bowley_skewness": bowley_skew,
        "kurtosis_excess": kurtosis_excess,
        "outlier_low_frac": outlier_low_frac,
        "outlier_high_frac": outlier_high_frac,
        "max_over_p99": safe_div(float(clean.max()), q99),
        "p99_over_median": safe_div(q99, abs(q50)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "strictly_positive": bool(np.all(clean > 0)),
        "suggested_transform": suggested_transform,
        "suggested_scaler": suggested_scaler,
        "suggested_lambda": suggested_lambda,
        "transform_reason": transform_reason,
    }


def compute_feature_distribution_diagnostics(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
    sample_size: int,
    rng: np.random.Generator,
    mode_bins: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
    rows: List[MutableMapping[str, object]] = []
    sample_cache: Dict[Tuple[str, str, str], np.ndarray] = {}

    for month in months:
        for feature in tqdm(features, desc=f"Distribution diagnostics {month}", leave=False):
            arr = np.load(paths.raw_dir / month / f"{feature}.npy", mmap_mode="r")
            sample = sample_flat_values(arr, sample_size, rng).astype(np.float32)
            sample_cache[("train_raw", month, feature)] = sample
            rows.append(
                describe_distribution_sample(
                    sample=sample,
                    feature=feature,
                    split="train_raw",
                    month=month,
                    month_label_value=month_label(month),
                    mode_bins=mode_bins,
                )
            )

    for feature in tqdm(features, desc="Distribution diagnostics test", leave=False):
        arr = np.load(paths.test_dir / f"{feature}.npy", mmap_mode="r")
        sample = sample_flat_values(arr, sample_size, rng).astype(np.float32)
        sample_cache[("test_in", "TEST_IN", feature)] = sample
        rows.append(
            describe_distribution_sample(
                sample=sample,
                feature=feature,
                split="test_in",
                month="TEST_IN",
                month_label_value=month_label("TEST_IN"),
                mode_bins=mode_bins,
            )
        )

    for feature in features:
        train_parts = [sample_cache[("train_raw", month, feature)] for month in months]
        train_all = np.concatenate(train_parts)
        if train_all.size > sample_size * 2:
            idx = rng.choice(train_all.size, size=sample_size * 2, replace=False)
            train_all = train_all[idx]
        rows.append(
            describe_distribution_sample(
                sample=train_all,
                feature=feature,
                split="train_all",
                month="TRAIN_ALL",
                month_label_value="Train Aggregate",
                mode_bins=mode_bins,
            )
        )

    df = pd.DataFrame(rows)
    return df, sample_cache


def build_feature_transform_recommendations(
    distribution_df: pd.DataFrame,
) -> pd.DataFrame:
    train_df = distribution_df[distribution_df["split"] == "train_raw"].copy()
    rows: List[MutableMapping[str, object]] = []

    for feature, grp in train_df.groupby("feature"):
        transform_mode = grp["suggested_transform"].mode()
        scaler_mode = grp["suggested_scaler"].mode()
        reason_mode = grp["transform_reason"].mode()
        rows.append(
            {
                "feature": feature,
                "category": feature_category(feature),
                "mean_train_skewness": float(grp["skewness"].mean()),
                "max_abs_train_skewness": float(grp["skewness"].abs().max()),
                "mean_train_kurtosis_excess": float(grp["kurtosis_excess"].mean()),
                "mean_zero_frac": float(grp["zero_frac"].mean()),
                "mean_outlier_frac": float((grp["outlier_low_frac"] + grp["outlier_high_frac"]).mean()),
                "mean_coeff_var": float(grp["coeff_var"].mean()),
                "majority_transform": transform_mode.iloc[0] if not transform_mode.empty else "",
                "majority_scaler": scaler_mode.iloc[0] if not scaler_mode.empty else "",
                "majority_reason": reason_mode.iloc[0] if not reason_mode.empty else "",
                "n_months_suggesting_nonlinear_transform": int(grp["suggested_transform"].isin(["log1p", "box_cox", "yeo_johnson"]).sum()),
                "n_months_suggesting_robust_scaler": int((grp["suggested_scaler"] == "robust").sum()),
                "n_months_suggesting_max_clip_or_robust": int((grp["suggested_scaler"] == "max_clip_or_robust").sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["n_months_suggesting_nonlinear_transform", "max_abs_train_skewness", "mean_outlier_frac"],
        ascending=[False, False, False],
    )


def compute_feature_temporal_diagnostics(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[MutableMapping[str, object]] = []
    hourly_rows: List[MutableMapping[str, object]] = []

    for month in spatial_mean_series:
        ts = timestamps_by_month[month]
        hours = ts.hour
        for feature in features:
            series = np.asarray(spatial_mean_series[month][feature], dtype=np.float64)
            grouped = pd.DataFrame({"hour": hours, "value": series}).groupby("hour")["value"]
            hour_means = grouped.mean()
            peak_hour = int(hour_means.idxmax())
            trough_hour = int(hour_means.idxmin())
            summary_rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "feature": feature,
                    "category": feature_category(feature),
                    "spatial_mean_mean": float(series.mean()),
                    "spatial_mean_std": float(series.std()),
                    "temporal_cv": safe_div(float(series.std()), abs(float(series.mean()))),
                    "lag1_autocorr": safe_corr(series[:-1], series[1:]) if series.size > 1 else float("nan"),
                    "lag6_autocorr": safe_corr(series[:-6], series[6:]) if series.size > 6 else float("nan"),
                    "lag12_autocorr": safe_corr(series[:-12], series[12:]) if series.size > 12 else float("nan"),
                    "lag24_autocorr": safe_corr(series[:-24], series[24:]) if series.size > 24 else float("nan"),
                    "diurnal_peak_hour": peak_hour,
                    "diurnal_trough_hour": trough_hour,
                    "diurnal_amplitude": float(hour_means.max() - hour_means.min()),
                    "peak_to_trough_ratio": safe_div(float(hour_means.max()), abs(float(hour_means.min()))),
                    "trend_slope_per_day": float(np.polyfit(np.arange(series.size), series, 1)[0] * 24.0) if series.size > 1 else float("nan"),
                }
            )

            for hour, values in grouped:
                hourly_rows.append(
                    {
                        "month": month,
                        "month_label": month_label(month),
                        "feature": feature,
                        "category": feature_category(feature),
                        "hour": int(hour),
                        "mean_spatial_mean": float(values.mean()),
                        "std_spatial_mean": float(values.std(ddof=0)),
                        "count": int(values.size),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(hourly_rows)


def compute_feature_spatial_diagnostics(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []

    for month in tqdm(months, desc="Spatial diagnostics", leave=False):
        for feature in features:
            arr = np.load(paths.raw_dir / month / f"{feature}.npy", mmap_mode="r")
            mean_map = np.asarray(arr.mean(axis=0), dtype=np.float64)
            temporal_std_map = np.asarray(arr.std(axis=0), dtype=np.float64)
            q95_mean = float(np.quantile(mean_map, 0.95))
            q99_mean = float(np.quantile(mean_map, 0.99))
            rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "feature": feature,
                    "category": feature_category(feature),
                    "mean_of_time_mean_map": float(mean_map.mean()),
                    "std_of_time_mean_map": float(mean_map.std()),
                    "spatial_cv_of_time_mean_map": safe_div(float(mean_map.std()), abs(float(mean_map.mean()))),
                    "p95_of_time_mean_map": q95_mean,
                    "p99_of_time_mean_map": q99_mean,
                    "top5pct_mass_share": safe_div(float(mean_map[mean_map >= q95_mean].sum()), float(mean_map.sum())),
                    "top1pct_mass_share": safe_div(float(mean_map[mean_map >= q99_mean].sum()), float(mean_map.sum())),
                    "mean_of_temporal_std_map": float(temporal_std_map.mean()),
                    "p95_of_temporal_std_map": float(np.quantile(temporal_std_map, 0.95)),
                    "std_of_temporal_std_map": float(temporal_std_map.std()),
                }
            )

    return pd.DataFrame(rows)


def compute_monthwise_shift_to_test(
    months: Sequence[str],
    features: Sequence[str],
    sample_cache: Mapping[Tuple[str, str, str], np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[MutableMapping[str, object]] = []

    for feature in features:
        test_sample = np.asarray(sample_cache[("test_in", "TEST_IN", feature)], dtype=np.float64)
        for month in months:
            train_sample = np.asarray(sample_cache[("train_raw", month, feature)], dtype=np.float64)
            ks = ks_2samp(train_sample, test_sample, alternative="two-sided", method="auto")
            rows.append(
                {
                    "feature": feature,
                    "month": month,
                    "month_label": month_label(month),
                    "category": feature_category(feature),
                    "ks_statistic": float(ks.statistic),
                    "wasserstein_distance": float(wasserstein_distance(train_sample, test_sample)),
                    "mean_ratio_test_over_month": safe_div(float(test_sample.mean()), float(train_sample.mean())),
                    "p95_ratio_test_over_month": safe_div(
                        float(np.quantile(test_sample, 0.95)),
                        float(np.quantile(train_sample, 0.95)),
                    ),
                }
            )

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby(["month", "month_label"], as_index=False)
        .agg(
            mean_ks_statistic=("ks_statistic", "mean"),
            median_ks_statistic=("ks_statistic", "median"),
            mean_wasserstein_distance=("wasserstein_distance", "mean"),
            mean_abs_log_mean_ratio=("mean_ratio_test_over_month", lambda x: float(np.mean(np.abs(np.log(np.clip(x, 1e-8, None)))))),
        )
        .sort_values(["mean_ks_statistic", "mean_wasserstein_distance"], ascending=[True, True])
    )
    return detail_df, summary_df


def compute_feature_seasonality_summary(
    feature_summary: pd.DataFrame,
) -> pd.DataFrame:
    train_rows = feature_summary[feature_summary["split"] == "train_raw"].copy()
    grouped = (
        train_rows.groupby("feature", as_index=False)
        .agg(
            max_month_mean=("global_mean", "max"),
            min_month_mean=("global_mean", "min"),
            mean_month_mean=("global_mean", "mean"),
            std_month_mean=("global_mean", "std"),
            max_month_std=("global_std", "max"),
            min_month_std=("global_std", "min"),
        )
    )
    grouped["seasonal_mean_ratio"] = grouped["max_month_mean"] / grouped["min_month_mean"].replace(0, np.nan)
    grouped["seasonal_mean_cv"] = grouped["std_month_mean"] / grouped["mean_month_mean"].replace(0, np.nan)
    grouped["seasonal_abs_mean_ratio"] = (
        train_rows.groupby("feature")["global_mean"].apply(lambda x: np.max(np.abs(x)) / max(np.min(np.abs(x[np.abs(x) > 1e-12])), 1e-12)).values
    )
    grouped["feature"] = grouped["feature"].astype(str)
    grouped["category"] = grouped["feature"].map(feature_category)
    return grouped.sort_values("seasonal_mean_cv", ascending=False)


def compute_feature_forecastability_summary(
    lead_corr_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []
    for feature, grp in lead_corr_df.groupby("feature"):
        grp = grp.dropna(subset=["corr_feature_t_with_cpm25_t_plus_lead"]).copy()
        if grp.empty:
            continue
        best_abs_idx = grp["corr_feature_t_with_cpm25_t_plus_lead"].abs().idxmax()
        best_pos_idx = grp["corr_feature_t_with_cpm25_t_plus_lead"].idxmax()
        best_neg_idx = grp["corr_feature_t_with_cpm25_t_plus_lead"].idxmin()
        best_abs = grp.loc[best_abs_idx]
        best_pos = grp.loc[best_pos_idx]
        best_neg = grp.loc[best_neg_idx]
        rows.append(
            {
                "feature": feature,
                "category": feature_category(feature),
                "best_abs_corr_lead_hours": int(best_abs["lead_hours"]),
                "best_abs_corr_value": float(best_abs["corr_feature_t_with_cpm25_t_plus_lead"]),
                "best_positive_corr_lead_hours": int(best_pos["lead_hours"]),
                "best_positive_corr_value": float(best_pos["corr_feature_t_with_cpm25_t_plus_lead"]),
                "best_negative_corr_lead_hours": int(best_neg["lead_hours"]),
                "best_negative_corr_value": float(best_neg["corr_feature_t_with_cpm25_t_plus_lead"]),
                "mean_abs_corr_0_to_16h": float(
                    grp[grp["lead_hours"] <= 16]["corr_feature_t_with_cpm25_t_plus_lead"].abs().mean()
                ),
                "mean_abs_corr_0_to_24h": float(grp["corr_feature_t_with_cpm25_t_plus_lead"].abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_corr_0_to_16h", ascending=False)


def compute_feature_redundancy_vif(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    features: Sequence[str],
) -> pd.DataFrame:
    frames = []
    for month, series_map in spatial_mean_series.items():
        frames.append(pd.DataFrame({feature: np.asarray(series_map[feature], dtype=np.float64) for feature in features}))
    data = pd.concat(frames, ignore_index=True)
    x = data[features].to_numpy(dtype=np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / np.where(x.std(axis=0, keepdims=True) < 1e-12, 1.0, x.std(axis=0, keepdims=True))
    corr = np.corrcoef(x, rowvar=False)

    rows: List[MutableMapping[str, object]] = []
    for idx, feature in enumerate(features):
        y = x[:, idx]
        mask = np.ones(len(features), dtype=bool)
        mask[idx] = False
        x_other = x[:, mask]
        x_design = np.column_stack([np.ones(x_other.shape[0]), x_other])
        beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        y_hat = x_design @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vif = float(np.inf) if r2 >= 0.999999 else float(1.0 / max(1e-12, (1.0 - r2)))
        mean_abs_corr = float(np.mean(np.abs(np.delete(corr[idx], idx))))
        max_abs_corr = float(np.max(np.abs(np.delete(corr[idx], idx))))
        rows.append(
            {
                "feature": feature,
                "category": feature_category(feature),
                "vif": vif,
                "regression_r2_on_other_features": float(r2),
                "mean_abs_corr_with_other_features": mean_abs_corr,
                "max_abs_corr_with_other_features": max_abs_corr,
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def build_feature_role_recommendations(
    distribution_df: pd.DataFrame,
    temporal_diag_df: pd.DataFrame,
    spatial_diag_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    forecastability_df: pd.DataFrame,
) -> pd.DataFrame:
    dist_agg = (
        distribution_df[distribution_df["split"] == "train_raw"]
        .groupby("feature", as_index=False)
        .agg(
            mean_zero_frac=("zero_frac", "mean"),
            max_abs_skewness=("skewness", lambda x: float(np.max(np.abs(x)))),
            majority_transform=("suggested_transform", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
            majority_scaler=("suggested_scaler", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        )
    )
    temporal_agg = (
        temporal_diag_df.groupby("feature", as_index=False)
        .agg(
            mean_temporal_cv=("temporal_cv", "mean"),
            mean_lag24_autocorr=("lag24_autocorr", "mean"),
            mean_diurnal_amplitude=("diurnal_amplitude", "mean"),
        )
    )
    spatial_agg = (
        spatial_diag_df.groupby("feature", as_index=False)
        .agg(
            mean_top1pct_mass_share=("top1pct_mass_share", "mean"),
            mean_spatial_cv=("spatial_cv_of_time_mean_map", "mean"),
        )
    )
    merged = (
        dist_agg.merge(temporal_agg, on="feature", how="left")
        .merge(spatial_agg, on="feature", how="left")
        .merge(seasonality_df[["feature", "seasonal_mean_cv", "seasonal_abs_mean_ratio"]], on="feature", how="left")
        .merge(redundancy_df[["feature", "vif", "max_abs_corr_with_other_features"]], on="feature", how="left")
        .merge(forecastability_df[["feature", "best_abs_corr_lead_hours", "best_abs_corr_value", "mean_abs_corr_0_to_16h"]], on="feature", how="left")
    )
    merged["category"] = merged["feature"].map(feature_category)

    role_labels = []
    role_reasons = []
    for row in merged.itertuples(index=False):
        if row.feature == "cpm25":
            role = "autoregressive_anchor"
            reason = "Past PM2.5 is the strongest direct predictor of future PM2.5 and should remain a privileged input."
        elif row.category == "emission" and row.mean_zero_frac > 0.95:
            role = "static_sparse_prior"
            reason = "Extremely sparse emission-like field; treat as a static or slowly varying spatial prior."
        elif row.mean_temporal_cv > 0.7 and row.mean_lag24_autocorr < 0.9:
            role = "highly_dynamic_driver"
            reason = "High temporal variability with weaker day-scale persistence; useful as a dynamic driver."
        elif row.mean_lag24_autocorr > 0.95 and row.mean_diurnal_amplitude > 0:
            role = "stable_diurnal_driver"
            reason = "Highly persistent with strong diurnal structure; likely helpful through climatology-aware conditioning."
        elif row.max_abs_corr_with_other_features > 0.9 and row.vif > 10:
            role = "redundant_proxy"
            reason = "Very redundant with other features; channel pruning or grouped encoding may be reasonable."
        else:
            role = "mixed_signal"
            reason = "Carries some signal but does not cleanly fit a purely static or highly dynamic bucket."
        role_labels.append(role)
        role_reasons.append(reason)

    merged["recommended_role"] = role_labels
    merged["role_reason"] = role_reasons
    return merged.sort_values(
        ["recommended_role", "mean_abs_corr_0_to_16h", "mean_temporal_cv"],
        ascending=[True, False, False],
    )


def compute_feature_target_regime_response(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    months: Sequence[str],
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    regime_edges = [0.0, 0.50, 0.75, 0.90, 0.975, 1.0]
    regime_labels = [
        "clean_p00_50",
        "moderate_p50_75",
        "elevated_p75_90",
        "high_p90_97_5",
        "extreme_p97_5_100",
    ]
    profile_rows: List[MutableMapping[str, object]] = []
    summary_rows: List[MutableMapping[str, object]] = []

    for month in months:
        target = np.asarray(spatial_mean_series[month]["cpm25"], dtype=np.float64)
        if target.size == 0:
            continue
        thresholds = np.quantile(target, regime_edges)

        for feature in features:
            series = np.asarray(spatial_mean_series[month][feature], dtype=np.float64)
            overall_mean = float(series.mean())
            overall_std = float(series.std())
            spearman_val = float(spearmanr(series, target, nan_policy="omit").statistic)
            regime_means: List[float] = []

            for idx, label in enumerate(regime_labels):
                left = thresholds[idx]
                right = thresholds[idx + 1]
                if idx == len(regime_labels) - 1:
                    mask = (target >= left) & (target <= right)
                else:
                    mask = (target >= left) & (target < right)
                subset = series[mask]
                regime_mean = float(subset.mean()) if subset.size else float("nan")
                regime_median = float(np.median(subset)) if subset.size else float("nan")
                regime_std = float(subset.std()) if subset.size else float("nan")
                regime_means.append(regime_mean)
                profile_rows.append(
                    {
                        "month": month,
                        "month_label": month_label(month),
                        "feature": feature,
                        "category": feature_category(feature),
                        "regime_rank": idx,
                        "regime_label": label,
                        "target_regime_left_q": regime_edges[idx],
                        "target_regime_right_q": regime_edges[idx + 1],
                        "count": int(subset.size),
                        "regime_feature_mean": regime_mean,
                        "regime_feature_median": regime_median,
                        "regime_feature_std": regime_std,
                        "regime_feature_mean_minus_global": regime_mean - overall_mean if np.isfinite(regime_mean) else float("nan"),
                        "regime_feature_mean_zscore": safe_div(regime_mean - overall_mean, overall_std),
                    }
                )

            valid_idx = [idx for idx, value in enumerate(regime_means) if np.isfinite(value)]
            if valid_idx:
                regime_rank = np.asarray(valid_idx, dtype=np.float64)
                regime_value = np.asarray([regime_means[idx] for idx in valid_idx], dtype=np.float64)
                regime_monotonic_corr = safe_corr(regime_rank, regime_value)
                clean_mean = regime_means[0]
                extreme_mean = regime_means[-1]
                summary_rows.append(
                    {
                        "month": month,
                        "month_label": month_label(month),
                        "feature": feature,
                        "category": feature_category(feature),
                        "same_time_spearman_with_cpm25": spearman_val,
                        "extreme_minus_clean_std_units": safe_div(extreme_mean - clean_mean, overall_std),
                        "extreme_over_clean_mean_ratio": safe_div(extreme_mean, clean_mean),
                        "regime_mean_range_std_units": safe_div(
                            float(np.nanmax(regime_value) - np.nanmin(regime_value)),
                            overall_std,
                        ),
                        "regime_monotonic_corr": regime_monotonic_corr,
                    }
                )

    profile_df = pd.DataFrame(profile_rows)
    summary_detail_df = pd.DataFrame(summary_rows)
    if summary_detail_df.empty:
        return profile_df, summary_detail_df

    feature_summary_df = (
        summary_detail_df.groupby(["feature", "category"], as_index=False)
        .agg(
            mean_abs_same_time_spearman=("same_time_spearman_with_cpm25", lambda x: float(np.mean(np.abs(x)))),
            mean_same_time_spearman=("same_time_spearman_with_cpm25", "mean"),
            mean_abs_extreme_minus_clean_std_units=("extreme_minus_clean_std_units", lambda x: float(np.mean(np.abs(x)))),
            max_abs_extreme_minus_clean_std_units=("extreme_minus_clean_std_units", lambda x: float(np.max(np.abs(x)))),
            mean_extreme_over_clean_mean_ratio=("extreme_over_clean_mean_ratio", "mean"),
            mean_regime_mean_range_std_units=("regime_mean_range_std_units", "mean"),
            mean_regime_monotonic_corr=("regime_monotonic_corr", "mean"),
            positive_spearman_month_frac=("same_time_spearman_with_cpm25", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        .sort_values(
            ["mean_abs_extreme_minus_clean_std_units", "mean_abs_same_time_spearman"],
            ascending=[False, False],
        )
    )
    return profile_df, feature_summary_df


def compute_feature_pair_stability(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    months: Sequence[str],
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[MutableMapping[str, object]] = []
    summary_rows: List[MutableMapping[str, object]] = []

    for month in months:
        frame = pd.DataFrame({feature: np.asarray(spatial_mean_series[month][feature], dtype=np.float64) for feature in features})
        corr = frame.corr()
        for idx, feature_a in enumerate(features):
            for feature_b in features[idx + 1 :]:
                corr_value = float(corr.loc[feature_a, feature_b])
                detail_rows.append(
                    {
                        "month": month,
                        "month_label": month_label(month),
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pair_label": f"{feature_a} ~ {feature_b}",
                        "corr_value": corr_value,
                        "abs_corr_value": abs(corr_value),
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, detail_df

    for (feature_a, feature_b), grp in detail_df.groupby(["feature_a", "feature_b"]):
        corr_values = grp["corr_value"].to_numpy(dtype=np.float64)
        strong_signs = np.sign(corr_values[np.abs(corr_values) >= 0.2])
        sign_flip = int(np.unique(strong_signs).size > 1)
        summary_rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "pair_label": f"{feature_a} ~ {feature_b}",
                "category_a": feature_category(feature_a),
                "category_b": feature_category(feature_b),
                "mean_corr": float(np.mean(corr_values)),
                "mean_abs_corr": float(np.mean(np.abs(corr_values))),
                "corr_std_across_months": float(np.std(corr_values)),
                "corr_range_across_months": float(np.max(corr_values) - np.min(corr_values)),
                "min_corr": float(np.min(corr_values)),
                "max_corr": float(np.max(corr_values)),
                "strong_sign_flip_across_months": sign_flip,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["corr_std_across_months", "mean_abs_corr"],
        ascending=[False, False],
    )
    return detail_df, summary_df


def build_feature_normalization_blueprint(
    distribution_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    role_df: pd.DataFrame,
) -> pd.DataFrame:
    train_all = distribution_df[distribution_df["split"] == "train_all"].copy()
    merged = (
        train_all.merge(
            transform_df[
                [
                    "feature",
                    "majority_transform",
                    "majority_scaler",
                    "majority_reason",
                    "mean_zero_frac",
                ]
            ],
            on="feature",
            how="left",
        )
        .merge(
            role_df[
                [
                    "feature",
                    "recommended_role",
                    "mean_abs_corr_0_to_16h",
                ]
            ],
            on="feature",
            how="left",
        )
        .sort_values("feature")
    )

    recipe_rows: List[MutableMapping[str, object]] = []
    for row in merged.itertuples(index=False):
        lower_clip = row.p01 if row.outlier_low_frac > 0.01 else row.min
        upper_clip = row.p99 if row.outlier_high_frac > 0.01 else row.max
        center_stat = "median" if row.majority_scaler in {"robust", "max_clip_or_robust"} else "mean"
        scale_stat = "iqr" if row.majority_scaler == "robust" else ("std" if row.majority_scaler == "standardize" else "p99_after_clip")
        clip_strategy = (
            "winsorize_to_p01_p99"
            if (row.outlier_low_frac > 0.01 or row.outlier_high_frac > 0.01)
            else "no_extra_clip"
        )
        recipe = f"{row.majority_transform} -> {clip_strategy} -> center={center_stat} -> scale={scale_stat}"
        recipe_rows.append(
            {
                "feature": row.feature,
                "category": row.category,
                "recommended_role": row.recommended_role,
                "forecastability_score_0_to_16h": row.mean_abs_corr_0_to_16h,
                "mean": row.mean,
                "median": row.median,
                "mode_approx": row.mode_approx,
                "std": row.std,
                "mad_normal": row.mad_normal,
                "iqr": row.iqr,
                "p01": row.p01,
                "p99": row.p99,
                "min": row.min,
                "max": row.max,
                "skewness": row.skewness,
                "kurtosis_excess": row.kurtosis_excess,
                "zero_frac": row.zero_frac,
                "negative_frac": row.negative_frac,
                "outlier_low_frac": row.outlier_low_frac,
                "outlier_high_frac": row.outlier_high_frac,
                "recommended_transform": row.majority_transform,
                "recommended_scaler": row.majority_scaler,
                "recommended_clip_lower": lower_clip,
                "recommended_clip_upper": upper_clip,
                "recommended_center_stat": center_stat,
                "recommended_scale_stat": scale_stat,
                "recommended_recipe": recipe,
                "normalization_reason": row.majority_reason,
            }
        )
    return pd.DataFrame(recipe_rows)


def compute_episode_feature_signatures(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    episode_maps: Optional[Mapping[str, Mapping[str, np.ndarray]]],
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not episode_maps:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows: List[MutableMapping[str, object]] = []
    for month in months:
        month_payload = episode_maps.get(month)
        if not month_payload or "episode_mask" not in month_payload:
            continue

        grid_mask = np.asarray(month_payload["episode_mask"], dtype=bool)
        if grid_mask.size == 0:
            continue
        non_grid_mask = ~grid_mask
        hour_mask = np.asarray(month_payload["domain_any_episode"], dtype=bool)
        episode_points = np.asarray(month_payload["episode_points_per_timestep"], dtype=np.float64)

        for feature in features:
            arr = np.load(paths.raw_dir / month / f"{feature}.npy", mmap_mode="r")
            ep_count = int(grid_mask.sum())
            non_ep_count = int(non_grid_mask.sum())
            total_sum = float(arr.sum())
            ep_sum = float(np.asarray(arr[grid_mask], dtype=np.float64).sum()) if ep_count else 0.0
            ep_mean = float(ep_sum / ep_count) if ep_count else float("nan")
            non_ep_mean = float((total_sum - ep_sum) / non_ep_count) if non_ep_count else float("nan")
            arr_std = float(np.std(arr))
            ep_sample = sample_masked_values(arr, grid_mask, sample_size, rng)
            non_ep_sample = sample_masked_values(arr, non_grid_mask, sample_size, rng)

            feature_series = np.asarray(spatial_mean_series[month][feature], dtype=np.float64)
            ep_hour_series = feature_series[hour_mask]
            non_ep_hour_series = feature_series[~hour_mask]
            feature_series_std = float(feature_series.std())
            ep_hour_mean = float(ep_hour_series.mean()) if ep_hour_series.size else float("nan")
            non_ep_hour_mean = float(non_ep_hour_series.mean()) if non_ep_hour_series.size else float("nan")

            detail_rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "feature": feature,
                    "category": feature_category(feature),
                    "episode_grid_count": ep_count,
                    "non_episode_grid_count": non_ep_count,
                    "episode_grid_mean": ep_mean,
                    "non_episode_grid_mean": non_ep_mean,
                    "episode_grid_median": float(np.median(ep_sample)) if ep_sample.size else float("nan"),
                    "non_episode_grid_median": float(np.median(non_ep_sample)) if non_ep_sample.size else float("nan"),
                    "episode_grid_p95": float(np.quantile(ep_sample, 0.95)) if ep_sample.size else float("nan"),
                    "non_episode_grid_p95": float(np.quantile(non_ep_sample, 0.95)) if non_ep_sample.size else float("nan"),
                    "grid_episode_over_non_episode_mean_ratio": safe_div(ep_mean, non_ep_mean),
                    "grid_episode_minus_non_episode_std_units": safe_div(ep_mean - non_ep_mean, arr_std),
                    "episode_hour_mean_spatial_mean": ep_hour_mean,
                    "non_episode_hour_mean_spatial_mean": non_ep_hour_mean,
                    "episode_hour_over_non_episode_hour_ratio": safe_div(ep_hour_mean, non_ep_hour_mean),
                    "episode_hour_minus_non_episode_hour_std_units": safe_div(
                        ep_hour_mean - non_ep_hour_mean,
                        feature_series_std,
                    ),
                    "episode_presence_corr_with_episode_points": safe_corr(feature_series, episode_points),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, detail_df

    summary_df = (
        detail_df.groupby(["feature", "category"], as_index=False)
        .agg(
            mean_abs_grid_episode_shift_std_units=("grid_episode_minus_non_episode_std_units", lambda x: float(np.mean(np.abs(x)))),
            max_abs_grid_episode_shift_std_units=("grid_episode_minus_non_episode_std_units", lambda x: float(np.max(np.abs(x)))),
            mean_abs_episode_hour_shift_std_units=("episode_hour_minus_non_episode_hour_std_units", lambda x: float(np.mean(np.abs(x)))),
            mean_episode_presence_corr=("episode_presence_corr_with_episode_points", "mean"),
            mean_grid_episode_over_non_episode_ratio=("grid_episode_over_non_episode_mean_ratio", "mean"),
        )
        .sort_values(
            ["mean_abs_grid_episode_shift_std_units", "mean_abs_episode_hour_shift_std_units"],
            ascending=[False, False],
        )
    )
    return detail_df, summary_df


def apply_preview_transform(values: np.ndarray, transform_name: str, lambda_hint: float) -> np.ndarray:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return clean
    try:
        if transform_name in {"log1p", "identity_or_log1p"}:
            clipped = np.clip(clean, a_min=0.0, a_max=None)
            return np.log1p(clipped)
        if transform_name == "box_cox":
            strictly_positive = clean[clean > 0]
            if strictly_positive.size < 10:
                return clean
            lam = lambda_hint if np.isfinite(lambda_hint) else boxcox_normmax(strictly_positive, method="mle")
            return boxcox(strictly_positive, lmbda=lam)
        if transform_name == "yeo_johnson":
            return yeojohnson(clean, lmbda=(lambda_hint if np.isfinite(lambda_hint) else None))
    except Exception:
        return clean
    return clean


def plot_transform_previews(
    paths: AnalysisPaths,
    distribution_df: pd.DataFrame,
    sample_cache: Mapping[Tuple[str, str, str], np.ndarray],
    features: Sequence[str],
    months: Sequence[str],
    dpi: int,
) -> None:
    preview_dir = ensure_dir(paths.plot_dir / "transform_previews")
    train_all = distribution_df[distribution_df["split"] == "train_all"].copy()
    selected = train_all[
        train_all["suggested_transform"].isin(["box_cox", "log1p", "yeo_johnson", "identity_or_log1p"])
    ].copy()
    if selected.empty:
        selected = train_all.reindex(train_all["skewness"].abs().sort_values(ascending=False).index).head(6)
    else:
        selected = selected.reindex(selected["skewness"].abs().sort_values(ascending=False).index).head(8)

    for row in selected.itertuples(index=False):
        feature = row.feature
        parts = [np.asarray(sample_cache[("train_raw", month, feature)], dtype=np.float64) for month in months]
        raw = np.concatenate(parts)
        transformed = apply_preview_transform(raw, row.suggested_transform, row.suggested_lambda)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        sns.histplot(raw, bins=80, ax=axes[0, 0], color="#3366aa")
        axes[0, 0].set_title(f"{feature} raw histogram")
        axes[0, 0].set_xlabel("raw")

        sns.histplot(np.log1p(np.clip(raw, a_min=0.0, a_max=None)), bins=80, ax=axes[0, 1], color="#aa6633")
        axes[0, 1].set_title(f"{feature} log1p preview")
        axes[0, 1].set_xlabel("log1p")

        sns.histplot(transformed, bins=80, ax=axes[1, 0], color="#228866")
        axes[1, 0].set_title(f"{feature} recommended transform preview")
        axes[1, 0].set_xlabel(row.suggested_transform)

        summary_lines = [
            f"mean={row.mean:.4g}",
            f"median={row.median:.4g}",
            f"mode~={row.mode_approx:.4g}",
            f"skew={row.skewness:.3f}",
            f"kurtosis={row.kurtosis_excess:.3f}",
            f"zero_frac={row.zero_frac:.3f}",
            f"recommended={row.suggested_transform} + {row.suggested_scaler}",
        ]
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.03,
            0.97,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox={"facecolor": "#f7f7f7", "edgecolor": "#d0d7e2", "boxstyle": "round,pad=0.5"},
        )
        axes[1, 1].set_title(f"{feature} transform summary")
        fig.tight_layout()
        save_figure(fig, preview_dir / f"{feature}_transform_preview.png", dpi=dpi)


def plot_feature_role_map(
    paths: AnalysisPaths,
    role_df: pd.DataFrame,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=role_df,
        x="mean_temporal_cv",
        y="mean_zero_frac",
        size="mean_top1pct_mass_share",
        hue="recommended_role",
        ax=ax,
        sizes=(70, 500),
    )
    for row in role_df.itertuples(index=False):
        ax.text(row.mean_temporal_cv, row.mean_zero_frac, str(row.feature), fontsize=8, alpha=0.8)
    ax.set_title("Feature Role Map")
    ax.set_xlabel("Mean Temporal CV")
    ax.set_ylabel("Mean Zero Fraction")
    save_figure(fig, paths.plot_dir / "feature_role_map.png", dpi=dpi)


def plot_forecastability_and_redundancy(
    paths: AnalysisPaths,
    forecastability_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    rank_fore = forecastability_df.sort_values("mean_abs_corr_0_to_16h", ascending=False)
    sns.barplot(
        data=rank_fore,
        x="mean_abs_corr_0_to_16h",
        y="feature",
        hue="feature",
        legend=False,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Feature Forecastability vs Future PM2.5 (0-16h mean |corr|)")
    axes[0].set_xlabel("Mean |lead correlation|")
    axes[0].set_ylabel("")

    rank_vif = redundancy_df.sort_values("vif", ascending=False)
    sns.barplot(
        data=rank_vif,
        x="vif",
        y="feature",
        hue="feature",
        legend=False,
        ax=axes[1],
        palette="magma",
    )
    axes[1].set_title("Feature Redundancy (VIF on spatial-mean series)")
    axes[1].set_xlabel("VIF")
    axes[1].set_ylabel("")

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "forecastability_and_redundancy.png", dpi=dpi)


def plot_target_regime_response(
    paths: AnalysisPaths,
    regime_profile_df: pd.DataFrame,
    regime_summary_df: pd.DataFrame,
    dpi: int,
) -> None:
    if regime_profile_df.empty or regime_summary_df.empty:
        return

    extreme_pivot = (
        regime_profile_df[regime_profile_df["regime_label"] == "extreme_p97_5_100"]
        .pivot(index="feature", columns="month_label", values="regime_feature_mean_zscore")
        .reindex(list(ALL_FEATURES))
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    sns.heatmap(extreme_pivot, cmap="coolwarm", center=0.0, ax=axes[0])
    axes[0].set_title("Feature Mean Shift in Extreme PM2.5 Regime")
    axes[0].set_ylabel("")

    ranked = regime_summary_df.sort_values("mean_abs_extreme_minus_clean_std_units", ascending=False)
    sns.barplot(
        data=ranked,
        x="mean_abs_extreme_minus_clean_std_units",
        y="feature",
        hue="feature",
        legend=False,
        palette="coolwarm",
        ax=axes[1],
    )
    axes[1].set_title("Features Most Sensitive to PM2.5 Regime")
    axes[1].set_xlabel("Mean |extreme-clean| in feature std units")
    axes[1].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "target_regime_feature_response.png", dpi=dpi)


def plot_pairwise_correlation_stability(
    paths: AnalysisPaths,
    pair_detail_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    dpi: int,
) -> None:
    if pair_detail_df.empty or pair_summary_df.empty:
        return

    top_pairs = pair_summary_df.head(18)["pair_label"].tolist()
    top_detail = pair_detail_df[pair_detail_df["pair_label"].isin(top_pairs)].copy()
    pivot = top_detail.pivot(index="pair_label", columns="month_label", values="corr_value").reindex(top_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.barplot(
        data=pair_summary_df.head(18),
        x="corr_std_across_months",
        y="pair_label",
        hue="strong_sign_flip_across_months",
        dodge=False,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Most Season-Unstable Feature Pairs")
    axes[0].set_xlabel("Correlation std across months")
    axes[0].set_ylabel("")

    sns.heatmap(pivot, cmap="coolwarm", center=0.0, ax=axes[1])
    axes[1].set_title("Monthly Correlation for Most Unstable Pairs")
    axes[1].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "pairwise_correlation_stability.png", dpi=dpi)


def plot_episode_feature_signatures(
    paths: AnalysisPaths,
    episode_feature_detail_df: pd.DataFrame,
    episode_feature_summary_df: pd.DataFrame,
    dpi: int,
) -> None:
    if episode_feature_detail_df.empty or episode_feature_summary_df.empty:
        return

    shift_pivot = (
        episode_feature_detail_df.pivot(
            index="feature",
            columns="month_label",
            values="grid_episode_minus_non_episode_std_units",
        )
        .reindex(list(ALL_FEATURES))
    )
    corr_pivot = (
        episode_feature_detail_df.pivot(
            index="feature",
            columns="month_label",
            values="episode_presence_corr_with_episode_points",
        )
        .reindex(list(ALL_FEATURES))
    )
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    sns.heatmap(shift_pivot, cmap="coolwarm", center=0.0, ax=axes[0])
    axes[0].set_title("Grid-Level Feature Shift During Episodes")
    axes[0].set_ylabel("")

    sns.heatmap(corr_pivot, cmap="coolwarm", center=0.0, ax=axes[1])
    axes[1].set_title("Feature Spatial Mean vs Episode Intensity")
    axes[1].set_ylabel("")

    ranked = episode_feature_summary_df.sort_values("mean_abs_grid_episode_shift_std_units", ascending=False)
    sns.barplot(
        data=ranked,
        x="mean_abs_grid_episode_shift_std_units",
        y="feature",
        hue="feature",
        legend=False,
        palette="rocket",
        ax=axes[2],
    )
    axes[2].set_title("Features Most Different During Episodes")
    axes[2].set_xlabel("Mean |episode shift| in feature std units")
    axes[2].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "episode_feature_signatures.png", dpi=dpi)


def maybe_load_existing_episode_tables(paths: AnalysisPaths) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    summary_path = paths.table_dir / "episode_summary.csv"
    hourly_path = paths.table_dir / "episode_hourly_profile.csv"
    hotspots_path = paths.table_dir / "episode_hotspots.csv"
    if summary_path.exists() and hourly_path.exists() and hotspots_path.exists():
        return pd.read_csv(summary_path), pd.read_csv(hourly_path), pd.read_csv(hotspots_path)
    return None, None, None


def maybe_load_existing_episode_maps(
    paths: AnalysisPaths,
    months: Sequence[str],
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    map_bundle: Dict[str, Dict[str, np.ndarray]] = {}
    for month in months:
        mask_path = paths.array_dir / f"{month.lower()}_episode_mask.npy"
        residual_path = paths.array_dir / f"{month.lower()}_residual_std.npy"
        pm25_path = paths.raw_dir / month / "cpm25.npy"
        if not (mask_path.exists() and residual_path.exists() and pm25_path.exists()):
            return None

        mask = np.load(mask_path, mmap_mode="r")
        residual_std = np.load(residual_path, mmap_mode="r")
        pm25 = np.load(pm25_path, mmap_mode="r")
        episode_points_per_timestep = np.asarray(mask.sum(axis=(1, 2)), dtype=np.int64)
        domain_any_episode = (episode_points_per_timestep > 0).astype(np.uint8)
        map_bundle[month] = {
            "episode_mask": mask,
            "episode_frequency_map": np.asarray(mask.mean(axis=0), dtype=np.float64),
            "residual_std_map": np.asarray(residual_std, dtype=np.float64),
            "episode_points_per_timestep": episode_points_per_timestep,
            "domain_any_episode": domain_any_episode,
            "spatial_mean_pm25": np.asarray(pm25.mean(axis=(1, 2)), dtype=np.float64),
        }
    return map_bundle


def render_table_pages(
    df: pd.DataFrame,
    title: str,
    output_prefix: Path,
    max_rows: int,
    max_cols: int,
    dpi: int,
) -> None:
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return

    row_chunks = list(range(0, n_rows, max(1, max_rows)))
    col_chunks = list(range(0, n_cols, max(1, max_cols)))

    for row_page, row_start in enumerate(row_chunks, start=1):
        row_end = min(row_start + max_rows, n_rows)
        for col_page, col_start in enumerate(col_chunks, start=1):
            col_end = min(col_start + max_cols, n_cols)
            subdf = df.iloc[row_start:row_end, col_start:col_end].copy()
            cell_text = [[format_scalar(value) for value in row] for row in subdf.to_numpy()]
            col_labels = [str(col) for col in subdf.columns]

            fig_w = max(8, min(24, 1.65 * len(col_labels) + 2))
            fig_h = max(4.5, min(18, 0.36 * len(subdf) + 2.4))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")

            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(max(5.0, 10.5 - 0.08 * max(len(subdf), len(col_labels))))
            table.scale(1.0, 1.15)

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold", color="white")
                    cell.set_facecolor("#1f4e79")
                else:
                    cell.set_facecolor("#f5f7fb" if row % 2 == 0 else "white")
                    cell.set_edgecolor("#d0d7e2")

            ax.set_title(
                f"{title} | rows {row_start + 1}-{row_end} | cols {col_start + 1}-{col_end}",
                fontsize=11,
                pad=10,
            )
            output_path = output_prefix.parent / f"{output_prefix.name}__r{row_page:02d}_c{col_page:02d}.png"
            save_figure(fig, output_path, dpi=dpi)


def render_all_table_images(
    paths: AnalysisPaths,
    max_rows: int,
    max_cols: int,
    dpi: int,
) -> None:
    table_image_dir = ensure_dir(paths.plot_dir / "table_images")
    for csv_path in sorted(paths.table_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        render_table_pages(
            df=df,
            title=csv_path.stem,
            output_prefix=table_image_dir / csv_path.stem,
            max_rows=max_rows,
            max_cols=max_cols,
            dpi=dpi,
        )


def plot_distribution_heatmaps(
    paths: AnalysisPaths,
    distribution_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    month_shift_detail_df: pd.DataFrame,
    dpi: int,
) -> None:
    train_df = distribution_df[distribution_df["split"] == "train_raw"].copy()

    skew_pivot = train_df.pivot(index="feature", columns="month_label", values="skewness").loc[list(ALL_FEATURES)]
    zero_pivot = train_df.pivot(index="feature", columns="month_label", values="zero_frac").loc[list(ALL_FEATURES)]
    kurt_pivot = train_df.pivot(index="feature", columns="month_label", values="kurtosis_excess").loc[list(ALL_FEATURES)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    sns.heatmap(skew_pivot, cmap="coolwarm", center=0.0, ax=axes[0])
    axes[0].set_title("Train Skewness by Feature and Month")
    sns.heatmap(zero_pivot, cmap="Blues", ax=axes[1])
    axes[1].set_title("Zero Fraction by Feature and Month")
    sns.heatmap(kurt_pivot, cmap="magma", ax=axes[2])
    axes[2].set_title("Excess Kurtosis by Feature and Month")
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "distribution_heatmaps.png", dpi=dpi)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ranked = transform_df.sort_values(["n_months_suggesting_nonlinear_transform", "max_abs_train_skewness"], ascending=[False, False])
    sns.barplot(
        data=ranked,
        x="max_abs_train_skewness",
        y="feature",
        hue="majority_transform",
        dodge=False,
        ax=axes[0],
    )
    axes[0].set_title("Features Ranked by Maximum Absolute Train Skewness")
    axes[0].set_xlabel("Max |skewness|")
    axes[0].set_ylabel("")

    month_shift_pivot = month_shift_detail_df.pivot(index="feature", columns="month_label", values="ks_statistic").loc[list(ALL_FEATURES)]
    sns.heatmap(month_shift_pivot, cmap="rocket_r", ax=axes[1])
    axes[1].set_title("Month-to-Test Similarity (lower KS is better)")
    axes[1].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "skew_transform_and_month_similarity.png", dpi=dpi)


def plot_spatial_concentration_heatmap(
    paths: AnalysisPaths,
    spatial_diag_df: pd.DataFrame,
    dpi: int,
) -> None:
    pivot = spatial_diag_df.pivot(index="feature", columns="month_label", values="top1pct_mass_share").loc[list(ALL_FEATURES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Spatial Concentration: Top 1% Grid Cells Mass Share")
    ax.set_ylabel("")
    save_figure(fig, paths.plot_dir / "spatial_concentration_heatmap.png", dpi=dpi)


def plot_feature_diagnostic_book(
    paths: AnalysisPaths,
    features: Sequence[str],
    months: Sequence[str],
    sample_cache: Mapping[Tuple[str, str, str], np.ndarray],
    distribution_df: pd.DataFrame,
    temporal_hourly_df: pd.DataFrame,
    temporal_diag_df: pd.DataFrame,
    lead_corr_df: pd.DataFrame,
    month_shift_detail_df: pd.DataFrame,
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    dpi: int,
) -> None:
    feature_dir = ensure_dir(paths.plot_dir / "feature_diagnostics")

    for feature in features:
        fig, axes = plt.subplots(3, 2, figsize=(15, 16))
        axes = axes.ravel()

        dist_records = []
        for month in months:
            values = np.asarray(sample_cache[("train_raw", month, feature)], dtype=np.float64)
            if values.size > 4000:
                values = values[:4000]
            for value in values:
                dist_records.append({"dataset": month_label(month), "value": value})
        test_values = np.asarray(sample_cache[("test_in", "TEST_IN", feature)], dtype=np.float64)
        if test_values.size > 4000:
            test_values = test_values[:4000]
        for value in test_values:
            dist_records.append({"dataset": "Test Inputs", "value": value})
        dist_frame = pd.DataFrame(dist_records)
        sns.boxenplot(data=dist_frame, x="dataset", y="value", hue="dataset", legend=False, ax=axes[0])
        axes[0].set_title(f"{feature} distribution by month/test")
        axes[0].tick_params(axis="x", rotation=25)

        train_diag = distribution_df[(distribution_df["feature"] == feature) & (distribution_df["split"] == "train_raw")]
        sns.barplot(data=train_diag, x="month_label", y="skewness", hue="month_label", legend=False, ax=axes[1], palette="coolwarm")
        axes[1].set_title(f"{feature} skewness by train month")
        axes[1].tick_params(axis="x", rotation=25)

        for month in months:
            axes[2].plot(timestamps_by_month[month], spatial_mean_series[month][feature], linewidth=1.2, label=month_label(month))
        axes[2].set_title(f"{feature} spatial-mean time series")
        axes[2].legend(fontsize=8)

        hourly_subset = temporal_hourly_df[temporal_hourly_df["feature"] == feature]
        sns.lineplot(data=hourly_subset, x="hour", y="mean_spatial_mean", hue="month_label", marker="o", ax=axes[3])
        axes[3].set_title(f"{feature} hour-of-day profile")

        lead_subset = lead_corr_df[lead_corr_df["feature"] == feature]
        axes[4].plot(lead_subset["lead_hours"], lead_subset["corr_feature_t_with_cpm25_t_plus_lead"], marker="o")
        axes[4].axhline(0.0, color="black", linewidth=1)
        axes[4].set_title(f"{feature} lead correlation to future PM2.5")
        axes[4].set_xlabel("Lead Hours")
        axes[4].set_ylabel("Correlation")

        agg_row = distribution_df[(distribution_df["feature"] == feature) & (distribution_df["split"] == "train_all")].iloc[0]
        shift_rows = month_shift_detail_df[month_shift_detail_df["feature"] == feature].sort_values("ks_statistic")
        temporal_rows = temporal_diag_df[temporal_diag_df["feature"] == feature]
        text_lines = [
            f"Train mean: {agg_row['mean']:.4g}",
            f"Train median: {agg_row['median']:.4g}",
            f"Approx mode: {agg_row['mode_approx']:.4g} ({agg_row['mode_kind']})",
            f"Train skewness: {agg_row['skewness']:.3f}",
            f"Train kurtosis: {agg_row['kurtosis_excess']:.3f}",
            f"Zero frac: {agg_row['zero_frac']:.3f}",
            f"Suggested transform: {agg_row['suggested_transform']}",
            f"Suggested scaler: {agg_row['suggested_scaler']}",
            f"Closest month to test: {shift_rows.iloc[0]['month_label']} (KS={shift_rows.iloc[0]['ks_statistic']:.3f})",
            f"Mean lag24 autocorr: {temporal_rows['lag24_autocorr'].mean():.3f}",
            f"Mean diurnal amplitude: {temporal_rows['diurnal_amplitude'].mean():.4g}",
        ]
        axes[5].axis("off")
        axes[5].text(
            0.02,
            0.98,
            "\n".join(text_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox={"facecolor": "#f7f7f7", "edgecolor": "#d0d7e2", "boxstyle": "round,pad=0.5"},
        )
        axes[5].set_title(f"{feature} summary card")

        fig.tight_layout()
        save_figure(fig, feature_dir / f"{feature}_diagnostic.png", dpi=dpi)


def plot_episode_hotspot_geo(
    paths: AnalysisPaths,
    episode_hotspots_df: Optional[pd.DataFrame],
    dpi: int,
) -> None:
    if episode_hotspots_df is None or episode_hotspots_df.empty:
        return
    if episode_hotspots_df["latitude"].isna().all() or episode_hotspots_df["longitude"].isna().all():
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax, (month, subset) in zip(axes, episode_hotspots_df.groupby("month")):
        scatter = ax.scatter(
            subset["longitude"],
            subset["latitude"],
            c=subset["episode_frequency"],
            s=40 + 4 * (subset["rank"].max() - subset["rank"] + 1),
            cmap="Reds",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_title(f"{month_label(month)} hotspot geography")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "episode_hotspot_geography.png", dpi=dpi)


def render_preprocessing_guide(
    blueprint_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
) -> str:
    lines = [
        "# Preprocessing Guide",
        "",
        "This guide is generated from the EDA and is intended to help with normalization and transform choices.",
        "",
        "## Global Rules",
        "",
        "- Fit all transforms and scalers on the training split only to avoid leakage.",
        "- Prefer time-safe validation splits before fitting normalization statistics.",
        "- For strongly right-skewed positive variables, consider `log1p` or Box-Cox style warping.",
        "- For sign-changing skewed variables, prefer a Yeo-Johnson style transform.",
        "- For heavy outliers, robust centering/scaling is often safer than standardization alone.",
        "",
        "## Feature-Level Recommendations",
        "",
    ]

    merged = blueprint_df.merge(seasonality_df[["feature", "seasonal_mean_cv"]], on="feature", how="left")
    for row in merged.itertuples(index=False):
        seasonal_cv = row.seasonal_mean_cv if pd.notna(row.seasonal_mean_cv) else float("nan")
        lines.append(
            f"- {row.feature}: recipe={row.recommended_recipe}, "
            f"clip=[{format_scalar(row.recommended_clip_lower)}, {format_scalar(row.recommended_clip_upper)}], "
            f"skew={row.skewness:.3f}, zero_frac={row.zero_frac:.3f}, "
            f"seasonal_mean_cv={seasonal_cv:.3f}, "
            f"reason={row.normalization_reason}"
        )
    lines.append("")
    lines.append("## External References")
    lines.append("")
    lines.append(f"- STL decomposition background: {REFERENCE_LINKS['stl']}")
    lines.append(f"- Power transforms (Yeo-Johnson / Box-Cox): {REFERENCE_LINKS['power_transform']}")
    lines.append(f"- Robust scaling: {REFERENCE_LINKS['robust_scaler']}")
    lines.append(f"- Quantile transform for extreme tail reshaping: {REFERENCE_LINKS['quantile_transform']}")
    lines.append("")
    return "\n".join(lines)


def render_model_search_guide(
    month_shift_summary_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    forecastability_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    role_df: pd.DataFrame,
    regime_summary_df: pd.DataFrame,
    pair_stability_df: pd.DataFrame,
    episode_feature_summary_df: pd.DataFrame,
) -> str:
    closest_months = ", ".join(month_shift_summary_df["month_label"].tolist())
    highly_seasonal = seasonality_df.sort_values("seasonal_mean_cv", ascending=False).head(5)["feature"].tolist()
    nonlinear_features = transform_df[
        (transform_df["n_months_suggesting_nonlinear_transform"] > 0)
        | (transform_df["majority_transform"] == "identity_or_log1p")
    ]["feature"].tolist()
    strongest_drivers = forecastability_df.sort_values("mean_abs_corr_0_to_16h", ascending=False).head(6)["feature"].tolist()
    redundant = redundancy_df.sort_values("vif", ascending=False).head(5)["feature"].tolist()
    sparse_priors = role_df[role_df["recommended_role"] == "static_sparse_prior"]["feature"].tolist()
    regime_sensitive = regime_summary_df.sort_values("mean_abs_extreme_minus_clean_std_units", ascending=False).head(6)["feature"].tolist()
    unstable_pairs = pair_stability_df.head(5)["pair_label"].tolist() if not pair_stability_df.empty else []
    episode_sensitive = (
        episode_feature_summary_df.sort_values("mean_abs_grid_episode_shift_std_units", ascending=False).head(6)["feature"].tolist()
        if not episode_feature_summary_df.empty
        else []
    )

    lines = [
        "# Model Search Guide",
        "",
        "This guide converts the EDA into a practical experiment search order.",
        "",
        "## What The Data Suggests",
        "",
        "- The target field lives on a fixed 140x124 grid, so grid-native models remain the first class to try.",
        "- Severe leakage occurs under random overlapping window splits, so validation must be time-safe.",
        f"- Train months closest to test by distribution shift: {closest_months}.",
        f"- Most season-sensitive features: {', '.join(highly_seasonal)}.",
        f"- Features most likely to benefit from nonlinear transforms: {', '.join(nonlinear_features) if nonlinear_features else 'none flagged strongly'}.",
        f"- Strongest feature-time-series drivers of future PM2.5: {', '.join(strongest_drivers)}.",
        f"- Most redundant channels at the aggregate level: {', '.join(redundant)}.",
        f"- Sparse spatial priors worth treating specially: {', '.join(sparse_priors) if sparse_priors else 'none'}.",
        f"- Features that change the most between clean and extreme PM2.5 regimes: {', '.join(regime_sensitive) if regime_sensitive else 'none computed'}.",
        f"- Pairwise relationships that are least season-stable: {', '.join(unstable_pairs) if unstable_pairs else 'none computed'}.",
        f"- Features most episode-sensitive: {', '.join(episode_sensitive) if episode_sensitive else 'episode masks not available yet'}.",
        "",
        "## Search Order",
        "",
        "1. Strong grid baseline: improve the current FNO-style setup with better validation, better normalization, and an episode-aware objective.",
        "2. Local spike specialist: try ConvLSTM or a U-Net-temporal residual head because the horizon is short and episodes are spatially sharp.",
        "3. Hybrid global-local model: combine a spectral/global branch with a local convolutional recurrent branch.",
        "4. Transport-aware variant: if time permits, test graph-style or sparse-edge message passing over the grid or coarse superpixels. This is an inference from the data plus graph forecasting literature, not a direct requirement from the competition.",
        "5. Final ensemble: blend models with complementary behavior, especially one smooth/global model and one spike-sensitive/local model.",
        "",
        "## Why These Families",
        "",
        f"- ConvLSTM was proposed for spatiotemporal sequence forecasting: {REFERENCE_LINKS['convlstm']}",
        f"- Recent air-quality work continues to explore graph+recurrent hybrids for spatial dependency structure: {REFERENCE_LINKS['air_quality_gcn_lstm']}",
        "",
        "## Practical Ablations",
        "",
        "- Season embedding or month indicator.",
        "- Better train/validation split.",
        "- Raw vs transformed inputs for skewed features.",
        "- Emission channels as static priors vs repeated temporal channels.",
        "- Loss weighting or curriculum focused on episodic regions.",
        "",
        "## Recommendation",
        "",
        "Start with the strongest reliable baseline you can validate properly, then add only one modeling idea at a time. The EDA says preprocessing, validation design, and episode focus are likely to give better returns than jumping immediately to an exotic architecture.",
        "",
    ]
    return "\n".join(lines)


def render_deep_dive_report(
    base_report_text: str,
    distribution_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    blueprint_df: pd.DataFrame,
    temporal_diag_df: pd.DataFrame,
    spatial_diag_df: pd.DataFrame,
    month_shift_detail_df: pd.DataFrame,
    month_shift_summary_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    forecastability_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    role_df: pd.DataFrame,
    regime_summary_df: pd.DataFrame,
    pair_stability_df: pd.DataFrame,
    episode_feature_summary_df: pd.DataFrame,
    episode_summary_df: Optional[pd.DataFrame],
) -> str:
    train_all = distribution_df[distribution_df["split"] == "train_all"].copy()
    most_skewed = train_all.reindex(train_all["skewness"].abs().sort_values(ascending=False).index).head(8)
    most_outlier_heavy = train_all.sort_values(["outlier_high_frac", "max_over_p99"], ascending=False).head(8)
    most_persistent = (
        temporal_diag_df.groupby("feature", as_index=False)[["lag1_autocorr", "lag24_autocorr", "diurnal_amplitude"]]
        .mean()
        .sort_values("lag24_autocorr", ascending=False)
        .head(8)
    )
    spatially_concentrated = (
        spatial_diag_df.groupby("feature", as_index=False)[["top1pct_mass_share", "spatial_cv_of_time_mean_map"]]
        .mean()
        .sort_values("top1pct_mass_share", ascending=False)
        .head(8)
    )
    strongest_forecast = forecastability_df.sort_values("mean_abs_corr_0_to_16h", ascending=False).head(8)
    most_redundant = redundancy_df.sort_values("vif", ascending=False).head(8)
    normalization_preview = blueprint_df.head(8)
    regime_preview = regime_summary_df.head(8)
    pair_preview = pair_stability_df.head(8)
    episode_feature_preview = episode_feature_summary_df.head(8) if not episode_feature_summary_df.empty else pd.DataFrame()
    role_preview = role_df[[
        "feature",
        "recommended_role",
        "majority_transform",
        "majority_scaler",
        "mean_abs_corr_0_to_16h",
    ]].head(10)

    lines = [base_report_text, "", "## Deep Dive Additions", ""]
    lines.append("### Distribution Shape and Normalization")
    for row in most_skewed.itertuples(index=False):
        lines.append(
            f"- {row.feature}: skewness={row.skewness:.3f}, kurtosis={row.kurtosis_excess:.3f}, "
            f"mean={row.mean:.4g}, median={row.median:.4g}, approx_mode={row.mode_approx:.4g}, "
            f"transform={row.suggested_transform}, scaler={row.suggested_scaler}"
        )
    lines.append("")
    lines.append("### Outlier-Heaviness")
    for row in most_outlier_heavy.itertuples(index=False):
        lines.append(
            f"- {row.feature}: upper-outlier-frac={row.outlier_high_frac:.4f}, max/p99={row.max_over_p99:.3f}, "
            f"zero-frac={row.zero_frac:.3f}"
        )
    lines.append("")
    lines.append("### Temporal Behavior")
    for row in most_persistent.itertuples(index=False):
        lines.append(
            f"- {row.feature}: lag1={row.lag1_autocorr:.3f}, lag24={row.lag24_autocorr:.3f}, diurnal_amplitude={row.diurnal_amplitude:.4g}"
        )
    lines.append("")
    lines.append("### Spatial Concentration")
    for row in spatially_concentrated.itertuples(index=False):
        lines.append(
            f"- {row.feature}: mean top1pct mass share={row.top1pct_mass_share:.3f}, mean spatial CV={row.spatial_cv_of_time_mean_map:.3f}"
        )
    lines.append("")
    lines.append("### Which Train Month Looks Closest to Test?")
    for row in month_shift_summary_df.itertuples(index=False):
        lines.append(
            f"- {row.month_label}: mean KS={row.mean_ks_statistic:.4f}, median KS={row.median_ks_statistic:.4f}, "
            f"mean Wasserstein={row.mean_wasserstein_distance:.4g}"
        )
    lines.append("")
    lines.append("### Forecastability Ranking")
    for row in strongest_forecast.itertuples(index=False):
        lines.append(
            f"- {row.feature}: mean |corr| to future PM2.5 over 0-16h = {row.mean_abs_corr_0_to_16h:.3f}, best lead = {row.best_abs_corr_lead_hours}h"
        )
    lines.append("")
    lines.append("### Redundancy / Multicollinearity")
    for row in most_redundant.itertuples(index=False):
        lines.append(
            f"- {row.feature}: VIF={row.vif:.2f}, max |corr with another feature|={row.max_abs_corr_with_other_features:.3f}"
        )
    lines.append("")
    lines.append("### Regime Sensitivity")
    for row in regime_preview.itertuples(index=False):
        lines.append(
            f"- {row.feature}: mean |extreme-clean|={row.mean_abs_extreme_minus_clean_std_units:.3f} std, "
            f"mean |same-time spearman|={row.mean_abs_same_time_spearman:.3f}, "
            f"regime monotonicity={row.mean_regime_monotonic_corr:.3f}"
        )
    lines.append("")
    lines.append("### Pairwise Relationship Stability")
    for row in pair_preview.itertuples(index=False):
        lines.append(
            f"- {row.pair_label}: corr std across months={row.corr_std_across_months:.3f}, "
            f"range={row.corr_range_across_months:.3f}, sign flip={int(row.strong_sign_flip_across_months)}"
        )
    lines.append("")
    lines.append("### Seasonality Severity")
    for row in seasonality_df.head(8).itertuples(index=False):
        lines.append(
            f"- {row.feature}: seasonal mean ratio={row.seasonal_mean_ratio:.3f}, seasonal mean CV={row.seasonal_mean_cv:.3f}"
        )
    lines.append("")
    lines.append("### Normalization Blueprint")
    for row in normalization_preview.itertuples(index=False):
        lines.append(
            f"- {row.feature}: recipe={row.recommended_recipe}, clip=[{format_scalar(row.recommended_clip_lower)}, {format_scalar(row.recommended_clip_upper)}], "
            f"role={row.recommended_role}"
        )
    lines.append("")
    lines.append("### Role Recommendations")
    for row in role_preview.itertuples(index=False):
        lines.append(
            f"- {row.feature}: role={row.recommended_role}, transform={row.majority_transform}, scaler={row.majority_scaler}, future-PM2.5 score={row.mean_abs_corr_0_to_16h:.3f}"
        )
    lines.append("")
    if not episode_feature_preview.empty:
        lines.append("### Feature Behavior During Episodes")
        for row in episode_feature_preview.itertuples(index=False):
            lines.append(
                f"- {row.feature}: mean |grid-level episode shift|={row.mean_abs_grid_episode_shift_std_units:.3f} std, "
                f"mean episode-presence corr={row.mean_episode_presence_corr:.3f}"
            )
        lines.append("")
    if episode_summary_df is not None and not episode_summary_df.empty:
        lines.append("### Episode Context")
        for row in episode_summary_df.sort_values("episode_pm25_p95", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.month_label}: episode ratio={row.episode_ratio:.4f}, episode PM2.5 p95={row.episode_pm25_p95:.2f}, "
                f"max episodic grid points in one hour={row.max_episode_points_timestep}"
            )
        lines.append("")
    lines.append("### Artifact Expansion")
    lines.append("- Every CSV table is also rendered as paginated PNG images under `plots/table_images`.")
    lines.append("- Each feature now gets its own diagnostic figure under `plots/feature_diagnostics`.")
    lines.append("- Additional guides are saved as `preprocessing_guide.md` and `model_search_guide.md`.")
    lines.append("- Added regime-response, pair-stability, normalization-blueprint, and episode-signature diagnostics.")
    lines.append("")
    return "\n".join(lines)


def build_dataset_overview(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
) -> Mapping[str, object]:
    lat_long_path = paths.raw_dir / "lat_long.npy"
    lat_long_shape = None
    if lat_long_path.exists():
        lat_long_shape = list(np.load(lat_long_path, mmap_mode="r").shape)

    month_payload = []
    for month in months:
        ts = timestamps_by_month[month]
        month_payload.append(
            {
                "month": month,
                "label": month_label(month),
                "timesteps": int(len(ts)),
                "start": str(ts.min()),
                "end": str(ts.max()),
                "unique_step_minutes": sorted(
                    set(((ts[1:] - ts[:-1]).asi8 // (60 * 10**9)).tolist())
                ),
                "window_count_stride_1": compute_window_count(len(ts), WINDOW_HOURS, 1),
                "lookback_hours": LOOKBACK_HOURS,
                "forecast_hours": FORECAST_HOURS,
            }
        )

    test_payload = {}
    if paths.test_dir.exists():
        test_example = np.load(paths.test_dir / f"{features[0]}.npy", mmap_mode="r")
        test_payload = {
            "shape": list(test_example.shape),
            "n_samples": int(test_example.shape[0]),
            "lookback_hours": int(test_example.shape[1]),
            "spatial_shape": list(test_example.shape[-2:]),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(paths.raw_dir),
        "test_dir": str(paths.test_dir),
        "stats_file": str(paths.stats_file) if paths.stats_file else None,
        "months": month_payload,
        "features": list(features),
        "n_features": len(features),
        "grid_shape": [140, 124],
        "lat_long_shape": lat_long_shape,
        "test_inputs": test_payload,
        "problem_framing": {
            "lookback_hours": LOOKBACK_HOURS,
            "forecast_hours": FORECAST_HOURS,
            "window_hours": WINDOW_HOURS,
            "evaluation_focus": [
                "Global SMAPE",
                "Episode Correlation",
                "Episode SMAPE",
            ],
            "episode_definition": "STL residual > 3 * residual_std and PM2.5 > 1",
        },
    }


def summarize_train_features(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
    provided_minmax: Mapping[str, float],
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], Dict[str, pd.DatetimeIndex]]:
    rows: List[MutableMapping[str, object]] = []
    spatial_mean_series: Dict[str, Dict[str, np.ndarray]] = {}
    timestamps_by_month: Dict[str, pd.DatetimeIndex] = {}

    for month in months:
        timestamps_by_month[month] = load_timestamps(paths.raw_dir / month / "time.npy")
        spatial_mean_series[month] = {}

        for feature in tqdm(features, desc=f"Summarizing {month}", leave=False):
            arr = np.load(paths.raw_dir / month / f"{feature}.npy", mmap_mode="r")
            sample = sample_flat_values(arr, sample_size, rng)
            q01, q05, q50, q95, q99 = safe_quantiles(sample, [0.01, 0.05, 0.50, 0.95, 0.99])

            spatial_mean = np.asarray(arr.mean(axis=(1, 2)), dtype=np.float64)
            pixel_std = np.asarray(arr.std(axis=0), dtype=np.float64)
            spatial_mean_series[month][feature] = spatial_mean

            provided_min = provided_minmax.get(f"{feature}_min")
            provided_max = provided_minmax.get(f"{feature}_max")

            rows.append(
                {
                    "split": "train_raw",
                    "month": month,
                    "month_label": month_label(month),
                    "feature": feature,
                    "category": feature_category(feature),
                    "dtype": str(arr.dtype),
                    "shape": "x".join(map(str, arr.shape)),
                    "time_len": int(arr.shape[0]),
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "global_mean": float(arr.mean()),
                    "global_std": float(arr.std()),
                    "global_min": float(arr.min()),
                    "sample_p01": q01,
                    "sample_p05": q05,
                    "sample_p50": q50,
                    "sample_p95": q95,
                    "sample_p99": q99,
                    "global_max": float(arr.max()),
                    "sample_zero_frac": float(np.mean(sample == 0)) if sample.size else float("nan"),
                    "sample_positive_frac": float(np.mean(sample > 0)) if sample.size else float("nan"),
                    "sample_non_finite_frac": float(np.mean(~np.isfinite(sample))) if sample.size else float("nan"),
                    "spatial_mean_std_over_time": float(spatial_mean.std()),
                    "pixel_std_mean": float(pixel_std.mean()),
                    "pixel_std_p95": float(np.quantile(pixel_std, 0.95)),
                    "relative_temporal_variability": float(pixel_std.mean() / (abs(arr.mean()) + 1e-12)),
                    "provided_min": provided_min,
                    "provided_max": provided_max,
                    "sample_below_provided_min_frac": (
                        float(np.mean(sample < provided_min)) if (sample.size and provided_min is not None) else float("nan")
                    ),
                    "sample_above_provided_max_frac": (
                        float(np.mean(sample > provided_max)) if (sample.size and provided_max is not None) else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows), spatial_mean_series, timestamps_by_month


def summarize_test_inputs(
    paths: AnalysisPaths,
    features: Sequence[str],
    provided_minmax: Mapping[str, float],
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []

    for feature in tqdm(features, desc="Summarizing test inputs", leave=False):
        arr = np.load(paths.test_dir / f"{feature}.npy", mmap_mode="r")
        sample = sample_flat_values(arr, sample_size, rng)
        q01, q05, q50, q95, q99 = safe_quantiles(sample, [0.01, 0.05, 0.50, 0.95, 0.99])
        provided_min = provided_minmax.get(f"{feature}_min")
        provided_max = provided_minmax.get(f"{feature}_max")

        rows.append(
            {
                "split": "test_in",
                "month": "TEST_IN",
                "month_label": month_label("TEST_IN"),
                "feature": feature,
                "category": feature_category(feature),
                "dtype": str(arr.dtype),
                "shape": "x".join(map(str, arr.shape)),
                "n_samples": int(arr.shape[0]),
                "time_len": int(arr.shape[1]),
                "height": int(arr.shape[2]),
                "width": int(arr.shape[3]),
                "global_mean": float(arr.mean()),
                "global_std": float(arr.std()),
                "global_min": float(arr.min()),
                "sample_p01": q01,
                "sample_p05": q05,
                "sample_p50": q50,
                "sample_p95": q95,
                "sample_p99": q99,
                "global_max": float(arr.max()),
                "sample_zero_frac": float(np.mean(sample == 0)) if sample.size else float("nan"),
                "sample_positive_frac": float(np.mean(sample > 0)) if sample.size else float("nan"),
                "sample_non_finite_frac": float(np.mean(~np.isfinite(sample))) if sample.size else float("nan"),
                "provided_min": provided_min,
                "provided_max": provided_max,
                "sample_below_provided_min_frac": (
                    float(np.mean(sample < provided_min)) if (sample.size and provided_min is not None) else float("nan")
                ),
                "sample_above_provided_max_frac": (
                    float(np.mean(sample > provided_max)) if (sample.size and provided_max is not None) else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def build_normalization_audit(feature_summary: pd.DataFrame) -> pd.DataFrame:
    train_rows = feature_summary[feature_summary["split"] == "train_raw"].copy()
    grouped = (
        train_rows.groupby("feature", as_index=False)
        .agg(
            empirical_min=("global_min", "min"),
            empirical_max=("global_max", "max"),
            train_mean=("global_mean", "mean"),
            train_std=("global_std", "mean"),
            provided_min=("provided_min", "first"),
            provided_max=("provided_max", "first"),
            mean_above_provided_max_frac=("sample_above_provided_max_frac", "mean"),
            mean_below_provided_min_frac=("sample_below_provided_min_frac", "mean"),
        )
    )
    grouped["max_ratio_empirical_over_provided"] = grouped["empirical_max"] / grouped["provided_max"].replace(0, np.nan)
    grouped["min_gap_empirical_minus_provided"] = grouped["empirical_min"] - grouped["provided_min"]
    grouped["max_gap_empirical_minus_provided"] = grouped["empirical_max"] - grouped["provided_max"]
    grouped["normalization_flag"] = np.where(
        (grouped["empirical_max"] > grouped["provided_max"] * 1.05)
        | (grouped["mean_above_provided_max_frac"] > 0.001),
        "recompute_stats",
        "looks_ok",
    )
    return grouped.sort_values(["normalization_flag", "max_ratio_empirical_over_provided"], ascending=[True, False])


def compute_cpm25_temporal_analysis(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    lag_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_rows: List[MutableMapping[str, object]] = []
    hourly_rows: List[MutableMapping[str, object]] = []

    for month, features in spatial_mean_series.items():
        cpm25_series = np.asarray(features["cpm25"], dtype=np.float64)
        ts = timestamps_by_month[month]
        daily_cycle = pd.DataFrame({"timestamp": ts, "cpm25_spatial_mean": cpm25_series})
        daily_cycle["hour"] = daily_cycle["timestamp"].dt.hour

        grouped = daily_cycle.groupby("hour")["cpm25_spatial_mean"]
        for hour, stats in grouped:
            hourly_rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "hour": int(hour),
                    "mean_cpm25_spatial_mean": float(stats.mean()),
                    "std_cpm25_spatial_mean": float(stats.std(ddof=0)),
                    "count": int(stats.size),
                }
            )

        month_rows.append(
            {
                "month": month,
                "month_label": month_label(month),
                "mean_cpm25_spatial_mean": float(cpm25_series.mean()),
                "std_cpm25_spatial_mean": float(cpm25_series.std()),
                "min_cpm25_spatial_mean": float(cpm25_series.min()),
                "max_cpm25_spatial_mean": float(cpm25_series.max()),
                "hourly_cycle_amplitude": float(grouped.mean().max() - grouped.mean().min()),
            }
        )

    autocorr_rows = []
    for lag in range(1, lag_hours + 1):
        left_parts = []
        right_parts = []
        for month, features in spatial_mean_series.items():
            cpm25_series = np.asarray(features["cpm25"], dtype=np.float64)
            if cpm25_series.size <= lag:
                continue
            left_parts.append(cpm25_series[:-lag])
            right_parts.append(cpm25_series[lag:])
        if left_parts:
            autocorr_rows.append(
                {
                    "lag_hours": lag,
                    "autocorrelation": safe_corr(np.concatenate(left_parts), np.concatenate(right_parts)),
                }
            )

    return pd.DataFrame(month_rows), pd.DataFrame(hourly_rows), pd.DataFrame(autocorr_rows)


def compute_spatial_maps(
    paths: AnalysisPaths,
    months: Sequence[str],
    rng: np.random.Generator,
    sample_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    rows: List[MutableMapping[str, object]] = []
    map_bundle: Dict[str, Dict[str, np.ndarray]] = {}

    for month in tqdm(months, desc="Building PM2.5 spatial maps", leave=False):
        arr = np.load(paths.raw_dir / month / "cpm25.npy", mmap_mode="r")
        sample = sample_flat_values(arr, sample_size, rng)
        global_p95 = float(np.quantile(sample, 0.95)) if sample.size else float("nan")
        mean_map = np.asarray(arr.mean(axis=0), dtype=np.float64)
        std_map = np.asarray(arr.std(axis=0), dtype=np.float64)
        p95_map = np.asarray(np.quantile(arr, 0.95, axis=0), dtype=np.float64)
        hotspot_freq_map = np.asarray((arr > global_p95).mean(axis=0), dtype=np.float64)

        map_bundle[month] = {
            "mean_map": mean_map,
            "std_map": std_map,
            "p95_map": p95_map,
            "hotspot_freq_map": hotspot_freq_map,
        }

        rows.append(
            {
                "month": month,
                "month_label": month_label(month),
                "global_p95_threshold": global_p95,
                "mean_of_mean_map": float(mean_map.mean()),
                "max_of_mean_map": float(mean_map.max()),
                "mean_of_std_map": float(std_map.mean()),
                "max_of_std_map": float(std_map.max()),
                "max_hotspot_frequency": float(hotspot_freq_map.max()),
            }
        )

    return pd.DataFrame(rows), map_bundle


def compute_feature_relationships(
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    features: Sequence[str],
    lag_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_frames = []
    for month, series_map in spatial_mean_series.items():
        frame = pd.DataFrame({feature: np.asarray(series_map[feature], dtype=np.float64) for feature in features})
        frame["month"] = month
        frame["timestamp"] = timestamps_by_month[month]
        monthly_frames.append(frame)

    concat_df = pd.concat(monthly_frames, ignore_index=True)
    corr_matrix = concat_df[features].corr()
    corr_matrix_df = corr_matrix.reset_index().rename(columns={"index": "feature"})

    corr_with_target_rows = []
    for feature in features:
        raw_x = concat_df[feature].to_numpy()
        raw_y = concat_df["cpm25"].to_numpy()
        demeaned_x_parts = []
        demeaned_y_parts = []
        for month in spatial_mean_series:
            x = np.asarray(spatial_mean_series[month][feature], dtype=np.float64)
            y = np.asarray(spatial_mean_series[month]["cpm25"], dtype=np.float64)
            demeaned_x_parts.append(x - x.mean())
            demeaned_y_parts.append(y - y.mean())
        corr_with_target_rows.append(
            {
                "feature": feature,
                "corr_with_cpm25_spatial_mean": safe_corr(raw_x, raw_y),
                "within_month_corr_with_cpm25_spatial_mean": safe_corr(
                    np.concatenate(demeaned_x_parts),
                    np.concatenate(demeaned_y_parts),
                ),
            }
        )

    lead_rows = []
    for feature in features:
        for lead in range(0, lag_hours + 1):
            lhs = []
            rhs = []
            for month in spatial_mean_series:
                x = np.asarray(spatial_mean_series[month][feature], dtype=np.float64)
                y = np.asarray(spatial_mean_series[month]["cpm25"], dtype=np.float64)
                if x.size <= lead:
                    continue
                if lead == 0:
                    lhs.append(x)
                    rhs.append(y)
                else:
                    lhs.append(x[:-lead])
                    rhs.append(y[lead:])
            lead_rows.append(
                {
                    "feature": feature,
                    "lead_hours": lead,
                    "corr_feature_t_with_cpm25_t_plus_lead": (
                        safe_corr(np.concatenate(lhs), np.concatenate(rhs)) if lhs else float("nan")
                    ),
                }
            )

    return corr_matrix_df, pd.DataFrame(corr_with_target_rows), pd.DataFrame(lead_rows)


def compute_train_test_shift(
    paths: AnalysisPaths,
    months: Sequence[str],
    features: Sequence[str],
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[MutableMapping[str, object]] = []

    per_month_sample = max(sample_size // max(len(months), 1), 1)

    for feature in tqdm(features, desc="Computing train-test shift", leave=False):
        train_samples = []
        for month in months:
            arr = np.load(paths.raw_dir / month / f"{feature}.npy", mmap_mode="r")
            train_samples.append(sample_flat_values(arr, per_month_sample, rng))
        train_sample = np.concatenate(train_samples)
        test_sample = sample_flat_values(np.load(paths.test_dir / f"{feature}.npy", mmap_mode="r"), sample_size, rng)

        ks = ks_2samp(train_sample, test_sample, alternative="two-sided", method="auto")
        rows.append(
            {
                "feature": feature,
                "category": feature_category(feature),
                "train_mean": float(train_sample.mean()),
                "test_mean": float(test_sample.mean()),
                "mean_ratio_test_over_train": float(test_sample.mean() / (train_sample.mean() + 1e-12)),
                "train_p50": float(np.quantile(train_sample, 0.50)),
                "test_p50": float(np.quantile(test_sample, 0.50)),
                "train_p95": float(np.quantile(train_sample, 0.95)),
                "test_p95": float(np.quantile(test_sample, 0.95)),
                "p95_ratio_test_over_train": float(
                    np.quantile(test_sample, 0.95) / (np.quantile(train_sample, 0.95) + 1e-12)
                ),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_distance": float(wasserstein_distance(train_sample, test_sample)),
            }
        )

    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def build_window_leakage_tables(
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    stride: int,
    val_frac: float,
    split_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    month_rows = []
    overlap_rows = []

    for month, ts in timestamps_by_month.items():
        n_windows = compute_window_count(len(ts), WINDOW_HOURS, stride)
        leakage = audit_random_split_leakage(
            n_windows=n_windows,
            window_hours=WINDOW_HOURS,
            val_frac=val_frac,
            seed=split_seed,
        )
        month_rows.append({"month": month, "month_label": month_label(month), **leakage})

    for offset in range(0, WINDOW_HOURS + 2):
        overlap_hours = max(WINDOW_HOURS - offset, 0)
        overlap_rows.append(
            {
                "window_start_offset_hours": offset,
                "shared_timesteps": overlap_hours,
                "shared_fraction_of_window": overlap_hours / WINDOW_HOURS,
            }
        )

    return pd.DataFrame(month_rows), pd.DataFrame(overlap_rows)


def detect_episodes_exact(
    pm25_data: np.ndarray,
    period: int = 24,
    residual_sigma: float = 3.0,
    min_pm25: float = 1.0,
    workers: int = 1,
    chunk_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact helper-notebook logic:
        stl = STL(ts, period=24, robust=True)
        episode iff remainder > 3 * std(remainder) and ts > 1
    """
    t_len, height, width = pm25_data.shape
    data_2d = pm25_data.reshape(t_len, -1)

    residual_std = np.zeros(data_2d.shape[1], dtype=np.float64)
    mask_2d = np.zeros_like(data_2d, dtype=np.uint8)

    if workers <= 1 or os.name != "posix":
        for idx in tqdm(range(data_2d.shape[1]), desc="STL episode scan", leave=False):
            ts = np.asarray(data_2d[:, idx], dtype=np.float64)
            stl = STL(ts, period=period, robust=True)
            result = stl.fit()
            remainder = result.resid
            std_val = float(np.std(remainder) + 1e-8)
            residual_std[idx] = std_val
            mask_2d[:, idx] = ((remainder > residual_sigma * std_val) & (ts > min_pm25)).astype(np.uint8)
    else:
        global _EPISODE_DATA_2D, _EPISODE_PERIOD, _EPISODE_SIGMA, _EPISODE_MIN_PM25
        _EPISODE_DATA_2D = data_2d
        _EPISODE_PERIOD = period
        _EPISODE_SIGMA = residual_sigma
        _EPISODE_MIN_PM25 = min_pm25

        column_ranges = [
            (start, min(start + max(1, chunk_size), data_2d.shape[1]))
            for start in range(0, data_2d.shape[1], max(1, chunk_size))
        ]
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            iterator = executor.map(_episode_worker_chunk, column_ranges, chunksize=1)
            for start, end, residual_chunk, mask_chunk in tqdm(
                iterator,
                total=len(column_ranges),
                desc=f"Parallel STL scan ({workers} workers)",
                leave=False,
            ):
                residual_std[start:end] = residual_chunk
                mask_2d[:, start:end] = mask_chunk

    return mask_2d.reshape(t_len, height, width), residual_std.reshape(height, width)


def _episode_worker_chunk(column_range: Tuple[int, int]) -> Tuple[int, int, np.ndarray, np.ndarray]:
    start, end = column_range
    if _EPISODE_DATA_2D is None:
        raise RuntimeError("Episode worker data was not initialized.")

    data_2d = _EPISODE_DATA_2D
    t_len = data_2d.shape[0]
    residual_chunk = np.zeros(end - start, dtype=np.float64)
    mask_chunk = np.zeros((t_len, end - start), dtype=np.uint8)

    for local_idx, col_idx in enumerate(range(start, end)):
        ts = np.asarray(data_2d[:, col_idx], dtype=np.float64)
        stl = STL(ts, period=_EPISODE_PERIOD, robust=True)
        result = stl.fit()
        remainder = result.resid
        std_val = float(np.std(remainder) + 1e-8)
        residual_chunk[local_idx] = std_val
        mask_chunk[:, local_idx] = (
            (remainder > _EPISODE_SIGMA * std_val) & (ts > _EPISODE_MIN_PM25)
        ).astype(np.uint8)

    return start, end, residual_chunk, mask_chunk


def analyze_episodes(
    paths: AnalysisPaths,
    months: Sequence[str],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    lat_long: Optional[np.ndarray],
    top_hotspots: int,
    sample_size: int,
    rng: np.random.Generator,
    save_episode_arrays: bool,
    episode_workers: int,
    episode_chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    summary_rows: List[MutableMapping[str, object]] = []
    hourly_rows: List[MutableMapping[str, object]] = []
    hotspot_rows: List[MutableMapping[str, object]] = []
    map_bundle: Dict[str, Dict[str, np.ndarray]] = {}

    for month in months:
        pm25 = np.load(paths.raw_dir / month / "cpm25.npy", mmap_mode="r")
        mask, residual_std = detect_episodes_exact(
            pm25,
            workers=episode_workers,
            chunk_size=episode_chunk_size,
        )
        ts = timestamps_by_month[month]
        freq_map = np.asarray(mask.mean(axis=0), dtype=np.float64)
        episode_points_per_timestep = np.asarray(mask.sum(axis=(1, 2)), dtype=np.int64)
        domain_episode_any = (episode_points_per_timestep > 0).astype(np.uint8)
        run_lengths = consecutive_run_lengths(domain_episode_any)
        ep_count = int(mask.sum())
        total_count = int(mask.size)
        episode_sample = sample_masked_values(pm25, mask.astype(bool), sample_size, rng)
        pm25_sum = float(pm25.sum())
        ep_sum = float(np.asarray(pm25[mask.astype(bool)], dtype=np.float64).sum()) if ep_count else 0.0
        non_ep_count = total_count - ep_count
        non_ep_sum = pm25_sum - ep_sum

        map_bundle[month] = {
            "episode_mask": mask,
            "episode_frequency_map": freq_map,
            "residual_std_map": residual_std,
            "episode_points_per_timestep": episode_points_per_timestep,
            "domain_any_episode": domain_episode_any,
            "spatial_mean_pm25": np.asarray(pm25.mean(axis=(1, 2)), dtype=np.float64),
        }

        if save_episode_arrays:
            np.save(paths.array_dir / f"{month.lower()}_episode_mask.npy", mask)
            np.save(paths.array_dir / f"{month.lower()}_residual_std.npy", residual_std)

        summary_rows.append(
            {
                "month": month,
                "month_label": month_label(month),
                "episode_ratio": float(ep_count / total_count),
                "timesteps_with_any_episode_frac": float(domain_episode_any.mean()),
                "mean_episode_points_per_timestep": float(episode_points_per_timestep.mean()),
                "max_episode_points_timestep": int(episode_points_per_timestep.max()),
                "mean_episode_frequency_per_grid": float(freq_map.mean()),
                "max_episode_frequency_per_grid": float(freq_map.max()),
                "mean_residual_std": float(residual_std.mean()),
                "p95_residual_std": float(np.quantile(residual_std, 0.95)),
                "episode_pm25_mean": float(ep_sum / ep_count) if ep_count else float("nan"),
                "non_episode_pm25_mean": float(non_ep_sum / non_ep_count) if non_ep_count else float("nan"),
                "episode_pm25_p95": float(np.quantile(episode_sample, 0.95)) if episode_sample.size else float("nan"),
                "episode_to_non_episode_mean_ratio": (
                    float((ep_sum / ep_count) / (non_ep_sum / non_ep_count))
                    if ep_count and non_ep_count and non_ep_sum > 0
                    else float("nan")
                ),
                "mean_domain_episode_run_hours": float(np.mean(run_lengths)) if run_lengths else 0.0,
                "max_domain_episode_run_hours": int(max(run_lengths)) if run_lengths else 0,
            }
        )

        hourly_frame = pd.DataFrame(
            {
                "timestamp": ts,
                "hour": ts.hour,
                "episode_points": episode_points_per_timestep,
                "episode_ratio_this_timestep": episode_points_per_timestep / (pm25.shape[1] * pm25.shape[2]),
                "domain_mean_pm25": np.asarray(pm25.mean(axis=(1, 2)), dtype=np.float64),
            }
        )
        grouped = hourly_frame.groupby("hour")
        for hour, grp in grouped:
            hourly_rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "hour": int(hour),
                    "mean_episode_ratio": float(grp["episode_ratio_this_timestep"].mean()),
                    "mean_episode_points": float(grp["episode_points"].mean()),
                    "mean_domain_pm25": float(grp["domain_mean_pm25"].mean()),
                    "count": int(grp.shape[0]),
                }
            )

        flat_idx = np.argsort(freq_map.reshape(-1))[::-1][:top_hotspots]
        rows, cols = np.unravel_index(flat_idx, freq_map.shape)
        for rank, (r, c, flat) in enumerate(zip(rows, cols, flat_idx), start=1):
            lat = lon = float("nan")
            if lat_long is not None and lat_long.shape[-1] >= 2:
                lat = float(lat_long[r, c, 0])
                lon = float(lat_long[r, c, 1])
            hotspot_rows.append(
                {
                    "month": month,
                    "month_label": month_label(month),
                    "rank": rank,
                    "row": int(r),
                    "col": int(c),
                    "flat_index": int(flat),
                    "episode_frequency": float(freq_map[r, c]),
                    "residual_std": float(residual_std[r, c]),
                    "latitude": lat,
                    "longitude": lon,
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(hourly_rows),
        pd.DataFrame(hotspot_rows),
        map_bundle,
    )


def plot_cpm25_distribution(
    paths: AnalysisPaths,
    months: Sequence[str],
    sample_size: int,
    rng: np.random.Generator,
    dpi: int,
) -> None:
    records = []
    for month in months:
        sample = sample_flat_values(np.load(paths.raw_dir / month / "cpm25.npy", mmap_mode="r"), sample_size, rng)
        for value in np.log1p(sample):
            records.append({"dataset": month_label(month), "log1p_cpm25": value})
    test_sample = sample_flat_values(np.load(paths.test_dir / "cpm25.npy", mmap_mode="r"), sample_size, rng)
    for value in np.log1p(test_sample):
        records.append({"dataset": month_label("TEST_IN"), "log1p_cpm25": value})

    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxenplot(
        data=frame,
        x="dataset",
        y="log1p_cpm25",
        hue="dataset",
        legend=False,
        ax=ax,
        palette="flare",
    )
    ax.set_title("PM2.5 Distribution by Month / Test Inputs (log1p scale)")
    ax.set_xlabel("")
    ax.set_ylabel("log1p(PM2.5)")
    save_figure(fig, paths.plot_dir / "cpm25_distribution_log1p.png", dpi=dpi)


def plot_temporal_panels(
    paths: AnalysisPaths,
    temporal_month_df: pd.DataFrame,
    hourly_profile_df: pd.DataFrame,
    autocorr_df: pd.DataFrame,
    spatial_mean_series: Mapping[str, Mapping[str, np.ndarray]],
    timestamps_by_month: Mapping[str, pd.DatetimeIndex],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))

    for month in spatial_mean_series:
        axes[0].plot(
            timestamps_by_month[month],
            spatial_mean_series[month]["cpm25"],
            label=month_label(month),
            linewidth=1.5,
        )
    axes[0].set_title("Spatial Mean PM2.5 Over Time")
    axes[0].set_ylabel("PM2.5")
    axes[0].legend()

    sns.lineplot(
        data=hourly_profile_df,
        x="hour",
        y="mean_cpm25_spatial_mean",
        hue="month_label",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Mean Hour-of-Day PM2.5 Spatial Mean")
    axes[1].set_ylabel("PM2.5")

    axes[2].plot(autocorr_df["lag_hours"], autocorr_df["autocorrelation"], marker="o")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("PM2.5 Spatial Mean Autocorrelation")
    axes[2].set_xlabel("Lag (hours)")
    axes[2].set_ylabel("Correlation")

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "cpm25_temporal_panels.png", dpi=dpi)


def plot_spatial_maps(
    paths: AnalysisPaths,
    months: Sequence[str],
    spatial_maps: Mapping[str, Mapping[str, np.ndarray]],
    dpi: int,
) -> None:
    metrics = [
        ("mean_map", "Mean PM2.5"),
        ("std_map", "Temporal Std PM2.5"),
        ("hotspot_freq_map", "Freq(PM2.5 > global P95)"),
    ]
    fig, axes = plt.subplots(len(months), len(metrics), figsize=(15, 4 * len(months)))
    if len(months) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, month in enumerate(months):
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data = spatial_maps[month][metric_key]
            vmax = np.quantile(data, 0.99) if np.isfinite(data).any() else None
            im = ax.imshow(data, origin="lower", aspect="auto", cmap="YlOrRd", vmax=vmax)
            ax.set_title(f"{month_label(month)} | {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "cpm25_spatial_maps.png", dpi=dpi)


def plot_feature_relationships(
    paths: AnalysisPaths,
    corr_matrix_df: pd.DataFrame,
    corr_with_target_df: pd.DataFrame,
    lead_corr_df: pd.DataFrame,
    features: Sequence[str],
    dpi: int,
) -> None:
    corr_matrix = corr_matrix_df.set_index("feature")[list(features)]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Correlation Matrix of Spatial Mean Feature Time Series")
    save_figure(fig, paths.plot_dir / "feature_correlation_matrix.png", dpi=dpi)

    pivot = lead_corr_df.pivot(index="feature", columns="lead_hours", values="corr_feature_t_with_cpm25_t_plus_lead")
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(pivot.loc[list(features)], cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Lead Correlation: feature(t) vs PM2.5 spatial mean(t + lead)")
    ax.set_xlabel("Lead Hours")
    ax.set_ylabel("")
    save_figure(fig, paths.plot_dir / "feature_lead_correlation_heatmap.png", dpi=dpi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ranked = corr_with_target_df.sort_values("within_month_corr_with_cpm25_spatial_mean", ascending=False)
    sns.barplot(
        data=ranked,
        x="within_month_corr_with_cpm25_spatial_mean",
        y="feature",
        hue="feature",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Within-Month Correlation with PM2.5 Spatial Mean")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("")
    save_figure(fig, paths.plot_dir / "feature_corr_with_cpm25.png", dpi=dpi)


def plot_shift_and_staticness(
    paths: AnalysisPaths,
    shift_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    shift_ranked = shift_df.sort_values("ks_statistic", ascending=False)
    sns.barplot(
        data=shift_ranked,
        x="ks_statistic",
        y="feature",
        hue="feature",
        legend=False,
        ax=axes[0],
        palette="rocket",
    )
    axes[0].set_title("Train vs Test Shift by Feature (KS Statistic)")
    axes[0].set_xlabel("KS Statistic")
    axes[0].set_ylabel("")

    variability = (
        feature_summary[feature_summary["split"] == "train_raw"]
        .groupby("feature", as_index=False)["relative_temporal_variability"]
        .mean()
        .sort_values("relative_temporal_variability", ascending=False)
    )
    sns.barplot(
        data=variability,
        x="relative_temporal_variability",
        y="feature",
        hue="feature",
        legend=False,
        ax=axes[1],
        palette="crest",
    )
    axes[1].set_title("Mean Relative Temporal Variability by Feature")
    axes[1].set_xlabel("pixel_std_mean / |global_mean|")
    axes[1].set_ylabel("")
    axes[1].set_xscale("log")

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "shift_and_staticness.png", dpi=dpi)


def plot_window_leakage(
    paths: AnalysisPaths,
    leakage_month_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(
        data=leakage_month_df,
        x="month_label",
        y="share_val_with_any_overlap",
        hue="month_label",
        legend=False,
        ax=axes[0],
        palette="magma",
    )
    axes[0].set_title("Fraction of Validation Windows Overlapping Train Windows")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of Val Windows")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(overlap_df["window_start_offset_hours"], overlap_df["shared_timesteps"], marker="o")
    axes[1].set_title("Shared Timesteps Between Two 26-Hour Windows")
    axes[1].set_xlabel("Difference in Window Start Time (hours)")
    axes[1].set_ylabel("Shared Timesteps")
    axes[1].axvline(25, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "window_leakage.png", dpi=dpi)


def plot_episode_panels(
    paths: AnalysisPaths,
    months: Sequence[str],
    episode_summary_df: pd.DataFrame,
    episode_hourly_df: pd.DataFrame,
    episode_maps: Mapping[str, Mapping[str, np.ndarray]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(months), 2, figsize=(12, 4 * len(months)))
    if len(months) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, month in enumerate(months):
        freq = episode_maps[month]["episode_frequency_map"]
        resid = episode_maps[month]["residual_std_map"]
        for col_idx, (data, title, cmap) in enumerate(
            [
                (freq, "Episode Frequency", "Reds"),
                (resid, "Residual Std", "viridis"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            vmax = np.quantile(data, 0.99) if np.isfinite(data).any() else None
            im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmax=vmax)
            ax.set_title(f"{month_label(month)} | {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "episode_spatial_panels.png", dpi=dpi)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for month in months:
        axes[0].plot(
            episode_maps[month]["episode_points_per_timestep"],
            label=month_label(month),
            linewidth=1.4,
        )
        axes[1].plot(
            episode_maps[month]["spatial_mean_pm25"],
            label=month_label(month),
            linewidth=1.4,
        )
    axes[0].set_title("Episode Grid Points Per Timestep")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[1].set_title("Spatial Mean PM2.5 Per Timestep")
    axes[1].set_ylabel("PM2.5")
    axes[1].set_xlabel("Hourly Timestep")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "episode_time_series.png", dpi=dpi)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(
        data=episode_hourly_df,
        x="hour",
        y="mean_episode_ratio",
        hue="month_label",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Hourly Episode Ratio")
    axes[0].set_ylabel("Episode Ratio")
    axes[0].set_xlabel("Hour")

    sns.barplot(
        data=episode_summary_df,
        x="month_label",
        y="episode_to_non_episode_mean_ratio",
        hue="month_label",
        legend=False,
        palette="mako",
        ax=axes[1],
    )
    axes[1].set_title("Episode / Non-Episode Mean PM2.5 Ratio")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Ratio")

    fig.tight_layout()
    save_figure(fig, paths.plot_dir / "episode_hourly_and_intensity.png", dpi=dpi)


def render_report(
    paths: AnalysisPaths,
    overview: Mapping[str, object],
    feature_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    normalization_audit: pd.DataFrame,
    temporal_month_df: pd.DataFrame,
    corr_with_target_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    leakage_month_df: pd.DataFrame,
    episode_summary_df: Optional[pd.DataFrame],
) -> str:
    cpm25_rows = feature_summary[(feature_summary["feature"] == "cpm25") & (feature_summary["split"] == "train_raw")]
    cpm25_rank = cpm25_rows.sort_values("global_mean", ascending=False)[["month_label", "global_mean", "sample_p95", "global_max"]]
    staticness = (
        feature_summary[feature_summary["split"] == "train_raw"]
        .groupby("feature", as_index=False)["relative_temporal_variability"]
        .mean()
        .sort_values("relative_temporal_variability")
    )
    top_shift = shift_df.sort_values("ks_statistic", ascending=False).head(6)
    bad_norm = normalization_audit[normalization_audit["normalization_flag"] == "recompute_stats"].sort_values(
        "max_ratio_empirical_over_provided", ascending=False
    )
    top_corr = corr_with_target_df.sort_values("within_month_corr_with_cpm25_spatial_mean", ascending=False).head(6)
    leakage_problem_months = leakage_month_df[leakage_month_df["share_val_with_any_overlap"] > 0.95]

    lines: List[str] = []
    lines.append("# Phase 2 PM2.5 EDA Report")
    lines.append("")
    lines.append(f"Generated: {overview['generated_at']}")
    lines.append("")
    lines.append("## Problem Lens")
    lines.append("")
    lines.append(
        "Phase 2 is not only a next-step forecasting task. The leaderboard also rewards how well the model tracks "
        "spatial structure and magnitude during extreme pollution episodes. That means the analysis should prioritize "
        "seasonality, distribution shift, episode rarity, and any preprocessing choice that can suppress sharp spikes."
    )
    lines.append("")
    lines.append("## Dataset Structure")
    lines.append("")
    for month_payload in overview["months"]:
        lines.append(
            f"- {month_payload['label']}: {month_payload['timesteps']} hourly steps from {month_payload['start']} to "
            f"{month_payload['end']} with {month_payload['window_count_stride_1']} sliding 26-hour windows at stride 1."
        )
    test_payload = overview.get("test_inputs", {})
    if test_payload:
        lines.append(
            f"- Test inputs: {test_payload['n_samples']} samples with shape {tuple(test_payload['shape'])}. "
            f"Only {test_payload['lookback_hours']} lookback hours are available at inference time."
        )
    lines.append("")
    lines.append("## Most Important Findings")
    lines.append("")
    lines.append("### 1. Seasonal heterogeneity is strong")
    for row in cpm25_rank.itertuples(index=False):
        lines.append(
            f"- {row.month_label}: mean={row.global_mean:.2f}, sample_p95={row.sample_p95:.2f}, max={row.global_max:.2f}"
        )
    lines.append("")
    lines.append("### 2. The provided normalization stats need auditing")
    if bad_norm.empty:
        lines.append("- No feature crossed the report threshold for obvious min/max mismatch.")
    else:
        for row in bad_norm.head(6).itertuples(index=False):
            lines.append(
                f"- {row.feature}: empirical_max={row.empirical_max:.4g}, provided_max={row.provided_max:.4g}, "
                f"ratio={row.max_ratio_empirical_over_provided:.3f}, mean_above_provided_max_frac={row.mean_above_provided_max_frac:.4f}"
            )
    lines.append("")
    lines.append("### 3. Several features show train-test shift")
    for row in top_shift.itertuples(index=False):
        lines.append(
            f"- {row.feature}: KS={row.ks_statistic:.4f}, mean_ratio={row.mean_ratio_test_over_train:.3f}, "
            f"p95_ratio={row.p95_ratio_test_over_train:.3f}"
        )
    lines.append("")
    lines.append("### 4. Some channels behave like quasi-static seasonal priors")
    for row in staticness.head(6).itertuples(index=False):
        lines.append(f"- {row.feature}: mean relative temporal variability={row.relative_temporal_variability:.6g}")
    lines.append("")
    lines.append("### 5. Feature relationships to PM2.5")
    for row in top_corr.itertuples(index=False):
        lines.append(
            f"- {row.feature}: within-month corr with PM2.5 spatial mean={row.within_month_corr_with_cpm25_spatial_mean:.3f}"
        )
    lines.append("")
    lines.append("### 6. Random overlapping splits are misleading")
    if leakage_problem_months.empty:
        lines.append("- The leakage audit did not exceed the warning threshold.")
    else:
        for row in leakage_problem_months.itertuples(index=False):
            lines.append(
                f"- {row.month_label}: {row.share_val_with_any_overlap:.3f} of validation windows overlap at least one train window; "
                f"mean nearest-window overlap={row.mean_overlap_hours_with_nearest_train_window:.2f} hours."
            )
    lines.append("")

    if episode_summary_df is not None and not episode_summary_df.empty:
        lines.append("### 7. Episodes are sparse but much more intense")
        for row in episode_summary_df.sort_values("episode_ratio", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.month_label}: episode_ratio={row.episode_ratio:.4f}, "
                f"timesteps_with_any_episode={row.timesteps_with_any_episode_frac:.4f}, "
                f"episode/non-episode mean ratio={row.episode_to_non_episode_mean_ratio:.2f}"
            )
        lines.append("")

    lines.append("## Modeling Takeaways")
    lines.append("")
    lines.append("- Use time-blocked or month-aware validation, not a random split over overlapping 26-hour windows.")
    lines.append("- Recompute normalization stats from the raw arrays; the provided stats can under-cover the true range.")
    lines.append("- Treat emission channels as strong static spatial priors and let dynamic meteorology carry temporal variation.")
    lines.append("- Keep architecture and loss choices episode-aware because rare spikes influence two of the three leaderboard terms.")
    lines.append(
        "- Expect regime change across seasons and between 2016 train months and 2017 test inputs; month/season conditioning and robust scaling can help."
    )
    lines.append("- Inspect the saved episode hotspot maps before committing to any model that visually over-smooths sharp plumes.")
    lines.append("")
    lines.append("## Output Artifacts")
    lines.append("")
    lines.append(f"- Tables: `{paths.table_dir}`")
    lines.append(f"- Plots: `{paths.plot_dir}`")
    lines.append(f"- Arrays: `{paths.array_dir}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = build_paths(args)

    if not paths.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {paths.raw_dir}")
    if not paths.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {paths.test_dir}")

    months = args.months if args.months else discover_months(paths.raw_dir)
    if not months:
        raise ValueError(f"No month folders found in {paths.raw_dir}")

    missing_months = [month for month in months if not (paths.raw_dir / month).exists()]
    if missing_months:
        raise FileNotFoundError(f"Requested month folders not found: {missing_months}")

    features = [feature for feature in ALL_FEATURES if (paths.raw_dir / months[0] / f"{feature}.npy").exists()]
    provided_minmax = load_provided_minmax(paths.stats_file)
    rng = np.random.default_rng(args.split_seed)

    sns.set_theme(style="whitegrid")

    feature_summary, spatial_mean_series, timestamps_by_month = summarize_train_features(
        paths=paths,
        months=months,
        features=features,
        provided_minmax=provided_minmax,
        sample_size=args.sample_size,
        rng=rng,
    )
    test_summary = summarize_test_inputs(
        paths=paths,
        features=features,
        provided_minmax=provided_minmax,
        sample_size=args.sample_size,
        rng=rng,
    )
    full_summary = pd.concat([feature_summary, test_summary], ignore_index=True)

    overview = build_dataset_overview(
        paths=paths,
        months=months,
        features=features,
        timestamps_by_month=timestamps_by_month,
    )
    normalization_audit = build_normalization_audit(feature_summary)
    temporal_month_df, hourly_profile_df, autocorr_df = compute_cpm25_temporal_analysis(
        spatial_mean_series=spatial_mean_series,
        timestamps_by_month=timestamps_by_month,
        lag_hours=args.lag_hours,
    )
    spatial_map_df, spatial_maps = compute_spatial_maps(
        paths=paths,
        months=months,
        rng=rng,
        sample_size=args.sample_size,
    )
    corr_matrix_df, corr_with_target_df, lead_corr_df = compute_feature_relationships(
        spatial_mean_series=spatial_mean_series,
        timestamps_by_month=timestamps_by_month,
        features=features,
        lag_hours=args.lag_hours,
    )
    shift_df = compute_train_test_shift(
        paths=paths,
        months=months,
        features=features,
        sample_size=args.sample_size,
        rng=rng,
    )
    distribution_diag_df, sample_cache = compute_feature_distribution_diagnostics(
        paths=paths,
        months=months,
        features=features,
        sample_size=args.sample_size,
        rng=rng,
        mode_bins=args.mode_bins,
    )
    transform_recommendations_df = build_feature_transform_recommendations(distribution_diag_df)
    feature_temporal_diag_df, feature_hourly_profile_df = compute_feature_temporal_diagnostics(
        spatial_mean_series=spatial_mean_series,
        timestamps_by_month=timestamps_by_month,
        features=features,
    )
    feature_spatial_diag_df = compute_feature_spatial_diagnostics(
        paths=paths,
        months=months,
        features=features,
    )
    forecastability_df = compute_feature_forecastability_summary(lead_corr_df)
    redundancy_df = compute_feature_redundancy_vif(
        spatial_mean_series=spatial_mean_series,
        features=features,
    )
    regime_profile_df, regime_summary_df = compute_feature_target_regime_response(
        spatial_mean_series=spatial_mean_series,
        months=months,
        features=features,
    )
    pair_detail_df, pair_stability_df = compute_feature_pair_stability(
        spatial_mean_series=spatial_mean_series,
        months=months,
        features=features,
    )
    month_shift_detail_df, month_shift_summary_df = compute_monthwise_shift_to_test(
        months=months,
        features=features,
        sample_cache=sample_cache,
    )
    seasonality_summary_df = compute_feature_seasonality_summary(feature_summary)
    feature_role_df = build_feature_role_recommendations(
        distribution_df=distribution_diag_df,
        temporal_diag_df=feature_temporal_diag_df,
        spatial_diag_df=feature_spatial_diag_df,
        seasonality_df=seasonality_summary_df,
        redundancy_df=redundancy_df,
        forecastability_df=forecastability_df,
    )
    normalization_blueprint_df = build_feature_normalization_blueprint(
        distribution_df=distribution_diag_df,
        transform_df=transform_recommendations_df,
        role_df=feature_role_df,
    )
    leakage_month_df, overlap_df = build_window_leakage_tables(
        timestamps_by_month=timestamps_by_month,
        stride=args.stride,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )

    lat_long = None
    lat_long_path = paths.raw_dir / "lat_long.npy"
    if lat_long_path.exists():
        lat_long = np.load(lat_long_path, mmap_mode="r")

    episode_summary_df: Optional[pd.DataFrame] = None
    episode_hourly_df: Optional[pd.DataFrame] = None
    episode_hotspots_df: Optional[pd.DataFrame] = None
    episode_maps: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    episode_feature_detail_df = pd.DataFrame()
    episode_feature_summary_df = pd.DataFrame()

    if not args.skip_episode:
        episode_summary_df, episode_hourly_df, episode_hotspots_df, episode_maps = analyze_episodes(
            paths=paths,
            months=months,
            timestamps_by_month=timestamps_by_month,
            lat_long=lat_long,
            top_hotspots=args.top_hotspots,
            sample_size=args.sample_size,
            rng=rng,
            save_episode_arrays=args.save_episode_arrays,
            episode_workers=args.episode_workers,
            episode_chunk_size=args.episode_chunk_size,
        )
    else:
        episode_summary_df, episode_hourly_df, episode_hotspots_df = maybe_load_existing_episode_tables(paths)
        episode_maps = maybe_load_existing_episode_maps(paths, months)

    if episode_maps is not None:
        episode_feature_detail_df, episode_feature_summary_df = compute_episode_feature_signatures(
            paths=paths,
            months=months,
            features=features,
            spatial_mean_series=spatial_mean_series,
            episode_maps=episode_maps,
            sample_size=args.sample_size,
            rng=rng,
        )

    save_json(overview, paths.output_dir / "overview.json")
    save_dataframe(full_summary, paths.table_dir / "feature_summary.csv")
    save_dataframe(normalization_audit, paths.table_dir / "normalization_audit.csv")
    save_dataframe(temporal_month_df, paths.table_dir / "cpm25_temporal_month_summary.csv")
    save_dataframe(hourly_profile_df, paths.table_dir / "cpm25_hourly_profile.csv")
    save_dataframe(autocorr_df, paths.table_dir / "cpm25_autocorrelation.csv")
    save_dataframe(spatial_map_df, paths.table_dir / "cpm25_spatial_summary.csv")
    save_dataframe(corr_matrix_df, paths.table_dir / "feature_correlation_matrix.csv")
    save_dataframe(corr_with_target_df, paths.table_dir / "feature_corr_with_cpm25.csv")
    save_dataframe(lead_corr_df, paths.table_dir / "feature_lead_correlations.csv")
    save_dataframe(shift_df, paths.table_dir / "train_test_shift.csv")
    save_dataframe(leakage_month_df, paths.table_dir / "window_leakage_months.csv")
    save_dataframe(overlap_df, paths.table_dir / "window_overlap_by_offset.csv")
    save_dataframe(distribution_diag_df, paths.table_dir / "feature_distribution_diagnostics.csv")
    save_dataframe(transform_recommendations_df, paths.table_dir / "feature_transform_recommendations.csv")
    save_dataframe(feature_temporal_diag_df, paths.table_dir / "feature_temporal_diagnostics.csv")
    save_dataframe(feature_hourly_profile_df, paths.table_dir / "feature_hourly_profile.csv")
    save_dataframe(feature_spatial_diag_df, paths.table_dir / "feature_spatial_diagnostics.csv")
    save_dataframe(month_shift_detail_df, paths.table_dir / "monthwise_train_test_shift.csv")
    save_dataframe(month_shift_summary_df, paths.table_dir / "monthwise_train_test_shift_summary.csv")
    save_dataframe(seasonality_summary_df, paths.table_dir / "feature_seasonality_summary.csv")
    save_dataframe(forecastability_df, paths.table_dir / "feature_forecastability_summary.csv")
    save_dataframe(redundancy_df, paths.table_dir / "feature_redundancy_vif.csv")
    save_dataframe(feature_role_df, paths.table_dir / "feature_role_recommendations.csv")
    save_dataframe(regime_profile_df, paths.table_dir / "feature_target_regime_profile.csv")
    save_dataframe(regime_summary_df, paths.table_dir / "feature_target_regime_summary.csv")
    save_dataframe(pair_detail_df, paths.table_dir / "feature_pair_monthly_correlation.csv")
    save_dataframe(pair_stability_df, paths.table_dir / "feature_pair_stability_summary.csv")
    save_dataframe(normalization_blueprint_df, paths.table_dir / "feature_normalization_blueprint.csv")

    if episode_summary_df is not None and episode_hourly_df is not None and episode_hotspots_df is not None:
        save_dataframe(episode_summary_df, paths.table_dir / "episode_summary.csv")
        save_dataframe(episode_hourly_df, paths.table_dir / "episode_hourly_profile.csv")
        save_dataframe(episode_hotspots_df, paths.table_dir / "episode_hotspots.csv")
    if not episode_feature_detail_df.empty:
        save_dataframe(episode_feature_detail_df, paths.table_dir / "episode_feature_signatures.csv")
        save_dataframe(episode_feature_summary_df, paths.table_dir / "episode_feature_signature_summary.csv")

    plot_cpm25_distribution(paths, months, args.sample_size, rng, args.dpi)
    plot_temporal_panels(
        paths=paths,
        temporal_month_df=temporal_month_df,
        hourly_profile_df=hourly_profile_df,
        autocorr_df=autocorr_df,
        spatial_mean_series=spatial_mean_series,
        timestamps_by_month=timestamps_by_month,
        dpi=args.dpi,
    )
    plot_spatial_maps(paths, months, spatial_maps, args.dpi)
    plot_feature_relationships(paths, corr_matrix_df, corr_with_target_df, lead_corr_df, features, args.dpi)
    plot_shift_and_staticness(paths, shift_df, feature_summary, args.dpi)
    plot_window_leakage(paths, leakage_month_df, overlap_df, args.dpi)
    plot_distribution_heatmaps(
        paths=paths,
        distribution_df=distribution_diag_df,
        transform_df=transform_recommendations_df,
        month_shift_detail_df=month_shift_detail_df,
        dpi=args.dpi,
    )
    plot_spatial_concentration_heatmap(paths, feature_spatial_diag_df, args.dpi)
    plot_feature_role_map(paths, feature_role_df, args.dpi)
    plot_forecastability_and_redundancy(paths, forecastability_df, redundancy_df, args.dpi)
    plot_target_regime_response(paths, regime_profile_df, regime_summary_df, args.dpi)
    plot_pairwise_correlation_stability(paths, pair_detail_df, pair_stability_df, args.dpi)
    plot_transform_previews(paths, distribution_diag_df, sample_cache, features, months, args.dpi)
    plot_feature_diagnostic_book(
        paths=paths,
        features=features,
        months=months,
        sample_cache=sample_cache,
        distribution_df=distribution_diag_df,
        temporal_hourly_df=feature_hourly_profile_df,
        temporal_diag_df=feature_temporal_diag_df,
        lead_corr_df=lead_corr_df,
        month_shift_detail_df=month_shift_detail_df,
        spatial_mean_series=spatial_mean_series,
        timestamps_by_month=timestamps_by_month,
        dpi=args.dpi,
    )
    plot_episode_hotspot_geo(paths, episode_hotspots_df, args.dpi)
    plot_episode_feature_signatures(paths, episode_feature_detail_df, episode_feature_summary_df, args.dpi)

    if episode_summary_df is not None and episode_hourly_df is not None and episode_maps is not None:
        plot_episode_panels(
            paths=paths,
            months=months,
            episode_summary_df=episode_summary_df,
            episode_hourly_df=episode_hourly_df,
            episode_maps=episode_maps,
            dpi=args.dpi,
        )

    summary_report_text = render_report(
        paths=paths,
        overview=overview,
        feature_summary=feature_summary,
        test_summary=test_summary,
        normalization_audit=normalization_audit,
        temporal_month_df=temporal_month_df,
        corr_with_target_df=corr_with_target_df,
        shift_df=shift_df,
        leakage_month_df=leakage_month_df,
        episode_summary_df=episode_summary_df,
    )
    report_text = render_deep_dive_report(
        base_report_text=summary_report_text,
        distribution_df=distribution_diag_df,
        transform_df=transform_recommendations_df,
        blueprint_df=normalization_blueprint_df,
        temporal_diag_df=feature_temporal_diag_df,
        spatial_diag_df=feature_spatial_diag_df,
        month_shift_detail_df=month_shift_detail_df,
        month_shift_summary_df=month_shift_summary_df,
        seasonality_df=seasonality_summary_df,
        forecastability_df=forecastability_df,
        redundancy_df=redundancy_df,
        role_df=feature_role_df,
        regime_summary_df=regime_summary_df,
        pair_stability_df=pair_stability_df,
        episode_feature_summary_df=episode_feature_summary_df,
        episode_summary_df=episode_summary_df,
    )
    report_path = paths.output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    (paths.output_dir / "report_summary.md").write_text(summary_report_text, encoding="utf-8")
    (paths.output_dir / "preprocessing_guide.md").write_text(
        render_preprocessing_guide(normalization_blueprint_df, seasonality_summary_df),
        encoding="utf-8",
    )
    (paths.output_dir / "model_search_guide.md").write_text(
        render_model_search_guide(
            month_shift_summary_df,
            transform_recommendations_df,
            seasonality_summary_df,
            forecastability_df,
            redundancy_df,
            feature_role_df,
            regime_summary_df,
            pair_stability_df,
            episode_feature_summary_df,
        ),
        encoding="utf-8",
    )

    render_all_table_images(
        paths=paths,
        max_rows=args.table_image_rows,
        max_cols=args.table_image_cols,
        dpi=args.dpi,
    )

    print(f"EDA complete. Report written to: {report_path}")
    print(f"Tables: {paths.table_dir}")
    print(f"Plots: {paths.plot_dir}")
    print(f"Arrays: {paths.array_dir}")


if __name__ == "__main__":
    main()

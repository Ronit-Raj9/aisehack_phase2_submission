"""Minimal preprocessing utility: feature-wise normalization only (no feature engineering).

This script fits train-only normalization stats for the 16 raw competition features,
stores a reusable artifact, and can optionally export normalized arrays.

Policy implemented (from project EDA decisions):
	- log1p + robust scaling: cpm25 + sparse emission channels
	- robust scaling: u10, v10, pblh, rain
	- standard scaling: q2, t2, swdown, psfc

Usage:
	python scripts/preprocess_data.py --config configs/train.yaml
	python scripts/preprocess_data.py --config configs/train.yaml --raw_root /path/to/raw
	python scripts/preprocess_data.py --config configs/train.yaml --export_normalized
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config

EPS = 1e-8


@dataclass(frozen=True)
class FeaturePolicy:
	transform: str
	scaler: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/train.yaml")
	parser.add_argument("--raw_root", type=str, default=None, help="Override cfg.paths.raw_path")
	parser.add_argument(
		"--out_dir",
		type=str,
		default=None,
		help="Directory to save stats and optional normalized arrays",
	)
	parser.add_argument(
		"--months",
		nargs="*",
		default=None,
		help="Optional list of months to use for fitting (default: cfg.data.months)",
	)
	parser.add_argument(
		"--export_normalized",
		action="store_true",
		help="If set, writes normalized arrays for selected months to out_dir/normalized",
	)
	return parser.parse_args()


def _fallback_raw_root(script_dir: Path) -> Path:
	candidate = (script_dir.parent.parent / "data" / "raw").resolve()
	if candidate.is_dir():
		return candidate
	candidate = (script_dir.parent.parent.parent / "data" / "raw").resolve()
	return candidate


def resolve_raw_root(cfg, arg_raw_root: str | None, script_dir: Path) -> Path:
	if arg_raw_root is not None:
		raw_root = Path(arg_raw_root).expanduser().resolve()
		if not raw_root.is_dir():
			raise FileNotFoundError(f"--raw_root not found: {raw_root}")
		return raw_root

	cfg_raw = getattr(getattr(cfg, "paths", object()), "raw_path", None)
	if cfg_raw is not None:
		cfg_path = Path(cfg_raw).expanduser()
		if cfg_path.is_dir():
			return cfg_path.resolve()

	fallback = _fallback_raw_root(script_dir)
	if fallback.is_dir():
		return fallback
	raise FileNotFoundError(
		"Could not resolve raw_root. Pass --raw_root explicitly or set cfg.paths.raw_path to a valid directory."
	)


def resolve_out_dir(arg_out_dir: str | None, script_dir: Path) -> Path:
	if arg_out_dir is not None:
		return Path(arg_out_dir).expanduser().resolve()
	return (script_dir.parent / "artifacts" / "preprocessing").resolve()


def get_raw_feature_order(cfg) -> List[str]:
	met = list(getattr(cfg.features, "met_variables_raw", []))
	persist = list(getattr(cfg.features, "persist_met_for_physics", []))
	emissions = list(getattr(cfg.features, "emission_variables_raw", []))

	ordered_met = list(met)
	for name in persist:
		if name not in ordered_met:
			ordered_met.append(name)

	ordered = ordered_met + emissions
	if len(ordered) != 16:
		raise ValueError(
			f"Expected 16 raw features, got {len(ordered)}: {ordered}. "
			"Check config feature lists."
		)
	return ordered


def get_feature_policy(feature: str) -> FeaturePolicy:
	log1p_robust = {
		"cpm25",
		"PM25",
		"NH3",
		"SO2",
		"NOx",
		"NMVOC_e",
		"NMVOC_finn",
		"bio",
	}
	robust_identity = {"u10", "v10", "pblh", "rain"}
	standard_identity = {"q2", "t2", "swdown", "psfc"}

	if feature in log1p_robust:
		return FeaturePolicy(transform="log1p", scaler="robust")
	if feature in robust_identity:
		return FeaturePolicy(transform="identity", scaler="robust")
	if feature in standard_identity:
		return FeaturePolicy(transform="identity", scaler="standard")
	raise KeyError(f"No normalization policy defined for feature '{feature}'")


def load_feature_vectors(raw_root: Path, months: Iterable[str], feature: str) -> np.ndarray:
	vectors: List[np.ndarray] = []
	for month in months:
		feature_path = raw_root / month / f"{feature}.npy"
		if not feature_path.is_file():
			raise FileNotFoundError(f"Missing feature file: {feature_path}")
		arr = np.load(feature_path, mmap_mode="r")
		vectors.append(np.asarray(arr, dtype=np.float32).reshape(-1))
	return np.concatenate(vectors, axis=0)


def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
	if transform == "identity":
		return x.astype(np.float32)
	if transform == "log1p":
		return np.log1p(np.maximum(x.astype(np.float32), 0.0)).astype(np.float32)
	raise ValueError(f"Unknown transform: {transform}")


def inverse_transform(x: np.ndarray, transform: str) -> np.ndarray:
	if transform == "identity":
		return x.astype(np.float32)
	if transform == "log1p":
		return np.expm1(x).astype(np.float32)
	raise ValueError(f"Unknown transform: {transform}")


def fit_standard_stats(x: np.ndarray) -> Dict[str, float]:
	mean = float(np.mean(x))
	std = float(np.std(x))
	if not np.isfinite(std) or std < EPS:
		std = 1.0
	return {"center": mean, "scale": std, "scale_name": "std"}


def fit_robust_stats(x: np.ndarray) -> Dict[str, float]:
	median = float(np.median(x))
	q1 = float(np.quantile(x, 0.25))
	q3 = float(np.quantile(x, 0.75))
	iqr = float(q3 - q1)
	p95 = float(np.quantile(x, 0.95))
	alt_scale = float(max(p95 - median, 0.0))
	std = float(np.std(x))
	min_meaningful_scale = max(EPS, 0.05 * std)
	if (not np.isfinite(iqr)) or (iqr < min_meaningful_scale):
		if np.isfinite(alt_scale) and alt_scale >= min_meaningful_scale:
			iqr = alt_scale
		elif np.isfinite(std) and std >= EPS:
			iqr = std
		else:
			iqr = 1.0
	return {
		"center": median,
		"scale": iqr,
		"scale_name": "iqr",
		"q1": q1,
		"q3": q3,
		"p95": p95,
	}


def normalize_with_stats(x: np.ndarray, center: float, scale: float) -> np.ndarray:
	return ((x.astype(np.float32) - center) / max(scale, EPS)).astype(np.float32)


def denormalize_with_stats(x: np.ndarray, center: float, scale: float) -> np.ndarray:
	return (x.astype(np.float32) * max(scale, EPS) + center).astype(np.float32)


def fit_feature_stats(raw_root: Path, months: List[str], feature: str) -> Dict[str, object]:
	policy = get_feature_policy(feature)
	raw_values = load_feature_vectors(raw_root, months, feature)
	transformed = apply_transform(raw_values, policy.transform)

	if policy.scaler == "standard":
		stats = fit_standard_stats(transformed)
	elif policy.scaler == "robust":
		stats = fit_robust_stats(transformed)
	else:
		raise ValueError(f"Unknown scaler: {policy.scaler}")

	normalized = normalize_with_stats(transformed, stats["center"], stats["scale"])
	summary = {
		"n": int(raw_values.size),
		"raw_min": float(np.min(raw_values)),
		"raw_max": float(np.max(raw_values)),
		"raw_mean": float(np.mean(raw_values)),
		"raw_std": float(np.std(raw_values)),
		"norm_mean": float(np.mean(normalized)),
		"norm_std": float(np.std(normalized)),
	}

	return {
		"feature": feature,
		"transform": policy.transform,
		"scaler": policy.scaler,
		"stats": stats,
		"summary": summary,
	}


def save_stats(out_dir: Path, payload: Dict[str, object]) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)
	stats_path = out_dir / "normalization_stats.json"
	with stats_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	return stats_path


def load_stats(stats_path: Path) -> Dict[str, object]:
	with stats_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def normalize_array(arr: np.ndarray, feature_payload: Dict[str, object]) -> np.ndarray:
	transform = feature_payload["transform"]
	center = float(feature_payload["stats"]["center"])
	scale = float(feature_payload["stats"]["scale"])
	transformed = apply_transform(arr.astype(np.float32), transform)
	return normalize_with_stats(transformed, center, scale)


def inverse_normalize_array(arr: np.ndarray, feature_payload: Dict[str, object]) -> np.ndarray:
	transform = feature_payload["transform"]
	center = float(feature_payload["stats"]["center"])
	scale = float(feature_payload["stats"]["scale"])
	denorm = denormalize_with_stats(arr.astype(np.float32), center, scale)
	return inverse_transform(denorm, transform)


def export_normalized_arrays(
	raw_root: Path,
	out_dir: Path,
	months: List[str],
	feature_order: List[str],
	features_payload: Dict[str, Dict[str, object]],
) -> Path:
	normalized_root = out_dir / "normalized"
	normalized_root.mkdir(parents=True, exist_ok=True)

	for month in months:
		month_in = raw_root / month
		month_out = normalized_root / month
		month_out.mkdir(parents=True, exist_ok=True)
		for feature in feature_order:
			src = month_in / f"{feature}.npy"
			if not src.is_file():
				raise FileNotFoundError(f"Missing feature file for export: {src}")
			arr = np.load(src)
			norm_arr = normalize_array(arr, features_payload[feature])
			np.save(month_out / f"{feature}.npy", norm_arr.astype(np.float32))

	return normalized_root


def main() -> None:
	args = parse_args()
	script_dir = Path(__file__).resolve().parent
	cfg = load_config(args.config)

	raw_root = resolve_raw_root(cfg, args.raw_root, script_dir)
	out_dir = resolve_out_dir(args.out_dir, script_dir)

	fit_months = list(args.months) if args.months else list(cfg.data.months)
	feature_order = get_raw_feature_order(cfg)

	print("=" * 72)
	print("Minimal normalization preprocessing")
	print(f"Config            : {Path(args.config).resolve()}")
	print(f"Raw root          : {raw_root}")
	print(f"Fit months        : {fit_months}")
	print(f"Feature count     : {len(feature_order)}")
	print(f"Output directory  : {out_dir}")
	print("=" * 72)

	features_payload: Dict[str, Dict[str, object]] = {}
	for feature in feature_order:
		payload = fit_feature_stats(raw_root, fit_months, feature)
		features_payload[feature] = payload
		s = payload["summary"]
		print(
			f"[{feature:10s}] transform={payload['transform']:8s} "
			f"scaler={payload['scaler']:8s} "
			f"norm_mean={s['norm_mean']:+.4f} norm_std={s['norm_std']:.4f}"
		)

	full_payload = {
		"version": 1,
		"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"fit_months": fit_months,
		"feature_order": feature_order,
		"features": features_payload,
		"notes": "Normalization only: no clipping, no feature engineering, no target transform.",
	}

	stats_path = save_stats(out_dir, full_payload)
	print(f"\nSaved stats artifact: {stats_path}")

	if args.export_normalized:
		normalized_root = export_normalized_arrays(
			raw_root=raw_root,
			out_dir=out_dir,
			months=fit_months,
			feature_order=feature_order,
			features_payload=features_payload,
		)
		print(f"Saved normalized arrays: {normalized_root}")

	print("Done.")


if __name__ == "__main__":
	main()

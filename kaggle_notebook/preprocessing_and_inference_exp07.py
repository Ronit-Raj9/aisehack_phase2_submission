"""End-to-end EXP07 preprocessing + inference runner.

This script reproduces the EXP07 preprocessing/inference path used in training notebooks,
then runs inference on test_in using a downloaded checkpoint (best.pt / last.pt).

Usage examples:
	python kaggle_notebook/preprocessing_and_inference_exp07.py \
		--model_path /path/to/best.pt

	python kaggle_notebook/preprocessing_and_inference_exp07.py \
		--model_path /path/to/last.pt \
		--raw_root /path/to/raw \
		--test_input /path/to/test_in \
		--work_dir /tmp/exp07_infer
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT_DEFAULT = Path("/kaggle/input/datasets/ronitraj1/ronit-pm25-phase2-src")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model_path",
		type=str,
		default="/kaggle/input/datasets/heyronnie/exp-07-1/experiments/exp07_hybrid_single_gpu_amp/checkpoints/best.pt",
		help="Path to downloaded best.pt/last.pt",
	)
	parser.add_argument("--prefer", type=str, default="best", choices=["best", "last"], help="Used only when --model_path is omitted")
	parser.add_argument(
		"--raw_root",
		type=str,
		default="/kaggle/input/datasets/ronitraj1/aisehack-finals/aisehack-theme-2/raw",
		help="Raw train data root (contains APRIL_16/, JULY_16/, etc.)",
	)
	parser.add_argument(
		"--test_input",
		type=str,
		default="/kaggle/input/datasets/ronitraj1/aisehack-finals/aisehack-theme-2/test_in",
		help="Test input folder (contains cpm25.npy etc.)",
	)
	parser.add_argument("--work_dir", type=str, default=None, help="Working output dir (runtime yaml, stats, preds)")
	parser.add_argument("--preds_out", type=str, default=None, help="Output .npy path for predictions")
	parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing if stats already exist")
	parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable used to invoke scripts")
	args, _ = parser.parse_known_args()
	return args


def _resolve_project_root() -> Path:
	# Prefer the Kaggle dataset copy that contains the known configs.
	if (PROJECT_ROOT_DEFAULT / "configs" / "train.yaml").is_file() and (PROJECT_ROOT_DEFAULT / "configs" / "infer.yaml").is_file():
		return PROJECT_ROOT_DEFAULT

	# Fall back to a writable working copy or current directory if needed.
	for candidate in [Path("/kaggle/working/ronit-pm25-phase2-src"), Path.cwd().resolve()]:
		if (candidate / "configs" / "train.yaml").is_file() and (candidate / "configs" / "infer.yaml").is_file():
			return candidate

	raise FileNotFoundError(
		"Could not resolve project root. Expected configs under /kaggle/input/datasets/ronitraj1/ronit-pm25-phase2-src or /kaggle/working/ronit-pm25-phase2-src."
	)


def _resolve_raw_root(raw_root_arg: str | None) -> Path:
	if raw_root_arg:
		root = Path(raw_root_arg).expanduser().resolve()
		if (root / "APRIL_16" / "cpm25.npy").is_file():
			return root
		raise FileNotFoundError(f"Invalid --raw_root: {root} (missing APRIL_16/cpm25.npy)")

	candidates = [
		Path("/kaggle/input/competitions/anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd/aisehack-theme-2/raw"),
		Path("/kaggle/input/datasets/ronitraj1/aisehack-finals/aisehack-theme-2/raw"),
		Path("/kaggle/input/datasets/ronitraj1/aisehack-finals/aisehack-theme-2"),
		Path("/kaggle/input/competitions/anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd/aisehack-theme-2"),
	]
	for candidate in candidates:
		if (candidate / "APRIL_16" / "cpm25.npy").is_file():
			return candidate.resolve()

	raise FileNotFoundError(
		"Could not auto-resolve raw root. Pass --raw_root explicitly to folder containing APRIL_16/cpm25.npy."
	)


def _resolve_test_input(test_input_arg: str | None) -> Path:
	if test_input_arg:
		test_dir = Path(test_input_arg).expanduser().resolve()
		if (test_dir / "cpm25.npy").is_file():
			return test_dir
		raise FileNotFoundError(f"Invalid --test_input: {test_dir} (missing cpm25.npy)")

	candidates = [
		Path("/kaggle/input/competitions/anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd/aisehack-theme-2/test_in"),
		Path("/kaggle/input/datasets/ronitraj1/aisehack-finals/aisehack-theme-2/test_in"),
	]
	for candidate in candidates:
		if (candidate / "cpm25.npy").is_file():
			return candidate.resolve()

	raise FileNotFoundError("Could not auto-resolve test input. Pass --test_input explicitly.")


def _resolve_model_path(model_path_arg: str | None, prefer: str, project_root: Path) -> Path:
	if model_path_arg:
		model_path = Path(model_path_arg).expanduser().resolve()
		if model_path.is_file():
			return model_path
		raise FileNotFoundError(f"--model_path not found: {model_path}")

	name = "best.pt" if prefer == "best" else "last.pt"
	candidates = [
		Path.cwd() / name,
		project_root / name,
		project_root / "downloaded_output" / "experiments" / "exp07_hybrid_single_gpu_amp" / "checkpoints" / name,
		project_root / "experiments" / "exp07_hybrid_single_gpu_amp" / "checkpoints" / name,
	]
	for candidate in candidates:
		if candidate.is_file():
			return candidate.resolve()

	raise FileNotFoundError(
		f"Could not auto-resolve {name}. Pass --model_path explicitly to downloaded checkpoint file."
	)


def _run_command(cmd: list[str], cwd: Path) -> None:
	print("RUN:", " ".join(cmd))
	env = os.environ.copy()
	env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
	subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
	args = parse_args()
	project_root = _resolve_project_root()
	raw_root = _resolve_raw_root(args.raw_root)
	test_input = _resolve_test_input(args.test_input)
	model_path = _resolve_model_path(args.model_path, args.prefer, project_root)

	default_work = Path("/kaggle/working") / "exp07_preprocess_infer"
	work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work.resolve()
	work_dir.mkdir(parents=True, exist_ok=True)

	exp_dir = work_dir / "experiments" / "exp07_hybrid_single_gpu_amp"
	preprocessing_dir = exp_dir / "preprocessing"
	preds_path = Path(args.preds_out).expanduser().resolve() if args.preds_out else (work_dir / "preds.npy")

	train_base = project_root / "configs" / "train.yaml"
	infer_base = project_root / "configs" / "infer.yaml"
	runtime_train_cfg = work_dir / "exp07_train_runtime.yaml"
	runtime_infer_cfg = work_dir / "exp07_infer_runtime.yaml"

	if not train_base.is_file() or not infer_base.is_file():
		raise FileNotFoundError("Missing base config files under configs/train.yaml or configs/infer.yaml")

	with open(train_base, "r", encoding="utf-8") as f:
		train_cfg = yaml.safe_load(f)
	with open(infer_base, "r", encoding="utf-8") as f:
		infer_cfg = yaml.safe_load(f)

	stats_path = preprocessing_dir / "normalization_stats.json"

	# Match EXP07 model/config settings
	train_cfg["paths"]["raw_path"] = str(raw_root)
	train_cfg["paths"]["save_dir"] = str(exp_dir / "logs" / "log.json")
	train_cfg["paths"]["model_save_path"] = str(exp_dir / "checkpoints" / "best.pt")
	train_cfg["preprocessing"] = {"stats_path": str(stats_path)}

	train_cfg["model"]["name"] = "hybrid_ensemble"
	train_cfg["model"]["width"] = 64
	train_cfg["model"]["modes1"] = 24
	train_cfg["model"]["modes2"] = 24
	train_cfg["model"]["convlstm_hidden"] = 64

	train_cfg["training"]["batch_size"] = 4
	train_cfg["training"]["use_amp"] = True
	train_cfg["training"]["max_grad_norm"] = 1.0

	train_cfg["data"]["partial_val_month"] = "JULY_16"
	train_cfg["data"]["partial_val_days"] = 20
	train_cfg["multi_gpu"] = {
		"enabled": False,
		"backend": "data_parallel",
		"min_gpus": 2,
		"device_ids": None,
	}

	infer_cfg["paths"]["input_loc"] = str(test_input)
	infer_cfg["paths"]["model_path"] = str(model_path)
	infer_cfg["paths"]["output_loc"] = str(preds_path)
	infer_cfg["preprocessing"] = {"stats_path": str(stats_path)}

	infer_cfg["model"]["name"] = "hybrid_ensemble"
	infer_cfg["model"]["width"] = 64
	infer_cfg["model"]["modes1"] = 24
	infer_cfg["model"]["modes2"] = 24
	infer_cfg["model"]["convlstm_hidden"] = 64
	infer_cfg["model"]["infer_season_idx"] = 2

	if "inference" not in infer_cfg or not isinstance(infer_cfg["inference"], dict):
		infer_cfg["inference"] = {}
	infer_cfg["inference"]["batch_size"] = 1

	with open(runtime_train_cfg, "w", encoding="utf-8") as f:
		yaml.safe_dump(train_cfg, f, sort_keys=False)
	with open(runtime_infer_cfg, "w", encoding="utf-8") as f:
		yaml.safe_dump(infer_cfg, f, sort_keys=False)

	print("Project root:", project_root)
	print("Raw root:", raw_root)
	print("Test input:", test_input)
	print("Model path:", model_path)
	print("Work dir:", work_dir)
	print("Runtime train config:", runtime_train_cfg)
	print("Runtime infer config:", runtime_infer_cfg)

	if not args.skip_preprocess:
		preprocessing_dir.mkdir(parents=True, exist_ok=True)
		preprocess_cmd = [
			args.python_bin,
			"scripts/preprocess_data.py",
			"--config",
			str(runtime_train_cfg),
			"--raw_root",
			str(raw_root),
			"--out_dir",
			str(preprocessing_dir),
		]
		_run_command(preprocess_cmd, cwd=project_root)
	else:
		print("Skipping preprocessing (--skip_preprocess).")

	if not stats_path.is_file():
		raise FileNotFoundError(f"Missing normalization stats after preprocessing: {stats_path}")

	infer_cmd = [
		args.python_bin,
		"scripts/infer.py",
		"--config",
		str(runtime_infer_cfg),
		"--input_loc",
		str(test_input),
		"--model_path",
		str(model_path),
		"--output_loc",
		str(preds_path),
	]
	_run_command(infer_cmd, cwd=project_root)

	if not preds_path.is_file():
		raise FileNotFoundError(f"Inference finished but output missing: {preds_path}")

	preds = np.load(preds_path)
	print("Predictions saved:", preds_path)
	print("Predictions shape:", preds.shape)
	if preds.shape != (218, 140, 124, 16):
		raise ValueError(f"Unexpected preds shape {preds.shape}; expected (218, 140, 124, 16)")
	print("✅ Inference completed with valid submission shape.")


if __name__ == "__main__":
	main()

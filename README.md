# PM2.5 Phase 2 — Spatio-Temporal Air Quality Forecasting

## Competition overview

Forecast PM2.5 concentration 16 hours ahead from 10 hours of lookback on a 140×124 grid. Inference only sees the lookback window (no future exogenous inputs).

## Metric

Composite of Global SMAPE, Episode SMAPE, and Episode Correlation (see competition page).

## Pipeline

1. **Train** — `scripts/train.py` reads competition **raw** monthly folders (`<MONTH>/*.npy`), builds sliding windows, derives in-window features (wind speed, rolling cpm25 stats, etc.) **without** normalizers or `preprocess_stats.json`.
2. **Infer** — `scripts/infer.py` loads test `.npy` (10-hour lookback per sample), builds the same channel stack, runs the checkpoint, writes `preds.npy` in physical concentration units.

Configs: [`configs/train.yaml`](configs/train.yaml), [`configs/infer.yaml`](configs/infer.yaml). Override raw data location with `--raw_path` (train) or patch `paths` in YAML.

```bash
pip install -r requirements.txt
cd Ronit_new
python scripts/train.py --config configs/train.yaml --raw_path /path/to/competition/raw
python scripts/infer.py --config configs/infer.yaml --model_path /path/to/best.pt --input_loc /path/to/test_in --output_loc /path/to/out/
```

On Kaggle, use `kaggle_notebook/exp01_baseline.ipynb` (copies paths into YAML) or run the same CLI from `/kaggle/working/...` after attaching the code dataset.

## Kaggle dataset push

```bash
cd /path/to/Ronit_new
make push MSG="describe what changed"
```

First time only: `make push-create` (requires `kaggle.json` and conda env with Kaggle CLI if you use the Makefile as written).

## Layout

```
├── configs/          train.yaml, infer.yaml
├── models/           FNO2D
├── scripts/          train.py, infer.py
├── src/
│   ├── data/         RawWindowDataset (raw → windows)
│   └── utils/        config, preprocessing, competition_metrics, optim, losses
├── kaggle_notebook/  exp01_baseline.ipynb
├── experiments/      checkpoints / logs (default under paths in YAML)
└── submissions/      preds outputs
```

## Experiments

Use a separate YAML per run (e.g. `configs/experiments/foo/train.yaml`) or copy `configs/train.yaml` into `experiments/<name>/config_snapshot.yaml` after each change. Keep **checkpoint + YAML + feature lists** together so inference matches training.

## Requirements note

Install PyTorch separately for your CUDA/CPU platform ([pytorch.org](https://pytorch.org)) if `pip install -r requirements.txt` does not pull the right wheel.

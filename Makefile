.PHONY: train infer push push-create status notebook-status download-preds leaderboard clean help

RONIT_DIR := $(shell pwd)
DATASET_SLUG := ronitraj1/ronit-pm25-phase2-src
NOTEBOOK_SLUG := ronitraj1/ronit-pm25-exp01-baseline
CONDA_ENV := aisehack

push-create:
	conda run -n $(CONDA_ENV) kaggle datasets create -p $(RONIT_DIR) --dir-mode zip
	@echo "Dataset created at https://www.kaggle.com/datasets/$(DATASET_SLUG)"

push:
	@if [ -z "$(MSG)" ]; then echo "MSG is required: make push MSG='describe change'"; exit 1; fi
	conda run -n $(CONDA_ENV) kaggle datasets version -p $(RONIT_DIR) --dir-mode zip -m "$(MSG)"
	@echo "https://www.kaggle.com/datasets/$(DATASET_SLUG)"

status:
	conda run -n $(CONDA_ENV) kaggle datasets status $(DATASET_SLUG)

notebook-status:
	conda run -n $(CONDA_ENV) kaggle kernels status $(NOTEBOOK_SLUG)

download-preds:
	conda run -n $(CONDA_ENV) kaggle kernels output $(NOTEBOOK_SLUG) -p $(RONIT_DIR)/submissions/
	@echo "preds.npy → submissions/"

leaderboard:
	conda run -n $(CONDA_ENV) kaggle competitions leaderboard aisehack-theme-2 --show

## Local / Kaggle working copy: run from repo root with raw data path set in configs/train.yaml or:
##   make train ARGS='--raw_path /path/to/raw'
train:
	cd $(RONIT_DIR) && python scripts/train.py $(ARGS)

infer:
	cd $(RONIT_DIR) && python scripts/infer.py $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "make train ARGS='--config configs/train.yaml --raw_path ...'"
	@echo "make infer ARGS='--config configs/infer.yaml --model_path ... --input_loc ...'"
	@echo "make push MSG='reason'"

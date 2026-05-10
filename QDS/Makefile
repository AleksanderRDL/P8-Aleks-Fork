SHELL := /bin/bash

REPO_ROOT := $(abspath ..)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python
QDS_REQUIREMENTS ?= requirements.txt

RESULTS_DIR ?= artifacts/results/smoke_synthetic
CSV_RESULTS_DIR ?= artifacts/results/smoke_cleaned_csv
CLEANED_CSV ?=

.PHONY: help check-env install install-cpu install-cuda test smoke smoke-csv

help:
	@echo "Targets:"
	@echo "  check-env   Print Python/package versions and run pip check"
	@echo "  install     Install QDS_REQUIREMENTS into PYTHON"
	@echo "  install-cpu Install the CPU/generic QDS profile"
	@echo "  install-cuda Install the CUDA cu130 QDS profile"
	@echo "  test        Run the QDS pytest suite"
	@echo "  smoke       Run a tiny synthetic training/evaluation experiment"
	@echo "  smoke-csv   Run a tiny cleaned-CSV training/evaluation experiment"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  QDS_REQUIREMENTS=$(QDS_REQUIREMENTS)"
	@echo "  CLEANED_CSV=$(CLEANED_CSV)"

check-env:
	$(PYTHON) -V
	$(PYTHON) -m pip check
	$(PYTHON) -c 'import matplotlib, numpy, pandas, pytest, torch, tqdm; print("torch", torch.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("matplotlib", matplotlib.__version__); print("pytest", pytest.__version__); print("tqdm", tqdm.__version__)'

install:
	$(PYTHON) -m pip install -r $(QDS_REQUIREMENTS)

install-cpu:
	$(PYTHON) -m pip install -r requirements-cpu.txt

install-cuda:
	$(PYTHON) -m pip install -r requirements-cuda-cu130.txt

test:
	$(PYTHON) -m pytest tests

smoke:
	$(PYTHON) -m src.experiments.run_ais_experiment \
		--n_ships 4 \
		--n_points 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(RESULTS_DIR)"

smoke-csv:
	@if [ -z "$(CLEANED_CSV)" ]; then echo "Set CLEANED_CSV to a cleaned AIS file, for example: make smoke-csv CLEANED_CSV=$(REPO_ROOT)/AISDATA/cleaned/<file-or-directory>"; exit 2; fi
	$(PYTHON) -m src.experiments.run_ais_experiment \
		--csv_path "$(CLEANED_CSV)" \
		--cache_dir artifacts/cache/smoke_cleaned_csv \
		--max_segments 8 \
		--max_points_per_segment 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(CSV_RESULTS_DIR)"

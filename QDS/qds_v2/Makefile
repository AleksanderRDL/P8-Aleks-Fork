SHELL := /bin/bash

REPO_ROOT := $(abspath ../..)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python

RESULTS_DIR ?= artifacts/results/smoke_synthetic
CSV_RESULTS_DIR ?= artifacts/results/smoke_cleaned_csv
CLEANED_CSV ?= $(REPO_ROOT)/AISDATA/cleaned/aisdk-2026-01-01.cleaned.csv

.PHONY: help check-env install test smoke smoke-csv

help:
	@echo "Targets:"
	@echo "  check-env   Print Python/package versions and run pip check"
	@echo "  install     Install pinned qds_v2 requirements into PYTHON"
	@echo "  test        Run the qds_v2 pytest suite"
	@echo "  smoke       Run a tiny synthetic training/evaluation experiment"
	@echo "  smoke-csv   Run a tiny cleaned-CSV training/evaluation experiment"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  CLEANED_CSV=$(CLEANED_CSV)"

check-env:
	$(PYTHON) -V
	$(PYTHON) -m pip check
	$(PYTHON) -c 'import matplotlib, numpy, pandas, pytest, torch, tqdm; print("torch", torch.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("matplotlib", matplotlib.__version__); print("pytest", pytest.__version__); print("tqdm", tqdm.__version__)'

install:
	$(PYTHON) -m pip install -r requirements.txt

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
		$(PYTHON) -m src.experiments.run_ais_experiment \
		--csv_path "$(CLEANED_CSV)" \
		--max_segments 8 \
		--max_points_per_segment 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(CSV_RESULTS_DIR)"

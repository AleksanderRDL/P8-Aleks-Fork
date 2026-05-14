SHELL := /bin/bash

REPO_ROOT := $(abspath ..)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python
TYPECHECK_PATHS ?= data evaluation experiments models queries simplification training scripts tests

RESULTS_DIR ?= artifacts/results/smoke_synthetic
CSV_RESULTS_DIR ?= artifacts/results/smoke_cleaned_csv
CLEANED_CSV ?=
BENCHMARK_SESSION ?= qds-range-benchmark
BENCHMARK_QUEUE_SESSION ?= qds-benchmark-queue
BENCHMARK_RUN_ID ?=
BENCHMARK_QUEUE_ID ?=
BENCHMARK_SEEDS ?= 42,43,44
BENCHMARK_CHILD_EXTRA_ARGS ?=
BENCHMARK_PLAN_FILE ?=
BENCHMARK_CONTINUE_ON_FAILURE ?= 0
ATTACH ?=
BENCHMARK_PROFILE ?= range_workload_aware_diagnostic
BENCHMARK_FAMILY ?= artifacts/benchmarks/legacy_range_workload_aware_diagnostic
BENCHMARK_CACHE ?= artifacts/cache/legacy_range_workload_aware_diagnostic
CONFIRM ?=

.PHONY: help check-env install lint test typecheck smoke smoke-csv benchmark-preflight benchmark-queue-preflight list-runs clean-smoke-artifacts range-benchmark-tmux range-benchmark-queue-tmux

help:
	@echo "Targets:"
	@echo "  check-env   Print Python/package versions and run pip check"
	@echo "  install     Install the root project with dev dependencies into PYTHON"
	@echo "  lint        Run Ruff correctness lint"
	@echo "  test        Run the QDS pytest suite"
	@echo "  typecheck   Run Pyright on TYPECHECK_PATHS"
	@echo "  smoke       Run a tiny synthetic training/evaluation experiment"
	@echo "  smoke-csv   Run a tiny cleaned-CSV training/evaluation experiment"
	@echo "  benchmark-preflight  Check tmux/GPU/data/artifact prerequisites"
	@echo "  benchmark-queue-preflight  Check prerequisites using BENCHMARK_QUEUE_SESSION"
	@echo "  list-runs   List benchmark runs from BENCHMARK_FAMILY"
	@echo "  clean-smoke-artifacts  Dry-run smoke artifact cleanup; set CONFIRM=1 to delete"
	@echo "  range-benchmark-tmux  Launch the range workload-aware diagnostic benchmark in tmux with monitoring"
	@echo "  range-benchmark-queue-tmux  Launch a sequential multi-seed range benchmark queue in tmux"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  TYPECHECK_PATHS=$(TYPECHECK_PATHS)"
	@echo "  CLEANED_CSV=$(CLEANED_CSV)"
	@echo "  BENCHMARK_SESSION=$(BENCHMARK_SESSION)"
	@echo "  BENCHMARK_QUEUE_SESSION=$(BENCHMARK_QUEUE_SESSION)"
	@echo "  BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID)"
	@echo "  BENCHMARK_QUEUE_ID=$(BENCHMARK_QUEUE_ID)"
	@echo "  BENCHMARK_SEEDS=$(BENCHMARK_SEEDS)"
	@echo "  BENCHMARK_CHILD_EXTRA_ARGS=$(BENCHMARK_CHILD_EXTRA_ARGS)"
	@echo "  BENCHMARK_PLAN_FILE=$(BENCHMARK_PLAN_FILE)"
	@echo "  BENCHMARK_CONTINUE_ON_FAILURE=$(BENCHMARK_CONTINUE_ON_FAILURE)"
	@echo "  ATTACH=$(ATTACH)"
	@echo "  BENCHMARK_PROFILE=$(BENCHMARK_PROFILE)"
	@echo "  BENCHMARK_FAMILY=$(BENCHMARK_FAMILY)"
	@echo "  BENCHMARK_CACHE=$(BENCHMARK_CACHE)"

check-env:
	$(PYTHON) -V
	$(PYTHON) -m pip check
	$(PYTHON) -c 'from importlib.metadata import version; import numpy, pandas, pyarrow, pyright, pyspark, pytest, torch; print("torch", torch.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("pyarrow", pyarrow.__version__); print("pyspark", pyspark.__version__); print("python-dotenv", version("python-dotenv")); print("psycopg", version("psycopg")); print("pytest", pytest.__version__); print("pyright", pyright.__version__); print("ruff", version("ruff"))'

install:
	cd $(REPO_ROOT) && $(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check data evaluation experiments models queries simplification training scripts tests

test:
	$(PYTHON) -m pytest tests

typecheck:
	$(PYTHON) -m pyright $(TYPECHECK_PATHS)

smoke:
	$(PYTHON) -m experiments.run_ais_experiment \
		--n_ships 4 \
		--n_points 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(RESULTS_DIR)"

smoke-csv:
	@if [ -z "$(CLEANED_CSV)" ]; then echo "Set CLEANED_CSV to a cleaned AIS file, for example: make smoke-csv CLEANED_CSV=$(REPO_ROOT)/AISDATA/cleaned/<file-or-directory>"; exit 2; fi
	$(PYTHON) -m experiments.run_ais_experiment \
		--csv_path "$(CLEANED_CSV)" \
		--cache_dir artifacts/cache/smoke_cleaned_csv \
		--max_segments 8 \
		--max_points_per_segment 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(CSV_RESULTS_DIR)"

benchmark-preflight:
	SESSION="$(BENCHMARK_SESSION)" PYTHON="$(PYTHON)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/benchmark_preflight.sh

benchmark-queue-preflight:
	SESSION="$(BENCHMARK_QUEUE_SESSION)" PYTHON="$(PYTHON)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/benchmark_preflight.sh

list-runs:
	$(PYTHON) scripts/list_benchmark_runs.py --family "$(BENCHMARK_FAMILY)"

clean-smoke-artifacts:
	scripts/clean_smoke_artifacts.sh $(if $(CONFIRM),--yes,)

range-benchmark-tmux:
	SESSION="$(BENCHMARK_SESSION)" RUN_ID="$(BENCHMARK_RUN_ID)" ATTACH="$(ATTACH)" PROFILE="$(BENCHMARK_PROFILE)" PYTHON="$(PYTHON)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/run_range_benchmark_tmux.sh

range-benchmark-queue-tmux:
	SESSION="$(BENCHMARK_QUEUE_SESSION)" QUEUE_ID="$(BENCHMARK_QUEUE_ID)" ATTACH="$(ATTACH)" PROFILE="$(BENCHMARK_PROFILE)" SEEDS="$(BENCHMARK_SEEDS)" CHILD_EXTRA_ARGS="$(BENCHMARK_CHILD_EXTRA_ARGS)" PLAN_FILE="$(BENCHMARK_PLAN_FILE)" CONTINUE_ON_FAILURE="$(BENCHMARK_CONTINUE_ON_FAILURE)" PYTHON="$(PYTHON)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/run_benchmark_queue_tmux.sh

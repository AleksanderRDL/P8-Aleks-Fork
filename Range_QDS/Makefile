SHELL := /bin/bash

REPO_ROOT := $(abspath ..)
UV ?= uv
UV_GROUP ?= dev
UV_GROUP_FLAGS ?= --group $(UV_GROUP)
UV_RUN := cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) --

TYPECHECK_PATHS ?= Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests
TEST_PATHS ?= Range_QDS/tests
LINT_PATHS ?= Range_QDS/data Range_QDS/evaluation Range_QDS/experiments Range_QDS/models Range_QDS/queries Range_QDS/simplification Range_QDS/training Range_QDS/scripts Range_QDS/tests
RUN ?= artifacts/results/latest/example_run.json

RESULTS_DIR ?= Range_QDS/artifacts/results/smoke_synthetic
CSV_RESULTS_DIR ?= Range_QDS/artifacts/results/smoke_cleaned_csv
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

.PHONY: help setup sync lock-check check-env lint lint-yaml test typecheck smoke smoke-csv benchmark-preflight benchmark-queue-preflight list-runs clean-smoke-artifacts range-benchmark-tmux range-benchmark-queue-tmux inspect-run inspect-gates inspect-scores inspect-causality inspect-generator inspect-predictability summarize-run test-property test-regression

help:
	@echo "Targets:"
	@echo "  setup       Alias for sync"
	@echo "  sync        Sync the uv environment with dependency group $(UV_GROUP)"
	@echo "  lock-check  Verify uv.lock is current"
	@echo "  check-env   Print uv/Python/package versions and run pip check"
	@echo "  lint        Run Ruff correctness lint"
	@echo "  lint-yaml   Run yamllint on repository YAML files"
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
	@echo "  inspect-run  Print compact jq run summary from RUN"
	@echo "  inspect-gates  Print jq gate diagnostics from RUN"
	@echo "  inspect-scores  Print jq QueryUsefulV1 scores from RUN"
	@echo "  inspect-causality  Print jq causality diagnostics from RUN"
	@echo "  inspect-generator  Print jq generator diagnostics from RUN"
	@echo "  inspect-predictability  Print jq predictability diagnostics from RUN"
	@echo "  summarize-run  Print Rich run summary from RUN"
	@echo "  test-property  Run Hypothesis property tests"
	@echo "  test-regression  Run report/schema regression tests"
	@echo ""
	@echo "Variables:"
	@echo "  UV=$(UV)"
	@echo "  UV_GROUP=$(UV_GROUP)"
	@echo "  TYPECHECK_PATHS=$(TYPECHECK_PATHS)"
	@echo "  RUN=$(RUN)"
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

setup: sync

sync:
	cd $(REPO_ROOT) && $(UV) sync $(UV_GROUP_FLAGS)

lock-check:
	cd $(REPO_ROOT) && $(UV) lock --check

check-env:
	cd $(REPO_ROOT) && $(UV) --version
	$(UV_RUN) python -V
	$(UV_RUN) python -m pip check
	$(UV_RUN) python -c 'from importlib.metadata import version; import numpy, pandas, pyarrow, pyright, pyspark, pytest, torch; print("torch", torch.__version__); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("pyarrow", pyarrow.__version__); print("pyspark", pyspark.__version__); print("python-dotenv", version("python-dotenv")); print("psycopg", version("psycopg")); print("pytest", pytest.__version__); print("pyright", pyright.__version__); print("ruff", version("ruff"))'

lint:
	$(UV_RUN) ruff check $(LINT_PATHS)

lint-yaml:
	$(UV_RUN) yamllint .

test:
	$(UV_RUN) pytest $(TEST_PATHS)

typecheck:
	$(UV_RUN) pyright $(TYPECHECK_PATHS)

smoke:
	$(UV_RUN) python -m experiments.run_ais_experiment \
		--n_ships 4 \
		--n_points 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(RESULTS_DIR)"

smoke-csv:
	@if [ -z "$(CLEANED_CSV)" ]; then echo "Set CLEANED_CSV to a cleaned AIS file, for example: make smoke-csv CLEANED_CSV=$(REPO_ROOT)/AISDATA/cleaned/<file-or-directory>"; exit 2; fi
	$(UV_RUN) python -m experiments.run_ais_experiment \
		--csv_path "$(CLEANED_CSV)" \
		--cache_dir Range_QDS/artifacts/cache/smoke_cleaned_csv \
		--max_segments 8 \
		--max_points_per_segment 24 \
		--n_queries 4 \
		--epochs 1 \
		--workload range \
		--compression_ratio 0.5 \
		--results_dir "$(CSV_RESULTS_DIR)"

benchmark-preflight:
	SESSION="$(BENCHMARK_SESSION)" UV="$(UV)" UV_GROUP="$(UV_GROUP)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/benchmark_preflight.sh

benchmark-queue-preflight:
	SESSION="$(BENCHMARK_QUEUE_SESSION)" UV="$(UV)" UV_GROUP="$(UV_GROUP)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/benchmark_preflight.sh

list-runs:
	$(UV_RUN) python Range_QDS/scripts/list_benchmark_runs.py --family "Range_QDS/$(BENCHMARK_FAMILY)"

clean-smoke-artifacts:
	scripts/clean_smoke_artifacts.sh $(if $(CONFIRM),--yes,)

range-benchmark-tmux:
	SESSION="$(BENCHMARK_SESSION)" RUN_ID="$(BENCHMARK_RUN_ID)" ATTACH="$(ATTACH)" PROFILE="$(BENCHMARK_PROFILE)" UV="$(UV)" UV_GROUP="$(UV_GROUP)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/run_range_benchmark_tmux.sh

range-benchmark-queue-tmux:
	SESSION="$(BENCHMARK_QUEUE_SESSION)" QUEUE_ID="$(BENCHMARK_QUEUE_ID)" ATTACH="$(ATTACH)" PROFILE="$(BENCHMARK_PROFILE)" SEEDS="$(BENCHMARK_SEEDS)" CHILD_EXTRA_ARGS="$(BENCHMARK_CHILD_EXTRA_ARGS)" PLAN_FILE="$(BENCHMARK_PLAN_FILE)" CONTINUE_ON_FAILURE="$(BENCHMARK_CONTINUE_ON_FAILURE)" UV="$(UV)" UV_GROUP="$(UV_GROUP)" ARTIFACT_ROOT="$(BENCHMARK_FAMILY)" CACHE_DIR="$(BENCHMARK_CACHE)" scripts/run_benchmark_queue_tmux.sh

inspect-run:
	jq -f scripts/jq/run_summary.jq $(RUN)

inspect-gates:
	jq -f scripts/jq/gates.jq $(RUN)

inspect-scores:
	jq -f scripts/jq/scores.jq $(RUN)

inspect-causality:
	jq -f scripts/jq/causality.jq $(RUN)

inspect-generator:
	jq -f scripts/jq/generator_health.jq $(RUN)

inspect-predictability:
	jq -f scripts/jq/predictability.jq $(RUN)

summarize-run:
	@run="$(RUN)"; case "$$run" in /*|Range_QDS/*) root_run="$$run" ;; *) root_run="Range_QDS/$$run" ;; esac; cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) -- python Range_QDS/scripts/summarize_run.py "$$root_run"

test-property:
	$(UV_RUN) pytest Range_QDS/tests/property

test-regression:
	$(UV_RUN) pytest Range_QDS/tests/regression

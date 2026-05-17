SHELL := /bin/bash

REPO_ROOT := $(abspath .)
RANGE_QDS_DIR := $(REPO_ROOT)/Range_QDS
UV ?= uv
UV_GROUP ?= dev
UV_GROUP_FLAGS ?= --group $(UV_GROUP)
UV_RUN := cd $(REPO_ROOT) && $(UV) run $(UV_GROUP_FLAGS) --
CSV ?=
QUERY_ARGS ?= --help

.PHONY: help setup sync lock-check check-env pipeline qds-check-env lint lint-yaml test typecheck smoke smoke-csv db-up db-down db-reset db-logs db-smoke db-import db-query

help:
	@echo "Targets:"
	@echo "  setup            Alias for sync"
	@echo "  sync             Sync the uv environment with dependency group $(UV_GROUP)"
	@echo "  lock-check       Verify uv.lock is current"
	@echo "  check-env        Print uv/Python versions and run pip check"
	@echo "  pipeline         Run AIS cleaning pipeline"
	@echo "  qds-check-env    Print QDS Python/package versions and run pip check"
	@echo "  lint             Run QDS Ruff correctness lint"
	@echo "  lint-yaml        Run yamllint on repository YAML files"
	@echo "  test             Run the QDS pytest suite"
	@echo "  typecheck        Run QDS Pyright"
	@echo "  smoke            Run a tiny QDS synthetic training/evaluation experiment"
	@echo "  smoke-csv        Run a tiny QDS cleaned-CSV experiment"
	@echo "  db-up            Start PostGIS service"
	@echo "  db-down          Stop PostGIS service"
	@echo "  db-reset         Recreate PostGIS volume and schema"
	@echo "  db-logs          Tail PostGIS logs"
	@echo "  db-smoke         Run DB smoke test"
	@echo "  db-import        Import cleaned AIS CSV (override with CSV=...)"
	@echo "  db-query         Run range query script (override with QUERY_ARGS=...)"

setup: sync

sync:
	cd $(REPO_ROOT) && $(UV) sync $(UV_GROUP_FLAGS)

lock-check:
	cd $(REPO_ROOT) && $(UV) lock --check

check-env:
	cd $(REPO_ROOT) && $(UV) --version
	$(UV_RUN) python -V
	$(UV_RUN) python -m pip check

pipeline:
	$(UV_RUN) python main.py

qds-check-env:
	$(MAKE) -C $(RANGE_QDS_DIR) check-env UV="$(UV)" UV_GROUP="$(UV_GROUP)"

lint:
	$(MAKE) -C $(RANGE_QDS_DIR) lint UV="$(UV)" UV_GROUP="$(UV_GROUP)"

lint-yaml:
	$(UV_RUN) yamllint .

test:
	$(MAKE) -C $(RANGE_QDS_DIR) test UV="$(UV)" UV_GROUP="$(UV_GROUP)"

typecheck:
	$(MAKE) -C $(RANGE_QDS_DIR) typecheck UV="$(UV)" UV_GROUP="$(UV_GROUP)"

smoke:
	$(MAKE) -C $(RANGE_QDS_DIR) smoke UV="$(UV)" UV_GROUP="$(UV_GROUP)"

smoke-csv:
	$(MAKE) -C $(RANGE_QDS_DIR) smoke-csv UV="$(UV)" UV_GROUP="$(UV_GROUP)" CLEANED_CSV="$(CLEANED_CSV)"

db-up:
	docker compose -f db/compose.yaml up -d

db-down:
	docker compose -f db/compose.yaml down

db-reset:
	docker compose -f db/compose.yaml down -v
	docker compose -f db/compose.yaml up -d

db-logs:
	docker compose -f db/compose.yaml logs -f postgis

db-smoke:
	$(UV_RUN) python db/smoke_test_db.py

db-import:
	@if [ -z "$(CSV)" ]; then echo "Set CSV to a cleaned AIS file, for example: make db-import CSV=AISDATA/cleaned/<file-or-directory>"; exit 2; fi
	$(UV_RUN) python db/import_ais_csv.py $(CSV)

db-query:
	$(UV_RUN) python db/run_range_query.py $(QUERY_ARGS)

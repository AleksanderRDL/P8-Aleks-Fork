SHELL := /bin/bash

REPO_ROOT := $(abspath .)
QDS_DIR := $(REPO_ROOT)/QDS
PYTHON ?= python
QDS_PYTHON ?= $(REPO_ROOT)/.venv/bin/python
CSV ?=
QUERY_ARGS ?= --help

.PHONY: help setup install pipeline qds-check-env test typecheck smoke smoke-csv db-up db-down db-reset db-logs db-smoke db-import db-query

help:
	@echo "Targets:"
	@echo "  setup            Create .venv and install Python dependencies"
	@echo "  install          Install Python dependencies into current interpreter"
	@echo "  pipeline         Run AIS cleaning pipeline"
	@echo "  qds-check-env    Print QDS Python/package versions and run pip check"
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

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) main.py

qds-check-env:
	$(MAKE) -C $(QDS_DIR) check-env PYTHON="$(QDS_PYTHON)"

test:
	$(MAKE) -C $(QDS_DIR) test PYTHON="$(QDS_PYTHON)"

typecheck:
	$(MAKE) -C $(QDS_DIR) typecheck PYTHON="$(QDS_PYTHON)"

smoke:
	$(MAKE) -C $(QDS_DIR) smoke PYTHON="$(QDS_PYTHON)"

smoke-csv:
	$(MAKE) -C $(QDS_DIR) smoke-csv PYTHON="$(QDS_PYTHON)" CLEANED_CSV="$(CLEANED_CSV)"

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
	$(PYTHON) db/smoke_test_db.py

db-import:
	@if [ -z "$(CSV)" ]; then echo "Set CSV to a cleaned AIS file, for example: make db-import CSV=AISDATA/cleaned/<file-or-directory>"; exit 2; fi
	$(PYTHON) db/import_ais_csv.py $(CSV)

db-query:
	$(PYTHON) db/run_range_query.py $(QUERY_ARGS)

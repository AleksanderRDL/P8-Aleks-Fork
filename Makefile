SHELL := /bin/bash

PYTHON ?= python
CSV ?=
QUERY_ARGS ?= --help

.PHONY: help setup install pipeline db-up db-down db-reset db-logs db-smoke db-import db-query

help:
	@echo "Targets:"
	@echo "  setup            Create .venv and install Python dependencies"
	@echo "  install          Install Python dependencies into current interpreter"
	@echo "  pipeline         Run AIS cleaning pipeline"
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

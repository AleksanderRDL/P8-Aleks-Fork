# AIS-QDS v2

AIS-QDS v2 is the current shift-aware rebuild of the AIS query-driven simplification pipeline. It loads AIS trajectories, generates typed query workloads, trains a query-conditioned ranking model, and evaluates the resulting simplifier against learned and geometric baselines under matched and shifted workloads.

## What Is In This Folder

- `requirements.txt` - Python dependencies for the v2 stack.
- `src/` - package code for loading data, building queries, training models, and running experiments.
- `tests/` - regression tests that guard the rebuild.
- `results/` - retained reference outputs and benchmark artifacts.

## Quick Start

```bash
cd QDS
pip install -r requirements.txt
python -m src.experiments.run_ais_experiment --n_queries 128 --epochs 6 --workload range
```

## Environment And Smoke Checks

The sprint environment is the repository-level virtual environment at `../.venv`
when commands are run from `QDS`. Requirements are pinned in
`requirements.txt` for the local QDS checks.

```bash
cd QDS
../.venv/bin/python -m pip install -r requirements.txt
make check-env
make test
```

Use the local Makefile for repeatable smoke runs:

```bash
# Tiny synthetic train/eval run. Outputs go to artifacts/results/smoke_synthetic.
make smoke

# Tiny cleaned-CSV smoke run against AISDATA/cleaned with segmented Parquet cache.
make smoke-csv
```

## Dependency Profiles

The repository now keeps dependency concerns separate:

- Root/base AIS pipeline and database tooling: [`../requirements.txt`](../requirements.txt).
- QDS shared non-Torch dependencies: [`requirements-common.txt`](requirements-common.txt).
- QDS CPU/generic Torch profile: [`requirements-cpu.txt`](requirements-cpu.txt).
- QDS CUDA reference profile: [`requirements-cuda-cu130.txt`](requirements-cuda-cu130.txt).

[`requirements.txt`](requirements.txt) remains a compatibility alias for the
current QDS CUDA sprint profile. The reference CUDA stack observed for this
machine is:

```text
torch 2.11.0+cu130
CUDA runtime 13.0
triton 3.6.0
```

Install the intended profile explicitly when changing environments:

```bash
cd QDS
../.venv/bin/python -m pip install -r requirements-cuda-cu130.txt
# or, for CPU/generic environments:
../.venv/bin/python -m pip install -r requirements-cpu.txt
```

Check the active environment before benchmarking:

```bash
cd QDS
../.venv/bin/python -m pip check
../.venv/bin/python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
```

The current `.venv` uses Python 3.14. Keep it as the local reference unless it
blocks CUDA package availability; if that happens, create a separate Python
3.12 environment instead of mutating this one.

## Runtime Benchmarks And GPU Telemetry

Use the runtime benchmark wrapper before accepting optimization changes. It
runs the existing experiment/inference CLIs, captures stdout, parses phase and
epoch timings, records git state and dependency versions, and writes a stable
JSON artifact.

```bash
cd QDS
../.venv/bin/python -m src.experiments.benchmark_runtime --help

# Cheap synthetic training benchmark.
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile small \
  --results_dir artifacts/benchmarks/runtime_small

# Same profile with TF32-enabled matmul for paired speed/F1 checks.
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile small \
  --results_dir artifacts/benchmarks/runtime_small_tf32 \
  --float32_matmul_precision high \
  --allow_tf32

# Same profile with CUDA BF16 autocast. Losses and metrics remain FP32.
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile small \
  --results_dir artifacts/benchmarks/runtime_small_bf16 \
  --amp_mode bf16

# Training batch-size sweep. Use a larger profile/dataset for final decisions.
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile medium \
  --train_batch_sizes 16,32,64,128 \
  --results_dir artifacts/benchmarks/batch_size_sweep

# Pure-workload matrix across runtime/batch/checkpoint-F1 variants.
../.venv/bin/python -m src.experiments.benchmark_matrix \
  --profile medium \
  --workloads range,knn,similarity,clustering \
  --results_dir artifacts/benchmarks/pure_workload_matrix

# Same matrix shape on cleaned AIS data with loader caps/cache.
../.venv/bin/python -m src.experiments.benchmark_matrix \
  --csv_path ../AISDATA/cleaned/<cleaned-ais-file-or-directory> \
  --cache_dir artifacts/cache/pure_workload_matrix \
  --max_points_per_segment 500 \
  --max_segments 64 \
  --workloads range,knn,similarity,clustering \
  --results_dir artifacts/benchmarks/pure_workload_matrix_csv

# Saved-checkpoint inference benchmark on a cleaned CSV.
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode inference \
  --checkpoint artifacts/benchmarks/runtime_small/benchmark_model.pt \
  --inference_csv_path ../AISDATA/cleaned/<cleaned-ais-file.csv> \
  --results_dir artifacts/benchmarks/inference_small \
  --inference_extra_args "--max_segments 8 --max_points_per_segment 64 --n_queries 16 --inference_device auto --inference_batch_size 32"
```

Benchmark artifacts are written under the selected `--results_dir`, including
`benchmark_runtime.json` and child command stdout logs. The JSON records Python,
Torch, CUDA runtime, Triton, TF32/matmul settings, AMP mode, train batch size
from the run config, phase timings, epoch timings, final F1 metrics, full child
commands, seed, git commit, and dirty status.
When `--train_batch_sizes` is provided, the artifact also includes a
`train_batch_size_sweep` table with epoch-time summary, peak CUDA memory, and
final F1 fields per batch size.

Training and inference CLIs expose the same runtime precision knobs:
`--float32_matmul_precision {highest,high,medium}` and
`--allow_tf32` / `--no-allow_tf32`, plus `--amp_mode {off,bf16,fp16}`.
`--inference_batch_size` tunes MLQDS window batches separately from
`--train_batch_size` for matched evaluation, saved-checkpoint inference, and
validation query-F1 diagnostics. Defaults preserve the FP32 baseline
(`highest`, TF32 off, AMP off). Use `high` plus `--allow_tf32` for RTX TF32
sweeps, and `--amp_mode bf16` for CUDA autocast sweeps after checking for NaNs,
collapse warnings, and final F1 drift.

While a training run is active, measure live GPU utilization from another shell:

```bash
watch -n 0.5 nvidia-smi
```

For a streaming console view:

```bash
nvidia-smi dmon
```

For one-shot telemetry matching the benchmark artifact fields:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu,utilization.memory --format=csv
```

If `nvidia-smi` is unavailable or blocked, `benchmark_runtime.json` records an
explicit `gpu_telemetry.unavailable_reason`.

The cleaned-CSV smoke target uses `--max_points_per_segment` and
`--max_segments` so it validates the real loader without turning into a full
research run. New sprint artifacts should be written under `artifacts/` or
another external run directory, not under `src/models/saved_models`.

To run on a CSV instead of synthetic data, point the CSV path at a cleaned AIS file under `../AISDATA/cleaned/`:

```bash
python -m src.experiments.run_ais_experiment \
  --csv_path ../AISDATA/cleaned/<cleaned-ais-file-or-directory> \
  --max_points_per_segment 500 \
  --max_time_gap_seconds 3600 \
  --n_queries 250 \
  --epochs 20 \
  --workload range \
  --compression_ratio 0.20 \
  --model_type baseline
```

To train on one cleaned CSV and evaluate/clean a different cleaned CSV without trajectory splitting:

```bash
python -m src.experiments.run_ais_experiment \
  --train_csv_path ../AISDATA/cleaned/train.csv \
  --eval_csv_path ../AISDATA/cleaned/eval.csv \
  --query_coverage 0.30 \
  --max_queries 1000 \
  --range_spatial_fraction 0.02 \
  --range_time_fraction 0.04 \
  --epochs 20 \
  --lr 0.0005 \
  --pointwise_loss_weight 0.25 \
  --gradient_clip_norm 1.0 \
  --workload range \
  --compression_ratio 0.20 \
  --model_type baseline
```

`--query_coverage` accepts either `0.30` or `30` for 30% point coverage. Coverage mode still emits exactly `--n_queries`; the target coverage is used to bias later query anchors toward points that have not yet been covered. `--max_queries` is accepted only as a positive legacy safety parameter. When `--eval_csv_path` is used and `--save_simplified_dir` is not set, the experiment writes only the eval-set simplified CSV under `AISDATA/ML_processed_AIS_files` as `ML_simplified_eval.csv`.

For local smoke runs on large cleaned AIS files, `--max_points_per_segment`
downsamples each loaded trajectory segment, `--max_segments` caps the loader
output during segmentation, and `--max_trajectories` remains as a legacy
post-load cap. CSV loading splits MMSI tracks by default whenever consecutive
points are more than `--max_time_gap_seconds 3600` seconds apart; set this to
`0` to disable gap-based segmentation for compatibility checks. Benchmark runs
should record any caps explicitly or leave them unset.

Add `--cache_dir artifacts/cache/<run-name>` to CSV runs to reuse segmented
Parquet caches across experiments. Use `--refresh_cache` when changing loader
code or when you want to force a rebuild of a matching source/config entry.

Range and kNN workloads focus query anchors on dense areas with a 70/30 sampler: 70% density-map weighted by lat/lon grid cell occupancy, 30% uniform from all points. The same sampler is used by coverage-targeted generation, so coverage still controls when generation stops while density controls where range/kNN queries are anchored.

For range-heavy runs, `--range_spatial_fraction` and `--range_time_fraction` control range-box half-widths as fractions of the dataset latitude/longitude and time spans. Lower these when you want many range queries without covering most of the dataset; for example, `0.02` spatial and `0.04` time gives smaller local boxes than the default `0.08` and `0.15`.

Phase 2 range diagnostics can be enabled without changing the default generator.
Use optional acceptance filters such as `--range_min_point_hits`,
`--range_max_point_hit_fraction`, `--range_max_trajectory_hit_fraction`,
`--range_max_box_volume_fraction`, `--range_duplicate_iou_threshold`, and
`--range_acceptance_max_attempts` to reject broad, duplicate, or uninformative
range boxes during generation. Each run now writes range workload diagnostics
and range label/baseline signal diagnostics into the result directory.

Use `--model_type turn_aware` to include the extra `turn_score` point feature.
Experiment runs now train one model per pure query workload. Use
`--workload {range,knn,similarity,clustering}` for the common path; explicit
`--train_workload_mix` and `--eval_workload_mix` overrides must also contain a
single positive query type.

Training uses a ranking loss plus balanced pointwise BCE supervision. Exact
final query F1 is the default checkpoint selection metric; it creates a
held-out validation workload and restores the epoch with the best validation
query F1. Use `--checkpoint_f1_variant combined` in matrix runs to compare the
legacy answer/support product against the default pure answer-set F1. If
diagnostics collapse to `pred_std=0`, prefer lowering `--lr`, keeping
`--gradient_clip_norm` enabled, and increasing query diversity before changing
the model architecture.

## Architecture At A Glance

1. `src/data/` loads AIS CSV files or generates deterministic synthetic trajectories.
2. `src/queries/` builds typed query workloads and executes range, kNN, similarity, and clustering queries.
3. `src/training/` computes typed F1-contribution labels, trains the model, restores the selected checkpoint epoch, and persists the scaler and checkpoint artifacts.
4. `src/models/` contains the query-conditioned trajectory transformer and the turn-aware variant.
5. `src/simplification/` keeps the highest-scoring points per trajectory with deterministic tie-breaking.
6. `src/evaluation/` runs learned and baseline methods and reports aggregate and per-type F1 scores.
7. `src/experiments/` wires the full pipeline together through the CLI.
8. `src/visualization/` is a minimal extension point for plotting hooks; current runs write JSON, CSV, GeoJSON, and fixed-width tables.

## Outputs

The experiment runner writes the core run files into `results/` or the
directory passed with `--results_dir`:

- `example_run.json` - config, workload mixes, per-method metrics, training history, and selected-checkpoint metadata.
- `matched_table.txt` - fixed-width comparison table for the evaluation workload.
- `shift_table.txt` - shift table comparing the train workload against the eval workload.
- `geometric_distortion_table.txt` - SED/PED, length preservation, and combined geometric/F1 reporting.
- `range_workload_diagnostics.json` - Phase 2 range workload, label, Oracle, and baseline diagnostics for train/eval/selection workloads.
- `range_query_diagnostics.jsonl` - one JSON record per generated range query with hit counts, footprint, broad-query flag, and duplicate-query flag.

## Validation

The `tests/` folder focuses on the rebuild-specific regressions:

- `test_beats_random_in_distribution.py` - in-distribution performance guard.
- `test_no_cross_trajectory_attention_leakage.py` - attention leakage guard.
- `test_query_type_ids_required.py` - query type ID contract.
- `test_scaler_persisted.py` - scaler persistence.
- `test_topk_no_positional_bias.py` - deterministic top-k behavior.
- `test_training_does_not_collapse.py` - training stability.

# Experiments Module

This module is the orchestration layer for AIS-QDS runs. It converts CLI
arguments into structured config, loads data, generates pure query workloads,
trains MLQDS, evaluates baselines, and writes run artifacts.

## Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | Shared `argparse` parser for training/evaluation runs. |
| `experiment_config.py` | Data, query, model, baseline, workload, and seed config dataclasses. |
| `experiment_pipeline_helpers.py` | Split, workload generation, diagnostics reuse, training, evaluation, and result dumping. |
| `benchmark_profiles.py` | Central benchmark profile constants such as `range_real_usecase`. |
| `benchmark_matrix.py` | Range-focused benchmark matrix runner. |
| `benchmark_runtime.py` | Runtime wrapper for train/inference timing experiments. |
| `run_ais_experiment.py` | Main train/evaluate entry point. |
| `run_inference.py` | Evaluate a saved checkpoint on a CSV without retraining. |

## Entry Points

Use `--help` on each entry point for the full current argument list.

```bash
../.venv/bin/python -m src.experiments.run_ais_experiment --help
../.venv/bin/python -m src.experiments.run_inference --help
../.venv/bin/python -m src.experiments.benchmark_matrix --help
../.venv/bin/python -m src.experiments.benchmark_runtime --help
```

Common argument groups:

- Data: `--csv_path`, `--train_csv_path`, `--validation_csv_path`, `--eval_csv_path`, segmentation caps, `--cache_dir`, `--refresh_cache`.
- Query workload: `--workload`, `--n_queries`, `--query_coverage`, range footprint/acceptance controls.
- Training: epochs, LR, ranking sampler, batch sizes, query chunking, checkpoint selection, early stopping.
- Runtime: `--float32_matmul_precision`, `--allow_tf32`, `--amp_mode`, inference device/batch size.
- Outputs: `--results_dir`, `--save_model`, `--save_queries_dir`, `--save_simplified_dir`.

Training-specific behavior is documented in
[`../training/README.md`](../training/README.md). Query-generation behavior is
documented in [`../queries/README.md`](../queries/README.md).

## Data Modes

If `--train_csv_path`, `--validation_csv_path`, and `--eval_csv_path` are
supplied, the three CSVs are used as training, checkpoint validation, and final
evaluation days. If only train/eval CSVs are supplied, checkpoint validation is
split out of the train CSV for compatibility. If only `--csv_path` is supplied,
one dataset is split by trajectory. If all CSV paths are omitted, the CLI
generates deterministic synthetic data.

CSV loading segments each MMSI by temporal continuity. The default
`--max_time_gap_seconds 3600` starts a new segment after a one-hour gap; use
`0` only for compatibility checks. `--cache_dir` persists post-segmentation
Parquet data keyed by source file identity and segmentation config.

## Pipeline

1. Resolve train/selection/eval trajectory sets.
2. Generate independent typed workloads for train, eval, and checkpoint selection.
3. Compute range diagnostics and reusable labels/query caches when applicable.
4. Train MLQDS and restore the best checkpoint according to the active selection metric.
5. Evaluate MLQDS, uniform, Douglas-Peucker, and label Oracle on the eval workload.
6. Write tables, JSON diagnostics, optional GeoJSON queries, and optional simplified CSVs.

Experiment entrypoints train and evaluate one model per pure workload. Use
`--workload {range,knn,similarity,clustering}`.

## Real-Usecase Range Profile

`benchmark_matrix.py --profile range_real_usecase` is the current benchmark
baseline. It selects the first three sorted cleaned CSV files from `--csv_path`
as train/checkpoint-validation/eval days unless explicit paths are provided.

Profile shape:

| Setting | Value |
| --- | --- |
| Workload | pure `range` |
| Data split | first sorted cleaned CSV = train, second = checkpoint validation, third = final eval |
| Query generation | Start at `80`, continue until `0.20` coverage, cap at `max_queries=2048` |
| Range footprint | `range_spatial_km=2.2`, `range_time_hours=5.0` fixed half-windows (`range_footprint_jitter=0.0`) |
| Compression | `0.05` retained points |
| Epoch budget | `20` with `early_stopping_patience=5` |
| Checkpoint selection | `checkpoint_selection_metric=f1`, `checkpoint_f1_variant=range_usefulness` by default |
| Loss objective | `loss_objective=budget_topk`, `budget_loss_ratios=0.01,0.02,0.05,0.10`, `residual_label_mode=temporal` |
| MLQDS scoring | pure workload `rank` mode, `mlqds_temporal_fraction=0.50`, `mlqds_diversity_bonus=0.0`, `mlqds_score_temperature=1.0` |
| Attention chunk | `query_chunk_size=2048`; the profile uses the same value for `max_queries` |
| Range labels | `range_label_mode=usefulness`, `range_boundary_prior_weight=0.0` |
| Range diagnostics | `range_diagnostics_mode=cached` when `--cache_dir` is set |
| Diagnostics | exact validation every eligible epoch by default: `f1_diagnostic_every=1`, `checkpoint_full_f1_every=1`, `checkpoint_candidate_pool_size=1`, no smoothing (`checkpoint_smoothing_window=1`) |
| Runtime variant | `tf32_bf16_bs32_inf32` by default |
| Ranking sampler | `vectorized` by default |
| Caps | leave `max_points_per_segment`, `max_segments`, and `max_trajectories` unset |

Use matrix variant `tf32_bf16_bs32_inf32_point_f1_labels` as the direct
ablation for the old range point-F1 label target. Use
`tf32_bf16_bs32_inf32_ranking_bce` as the legacy pairwise-loss ablation.
Use `tf32_bf16_bs64_inf32` and `tf32_bf16_bs128_inf32` to test whether the
batched budget-top-k loss can use more available VRAM efficiently.
Use `tf32_bf16_bs32_inf32_temporal000`,
`tf32_bf16_bs32_inf32_temporal050`, and
`tf32_bf16_bs32_inf32_temporal075` as temporal-spine ablations. The default is
`0.50`; `0.75` is an explicit high-scaffold comparison rather than the
baseline.
Use `tf32_bf16_bs32_inf32_residual_none` to test whether training on all labels
beats temporal-residual learned-fill training.
Use `tf32_bf16_bs32_inf32_diversity005` to test the old spacing bonus against
the no-diversity default.

`query_coverage` is a target used for workload generation and diagnostics.
For the real-usecase profile, `n_queries` is only the minimum workload size;
generation continues until the target is reached or `max_queries` is hit.
`query_generation_diagnostics` records the minimum query count, max query cap,
final generated query count, per-type query counts, final coverage, and stop
reason for each split.
When `--cache_dir` is set, generated workloads are cached under
`<cache_dir>/workloads/` by data fingerprint, query config, seed, and workload
map. The real-usecase profile also enables `--range_diagnostics_mode cached`,
which stores range workload summaries, per-query diagnostic rows, and training
label tensors under `<cache_dir>/range_diagnostics/`. Final matched evaluation
remains exact; stale or incomplete diagnostics cache entries are ignored and
recomputed. Use `--refresh_cache` to force regeneration.
For `range_label_mode=usefulness`, label diagnostics also include
`component_positive_label_mass_fraction`, which shows whether the local proxy
is mostly driven by point, ship, crossing, temporal, gap, turn, or shape
supervision. Component mass is reported before the final training-label clamp.

Direct matrix run:

```bash
../.venv/bin/python -m src.experiments.benchmark_matrix \
  --profile range_real_usecase \
  --workloads range \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_real_usecase \
  --variants tf32_bf16_bs32_inf32 \
  --results_dir artifacts/benchmarks/range_real_usecase/runs/manual_range_real_usecase_a \
  --run_id manual_range_real_usecase_a
```

Use `--no_cache_warmup` only when intentionally measuring cold-cache behavior.
Before changing range footprint or target coverage, estimate coverage/cap
behavior first:

```bash
../.venv/bin/python scripts/estimate_range_coverage.py \
  --csv_path ../AISDATA/cleaned \
  --cache_dir artifacts/cache/range_real_usecase \
  --query_counts 80,384,512,640,1024,2048 \
  --sample_stride 20 \
  --target_coverage 0.20 \
  --range_spatial_km 2.2 \
  --range_time_hours 5.0 \
  --range_footprint_jitter 0.0
```

## Long Runs

Use tmux launchers for long local runs. They keep training alive outside the
IDE session and capture system telemetry beside the run.

```bash
make benchmark-preflight
ATTACH=0 BENCHMARK_RUN_ID=range_real_usecase_a make range-benchmark-tmux
```

Queue rows use a tab-separated plan file:

```text
range_real_usecase_base_seed42	42
range_real_usecase_rank_tie_seed42	42	--mlqds_score_mode rank_tie
range_real_usecase_pairs192_seed42	42	--ranking_pairs_per_type 192
```

Launch a sequential queue:

```bash
ATTACH=0 \
  BENCHMARK_PLAN_FILE=artifacts/benchmarks/range_real_usecase/queues/my_plan.tsv \
  BENCHMARK_CONTINUE_ON_FAILURE=1 \
  make range-benchmark-queue-tmux
```

The queue launcher validates each row's child args against
`run_ais_experiment` before tmux starts, so unsupported benchmark knobs fail
fast.

The monitor pane writes RAM/swap, disk, top RSS processes, GPU utilization,
GPU memory, temperature, power draw, clocks, CUDA processes, and relevant
kernel markers. If a launcher observes an abnormal exit, it marks stale
`run_status.json` files as failed.

## Artifacts And Timing

For comparisons, start with `benchmark_matrix.md` or `benchmark_matrix.csv`.
The matrix includes train-split usefulness-label component mass fractions so
quality deltas can be checked against the supervision mix used by that run.
It also includes the training target residual label-mass fraction at the run
compression ratio, which helps identify cases where the temporal base consumes
most of the useful signal before learned fill is trained.
For model behavior, inspect the variant `example_run.json`,
`matched_table.txt`, `range_usefulness_table.txt`, and
`range_workload_diagnostics.json`. `RangePointF1` is the retained in-box point
metric; `RangeUseful` is the current versioned audit score for range-local
usefulness, including ship presence, per-ship coverage, entry/exit,
crossing-bracket preservation, temporal span, gap coverage, route-change
coverage, and local shape fidelity.
`learned_fill_diagnostics.json` compares MLQDS against the same temporal base
with random fill and oracle-label fill, so failures can be separated into
temporal-base, learned-fill, and scoring issues. The matrix CSV also reports
temporal-random-fill usefulness, MLQDS-vs-random-fill usefulness, and the
oracle-fill gap so residual-fill failures are visible in compact tables.
`range_residual_objective_summary.json` collects the same learned-fill deltas
with the train label component mix and residual target mass split, so a single
run can be inspected without opening the full `example_run.json`. It also
records that `TemporalOracleFill` is an additive-label residual-fill reference,
not an exact retained-set optimum.
`training_target_diagnostics` inside `example_run.json` records the effective
residual-label budgets and label-mass split used by the loss, while
`range_workload_distribution_comparison.json` compares train/selection/eval
query coverage and hit distributions. `collapse_warning_any` means any epoch
collapsed; `best_epoch_collapse_warning` is the selected-checkpoint signal.
`best_selection_score` is the canonical checkpoint score, while `best_f1` is
kept for compatibility with older artifacts.
Use `--range_audit_compression_ratios 0.01,0.02,0.05,0.10` when you want the
same range-usefulness components rerun across multiple retained-point budgets;
it is disabled by default because it reruns method evaluation.
Artifact layout and cleanup rules live in
[`../../artifacts/README.md`](../../artifacts/README.md).

Use `benchmark_runtime.py` only for targeted train/inference timing studies
such as batch-size sweeps. Use `benchmark_matrix.py` for model-quality
benchmark runs.

```bash
../.venv/bin/python -m src.experiments.benchmark_runtime \
  --mode train \
  --profile range_real_usecase \
  --train_extra_args "--train_csv_path ../AISDATA/cleaned/aisdk-2026-02-02_cleaned.csv --validation_csv_path ../AISDATA/cleaned/aisdk-2026-02-03_cleaned.csv --eval_csv_path ../AISDATA/cleaned/aisdk-2026-02-04_cleaned.csv --cache_dir artifacts/cache/range_real_usecase" \
  --train_batch_sizes 16,32,64 \
  --results_dir artifacts/benchmarks/runtime_range_real_usecase
```

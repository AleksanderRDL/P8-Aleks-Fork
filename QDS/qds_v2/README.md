# AIS-QDS v2

AIS-QDS v2 is the current shift-aware rebuild of the AIS query-driven simplification pipeline. It loads AIS trajectories, generates typed query workloads, trains a query-conditioned ranking model, and evaluates the resulting simplifier against learned and geometric baselines under matched and shifted workloads.

## What Is In This Folder

- `requirements.txt` - Python dependencies for the v2 stack.
- `src/` - package code for loading data, building queries, training models, and running experiments.
- `tests/` - regression tests that guard the rebuild.
- `results/` - example outputs from the experiment runner.

## Quick Start

```bash
cd qds_v2
pip install -r requirements.txt
python -m src.experiments.run_ais_experiment --n_queries 128 --epochs 6 --workload mixed
```

To run on a CSV instead of synthetic data:

```bash
python -m src.experiments.run_ais_experiment \
  --csv_path "C:\path\to\ais.csv" \
  --n_queries 250 \
  --epochs 20 \
  --workload mixed \
  --compression_ratio 0.20 \
  --model_type baseline
```

Use `--model_type turn_aware` to include the extra `turn_score` point feature. Workload mixes can be overridden with `--train_workload_mix` and `--eval_workload_mix` (or the `..._mix_train` / `..._mix_eval` aliases).

## Architecture At A Glance

1. `src/data/` loads AIS CSV files or generates deterministic synthetic trajectories.
2. `src/queries/` builds typed query workloads and executes range, kNN, similarity, and clustering queries.
3. `src/training/` computes typed importance labels, trains the model, and persists the scaler and checkpoint artifacts.
4. `src/models/` contains the query-conditioned trajectory transformer and the turn-aware variant.
5. `src/simplification/` keeps the highest-scoring points per trajectory with deterministic tie-breaking.
6. `src/evaluation/` runs learned and baseline methods and reports aggregate and per-type errors.
7. `src/experiments/` wires the full pipeline together through the CLI.
8. `src/visualization/` is currently a placeholder package for future plotting hooks.

## Outputs

The experiment runner writes three files into `results/`:

- `example_run.json` - config, workload mixes, per-method metrics, and training history.
- `matched_table.txt` - fixed-width comparison table for the evaluation workload.
- `shift_table.txt` - shift table comparing the train workload against the eval workload.

## Validation

The `tests/` folder focuses on the rebuild-specific regressions:

- `test_beats_random_in_distribution.py` - in-distribution performance guard.
- `test_no_cross_trajectory_attention_leakage.py` - attention leakage guard.
- `test_query_type_ids_required.py` - query type ID contract.
- `test_scaler_persisted.py` - scaler persistence.
- `test_topk_no_positional_bias.py` - deterministic top-k behavior.
- `test_training_does_not_collapse.py` - training stability.

For the legacy v1 system, see [qds_project/README.md](../qds_project/README.md).

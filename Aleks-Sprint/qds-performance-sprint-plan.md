# QDS Performance And Reliability Sprint Plan

## Purpose

This sprint plan captures the cleanup, benchmarking, and optimization work for
making QDS easier to iterate on with the RTX 5060 Ti 16GB machine.

The goal is not to preserve the current model behavior as a gold standard. The
current models are still weak enough that speedups are valuable mainly because
they make model-quality iteration cheaper and faster. The plan should therefore
optimize for two outcomes at the same time:

1. faster, more reliable experimentation
2. better evidence about which model/training changes actually improve QDS

## Current Starting Point

Observed on 2026-05-10:

- Local QDS environment: `torch 2.11.0+cu130`, CUDA runtime `13.0`,
  `triton 3.6.0`.
- The observed `.venv` uses Python 3.14, which may make some CUDA package
  availability more fragile than a more common Python version.
- `QDS/requirements.txt` now aliases the CUDA reference profile, while
  separate QDS common, CPU, and CUDA requirement profiles document the intended
  stacks.
- `../.venv/bin/python -m pytest tests` passes from inside `QDS`.
- `.venv/bin/python -m pytest QDS/tests` now passes from the repository root
  through the root `pyproject.toml` pytest configuration.
- GPU telemetry is now visible from the benchmark wrapper on the local shell:
  Torch and `nvidia-smi` report an RTX 5060 Ti with 16GB class memory. Real
  utilization under larger training loads still needs measurement.
- TF32 matmul is currently disabled in the observed environment.
- Existing QDS runs are useful reference points for timing and sanity checks,
  but they should not be treated as "good model" regression baselines.

## Sprint Goal

Make QDS faster, easier to benchmark, and less ambiguous so model changes can
be tested more often and with better evidence.

The sprint should proceed in this order:

1. Fix reproducibility and test setup.
2. Add measurement and benchmark hooks.
3. Apply low-risk speedups.
4. Add RTX 5060 Ti specific tuning controls.
5. Tackle behavior-sensitive training-loop refactors.

## Plan Status

Status: in progress. Phase 1 tasks 1 and 2 were implemented on 2026-05-10.

This plan is intended to guide implementation. Individual tasks can still be
split into smaller tickets as evidence comes in from benchmarks.

## Guiding Principles

### Reference Points, Not Gold Baselines

The current implementation is not known-good enough to use as a strict model
quality regression target. Use current runs as reference points for:

- runtime
- phase timing
- collapse diagnostics
- rough F1 sanity checks
- geometry metric sanity checks

Do not reject a change only because it changes model behavior. Reject or revise
a change when it makes runtime worse without improving evidence quality, causes
obvious training collapse, introduces NaNs, breaks core invariants, or makes the
benchmark harder to interpret.

### Define Quality Gates Per Experiment

Each benchmark should state what counts as acceptable before it runs. Examples:

- no NaNs
- no persistent collapse warnings
- nonzero prediction spread
- per-type F1 reported, not only aggregate F1
- geometry metrics still reported
- command, seed, dependency versions, GPU settings, and git status recorded

Because there is no trusted "best" run yet, the first benchmark harness should
focus on comparability and observability rather than strict pass/fail F1
thresholds.

### Measure End-To-End And Phase-By-Phase

Track both complete runtime and major phase timings:

- data load/cache
- query generation
- diagnostics
- training
- validation F1
- matched evaluation
- shift evaluation
- writing artifacts

This avoids mistaking a faster training step for a faster experiment if
diagnostics or evaluation dominate wall time.

### Keep Diagnostics Available

Expensive diagnostics can have a fast/off switch for iteration, but they should
not disappear. The project still needs diagnostics to detect weak workloads,
collapsed models, misleading aggregate F1, and geometry tradeoffs.

### Keep Speed And Research Work Distinct

Some tasks improve the speed and reliability of experiments. Other tasks may
change the model, workload semantics, or training target. Keep those separate
in run notes so it is clear whether a result changed because the system got
faster, because the model got better, or because the benchmark changed.

Examples of model-quality work:

- similarity query generation
- checkpoint selection policy
- training objective changes
- label construction changes
- simplification budget policy

Examples of speed/reliability work:

- package setup
- benchmark logging
- CUDA device placement
- TF32/BF16 controls
- evaluation caching
- vectorized ranking loss

### Final F1 Is The Learning North Star

The desired training direction is to learn from final retained-set query
performance directly, not to stop at proxy per-point labels. Proxy labels remain
useful diagnostics and warm-start signals, but the eventual model objective
should be aligned with the same post-simplification workload F1 used for final
evaluation.

### Stop Or Shelve Rules

Pause, revert, or isolate a change if it:

- introduces NaNs
- creates persistent collapse warnings
- breaks core tests
- makes benchmark output less complete
- slows end-to-end runtime without improving model-quality evidence
- changes workload semantics without being documented as research work
- makes the environment harder to recreate

### Python Version Decision

The current `.venv` uses Python 3.14. Keep that as the current working
reference unless it blocks CUDA package availability. If a future CUDA stack
experiment needs wheels that are missing or unstable on Python 3.14, create a
separate Python 3.12 environment rather than mutating the current environment.

### Benchmark Artifact Schema

Each benchmark artifact should include at least:

- timestamp
- git commit
- git dirty-status summary
- full command
- working directory
- dataset description
- seed
- Python version
- Torch version
- Torch CUDA runtime
- Triton version
- GPU name and driver when visible
- TF32/matmul precision settings
- AMP/BF16/FP16 settings
- model config
- query config
- train/eval workload mix
- train batch size
- inference batch size
- phase timings
- epoch timings
- peak GPU memory when available
- aggregate F1
- per-type F1
- collapse diagnostics
- geometry metrics

### Default Benchmark Matrix

Use three run sizes so changes can be tested at the right cost:

```text
small:   quick correctness/runtime smoke, cheap enough to run often
medium:  representative iteration benchmark for most optimization work
serious: slower benchmark used before accepting major training or CUDA-stack changes
```

The exact commands should be recorded in the benchmark script or README once
the first implementation slice lands.

## Phase 1: Reproducibility And Measurement

Goal: make every later optimization comparable.

### 1. Fix Repository-Root Test And Import Setup

Task:

- Add a proper Python package/test setup so QDS tests can run from the
  repository root, not only from inside `QDS`.
- Candidate approaches:
  - add a `pyproject.toml` with package metadata and pytest path config
  - use an editable install for QDS
  - add a minimal pytest `pythonpath` configuration

Benefit:

- Removes a recurring local/CI friction point.
- Makes future benchmark and test commands less fragile.

Risk:

- Low.
- Could expose hidden assumptions about running commands from `QDS`.

Acceptance check:

```bash
.venv/bin/python -m pytest QDS/tests
```

Completion note, 2026-05-10:

- Added a root `pyproject.toml` with package discovery and pytest
  `pythonpath` configuration for `QDS`.
- Verified `.venv/bin/python -m pytest QDS/tests` from the repository root:
  `72 passed, 1 warning`.

### 2. Split And Document Requirements Profiles

Task:

- Separate environment concerns:
  - base repository dependencies
  - QDS CPU dependencies
  - QDS CUDA dependencies
- Document the current installable CUDA reference stack:
  `torch 2.11.0+cu130`, CUDA runtime `13.0`, `triton 3.6.0`.

Benefit:

- Makes dependency-stack experiments reversible.
- Reduces the chance of accidentally comparing different dependency stacks.
- Makes it clear whether a speedup came from code, configuration, or a new
  package stack.

Risk:

- Low.
- Needs care not to break existing `make` targets or the current `.venv`.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m pip check
../.venv/bin/python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
```

Completion note, 2026-05-10:

- Added `QDS/requirements-common.txt`, `QDS/requirements-cpu.txt`, and
  `QDS/requirements-cuda-cu130.txt`.
- Kept `QDS/requirements.txt` as the current CUDA sprint compatibility alias.
- Updated `QDS/Makefile` with `QDS_REQUIREMENTS`, `install-cpu`, and
  `install-cuda`.
- Documented the profiles and current CUDA reference stack in `QDS/README.md`.
- Verified `pip check` and the active stack:
  `torch 2.11.0+cu130`, CUDA runtime `13.0`, `triton 3.6.0`.

### 3. Add A Stable Benchmark Script

Task:

- Add a benchmark entrypoint for:
  - one representative training run
  - one representative inference/evaluation run
- Log:
  - Python version
  - Torch version
  - CUDA runtime
  - Triton version
  - GPU name when visible
  - TF32 setting
  - matmul precision
  - AMP/BF16/FP16 setting
  - batch size
  - phase timings
  - epoch timings
  - final F1 metrics
  - full command
  - seed
  - git commit and dirty-status summary

Benefit:

- Prevents "optimization by vibes."
- Gives a clean reference point before touching training logic or installing
  alternate dependency stacks.

Risk:

- Low.
- Must use fixed seeds and a stable dataset slice to be meaningful.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m src.experiments.benchmark_runtime --help
```

Completion note, 2026-05-10:

- Added `src.experiments.benchmark_runtime`.
- The wrapper can run training, saved-checkpoint inference, or both through the
  existing `run_ais_experiment` and `run_inference` CLIs.
- It writes `benchmark_runtime.json` plus child stdout logs under
  `--results_dir`.
- The artifact records Python, Torch, CUDA runtime, Triton, visible GPU
  metadata, TF32/matmul settings, AMP intent, git commit/dirty status, full
  child commands, seed, parsed phase timings, parsed epoch timings, and final
  matched-workload F1 metrics.
- Verified `../.venv/bin/python -m src.experiments.benchmark_runtime --help`.
- Verified a small synthetic train benchmark:
  `../.venv/bin/python -m src.experiments.benchmark_runtime --mode train --profile small --results_dir artifacts/benchmarks/smoke_runtime --seed 123`.

### 4. Add GPU Utilization Guidance

Task:

- Add a short documented workflow for measuring utilization during training:

```bash
watch -n 0.5 nvidia-smi
```

or:

```bash
nvidia-smi dmon
```

Benefit:

- Tells whether the bottleneck is GPU compute, CPU/control flow, data/query
  generation, or evaluation.

Risk:

- Low.
- WSL and driver permissions may limit visibility depending on where commands
  are run.

Acceptance check:

- The normal training shell can see the RTX 5060 Ti with `nvidia-smi`, or the
  benchmark output clearly notes that GPU telemetry was unavailable.

Completion note, 2026-05-10:

- Added runtime benchmark and GPU telemetry guidance to `QDS/README.md`.
- The benchmark wrapper probes `nvidia-smi` and records either GPU rows or an
  explicit unavailable reason.
- On the local shell, the smoke benchmark artifact recorded:
  `NVIDIA GeForce RTX 5060 Ti`, driver `596.36`, and Torch CUDA availability
  with `torch 2.11.0+cu130`.

## Phase 2: Low-Risk Speedups

Goal: remove obvious inefficiencies without changing the research logic.

### 5. Disable Unused Attention Weight Return

Task:

- Pass `need_weights=False` in the `nn.MultiheadAttention` call inside
  `chunked_cross_attention_context`.

Benefit:

- You discard attention weights today.
- This may unlock faster PyTorch attention paths.
- Very small code change.

Risk:

- Low.
- Need to verify outputs/tests still pass.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m pytest tests/test_attention_context_scaling.py
```

Completion note, 2026-05-10:

- Added `need_weights=False` to the point-to-query `MultiheadAttention` call in
  `chunked_cross_attention_context`.
- Verified:
  `../.venv/bin/python -m pytest tests/test_attention_context_scaling.py tests/test_positional_encoding_cache.py`.

### 6. Cache Positional Encodings

Task:

- Avoid rebuilding sinusoidal positional encodings on every model forward for
  the fixed window sizes used in training and inference.
- Cache by `(length, device, dtype)` or store a reusable buffer and slice it.

Benefit:

- Reduces repeated allocation and trigonometric work.
- Helps both training and inference.

Risk:

- Low.
- Must handle device and dtype correctly.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m pytest tests/test_no_cross_trajectory_attention_leakage.py tests/test_training_does_not_collapse.py
```

Completion note, 2026-05-10:

- Added a non-persistent positional-encoding cache buffer to
  `TrajectoryQDSModel`.
- The cache reuses encodings for matching `(length <= cached_length, device,
  dtype)` requests and rebuilds when length, device, dtype, or embedding
  dimension require it.
- Added `tests/test_positional_encoding_cache.py`.
- Verified:
  `../.venv/bin/python -m pytest tests/test_no_cross_trajectory_attention_leakage.py tests/test_training_does_not_collapse.py`.

### 7. Run MLQDS Inference/Evaluation On CUDA

Task:

- Ensure model-based simplification can run on CUDA when available.
- Today the trained model is restored to CPU after training, and MLQDS
  prediction likely runs on CPU unless explicitly moved.
- Add a device-aware prediction path for:
  - `MLQDSMethod.simplify`
  - validation F1 diagnostics
  - saved-checkpoint inference

Benefit:

- Potentially large speedup for evaluation and validation diagnostics.
- Makes GPU benchmarking more honest.

Risk:

- Medium-low.
- Need careful movement of model, points, queries, masks, and final CPU metrics.
- Evaluation metrics may still run on CPU, which is fine as long as boundaries
  are clear.

Acceptance check:

- CPU and CUDA outputs should match within a small tolerance on the same
  checkpoint and workload.

Completion note, 2026-05-10:

- Added device-aware `windowed_predict(..., device=...)` and
  `forward_predict(..., device=...)`.
- `MLQDSMethod.simplify` now defaults to CUDA when available and keeps the
  retained mask on the original points device for existing evaluation metrics.
- Added `run_inference --inference_device {auto,cpu,cuda}` for saved-checkpoint
  inference control.
- Made trajectory window construction preserve tensor devices for padding,
  masks, indices, and trajectory IDs.
- Added CUDA regression checks comparing CPU and CUDA prediction outputs on the
  same model/artifact within tolerance.
- Verified:
  `../.venv/bin/python -m pytest tests/test_scaler_persisted.py`,
  `.venv/bin/python -m pytest QDS/tests`, and a small
  `benchmark_runtime --mode train --profile small` smoke.

### 8. Replace Range Point-F1 Tuple Sets With Mask Math

Task:

- The range F1 path currently converts hit points into Python tuple sets.
- Since simplified points are selected from the original points, range F1 can
  likely be computed from boolean masks and retained masks.

Benefit:

- Faster range-heavy evaluation.
- Less Python object allocation.

Risk:

- Low-medium.
- Must preserve duplicate-point semantics if duplicate AIS rows remain after
  cleaning.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m pytest tests/test_range_point_evaluation.py tests/test_metrics.py
```

Completion note, 2026-05-10:

- Replaced Python tuple-set construction in range point-F1 with boolean mask
  math over the original `retained_mask` and the range query box mask.
- Made full/simplified trajectory views lazy inside `score_retained_mask`, so
  range-only scoring avoids building simplified point tensors entirely.
- Range F1 now scores retained point instances directly, so exact duplicate
  AIS rows count as separate query-hit mass instead of being collapsed by row
  value.
- Added a duplicate-row regression in `tests/test_range_point_evaluation.py`.
- Verified:
  `../.venv/bin/python -m pytest tests/test_range_point_evaluation.py tests/test_metrics.py`
  and `.venv/bin/python -m pytest QDS/tests`.

### 9. Cache Reusable Evaluation Query Results

Task:

- Cache full-data query answers and reusable support masks during
  `score_retained_mask`.
- Reuse them across methods in matched evaluation.

Benefit:

- Avoids rerunning the full-data side of the same queries for every method.
- Especially valuable for similarity and clustering workloads.

Risk:

- Medium.
- Needs clean cache ownership so results cannot leak between datasets or
  workloads.

Acceptance check:

- Existing matched tables should be numerically unchanged.
- Evaluation wall time should drop on mixed/global workloads.

Completion note, 2026-05-10:

- Added caller-owned `EvaluationQueryCache` support to `score_retained_mask`
  and `evaluate_method`.
- The cache is scoped to one points tensor, boundary list, and typed-query
  list; reuse with a different workload raises immediately instead of leaking
  stale answers.
- Cached data includes full-data query answers, full-data trajectory views, and
  support masks for range, kNN, similarity, and clustering queries. Simplified
  query answers still run per retained mask/method.
- Matched experiment evaluation, saved-checkpoint inference, range signal
  diagnostics, and validation query-F1 diagnostics now share the appropriate
  cache for their evaluation scope.
- Added regression tests that verify full-data query execution is reused across
  retained masks and that mismatched workloads are rejected.
- Verified:
  `../.venv/bin/python -m pytest tests/test_metrics.py tests/test_range_point_evaluation.py`
  and `.venv/bin/python -m pytest QDS/tests`.

## Phase 3: RTX 5060 Ti / CUDA-Specific Tuning

Goal: use the hardware better with the current project setup.

### 10. Add TF32 / Matmul Precision Controls

Task:

- Add config/CLI controls for:

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
```

Benefit:

- Likely useful for FP32 linear and attention-heavy work on modern NVIDIA GPUs.
- Low implementation effort.

Risk:

- Low.
- Small numeric differences are expected.
- Benchmark final F1 stability, not only speed.

Acceptance check:

- Benchmark FP32 default vs TF32-enabled with same seed and dataset.

Completion note, 2026-05-10:

- Added shared torch runtime precision helper in
  `QDS/src/experiments/torch_runtime.py`.
- Added config and CLI controls:
  `--float32_matmul_precision {highest,high,medium}` and
  `--allow_tf32` / `--no-allow_tf32`.
- Defaults preserve the previous FP32 baseline (`highest`, TF32 off).
- `run_ais_experiment` applies settings before training and records them in
  `example_run.json`; saved checkpoints persist the desired settings through
  `ModelConfig`.
- `run_inference` can override settings or reuse checkpoint settings by
  default, and records active runtime precision in `inference_run.json`.
- `benchmark_runtime` applies the same settings in the wrapper, forwards them
  to child train/inference commands, and records the effective torch runtime.
- Added tests for runtime application and config round-tripping, including
  legacy config defaults.
- Verified:
  `../.venv/bin/python -m pytest tests/test_torch_runtime_controls.py tests/test_scaler_persisted.py`.
- Acceptance smoke:
  `benchmark_runtime --mode train --profile small --seed 125` was run once
  with defaults and once with `--float32_matmul_precision high --allow_tf32`.
  Both artifacts recorded the requested settings and identical small-smoke
  aggregate F1 values for matched methods.

### 11. Sweep Training Batch Size

Task:

- Benchmark `train_batch_size` values such as:

```text
16, 32, 64, 128
```

- Stop when VRAM, instability, or diminishing returns appear.

Benefit:

- May improve GPU utilization without changing model logic.
- Particularly relevant on a 16GB GPU.

Risk:

- Low-medium.
- Larger batches can slightly change optimization dynamics.

Acceptance check:

- Compare epoch time, peak memory, and final validation F1.

Completion note, 2026-05-10:

- Exposed `train_batch_size` through `build_experiment_config` and the
  training CLI via `--train_batch_size`.
- Added CUDA peak-memory snapshots around the training phase and writes them
  to `example_run.json` under `cuda_memory.training`.
- Extended `benchmark_runtime` with `--train_batch_sizes`, which runs one
  training child process per requested batch size and writes a compact
  `train_batch_size_sweep` comparison table.
- Sweep rows include return code, elapsed time, epoch-time mean/min/max, peak
  CUDA allocated/reserved memory, `best_f1`, and MLQDS aggregate F1.
- The sweep stops on first failure by default, with
  `--sweep_continue_on_failure` available for exploratory runs.
- Added tests for batch-size config round-trip, sweep parsing, and summary
  extraction.
- Verified:
  `../.venv/bin/python -m pytest tests/test_torch_runtime_controls.py`.
- Acceptance smoke:
  `benchmark_runtime --mode train --profile small --seed 126 --train_batch_sizes 4,8`
  produced two sweep rows with epoch time, peak CUDA memory, and identical
  small-smoke MLQDS aggregate F1.

### 12. Add Optional BF16 Autocast

Task:

- Add an opt-in autocast mode for CUDA training and inference.
- Prefer BF16 first, because it is usually easier to stabilize than FP16.

Benefit:

- Can use tensor cores more directly on Blackwell-class hardware.
- May be one of the most important hardware-specific tests if the model remains
  FP32 today.

Risk:

- Medium.
- Ranking loss and BCE stability must be checked.
- Might require keeping some loss/stat operations in FP32.

Acceptance check:

- Compare FP32/TF32 vs BF16 speed and F1 stability.
- Watch for NaNs, collapse warnings, or severe metric drift.

Completion note, 2026-05-10:

- Added `amp_mode` to `ModelConfig`, `build_experiment_config`, the training
  CLI, saved-checkpoint inference, and the benchmark wrapper. Defaults keep
  autocast off.
- Added shared AMP helpers in `QDS/src/experiments/torch_runtime.py` for mode
  validation, CUDA-only autocast context creation, and runtime metadata.
- Training and validation-F1 diagnostics now wrap model forwards in optional
  autocast, then cast predictions back to FP32 before ranking loss, balanced
  BCE, diagnostic statistics, and retained-set scoring. FP16 uses CUDA
  `GradScaler`; BF16 does not.
- `windowed_predict`, `forward_predict`, and `MLQDSMethod` accept `amp_mode`,
  so matched evaluation and saved-checkpoint inference use the same setting.
- `example_run.json`, `inference_run.json`, and benchmark artifacts now record
  effective AMP metadata alongside TF32/matmul settings.
- Documented `--amp_mode {off,bf16,fp16}` in the QDS, experiments, and
  training READMEs.
- Verified:
  `../.venv/bin/python -m py_compile src/experiments/torch_runtime.py src/experiments/run_inference.py src/experiments/benchmark_runtime.py src/training/train_model.py src/training/training_pipeline.py src/evaluation/baselines.py`
  and
  `../.venv/bin/python -m pytest tests/test_torch_runtime_controls.py tests/test_scaler_persisted.py`.
- Acceptance smoke:
  paired `benchmark_runtime --mode train --profile small --seed 127` runs with
  TF32-enabled FP32 (`--float32_matmul_precision high --allow_tf32`) and BF16
  (`--float32_matmul_precision high --allow_tf32 --amp_mode bf16`) both
  completed on the RTX 5060 Ti. BF16 recorded CUDA autocast enabled, no collapse
  warnings, lower training peak allocation (`42.7 MiB` vs `58.5 MiB`), similar
  elapsed time (`3.39s` vs `3.31s`), and a small-smoke MLQDS aggregate F1 of
  `0.5242` vs `0.4622`. Treat this as a stability/runtime smoke only; larger
  workloads are still needed before judging model-quality drift.

### 13. Add Inference Batch-Size Controls

Task:

- Expose `windowed_predict` batch size through config/CLI for inference and
  validation diagnostics.

Benefit:

- Allows separate tuning of training and inference throughput.

Risk:

- Low.
- Mainly a memory tuning issue.

Acceptance check:

- Larger inference batches should produce identical predictions within
  tolerance.

Completion note, 2026-05-10:

- Added `inference_batch_size` to `ModelConfig`, `build_experiment_config`,
  the training CLI, saved-checkpoint inference, and benchmark summary metadata.
  The default is `16`, matching the previous `windowed_predict` default.
- `MLQDSMethod`, `windowed_predict`, and `forward_predict` now expose the
  inference batch size through the model-evaluation path.
- Validation query-F1 diagnostics now use `model_config.inference_batch_size`
  instead of `train_batch_size`, so optimizer-batch and inference-batch tuning
  are separate.
- Matched evaluation, shifted-workload evaluation, simplified CSV fallback
  evaluation, and saved-checkpoint inference all pass the configured inference
  batch size into MLQDS prediction.
- Documented `--inference_batch_size` in the QDS, experiments, and training
  READMEs.
- Verified:
  `../.venv/bin/python -m py_compile src/experiments/experiment_config.py src/experiments/experiment_cli.py src/experiments/run_ais_experiment.py src/experiments/run_inference.py src/experiments/experiment_pipeline_helpers.py src/experiments/benchmark_runtime.py src/evaluation/baselines.py src/training/train_model.py src/training/training_pipeline.py`
  and
  `../.venv/bin/python -m pytest tests/test_scaler_persisted.py tests/test_torch_runtime_controls.py`.
- Acceptance smoke:
  paired tiny `run_ais_experiment` runs with `--inference_batch_size 1` and
  `--inference_batch_size 4` produced identical `best_loss`
  (`0.13907331228256226`), identical restored `best_epoch` (`7`), and identical
  MLQDS aggregate F1 (`0.3333333333333333`). MLQDS eval latency dropped from
  `25.57 ms` to `18.61 ms` in this small smoke.

## Phase 4: Behavior-Sensitive Training Refactors

Goal: improve the actual training loop once the safer changes are measured.

### 14. Vectorize Ranking Pair Sampling On Device

Task:

- Replace CPU copies and per-pair `.item()` sampling in ranking loss with a
  vectorized tensor implementation.
- Preserve the same high-level sampling intent:
  - top-quantile point vs valid point
  - skip equal-target pairs
  - margin ranking loss

Benefit:

- Likely one of the highest-impact training-loop optimizations.
- Removes GPU-host synchronization and Python loops from the inner training
  path.

Risk:

- Medium-high.
- Exact RNG sequence will probably change.
- Training curves and tests need careful review.

Acceptance check:

```bash
cd QDS
../.venv/bin/python -m pytest tests/test_training_does_not_collapse.py
```

Then run a benchmark comparing:

- old ranking loss
- vectorized ranking loss
- final F1
- epoch time
- collapse diagnostics

### 15. Make Heavy Diagnostics Optional Or Reused

Task:

- Make range diagnostics, Oracle diagnostics, and validation F1 diagnostics
  clearly configurable.
- Reuse labels/query results where possible.

Benefit:

- Speeds up sweeps and repeated experiments.
- Keeps full diagnostics available when needed.

Risk:

- Medium.
- Diagnostics are valuable; do not hide important warnings by default in serious
  runs.

Acceptance check:

- Fast mode and full diagnostic mode should both produce valid result JSON.

### 16. Revisit Checkpoint Selection Defaults

Task:

- Review whether `checkpoint_selection_metric="loss"` should remain the
  default for serious runs.
- Consider documenting or defaulting benchmark scripts to `f1` or
  `uniform_gap`.

Benefit:

- Reduces the risk of selecting collapsed or low-utility checkpoints.

Risk:

- Medium.
- F1/uniform-gap selection is slower because it runs validation query
  evaluation.

Acceptance check:

- Benchmark scripts clearly state which checkpoint criterion they use.

### 17. Diagnose Similarity Query Generation

Task:

- Add diagnostics for similarity workloads before changing logic.
- Specifically inspect whether generated query anchors and reference snippets
  describe coherent query intent.

Benefit:

- Could improve training signal and final F1.

Risk:

- High if behavior changes are made too early.
- This is model-quality research work, not pure performance work.

Acceptance check:

- Produce a diagnostic report showing similarity answer-set sizes, support
  density, and Oracle gap over baselines.

## Recommended First Implementation Slice

The first practical slice should be:

1. Fix repo-root test/package setup.
2. Split/document requirements profiles.
3. Add the benchmark script.
4. Add `need_weights=False` to cross-attention.
5. Cache positional encodings.
6. Add device-aware MLQDS inference/evaluation.
7. Add TF32 controls.
8. Sweep training batch size.

This gives a strong performance foundation without changing the most
behavior-sensitive learning logic.

## Optional Spare-Time CUDA Stack Experiment

This is not a sprint goal. Treat it as a fun end-of-sprint experiment only if
the main plan is already producing useful benchmark results.

If there is time to spare, create a separate throwaway environment and compare
an alternate PyTorch/CUDA stack against the same benchmark artifact schema.
Do not mutate the current working `.venv`.

Only keep notes from that experiment if it produces a clear speed or iteration
benefit without making installation, reproducibility, or model-quality
interpretation worse.

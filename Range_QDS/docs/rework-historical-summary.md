# Range_QDS Query-Driven Rework Progress Summary

This document is the condensed historical progress record for the Range_QDS query-driven rework so far.


The active operating guide is:

```text
Range_QDS/docs/query-driven-rework-guide.md
```

---

## 1. Project goal

The project is building **query-driven, workload-blind AIS trajectory simplification**.

The model may train from generated or historical range workloads, but it must compress validation/eval trajectories before future validation/eval queries are known. Final success requires frozen retained masks, a stable future-query workload profile, learned model causality, and wins over uniform temporal sampling and Douglas-Peucker on `QueryUsefulV1`.

---

## 2. Historical implementation summary

### Repository and protocol cleanup

Completed:

- Standardized the project path around `Range_QDS/`.
- Moved the redesign guide into `Range_QDS/docs/` as the canonical source of truth.
- Marked old `RangeUseful` / scalar-target profiles as legacy diagnostics.
- Added final-claim separation in reports:
  - `final_claim_summary`
  - `diagnostic_summary`
  - `legacy_range_useful_summary`
  - `learning_causality_summary`
- Blocked historical-prior KNN and `range_aware` diagnostic paths from final success claims.
- Added tests for path hygiene, legacy guards, model metadata, target separation, and final-claim reporting.

### Query-driven candidate path

Implemented candidate path:

```text
workload_profile_id              = range_workload_v1
model_type                       = workload_blind_range_v2
range_training_target_mode        = query_useful_v1_factorized
selector_type                    = learned_segment_budget_v1
checkpoint_score_variant          = query_useful_v1
primary metric                   = QueryUsefulV1
```

Implemented components:

```text
range_workload_v1
QueryUsefulV1
factorized QueryUsefulV1 targets
train-derived query-prior fields
workload_blind_range_v2
learned_segment_budget_v1
frozen-mask protocol
predictability/signature/stability/causality/global-sanity gates
benchmark-level final-grid summary
```

### Important bug fixes and guardrails

Fixed or added:

- Endpoint retention sanity checks.
- Validation checkpoint scoring with the segment-budget head.
- Neutral no-segment-head ablation.
- Disabled-head neutralization in the factorized model.
- Out-of-extent prior sampling no longer clamps into fake prior mass unless explicitly using nearest mode.
- Geometry tie-breaker uses full-trajectory anchors.
- No-prior ablation preserves train extent using zeroed prior fields.
- Validation checkpoint scoring includes global-sanity penalty inputs.
- Query-local interpolation fidelity requires retained in-query evidence.
- Canonical segment IDs are used for segment-head loss and diagnostics.
- Workload-profile query plans preserve planned anchor/footprint quotas.
- Final workload-stability gate rejects tiny calibrated final workloads and exhausted generation.
- Benchmark final-grid summary blocks on support overlap and prior-predictive alignment.

### Integration verification

Current verification result:

```text
git diff --check: passed
ruff: passed
pyright: passed
tests/test_query_driven_rework.py: passed
tests/test_query_coverage_generation.py: passed
tests/test_benchmark_runner.py: passed
full test suite: passed
```

Known non-blocking warning:

```text
PyTorch nested tensor prototype warning from torch.nn.modules.transformer
```

This is not a project failure.

---

## 3. Historical research findings

Earlier workload-blind approaches were useful but insufficient.

Key findings:

- Uniform temporal sampling is a strong baseline because it naturally preserves time coverage, gap behavior, and rough shape.
- Douglas-Peucker is weaker for query utility than uniform in many range-query cells, but still matters as a final baseline.
- Query-aware `range_aware` can beat baselines, proving the workload contains exploitable signal when queries are known. It is diagnostic only.
- Scalar expected-usefulness labels, retained-frequency labels, component blends, continuity blends, local-swap targets, and teacher-student distillation did not produce robust final success.
- Historical-prior KNN had weak useful signal but is not acceptable as final learned model success.
- Large temporal scaffold ratios can hide model weakness and are not acceptable final evidence.
- The final direction should remain factorized, query-driven, and trainable.

---

## 4. Current strict debug probe

Latest strict debug probe configuration:

```text
synthetic_route_families = 1
n_ships                  = 8
n_points                 = 128
seed                     = 2323
coverage target           = 0.10
compression ratio         = 0.05
n_queries                 = 16
range_train_workload_replicates = 4
workload_profile_id       = range_workload_v1
coverage_calibration_mode = profile_sampled_query_count
workload_stability_gate_mode = final
model_type                = workload_blind_range_v2
selector_type             = learned_segment_budget_v1
```

Result:

```text
QueryUsefulV1
MLQDS:           0.0645
Uniform:         0.1190
Douglas-Peucker: 0.1478

RangeUsefulLegacy
MLQDS:           0.0369
Uniform:         0.0936
Douglas-Peucker: 0.1228
```

Final claim blocked by:

```text
workload_stability_gate
predictability_gate
prior_predictive_alignment_gate
workload_signature_gate
learning_causality_ablations
global_sanity_gates
full_coverage_compression_grid
```

Passing gates:

```text
support_overlap_gate = true
target_diffusion_gate = true
```

---

## 5. Current diagnosis

### First blocker: workload generation health

The strict profile cannot currently generate accepted train/selection workloads cleanly under the requested settings.

Observed:

```text
Train:     accepted 11/16, exhausted 6000 attempts, 5989 rejected
Selection: accepted 11/16, exhausted 6000 attempts, 5989 rejected
Eval:      accepted 16/16, 3305 attempts, 3289 rejected
```

Dominant rejection reason:

```text
too_broad
```

Secondary pressure:

```text
coverage_overshoot
```

Conclusion: accepted workloads are a filtered subset of the intended profile. Model quality should not be judged until this is fixed.

### Second blocker: workload-signature drift

Observed train-vs-eval signature failures:

```text
anchor-family L1:   0.3295
footprint-family L1:0.1477
point-hit KS:       1.0000
ship-hit KS:        0.5966
```

Planned family quotas are not enough. Accepted workload distributions must also match.

### Third blocker: prior predictability

Aggregate train-derived prior predictability is poor:

```text
Spearman:     0.0332
Kendall tau: -0.0316
PR-AUC lift:  0.9440
lift@1%:      0.0
lift@2%:      0.0
lift@5%:      0.0
lift@10%:     0.0
```

Per-head predictability:

```text
query_hit_probability:        spearman 0.0343, lift@5% 0.0
conditional_behavior_utility: spearman -0.0264, lift@5% 0.0
boundary_event_utility:       spearman 0.1245, lift@5% 0.0639
replacement_representative:   spearman 0.0311, lift@5% 0.0
segment_budget_target:        spearman 0.3177, lift@5% 1.1791
```

Conclusion: segment-level structure has some signal, but future query-hit mass is not transferring under the current workload samples.

### Fourth blocker: learned model causality

The trained model is worse than controls in the strict probe.

```text
MLQDS QueryUsefulV1:                       0.0645
shuffled-score control delta:             -0.1636
untrained-model control delta:            -0.0522
no-segment-budget-head delta:             -0.0564
prior-field-only delta:                   -0.0007
no-query-prior-features delta:             0.0000
no-behavior-head delta:                    0.0002
```

Conclusion: current trained scores are harmful. Do not tune architecture before fixing workload generation and prior predictability.

### Fifth blocker: global sanity

Endpoint and SED sanity are acceptable, but length preservation fails:

```text
endpoint_sanity = 1.0
avg_sed_ratio_vs_uniform = 0.9019
avg_length_preserved = 0.5907
required length range = [0.80, 1.20]
```
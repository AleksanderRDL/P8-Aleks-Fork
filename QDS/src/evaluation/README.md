# Evaluation Module

This module evaluates retained-point sets for a pure workload and compares
MLQDS against temporal, geometric, and oracle-style baselines.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | `MLQDSMethod`, `UniformTemporalMethod`, `DouglasPeuckerMethod`, `OracleMethod`, and diagnostic score hybrids. |
| `metrics.py` | F1 helpers and the `MethodEvaluation` container. |
| `query_cache.py` | Reusable retained-independent query/audit caches. |
| `evaluate_methods.py` | Method execution and retained-mask scoring. |
| `tables.py` | Fixed-width comparison and audit tables. |

## Methods

- `MLQDSMethod` uses the trained model, persisted scaler, explicit workload
  query set, canonical score conversion, and trajectory-local simplification.
- `UniformTemporalMethod` keeps evenly spaced points per trajectory.
- `DouglasPeuckerMethod` recursively keeps highest-error geometry points until
  the compression budget is filled.
- `OracleMethod` greedily keeps high additive-label points for the workload. It
  is a diagnostic upper reference, not an exact combinatorial optimum.
- `ScoreHybridMethod` supports residual-fill diagnostics by applying the same
  temporal base as MLQDS and filling the learned budget with supplied scores.

## Range Metrics

Range tables report two top-level scores:

- `RangePointF1`: retained in-box point-hit F1. This is the old answer metric
  renamed so tables no longer imply it captures full range usefulness.
- `RangeUseful`: versioned range-local usefulness audit score. Current
  components are point hits, ship presence, per-ship coverage, sampled
  entry/exit support, crossing brackets, temporal span, gap coverage,
  route-change coverage, and local shape fidelity.

`RangeUseful` is the canonical checkpoint target for range training, but it is
still an evolving audit objective rather than a mathematically final target.
See [`../../../Aleks-Sprint/range-objective-redesign.md`](../../../Aleks-Sprint/range-objective-redesign.md)
for the rationale.

Important component semantics:

- `ShipF1` asks whether each hit ship is represented at all.
- `ShipCov` averages per-ship point-subset coverage so dense ships do not hide
  sparse ship failures.
- `EntryExitF1` uses sampled AIS entry/exit points, not interpolated true
  boundary crossings.
- `CrossingF1` scores AIS point pairs that bracket box-boundary crossings or
  between-sample pass-throughs.
- `TemporalCov` scores retained in-query time span.
- `GapCov` penalizes large missing runs between retained in-query points.
- `TurnCov` scores route-change support inside the query.
- `ShapeScore` scores range-local route fidelity with shortcut-error style
  geometry penalties.

Non-range workloads still report answer-set `AnswerF1`/`CombinedF1` for
legacy ablations, but the current benchmark workflow is range-only.

## Reporting

- `print_method_comparison_table` prints the compact method comparison.
- `print_range_usefulness_table` prints range audit components.
- `print_geometric_distortion_table` prints SED/PED, length preservation, and
  `F1xLen`.
- `AvgPtGap` is lower-is-better; F1, usefulness, and length preservation are
  higher-is-better.

Final benchmark audits should stay exact. `EvaluationQueryCache` precomputes
retained-independent query support and is reused across MLQDS, baselines,
oracle diagnostics, and compression-ratio audits. Sampling is acceptable for
checkpoint diagnostics, not for final reported benchmark metrics.

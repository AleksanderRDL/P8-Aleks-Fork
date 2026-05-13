# Evaluation Module

Evaluates frozen retained-point masks against pure workloads and compares MLQDS
with baseline and diagnostic methods.

## Files

| File | Purpose |
| --- | --- |
| `baselines.py` | MLQDS, uniform temporal, Douglas-Peucker, oracle, and score-hybrid methods. |
| `metrics.py` | F1 helpers, range audits, and `MethodEvaluation`. |
| `query_cache.py` | Retained-independent query/audit cache. |
| `evaluate_methods.py` | Method execution and retained-mask scoring. |
| `tables.py` | Text tables for reports. |

## Methods

- `MLQDSMethod`: trained model plus persisted scaler and trajectory-local
  simplification.
- `UniformTemporalMethod`: evenly spaced points per trajectory.
- `DouglasPeuckerMethod`: geometry baseline that keeps highest-error points.
- `OracleMethod`: additive-label upper reference, not an exact optimum.
- `ScoreHybridMethod`: temporal-base residual-fill diagnostics.

## Range Metrics

Primary range scores:

- `RangeUseful`: current aggregate range usefulness audit and preferred range
  checkpoint target.
- `RangePointF1`: retained in-box point-hit F1. Useful, but too narrow for
  final claims.

Range audit components:

| Component | Meaning |
| --- | --- |
| `ShipF1` | Whether hit ships remain represented. |
| `ShipCov` | Per-ship point-subset coverage. |
| `EntryExitF1` | Sampled AIS entry/exit support. |
| `CrossingF1` | Point pairs bracketing range-boundary crossings. |
| `TemporalCov` | Retained time span inside the query. |
| `GapCov` | Penalty for large missing runs. |
| `TurnCov` | Route-change support. |
| `ShapeScore` | Range-local route fidelity. |

Non-range workloads still report answer-set `AnswerF1` and `CombinedF1` for
legacy ablations. Current benchmark work is range-only.

## Reporting Rules

- Final benchmark audits should use exact retained-mask scoring.
- Checkpoint diagnostics may use cheaper sampling where explicitly configured.
- `EvaluationQueryCache` should be reused across MLQDS, baselines, oracle
  diagnostics, and compression-ratio audits.
- Report component tables with aggregate scores. Aggregate-only improvements
  are not enough to understand failures.

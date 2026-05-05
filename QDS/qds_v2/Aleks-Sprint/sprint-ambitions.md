# Sprint Ambitions: Query-Driven Simplification Models

## Context

The current QDS v2 work lives in `QDS/qds_v2`. It is a Python research pipeline for machine learned query-driven simplification of AIS trajectories.

The pipeline currently follows this flow:

1. Load AIS trajectories or generate synthetic AIS data.
2. Generate typed query workloads.
3. Build per-point, per-query-type importance labels.
4. Train a query-conditioned trajectory model.
5. Simplify trajectories by retaining a fixed compression budget per trajectory.
6. Evaluate simplified trajectories against original trajectories with query F1 and geometric distortion metrics.

The four supported query workloads are:

- `range`
- `knn`
- `similarity`
- `clustering`

## Current Findings

The current system is structurally coherent, but the learned scorer is still the weak part.

The model is not directly trained to maximize final query F1. Final query F1 is measured only after scoring, top-k simplification, query execution, and answer comparison. Instead, training first builds heuristic per-point labels derived from query behavior, then teaches the model to rank those labeled points highly.

This proxy is useful, but imperfect. It can label many points in the same useful region even though only a few retained representatives are needed under the final compression budget.

Recent code already addresses several earlier issues:

- Oracle is now evaluated on the same eval/test workload as MLQDS and baselines.
- Coverage-targeted query generation keeps `n_queries` as a minimum.
- Range evaluation is point-aware.
- kNN, similarity, and clustering evaluation include retained point-support penalties in addition to answer-set F1.
- Training includes collapse diagnostics, pointwise BCE, gradient clipping, and optional validation F1 checkpoint selection.
- MLQDS can use a temporal base plus learned score fill through `mlqds_temporal_fraction`.

Recent results show that a temporal-hybrid MLQDS setup can slightly beat Random and new uniform temporal sampling, but pure learned scoring still struggles. The sprint should therefore focus on making the learned query-specific scorer genuinely useful, not only relying on the temporal base.

## Primary Objective

The primary objective is to train four specialized machine learned query-driven simplification models:

1. A Range-QDS model for range query workloads.
2. A kNN-QDS model for kNN query workloads.
3. A Similarity-QDS model for similarity query workloads.
4. A Clustering-QDS model for clustering query workloads.

Each model should be trained, selected, and evaluated primarily on its own query workload.

Mixed-workload or cross-workload evaluation can still be useful as diagnostics, but it is not the main success criterion for this sprint.

## Desired Behavior

A successful MLQDS model should learn the marginal value of retaining each point under a fixed compression budget.

In practice, this means the model should select points that preserve query answers after simplification, not merely points that look geometrically important or points that individually fall inside many query regions.

The desired behavior is:

1. **Budget-aware selection**

   The model should understand that only a limited fraction of points can be retained. If many adjacent points provide the same query value, the model should avoid wasting the budget on all of them.

2. **Query-answer preservation**

   The simplified trajectories should answer the target query workload as similarly as possible to the original trajectories.

3. **Workload-specific specialization**

   Each of the four models should specialize in one query workload:

   - Range-QDS should preserve range query point hits.
   - kNN-QDS should preserve nearest representative points and returned trajectory IDs.
   - Similarity-QDS should preserve trajectory snippets, local shape, and movement patterns needed for similarity matching.
   - Clustering-QDS should preserve trajectory representatives and co-membership structure.

4. **Non-redundant retention**

   The model should avoid selecting many neighboring points with nearly identical query value. It should spread retained points across useful regions and trajectories.

5. **Baseline superiority at equal compression**

   Each specialized model must beat typical simplification methods at the same compression ratio:

   - Random
   - new uniform temporal sampling
   - Douglas-Peucker

6. **Per-type success**

   A model should not hide weak performance behind aggregate scores. For this sprint, each specialized model should win on its own query type.

7. **Geometry should remain reasonable**

   Query F1 is the primary objective, but simplified trajectories should not become geometrically broken. Length loss, SED/PED distortion, temporal spacing, and trajectory shape should remain within acceptable bounds.

## Success Criteria

The sprint should be judged by workload-specific wins:

- Range-QDS beats baselines on range F1.
- kNN-QDS beats baselines on kNN F1.
- Similarity-QDS beats baselines on similarity F1.
- Clustering-QDS beats baselines on clustering F1.

All comparisons must use the same compression ratio.

Useful secondary criteria:

- The learned model should beat Random and new uniform temporal without depending entirely on a large temporal base.
- Oracle should remain substantially above all methods, confirming that the label/evaluation setup still contains learnable signal.
- Validation checkpoint selection should prefer models that improve final query F1, not merely training loss.
- Geometric distortion should be reported alongside query F1.

## Recommended Sprint Direction

Prioritize proving one specialized model at a time.

Suggested order:

1. Range-QDS
2. kNN-QDS
3. Similarity-QDS
4. Clustering-QDS

Range and kNN are good first targets because their query behavior is more local and easier to inspect. Similarity and clustering are more global and may need stronger trajectory-level or set-aware supervision.

The key technical direction is to move from point-label prediction toward set-aware and budget-aware learning. The model should learn which retained subset preserves query answers, not only which individual points receive heuristic positive labels.

## Data And Pretraining Layer Findings

There is no separate `pretrain` module in `qds_v2`. The current data/pretraining layer is the combination of:

- AIS CSV loading in `src/data/ais_loader.py`.
- Trajectory flattening and boundary construction in `src/data/trajectory_dataset.py`.
- Query workload generation in `src/queries/query_generator.py`.
- Query-derived importance label construction in `src/training/importance_labels.py`.
- Feature scaling and window batching in `src/training/scaler.py` and `src/training/trajectory_batching.py`.

The cleaned AIS files in `AISDATA/cleaned` match the current v2 loader assumptions reasonably well. They contain columns such as `MMSI`, `# Timestamp`, `Latitude`, `Longitude`, `SOG`, and `COG`, which the loader can resolve through aliases.

The upstream cleaning pipeline already performs useful work:

- Keeps Class A AIS rows.
- Removes duplicate output rows.
- Trims repeated stationary points.
- Fills or removes undefined ship types.
- Removes selected ship types.
- Removes GPS outliers.

This is a solid start, but the model-specific data layer still needs work for the sprint ambition.

### Observed Data Characteristics

On `AISDATA/cleaned/aisdk-2026-01-01.cleaned.csv`, the cleaned daily file contains:

- `3,328,209` rows.
- `1,533` unique MMSIs.
- Median MMSI length of `338` points.
- 75th percentile MMSI length of `3,156` points.
- 99th percentile MMSI length of `14,234` points.
- Maximum MMSI length of `27,139` points.
- `715` MMSIs with more than `500` points.
- `32,367` rows with missing or invalid `COG`.
- `29,001` duplicate timestamp occurrences within MMSI.
- Maximum within-MMSI time gap of `16.9` hours.
- `22` MMSIs with a max time gap above `6` hours.

These numbers matter because the current loader treats each MMSI in the loaded file as one trajectory.

### Main Risk: Trajectory Definition

The current trajectory unit is effectively:

> one MMSI over the loaded file equals one trajectory

For query-driven simplification, this is likely too coarse. A vessel can have separate trips, long inactive periods, or large temporal gaps inside one daily MMSI track.

This affects the four specialized models differently:

- Range-QDS may tolerate this better because range labels are local point hits.
- kNN-QDS can become noisy because the answer unit is the whole trajectory ID, even if only one local part of a long vessel track is relevant.
- Similarity-QDS is especially sensitive because unrelated movement segments inside one MMSI can dilute trajectory-level similarity.
- Clustering-QDS is also sensitive because trajectory representatives and co-membership structure depend heavily on the trajectory unit.

The data layer should therefore segment trajectories by MMSI plus temporal continuity, not just MMSI.

### Loader And Feature Issues

The current loader drops rows with missing or invalid COG because COG is used as the heading feature. This prevents NaNs from reaching the scaler, but it also removes real AIS positions and can create artificial gaps.

A better approach may be:

- Use COG when valid.
- Derive movement bearing from consecutive latitude/longitude positions when COG is missing.
- Add a missing-heading flag if needed.

The current feature set is also thin:

- time
- latitude
- longitude
- speed
- COG/heading
- `is_start`
- `is_end`
- optional `turn_score`

For the four specialized models, richer local features may help:

- delta time to previous/next point
- distance to previous/next point
- implied speed
- acceleration or speed change
- movement bearing from coordinates
- heading change
- local turn intensity
- local point density
- normalized progress through segment
- ship type, if useful and encoded safely

Range and kNN may work with simpler features. Similarity and clustering likely need stronger trajectory-shape and local-context features.

### Scaling And Runtime Issues

The loader currently reads the full CSV with Pandas. The cleaned daily files are roughly `268-377 MB` each, with `3.3-4.7 million` rows per day.

This can work for small experiments, but it is not ideal for training four specialized models across multiple days.

The sprint should add a cached preprocessing stage:

```text
cleaned CSV -> validated trajectory segments -> cached tensors/parquet/pt artifacts
```

This would make repeated range/kNN/similarity/clustering experiments much faster and more reproducible.

The existing `max_points_per_ship` option exists in the loader but is not exposed in the experiment config or CLI. Large trajectory control should be exposed as normal run configuration.

Useful controls:

- `min_points_per_segment`
- `max_points_per_segment`
- `max_time_gap_seconds`
- `max_segments`
- optional per-day or per-MMSI sampling

### Query And Label Preparation Issues

The current label construction is still proxy-based and not budget-aware.

Current behavior:

- Range labels reward points inside query boxes.
- kNN labels representative in-window points from returned trajectories.
- Similarity labels reference-nearest support points from returned trajectories.
- Clustering labels points in trajectories that participate in non-noise co-membership structure.

This is useful scaffolding, but it does not fully satisfy the sprint ambition. The desired models should learn which retained subset preserves query answers under a fixed budget.

For specialized models, labels should become more query-type-specific and budget-aware:

- Range-QDS should avoid labeling redundant dense in-box points as equally valuable.
- kNN-QDS should focus on points that preserve nearest-trajectory membership.
- Similarity-QDS should focus on shape-defining snippets and reference-nearest local structures.
- Clustering-QDS should focus on trajectory representatives that preserve cluster co-membership.

Query coverage and label generation currently scan large point tensors. With millions of points and hundreds or thousands of queries, this can become expensive. Workloads and labels should be cached per query type and per dataset split.

### Environment Finding

The current local Python environments are not aligned:

- System Python has `numpy`, but not `torch`, `pandas`, or `pytest`.
- The repo `.venv` has `pandas` and `pytest`, but not `torch`.

This blocks local end-to-end training and test execution until the environment is fixed.

### Data Layer Work Needed For The Sprint

The four specialized-model ambition requires data/pretraining work, not only model changes.

Priority work:

1. Add a cached preprocessing stage from cleaned CSVs to validated trajectory segment artifacts.
2. Segment trajectories by MMSI plus time gap or voyage continuity.
3. Add data audit reports for row drops, invalid COG, duplicate timestamps, segment lengths, and time gaps.
4. Expose segment and sampling controls in `DataConfig` and the CLI.
5. Build query-type-specific workload and label configs for range, kNN, similarity, and clustering.
6. Cache generated workloads and labels per query type.
7. Track positive label fraction, Oracle F1, and baseline F1 for each specialized training set.
8. Make labels more budget-aware before adding more model complexity.
9. Add smoke tests using real cleaned CSV slices, not only synthetic trajectories.

The practical recommendation is:

> Fix trajectory segmentation and cached preprocessing before making major architecture changes.

The specialized models will only be as good as the trajectory units, query workloads, and label targets they are trained on.

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


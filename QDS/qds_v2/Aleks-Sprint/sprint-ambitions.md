# Sprint Ambitions: Query-Driven Simplification Models

## Executive Summary

The goal of this sprint is to move `QDS/qds_v2` from a promising prototype into a credible research pipeline for machine learned query-driven simplification of AIS trajectories.

The primary ambition is to train four specialized models:

1. **Range-QDS** for range query workloads.
2. **kNN-QDS** for k-nearest-neighbor query workloads.
3. **Similarity-QDS** for trajectory similarity workloads.
4. **Clustering-QDS** for clustering workloads.

Each specialist model should be trained, selected, and evaluated primarily on its own workload. Mixed-workload and cross-workload experiments can remain useful diagnostics, but they are not the main success criterion.

The current system has a coherent end-to-end structure, but it is not yet strong enough to support the desired claims. The main gaps are:

- trajectory units are too coarse because one MMSI-day is currently treated as one trajectory
- query generation is not yet workload-specific or difficulty-controlled enough
- label construction is still a proxy for final retained-set utility
- the model attends over full query workloads, which may not scale to the desired query volumes
- simplification uses fixed per-trajectory budgets, not workload-aware global budgets
- benchmarking is single-run oriented and does not yet provide four-model, multi-seed evidence against strong baselines
- environment, artifact, and visualization support need cleanup for reproducible sprint work

The practical sprint direction is:

> Build a reliable specialist-model pipeline before scaling architecture complexity.

The highest leverage sequence is:

1. Fix trajectory segmentation and cached preprocessing.
2. Build validated workload generators for each query type.
3. Improve labels and training diagnostics around retained-set utility.
4. Add a four-specialist, multi-seed benchmark runner.
5. Use visualization and per-query diagnostics to debug why models win or lose.

## Current Pipeline

The current QDS v2 work lives in `QDS/qds_v2`.

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

Several important improvements are already present:

- The local sprint branch now has documented environment commands, a Makefile,
  segmented CSV loading, Parquet-backed trajectory caching, and data audit
  metadata.
- Range workload acceptance filters and diagnostics are implemented for
  hit-count bands, broad boxes, duplicate boxes, query JSONL output, positive
  label signal, label Oracle, `uniform`, and Douglas-Peucker.
- Oracle is evaluated on the same eval/test workload as MLQDS and baselines.
- Coverage-targeted query generation keeps `n_queries` as a minimum.
- Range evaluation is point-aware.
- kNN, similarity, and clustering evaluation include retained point-support penalties in addition to answer-set F1.
- Training includes collapse diagnostics, pointwise BCE, gradient clipping,
  optional validation F1 checkpoint selection, uniform-gap checkpoint selection,
  and optional rolling checkpoint-score smoothing.
- MLQDS can use a temporal base plus learned score fill through `mlqds_temporal_fraction`.
- MLQDS now rank-normalizes each query-type head within each trajectory before
  workload mixing, so mixed workloads are less sensitive to uncalibrated head
  magnitudes.
- Result reporting now distinguishes query F1 from shape preservation through
  SED/PED, `LengthPres`, `F1xLen`, and explicit MLQDS gaps versus the main
  baselines.
- The primary result table now separates `AnswerF1` from `CombinedF1`, where
  `AnswerF1` is the pure query-answer score and `CombinedF1` is the legacy
  answer/support product retained for diagnostics and optional checkpoint
  selection.
- The geometric baseline is now a true recursive Douglas-Peucker implementation
  rather than the earlier first-last chord proxy.
- Large-tensor sampling and quantile safeguards are in place for combined-day
  AIS workloads. The incoming range sqrt-normalization change is intentionally
  not part of the accepted sprint baseline.

The remaining issue is not that the pipeline is conceptually broken. The issue is that the learned scorer is not yet reliably learning the retained subset that preserves final query answers.

## Core Diagnosis

The model is not directly trained to maximize final query F1.

Final query F1 is measured only after:

1. model scoring
2. top-k simplification
3. query execution on simplified trajectories
4. answer comparison against original trajectories

Training instead builds heuristic per-point labels from query behavior and teaches the model to rank those labeled points highly.

This proxy is useful, but imperfect. It can label many nearby points in the same useful region even though only a few representatives are needed under the final compression budget.

The desired learning problem is therefore not simply:

```text
predict high scores for positive proxy labels
```

It is:

```text
choose the retained subset that preserves the query workload under budget
```

Recent results suggest the temporal-hybrid MLQDS setup can slightly beat uniform temporal sampling in some runs, but pure learned scoring still struggles. The sprint should focus on making the learned query-specific scorer genuinely useful, not only relying on the temporal base.

## Success Criteria

The sprint should be judged by workload-specific wins:

- Range-QDS beats baselines on range F1.
- kNN-QDS beats baselines on kNN F1.
- Similarity-QDS beats baselines on similarity F1.
- Clustering-QDS beats baselines on clustering F1.

All comparisons must use the same compression ratio.

The primary comparison should be against:

- uniform temporal sampling
- true Douglas-Peucker
- label Oracle as a diagnostic upper reference

Secondary success criteria:

- The learned model should beat uniform temporal without depending entirely on a large temporal base.
- The label Oracle should remain substantially above all methods, confirming that the label/evaluation setup still contains learnable signal.
- Validation checkpoint selection should prefer models that improve final query F1, not merely proxy loss.
- Geometric distortion should be reported alongside query F1.
- Results should be repeated across seeds, not based on a single favorable run.

## Desired Model Behavior

A successful MLQDS model should learn the marginal value of retaining each point under a fixed compression budget.

In practice, this means the model should select points that preserve query answers after simplification, not merely points that look geometrically important or points that individually fall inside many query regions.

The desired behavior is:

1. **Budget-aware selection**

   The model should understand that only a limited fraction of points can be retained. If many adjacent points provide the same query value, the model should avoid wasting budget on all of them.

2. **Query-answer preservation**

   The simplified trajectories should answer the target query workload as similarly as possible to the original trajectories.

3. **Workload-specific specialization**

   Each specialist model should focus on one query workload:

   - Range-QDS should preserve range query point hits.
   - kNN-QDS should preserve nearest representative points and returned trajectory IDs.
   - Similarity-QDS should preserve snippets, local shape, and movement patterns needed for similarity matching.
   - Clustering-QDS should preserve trajectory representatives and co-membership structure.

4. **Non-redundant retention**

   The model should avoid selecting many neighboring points with nearly identical query value. It should spread retained points across useful regions and trajectories.

5. **Baseline superiority at equal compression**

   Each specialist model should beat typical simplification methods at the same compression ratio.

6. **Per-type success**

   A model should not hide weak performance behind aggregate scores. For this sprint, each specialist model should win on its own query type.

7. **Reasonable geometry**

   Query F1 is the primary objective, but simplified trajectories should not become geometrically broken. Length loss, SED/PED distortion, temporal spacing, and trajectory shape should remain within acceptable bounds.

## Sprint Strategy

Prioritize proving one specialist model at a time.

Suggested order:

1. Range-QDS
2. kNN-QDS
3. Similarity-QDS
4. Clustering-QDS

Range and kNN are good first targets because their query behavior is more local and easier to inspect. Similarity and clustering are more global and may need stronger trajectory-level or set-aware supervision.

The key technical direction is to move from point-label prediction toward set-aware and budget-aware learning. The model should learn which retained subset preserves query answers, not only which individual points receive heuristic positive labels.

## Data And Preprocessing

There is no separate `pretrain` module in `qds_v2`. The current data/pretraining layer is the combination of:

- AIS CSV loading in `src/data/ais_loader.py`
- trajectory flattening and boundary construction in `src/data/trajectory_dataset.py`
- query workload generation in `src/queries/query_generator.py`
- query-derived importance label construction in `src/training/importance_labels.py`
- feature scaling and window batching in `src/training/scaler.py` and `src/training/trajectory_batching.py`

The cleaned AIS files in `AISDATA/cleaned` match the current v2 loader assumptions reasonably well. They contain columns such as `MMSI`, `# Timestamp`, `Latitude`, `Longitude`, `SOG`, and `COG`, which the loader can resolve through aliases.

The upstream cleaning pipeline already performs useful work:

- keeps Class A AIS rows
- removes duplicate output rows
- trims repeated stationary points
- fills or removes undefined ship types
- removes selected ship types
- removes GPS outliers

This is a solid start, but the model-specific data layer still needs work.

### Observed Data Characteristics

On `AISDATA/cleaned/aisdk-2026-01-01.cleaned.csv`, the cleaned daily file contains:

- `3,328,209` rows
- `1,533` unique MMSIs
- median MMSI length of `338` points
- 75th percentile MMSI length of `3,156` points
- 99th percentile MMSI length of `14,234` points
- maximum MMSI length of `27,139` points
- `715` MMSIs with more than `500` points
- `32,367` rows with missing or invalid `COG`
- `29,001` duplicate timestamp occurrences within MMSI
- maximum within-MMSI time gap of `16.9` hours
- `22` MMSIs with a max time gap above `6` hours

These numbers matter because the current loader treats each MMSI in the loaded file as one trajectory.

### Main Risk: Trajectory Definition

The current trajectory unit is effectively:

```text
one MMSI over the loaded file equals one trajectory
```

For query-driven simplification, this is likely too coarse. A vessel can have separate trips, long inactive periods, or large temporal gaps inside one daily MMSI track.

This affects the specialist models differently:

- Range-QDS may tolerate this better because range labels are local point hits.
- kNN-QDS can become noisy because the answer unit is the whole trajectory ID, even if only one local part of a long vessel track is relevant.
- Similarity-QDS is especially sensitive because unrelated movement segments inside one MMSI can dilute trajectory-level similarity.
- Clustering-QDS is also sensitive because trajectory representatives and co-membership structure depend heavily on the trajectory unit.

The data layer should segment trajectories by MMSI plus temporal continuity, not just MMSI.

### Loader And Feature Issues

The current loader drops rows with missing or invalid COG because COG is used as the heading feature. This prevents NaNs from reaching the scaler, but it also removes real AIS positions and can create artificial gaps.

A better approach may be:

- use COG when valid
- derive movement bearing from consecutive latitude/longitude positions when COG is missing
- add a missing-heading flag if needed

The current feature set is thin:

- time
- latitude
- longitude
- speed
- COG/heading
- `is_start`
- `is_end`
- optional `turn_score`

For the specialist models, richer local features may help:

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

### Scaling And Caching

The loader currently reads the full CSV with Pandas. The cleaned daily files are roughly `268-377 MB` each, with `3.3-4.7 million` rows per day.

This can work for small experiments, but it is not ideal for training four specialist models across multiple days.

The sprint should add a cached preprocessing stage:

```text
cleaned CSV -> validated trajectory segments -> cached tensors/parquet/pt artifacts
```

Useful data controls:

- `min_points_per_segment`
- `max_points_per_segment`
- `max_time_gap_seconds`
- `max_segments`
- optional per-day or per-MMSI sampling

The existing `max_points_per_ship` option exists in the loader but is not exposed in the experiment config or CLI. Large trajectory control should be exposed as normal run configuration.

### Data Work Needed

Priority work:

1. Add cached preprocessing from cleaned CSVs to validated trajectory segment artifacts.
2. Segment trajectories by MMSI plus time gap or voyage continuity.
3. Add data audit reports for row drops, invalid COG, duplicate timestamps, segment lengths, and time gaps.
4. Expose segment and sampling controls in `DataConfig` and the CLI.
5. Cache generated workloads and labels per query type and dataset split.
6. Track positive label fraction, label Oracle F1, and baseline F1 for each specialist training set.
7. Add smoke tests using real cleaned CSV slices, not only synthetic trajectories.

Practical recommendation:

> Fix trajectory segmentation and cached preprocessing before making major architecture changes.

The specialist models will only be as good as the trajectory units, query workloads, and label targets they are trained on.

## Query Generation And Sampling

The query generator is a critical part of the training pipeline. For the four-specialist ambition, query quality matters as much as model quality.

The current generator already has useful behavior:

- supports all four query types
- normalizes workload mixes
- produces padded query feature tensors and type IDs
- uses density-biased anchor sampling for range and kNN
- keeps `n_queries` as a minimum when coverage mode is enabled
- allows range query footprint control through `range_spatial_fraction` and `range_time_fraction`
- allows configurable kNN `k` through `knn_k`
- has tests for coverage mode, density bias, smaller range footprints, and configured kNN `k`

This is enough for prototype experiments, but not enough for robust specialist MLQDS training.

### Main Query Sampling Risks

The current generator is too generic. It does not yet guarantee that generated queries are informative training examples.

Missing controls:

- query difficulty
- answer cardinality
- positive label density
- per-query label Oracle score
- empty/easy/hard query rejection
- query diversity across space, time, density, trajectory length, and vessel behavior

Point coverage is currently used as the main workload coverage signal. This is useful, but it is not the same as query-answer quality. A workload can cover many points while still producing weak or redundant training signal.

Specialist workloads should track:

- point coverage
- trajectory coverage
- answer set size
- support point count
- positive label fraction
- empty-query rate
- overly broad query rate
- repeated or near-duplicate query rate
- label Oracle F1

### Range Queries

Range query generation is currently the strongest and most controllable part of the generator.

Existing controls:

- spatial footprint fraction
- time footprint fraction
- density-biased anchors
- coverage targeting

Needed improvements:

- reject empty boxes
- reject boxes that hit too much of the dataset
- target useful hit-count bands
- track point-hit distribution
- avoid redundant overlapping boxes unless overlap is intentional

Range-QDS should train on boxes that produce meaningful point-level preservation pressure, not only broad boxes that reward uniform sampling.

### kNN Queries

kNN generation currently uses density-biased anchors and configurable `k`, which is good.

However, the time window is fixed as:

```text
t_half_window = 0.25 * dataset_time_span
```

On a one-day dataset, this is roughly a 6-hour half-window. That may be too broad for local kNN behavior.

Needed improvements:

- configurable kNN time-window distribution
- configurable `k` distribution instead of only one fixed `k`
- answer-cardinality and distance-margin diagnostics
- rejection of ambiguous queries where many trajectories are nearly tied
- distinction between dense-port kNN and open-water kNN

kNN-QDS should train on queries where retaining the correct local representative points actually matters.

### Similarity Queries

Similarity generation is currently weaker.

Current behavior:

- picks a random anchor point for the query centroid
- picks a random trajectory and 5-point reference snippet
- uses fixed radius fraction
- hardcodes `top_k=5`
- query features only include the mean of the reference snippet

Risks:

- the query centroid can be weakly related to the reference snippet
- a 5-point reference may be too short or too noisy
- mean reference features lose most shape information
- fixed `top_k` and radius may produce unstable difficulty

Needed improvements:

- configurable reference snippet length
- tie query centroid more directly to the reference trajectory or intended candidate region
- configurable similarity `top_k`
- configurable radius and time window distributions
- query acceptance based on answer count and distance separation
- hard-negative sampling for similar but wrong trajectories
- richer query encoding for shape, not only reference mean

Similarity-QDS likely needs the most generator work before model changes will pay off.

### Clustering Queries

Clustering generation currently reuses range boxes and adds:

```text
eps = 0.02 * max(dataset_lat_span, dataset_lon_span)
min_samples = random integer from 3 to 6
```

Risks:

- `eps` is in coordinate degrees, not kilometers
- global span-based `eps` changes meaning across datasets
- generated queries may produce no clusters or trivial clusters
- cluster structure can be unstable under small changes in query region

Needed improvements:

- calibrate `eps` in kilometers or local projected units
- tune `eps` using local density
- reject queries with no meaningful co-membership pairs
- track cluster count, noise fraction, and pair-count distribution
- generate both dense-region and sparse-region clustering workloads

Clustering-QDS should train on queries with stable and non-trivial cluster structure.

### Query Workload Caching

Generated workloads should be cached per:

- dataset split
- query type
- query generator config
- seed
- trajectory segmentation version

The cache should store:

- typed query dictionaries
- padded query features
- type IDs
- coverage metadata
- answer-set metadata
- label statistics
- label Oracle score
- baseline scores if available

Practical recommendation:

> Do not treat query generation as a minor utility. It is part of the learning objective.

If query sampling is noisy, too easy, too broad, or unbalanced, the learned model can fail even if the architecture is otherwise adequate.

## Data Volume Targets

The needed AIS data volume should be judged at two levels:

1. Number of cleaned AIS days.
2. Number of usable trajectory segments and query-label examples after segmentation.

Days alone are not enough, because one day can contain many long MMSI tracks, noisy gaps, stationary behavior, and uneven traffic density. The model should ultimately train on validated trajectory segments, not raw MMSI-day tracks.

### Minimum Viable Target

Use `5` cleaned AIS days:

```text
Train: 3 days
Validation: 1 day
Test: 1 day
```

This is enough to debug the four specialist training pipelines and check whether Range-QDS and kNN-QDS have learnable signal. It is not enough for strong research claims.

### Practical Sprint Target

Use `10` cleaned AIS days.

The current workspace already appears to contain `10` cleaned daily files in `AISDATA/cleaned`, so this is the most practical sprint target.

Suggested split:

```text
Train: Jan 01-Jan 07
Validation: Jan 08
Test: Jan 09-Jan 10
```

This should be enough to test whether each specialist model can beat uniform temporal sampling and Douglas-Peucker on same-region, same-period AIS data.

For this sprint, `10` cleaned days should be treated as the main target.

### Ideal Research Target

Use `30` cleaned AIS days.

Suggested split:

```text
Train: 20 days
Validation: 5 days
Test: 5 days
```

This gives more temporal variation and reduces overfitting to a few days of traffic, weather, port activity, dense regions, or specific MMSIs.

At the observed cleaned data scale, `30` days would likely mean more than `100M` cleaned AIS rows before segmentation. This makes cached preprocessing, segment sampling, and cached query-label artifacts mandatory.

### Target Query And Segment Scale

For specialist model training, target query-label examples per workload:

- Range-QDS: `10k-50k` generated range queries
- kNN-QDS: `10k-30k` generated kNN queries
- Similarity-QDS: `2k-10k` generated similarity queries
- Clustering-QDS: `2k-10k` generated clustering queries

Similarity and clustering can use fewer queries initially because their execution and label construction are heavier.

For trajectory segments:

- minimum useful scale: `10k+` usable trajectory segments
- preferred robust scale: `30k-100k` usable trajectory segments

These should be counted after MMSI/time-gap segmentation and quality filtering.

### Growth Plan

Start with the existing `10` cleaned days for the sprint.

Then run a learning-curve study:

```text
3 days -> 5 days -> 10 days -> 20 days -> 30 days
```

For each data volume, compare each specialist model against the same baselines at the same compression ratio.

If performance still improves meaningfully from `10` to `30` days, then data volume and preprocessing are still bottlenecks. If performance plateaus before `10` days, the bottleneck is more likely label quality, budget-aware training, model architecture, or query workload design.

## Labels, Model Training, And Simplification

This layer determines whether the four specialist models can actually learn behavior that beats the baselines, rather than only producing better diagnostics around the same failure modes.

### Label Construction

The current label construction is still proxy-based and not budget-aware.

Current behavior:

- Range labels reward points inside query boxes.
- kNN labels representative in-window points from returned trajectories.
- Similarity labels reference-nearest support points from returned trajectories.
- Clustering labels points in trajectories that participate in non-noise co-membership structure.

This is useful scaffolding, but it does not fully satisfy the sprint ambition. The desired models should learn which retained subset preserves query answers under a fixed budget.

For specialist models, labels should become more query-type-specific and budget-aware:

- Range-QDS should avoid labeling redundant dense in-box points as equally valuable.
- kNN-QDS should focus on points that preserve nearest-trajectory membership.
- Similarity-QDS should focus on shape-defining snippets and reference-nearest local structures.
- Clustering-QDS should focus on trajectory representatives that preserve cluster co-membership.

### Query-Conditioning Scale Risk

The current model conditions point scores on the full query workload. Each trajectory window attends over the generated query feature tensor.

This is workable for prototype workloads with hundreds of queries. It is not obviously scalable to the target query volumes listed above.

The current chunked cross-attention makes large workloads more memory-safe, but the computation still grows with query count. Once the workload is larger than the query chunk size, attention is also an approximation because each chunk is softmaxed independently and then averaged.

For large specialist workloads, the sprint likely needs one of these strategies:

- query sampling during training
- compact workload summary embeddings
- query clustering or query prototypes
- cached per-point workload labels without feeding every query through the model
- specialist models that consume compact query-type descriptors instead of full query lists

Practical recommendation:

> Large query volume cannot simply be added by increasing `n_queries`.

The model-conditioning path must be made compatible with the intended workload scale.

### Training Objective Risk

The current training objective is point-ranking based. It trains the model to assign higher scores to points with higher proxy labels.

Final evaluation depends on:

1. predicted point scores
2. top-k retained subset selection
3. query execution on the simplified data
4. answer-set F1 and point-support preservation

This creates a mismatch:

```text
training learns point label ranking
evaluation scores retained set behavior
```

For the sprint, this does not require fully differentiable query execution, but the training objective should become more budget-aware.

Useful intermediate improvements:

- train against points that survive top-k and improve F1
- add listwise or top-k-aware losses
- include hard negatives that steal budget but do not improve query F1
- add supervision for non-redundant representatives instead of dense positive regions
- evaluate validation checkpoints by query F1 or uniform-gap, not only proxy loss

The core learning goal is:

```text
choose the retained subset that preserves the query workload under budget
```

### Negative Supervision Gap

The current training loop skips type-specific windows that contain no positive label for that type.

This avoids all-zero collapse, but it also means the model gets weaker pressure to keep irrelevant regions low. False high scores in irrelevant regions can still steal top-k budget during simplification.

The sprint should add stronger negative/background supervision:

- sample negative-only windows
- include hard negatives near query boundaries
- include hard negatives from trajectories that are close but not returned
- track false-positive score mass in diagnostics

This is especially important for kNN and similarity, where many nearby but wrong points can look plausible.

### Simplification Policy Risk

The current simplifier uses per-trajectory top-k retention. Each trajectory keeps roughly the same fraction of points, and endpoints are always preserved.

This makes comparisons fair and simple, but it is not fully query-driven.

Current limitation:

```text
fixed compression ratio per trajectory
```

Desired future behavior:

```text
fixed global or workload-aware budget across the dataset
```

For true query-driven simplification, the model may need to spend more retained-point budget on trajectories that matter for the workload and less budget on trajectories that are irrelevant.

This matters especially for range and kNN workloads. If only a subset of vessels or regions matters, a per-trajectory budget wastes retained points on trajectories that do not help query F1.

The temporal-hybrid simplifier is useful because it prevents badly broken trajectories, but the sprint should not rely entirely on a large temporal base. The learned scorer should eventually make meaningful query-driven choices beyond uniform temporal coverage.

### Specialist Model Configuration

The current architecture already has four output heads, one per query type. That is useful for mixed-workload experiments.

However, the sprint objective is four specialized models, not one universal model.

For specialist training, each model should have:

- one primary workload config
- one primary validation workload
- one primary checkpoint-selection metric
- one primary result card
- optional cross-workload diagnostics

The default checkpoint selection should be aligned with the sprint goal. A default of training loss is useful for debugging, but final specialist runs should prefer:

- validation query F1
- validation uniform-gap selection

The current `turn_aware` model is not a fundamentally different architecture. It is the same query-conditioned model with an extra turn-score feature. It may be useful, but it should not distract from the core objective, labels, and simplification policy.

### Model And Training Work Needed

Priority work:

1. Decide how large specialist query workloads will condition the model without feeding every query through every forward pass.
2. Add query sampling, workload summaries, or query prototypes for scalable training.
3. Add stronger negative and hard-negative supervision.
4. Add top-k-aware or retained-subset-aware training diagnostics.
5. Use validation query F1 or uniform-gap selection for final specialist runs.
6. Investigate global or workload-aware budget allocation beyond fixed per-trajectory ratios.
7. Keep temporal-hybrid simplification as a stabilizer, but measure pure learned contribution separately.
8. Treat `turn_aware` as a feature variant, not a core architecture solution.

Practical recommendation:

> Do not move directly from better query generation to bigger model training.

The model-conditioning path, loss objective, and simplification policy must support the four-specialist ambition first.

## Benchmarking, Experimentation, And Metrics

The current benchmarking layer is useful for prototype experiments, but it is not yet strong enough to prove the sprint ambition.

The existing layer already has important foundations:

- matched-budget evaluation at the same compression ratio
- baselines for uniform temporal sampling, true recursive Douglas-Peucker geometry, and Oracle
- aggregate and per-query-type `AnswerF1` / `CombinedF1` reporting
- point-aware range query scoring
- kNN, similarity, and clustering scoring that combines answer-set F1 with retained point-support preservation
- geometric distortion reporting through SED, PED, length preservation, and `F1xLen`
- optional validation query-F1 checkpoint selection
- JSON and text table outputs for individual runs

This is enough to reveal whether a run is promising or failing. It is not enough to claim that MLQDS beats typical algorithms across range, kNN, similarity, and clustering workloads.

### Benchmark Matrix

The current experiment pipeline runs one model with one train workload mix and one eval workload mix.

For the sprint ambition, the core benchmark should be:

```text
Train model       Eval: range   Eval: kNN   Eval: similarity   Eval: clustering
Range-QDS
kNN-QDS
Similarity-QDS
Clustering-QDS
```

The diagonal is the primary success criterion. Cross-workload scores are useful diagnostics, but they should not define success.

The current shift table is not a reliable workload-shift benchmark yet. It only reports MLQDS aggregate F1, does not compare baselines, and the train-workload side evaluates train-generated queries against test points. That makes the number hard to interpret as true generalization.

The shift benchmark should instead use explicit eval workloads generated from the eval/test split for each query type and should include all baselines.

### Baselines

The baseline set is directionally correct, but it needs tightening before final claims.

Uniform temporal sampling is currently the strongest practical baseline and should remain a primary comparison target. In code and result JSON, keep the stable method key `uniform`.

Douglas-Peucker is now implemented as a true recursive baseline: endpoints are retained, the farthest point from the current segment chord is selected, and the process repeats until the compression budget is filled. This is a stronger and more credible geometric comparison than the earlier first-last chord proxy. The remaining risks are scale/performance on large AIS runs and the fact that the implementation uses lat/lon Euclidean geometry rather than a time-aware or haversine DP objective.

Oracle should be described carefully. The current Oracle uses heuristic query-derived labels directly, so it is a label Oracle diagnostic, not a theoretical final-F1 optimum. It is useful for checking whether the generated labels contain learnable signal, but it should not be presented as the absolute best possible simplification.

### Metrics And Result Cards

The F1 direction is correct: higher is better.

The current point-aware range metric is an improvement over trajectory-presence scoring because it prevents a method from preserving a whole range answer by keeping only one point from a hit trajectory.

The kNN, similarity, and clustering metrics also move in the right direction by multiplying answer-set agreement with retained support-point quality.

The metrics layer still needs more diagnostics:

- per-query F1 distributions, not only averages
- answer cardinality distributions
- empty-query and trivial-query rates
- broad-query rates
- per-query support point counts
- positive label fractions
- label Oracle score per query type
- best-baseline gap per query type
- confidence intervals or standard deviations across seeds

Each specialist result card should report:

- target workload F1
- best baseline F1
- gap to best baseline
- gap to label Oracle
- compression ratio
- SED and PED distortion
- length preservation and `F1xLen`
- simplification latency
- mean and standard deviation across seeds

### Runtime And Scale

Current latency measures the simplification call. For MLQDS this includes model inference and simplification, but it does not represent full end-to-end benchmark time including query generation, query execution, label construction, and geometric metric computation.

Final reports should separate:

- simplification latency
- query evaluation time
- geometric metric time
- training time
- workload and label generation time

Some current metric implementations may become expensive at larger AIS scale:

- Range point F1 builds Python sets from point rows.
- Clustering F1 builds co-membership pair sets, which can grow quadratically with cluster size.
- Geometric distortion scans removed points across all trajectories on CPU.

These are acceptable for small experiments, but larger 10-day and 30-day benchmarks will need caching, sampling, or vectorized implementations.

### Benchmarking Work Needed

Priority work:

1. Add a four-specialist benchmark runner.
2. Add a true train-model by eval-workload matrix.
3. Evaluate all baselines in every matrix cell.
4. Validate the true recursive Douglas-Peucker baseline on larger AIS runs.
5. Document Oracle as a label Oracle.
6. Add multi-seed aggregation with mean and standard deviation.
7. Add result fields for best baseline, gap to best baseline, and gap to Oracle.
8. Persist per-query scores and query diagnostics.
9. Separate simplification latency from full benchmark/evaluation runtime.
10. Add learning-curve benchmark support for increasing AIS day counts.

Minimum acceptable benchmark protocol:

```text
Models: Range-QDS, kNN-QDS, Similarity-QDS, Clustering-QDS
Baselines: uniform, true Douglas-Peucker, label Oracle
Seeds: at least 3
Compression: same ratio for all methods
Splits: fixed train/validation/test days
Metrics: per-type F1, best-baseline gap, Oracle gap, SED/PED, length preservation
```

Preferred benchmark protocol:

```text
Seeds: 5
Data volumes: 3, 5, 10, 20, 30 days
Result reporting: mean, standard deviation, best-baseline gap, and learning curves
```

Practical recommendation:

> Treat benchmarking as part of the model objective, not as an afterthought.

For this sprint, a model should only be considered successful when its specialist version beats the strongest baseline on its own workload at equal compression, across repeated seeds, with geometry remaining within acceptable bounds.

## Reproducibility, Artifacts, And Visualization

These areas do not define the learning objective, but they determine whether sprint results can be trusted, repeated, and debugged.

### Environment

The local sprint environment has moved from a blocker to a maintained dependency.

Current local sprint status:

- `qds_v2` has documented environment commands, pinned requirements, and Makefile
  targets for environment checks, tests, and smoke runs.
- Future work should keep those commands current as Phase 3 adds benchmark
  runner scripts and larger cached datasets.

### Artifact Hygiene

The repository currently tracks many saved model checkpoints under `src/models/saved_models`.

This is not ideal because checkpoints are experiment artifacts, not source code. Keeping them under `src/` makes the package heavier and makes it harder to distinguish code changes from run outputs.

Recommended direction:

- move saved checkpoints out of `src/`
- store run artifacts under a dedicated artifact/results directory
- add ignore rules for generated checkpoints
- keep only small intentional example artifacts if needed
- record artifact metadata in JSON rather than relying on file names

This matters because multi-seed, multi-workload benchmarks will generate many more checkpoints and result files.

### Visualization And Debugging

The v2 visualization module is currently a placeholder. Tables and JSON are useful, but they are not enough to debug query-driven simplification behavior.

The sprint should add visual debugging outputs for:

- query regions
- query answer sets
- label heatmaps
- retained vs removed points
- MLQDS vs baseline retained masks
- per-query failure examples
- range/kNN/similarity/clustering-specific retained-point behavior

The old `qds_project` contains visualization and workload-comparison ideas that can be mined, but v2 should build only the plots needed to debug the current specialist-model objective.

For each specialist model, visual inspection should answer:

```text
What did the model keep that the baseline did not?
Did those retained points actually help the target query workload?
Did the model waste budget on redundant or irrelevant points?
```

## Priority Roadmap

The findings above imply the following sprint roadmap.

### Phase 1: Make The Data Trustworthy

1. Add cached preprocessing artifacts.
2. Segment trajectories by MMSI plus temporal continuity.
3. Add data audit reports.
4. Expose segment and sampling controls.
5. Add real cleaned-CSV smoke tests.

### Phase 2: Make Workloads Trainable

1. Add query-type-specific generator configs.
2. Add query acceptance filters and difficulty controls.
3. Add answer-cardinality, positive-label, and label Oracle diagnostics.
4. Cache workloads and labels per dataset split.
5. Build one validated workload generator per specialist model.

### Phase 3: Make Learning Budget-Aware

1. Improve labels around non-redundant retained representatives.
2. Add stronger negative and hard-negative supervision.
3. Add top-k-aware or retained-subset-aware training diagnostics.
4. Decide how large query workloads condition the model.
5. Investigate global or workload-aware budget allocation.

### Phase 4: Make Claims Defensible

1. Add the four-specialist benchmark runner.
2. Add multi-seed aggregation.
3. Validate true recursive Douglas-Peucker as the geometric baseline at sprint scale.
4. Add best-baseline gap and label Oracle gap reporting.
5. Add learning-curve experiments across data volume.

### Phase 5: Make Results Reproducible And Debuggable

1. Fix the Python environment and dependency reproducibility.
2. Move generated checkpoints and large artifacts out of source-controlled model code.
3. Add visual debugging outputs for specialist workload behavior.
4. Record artifact metadata in structured result files.

## Definition Of Done

The sprint should be considered successful when the project can produce a repeatable benchmark report showing:

- four trained specialist models
- fixed train/validation/test day splits
- at least three seeds per model
- same compression ratio for all compared methods
- per-workload F1 and geometric distortion
- best-baseline gap for each specialist
- label Oracle gap for each specialist
- clear evidence of whether each specialist beats uniform and true Douglas-Peucker

The strongest version of the sprint result would show all four specialists beating the strongest baseline on their target workload. A still-useful sprint result would show this for Range-QDS and kNN-QDS, with clear diagnostics explaining what blocks Similarity-QDS and Clustering-QDS.

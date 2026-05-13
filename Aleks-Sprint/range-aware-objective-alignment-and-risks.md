# Range-Aware Objective Alignment And Risks

## Purpose

This note explains why the new `range_aware` model aligns better with the
range-query objective, what changed versus the earlier local-label setup, and
what risks remain.

## Target Objective

The project objective is not generic trajectory simplification. The target user
runs spatiotemporal range queries over simplified AIS data and should still be
able to understand ships and movement inside those queried areas/time windows.

The retained points should preserve:

- relevant in-query points
- ships represented in the query result
- enough per-ship coverage to interpret movement
- entry/exit behavior near query boundaries
- boundary crossings
- temporal coverage and gap behavior
- turns and local route shape inside the query

Global trajectory geometry is secondary. This is why `RangeUseful` and
`RangePointF1` matter more for this objective than global SED/PED alone.

## What The Earlier Setup Did

The earlier range training path had local/additive labels. Those labels told
the model which points were useful for the generated range workload during
training.

That is supervision. It helps update model weights. It is not available at
inference.

The earlier model also had query cross-attention. Points could attend to query
embeddings. That means it had some query context, but it still had to learn
geometric relationships such as "this point is inside this query box" from
separate point and query embeddings.

That was too weak in practice. The learned residual fill often failed to beat
uniform or `TemporalRandomFill` in a meaningful way.

## What `range_aware` Adds

`model_type=range_aware` augments each point with explicit relation features
computed from the current pure range workload:

- fraction of range queries containing the point
- weighted containment mass
- maximum weighted containment
- near-box proximity
- query-center proximity
- any-boundary proximity
- temporal-boundary proximity
- spatial-boundary proximity

These are input features, not labels. The model still learns a score function,
but it no longer has to rediscover basic range-query geometry from scratch.

This is a strong inductive bias. It is the main reason the learned model now
beats uniform and Douglas-Peucker on both `RangeUseful` and `RangePointF1`.

## Why This Aligns Better

Uniform temporal sampling preserves spacing. Douglas-Peucker preserves global
path geometry. Neither method knows which retained points are likely to answer
the range workload.

The range-aware model receives direct information about point/query relevance.
That lets it spend retained-point budget on points more likely to matter for
the intended range-query workload.

This is aligned with workload-aware QDS: the workload is part of the
simplification problem, not an after-the-fact hidden test.

## Supervision Versus Inference

Training/supervision:

- uses labels derived from the training workload
- compares predictions against those labels
- updates model weights
- may use a validation/selection workload to choose the best checkpoint

Inference/evaluation:

- does not update model weights
- does not get training labels
- receives input points plus the query workload
- computes relation features for that workload
- predicts point scores
- retains points according to the compression budget

So local labels teach the model. Range-aware relation features are inputs the
model can use when making predictions.

## Is Using The Eval Workload Cheating?

It depends on the task definition.

If the problem is workload-aware simplification, then using the workload at
inference is legitimate. The model is asked to simplify the dataset for a known
query workload. In that setting, using the eval workload to compute relation
features is not cheating; it is the intended input.

If the problem is workload-blind compression, then it is cheating. In that
setting, the simplified dataset must be produced before knowing the future
queries. A model that sees the eval queries before choosing retained points
would be using test-time information it should not have.

Current benchmark interpretation:

- training labels come from the train workload
- checkpoint selection uses a separate validation/selection workload
- final evaluation uses the eval workload as inference input
- model weights are not updated on eval queries
- retained points are workload-conditioned on eval queries

That is transductive/workload-aware inference. It is valid only if the product
claim is explicitly "simplify for this known range workload" or "query-workload
aware simplification." It should not be marketed as workload-blind compression.

## Downsides And Risks

### Workload Dependence

The retained set is optimized for the provided workload. If real user queries
are unknown, broader, narrower, differently distributed, or generated with
different spatial/time footprints, quality may drop.

### Query Distribution Overfit

The model may learn the query generator distribution, not general range-query
behavior. The 40% coverage sweep already showed weaker margins than 10% and
20% coverage.

### Runtime Cost

Range-aware feature construction scales with points times queries. High
coverage generated about 1.1k queries per split and made preprocessing and
evaluation expensive:

- train label prep: about 940s
- eval label prep: about 829s
- diagnostics: about 876s
- 5% MLQDS eval latency: about 133s

This is the biggest practical downside.

### Global Geometry Trade-Off

The model wins on range-query usefulness, but global SED/PED are still worse
than uniform and Douglas-Peucker. This is acceptable only if the user outcome
prioritizes query-local usefulness over global trajectory fidelity.

### Range-Specific Design

`range_aware` currently requires a pure range workload. It is not a general
solution for kNN, similarity, or clustering workloads.

### Strong Inductive Bias

The improvement is not a free lunch. We encoded geometric relation features
that are very close to what the range workload cares about. This is good
engineering if the objective is known range-query usefulness, but it makes the
model less generic.

## Optimization Opportunities Without Quality Loss

Some optimizations should preserve exact output semantics if implemented
carefully:

- cache range-aware point features per `(points, workload, feature_schema)`
- cache normalized model inputs for repeated evaluation of the same workload
- compute MLQDS predictions once per workload and reuse scores across
  compression-ratio audits
- reuse query-cache and range-audit support across audit ratios and methods
- cache high-coverage diagnostics aggressively
- vectorize or index exact range-relation feature construction

These are not quality trade-offs. They are engineering work. The main trade-off
is memory/disk usage and implementation complexity.

Approximate optimizations, such as sampling queries or using approximate
nearest boxes, may reduce quality and should be treated as separate ablations.

## Query Jitter

`range_footprint_jitter` already exists. It randomizes query footprint size
around the configured spatial/time footprint.

Introducing jitter is relevant for robustness if real user queries are not all
fixed-size boxes. It should be tested as an A/B, not blindly enabled:

- train/eval with no jitter: current controlled baseline
- train/eval with jitter: realistic distribution robustness test
- train with jitter, eval without jitter: checks whether jitter hurts the
  current target
- train without jitter, eval with jitter: checks brittleness

Likely trade-off: jitter may lower peak score on the current fixed-footprint
benchmark while improving robustness to varied query sizes.

## Checkpoint Selection: `uniform_gap`

The benchmark profile uses:

- `checkpoint_selection_metric=uniform_gap`
- `checkpoint_f1_variant=range_usefulness`

This still uses `RangeUseful` as the checkpoint validation score.

`uniform_gap` then adjusts that score by rewarding margin over uniform and
penalizing any active workload type that falls below uniform. In simplified
form:

`selection = RangeUseful + 0.5 * (RangeUseful - uniform_RangeUseful) - type_deficit - collapse_penalty`

For the current pure range workload, the type-deficit term is effectively
"do not select a checkpoint that loses to uniform on range."

So `uniform_gap` does not replace `RangeUseful`. It wraps `RangeUseful` with a
fair-baseline guardrail.

## Bottom Line

The current model is better aligned with the objective because it explicitly
receives range-query relation features and is evaluated on range-query
usefulness. That is a legitimate workload-aware QDS design.

It is not a workload-blind simplifier, not a general query-type solution, and
not free computationally. The result should be described as:

"A range-workload-aware simplification model that uses known range workload
geometry at inference to retain points that better preserve range-query
answers."

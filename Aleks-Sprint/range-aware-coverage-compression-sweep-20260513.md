# Range-Aware Coverage/Compression Sweep - 2026-05-13

## Scope

This sweep tests the strict learned range-aware profile across range-query
coverage targets and retained-point compression ratios.

Fixed settings:

- `model_type=range_aware`
- `mlqds_temporal_fraction=0.25`
- `mlqds_range_geometry_blend=0.0`
- `mlqds_diversity_bonus=0.0`
- primary training/eval compression ratio `0.05`
- audit compression ratios `0.01,0.02,0.05,0.10`
- original `2026-02-02/03/04` train/validation/eval split

Caveat: each coverage target trains one model at the profile's normal 5%
primary compression setting, then audits that model at multiple compression
ratios. This measures robustness across budgets; it is not separate retraining
for every compression target.

## Main Grid

`RangeUseful` by coverage and compression:

| query coverage | compression | MLQDS | uniform | Douglas-Peucker | TemporalRandomFill | Oracle | MLQDS vs uniform | MLQDS vs DP | MLQDS vs random fill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.01 | 0.4512 | 0.2202 | 0.2049 | 0.2036 | 0.6423 | +0.2310 | +0.2463 | +0.2476 |
| 0.10 | 0.02 | 0.5738 | 0.2848 | 0.2581 | 0.2513 | 0.7296 | +0.2890 | +0.3158 | +0.3225 |
| 0.10 | 0.05 | 0.7388 | 0.3729 | 0.3510 | 0.3441 | 0.8455 | +0.3659 | +0.3878 | +0.3946 |
| 0.10 | 0.10 | 0.8268 | 0.4523 | 0.4401 | 0.4284 | 0.9015 | +0.3745 | +0.3866 | +0.3984 |
| 0.20 | 0.01 | 0.3999 | 0.2173 | 0.2042 | 0.1984 | 0.5913 | +0.1826 | +0.1957 | +0.2015 |
| 0.20 | 0.02 | 0.5047 | 0.2822 | 0.2581 | 0.2505 | 0.6679 | +0.2225 | +0.2466 | +0.2542 |
| 0.20 | 0.05 | 0.6626 | 0.3721 | 0.3522 | 0.3427 | 0.7855 | +0.2905 | +0.3104 | +0.3200 |
| 0.20 | 0.10 | 0.7859 | 0.4527 | 0.4393 | 0.4276 | 0.8719 | +0.3332 | +0.3465 | +0.3583 |
| 0.40 | 0.01 | 0.2939 | 0.2004 | 0.1823 | 0.1806 | 0.5173 | +0.0936 | +0.1116 | +0.1134 |
| 0.40 | 0.02 | 0.3940 | 0.2713 | 0.2413 | 0.2362 | 0.5927 | +0.1227 | +0.1527 | +0.1578 |
| 0.40 | 0.05 | 0.5533 | 0.3687 | 0.3413 | 0.3335 | 0.7024 | +0.1846 | +0.2120 | +0.2198 |
| 0.40 | 0.10 | 0.6693 | 0.4509 | 0.4332 | 0.4238 | 0.7973 | +0.2184 | +0.2361 | +0.2456 |

The model wins every tested cell against uniform, Douglas-Peucker, and
TemporalRandomFill. The margin is largest at 10% query coverage and shrinks at
40% coverage.

## Component View

MLQDS detailed metrics:

| coverage | compression | PointF1 | EntryExitF1 | CrossingF1 | GapCov | ShapeScore | LengthPres | MLQDS latency ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.01 | 0.1887 | 0.7091 | 0.4097 | 0.3447 | 0.3526 | 0.9246 | 20968 |
| 0.10 | 0.02 | 0.3233 | 0.8158 | 0.5072 | 0.5137 | 0.4971 | 0.9528 | 22030 |
| 0.10 | 0.05 | 0.5601 | 0.9210 | 0.6037 | 0.7639 | 0.6770 | 0.9756 | 21278 |
| 0.10 | 0.10 | 0.7118 | 0.9565 | 0.6524 | 0.8787 | 0.7773 | 0.9849 | 22201 |
| 0.20 | 0.01 | 0.1291 | 0.6709 | 0.3824 | 0.2804 | 0.3160 | 0.9195 | 41732 |
| 0.20 | 0.02 | 0.2291 | 0.7807 | 0.4748 | 0.4156 | 0.4363 | 0.9485 | 41678 |
| 0.20 | 0.05 | 0.4332 | 0.8986 | 0.5888 | 0.6358 | 0.6222 | 0.9731 | 42441 |
| 0.20 | 0.10 | 0.6343 | 0.9504 | 0.6630 | 0.8072 | 0.7580 | 0.9836 | 43580 |
| 0.40 | 0.01 | 0.0818 | 0.4694 | 0.2521 | 0.2194 | 0.2076 | 0.9237 | 134419 |
| 0.40 | 0.02 | 0.1464 | 0.6101 | 0.3540 | 0.3300 | 0.3264 | 0.9529 | 133690 |
| 0.40 | 0.05 | 0.2922 | 0.7853 | 0.4936 | 0.5371 | 0.5453 | 0.9768 | 133120 |
| 0.40 | 0.10 | 0.4340 | 0.8766 | 0.5838 | 0.7078 | 0.7127 | 0.9856 | 132461 |

## Runtime/Scale

| coverage | train queries | eval queries | selection queries | best epoch | best validation usefulness | runtime s | workload gen s | train labels s | eval labels s | compression audit s | diagnostics s | MLQDS 5% latency ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 169 | 124 | 144 | 8 | 0.7410 | 714 | 33.7 | 147.0 | 99.2 | 106.0 | 130.7 | 21278 |
| 0.20 | 418 | 336 | 360 | 5 | 0.6600 | 572 | 0.3 | 0.1 | 0.1 | 188.2 | 0.4 | 42441 |
| 0.40 | 1152 | 1084 | 1094 | 5 | 0.5481 | 4431 | 216.4 | 940.1 | 829.4 | 526.0 | 876.0 | 133120 |

Coverage 40% is expensive because the query generator needs about 1.1k queries
per split to reach the target. That hits every expensive path: workload
generation, label prep, exact validation, audit evaluation, and diagnostics.

## Interpretation

The model is robust across the tested compression range. Even at 1% retained
points it beats the baselines under all three coverage targets.

Performance is not monotonic in query coverage. Lower coverage produces a more
selective query workload, and the range-aware features are highly informative.
At 40% coverage, queries become much less selective, the Oracle ceiling drops,
and the model margin shrinks. It still wins, but this is the weakest tested
regime.

For compression, the absolute score rises as retained budget increases. The
margin over baselines also generally grows with budget, especially from 1% to
5%. The 1% results are still useful but weaker, especially at 40% coverage.

The 20% coverage audit at 2% compression (`RangeUseful=0.5047`) beat the earlier
separately trained 2% run (`0.4767`). That suggests compression-specific
retraining is not automatically better with the current objective. The
budget-loss ratios and checkpoint selection matter more than simply setting the
primary compression ratio lower.

## Recommendations

- Keep `query_coverage=0.20` as the practical default. It is a strong middle
  target and already has warm caches.
- Treat 40% coverage as a stress test. It still wins, but the quality margin
  and runtime both get worse.
- Use `range_audit_compression_ratios=0.01,0.02,0.05,0.10` for final reporting,
  but not for every iteration. It adds real cost.
- Do not assume separate low-compression retraining improves results. Test it,
  but the first evidence says the 5% primary model generalizes well down to 2%.
- The next useful optimization target is high-coverage range label/diagnostic
  preparation. Coverage 40% spent about 940s on train labels, 829s on eval
  labels, and 876s on diagnostics.

# Range-Aware Epoch Sweep - 2026-05-13

## Purpose

Gather denser data for the current successful strict learned profile:

- `model_type=range_aware`
- `mlqds_temporal_fraction=0.25`
- `mlqds_range_geometry_blend=0.0`
- `mlqds_diversity_bonus=0.0`
- `compression_ratio=0.05`
- original `2026-02-02/03/04` train/validation/eval split

The question was whether increasing epochs improves the model, and where extra
epochs start giving diminishing returns.

## Run Summary

| run | lr | epochs | candidate pool | best epoch | best validation usefulness | final RangeUseful | final RangePointF1 | runtime | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `range_aware_t25_20260513` | 0.0005 | 8 | 2 | 5 | 0.6581 | 0.6602 | 0.4289 | 380s | strong baseline |
| `range_aware_t25_e16_pool4_20260513` | 0.0005 | 16 | 4 | 5 | 0.6595 | 0.6635 | 0.4294 | 1033s | best final score, but not because later epochs helped |
| `range_aware_t25_e16_pool4_lr25_20260513` | 0.00025 | 16 | 4 | 16 | 0.6591 | 0.6604 | 0.4250 | 1043s | slower convergence, no final improvement |

## Final Eval Details

| run | RangeUseful | vs uniform | vs DP | vs TemporalRandomFill | EntryExitF1 | CrossingF1 | GapCov | ShapeScore | LengthPres | AvgSED km | AvgPED km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `t25_8ep_pool2_lr5e-4` | 0.6602 | +0.2881 | +0.3080 | +0.3176 | 0.8942 | 0.5800 | 0.6407 | 0.6236 | 0.9739 | 0.0624 | 0.0346 |
| `t25_16ep_pool4_lr5e-4` | 0.6635 | +0.2913 | +0.3112 | +0.3208 | 0.8979 | 0.5847 | 0.6416 | 0.6304 | 0.9753 | 0.0626 | 0.0348 |
| `t25_16ep_pool4_lr2.5e-4` | 0.6604 | +0.2883 | +0.3082 | +0.3178 | 0.8922 | 0.5883 | 0.6381 | 0.6347 | 0.9763 | 0.0600 | 0.0342 |

Uniform `RangeUseful=0.3721`; Douglas-Peucker `RangeUseful=0.3522`;
TemporalRandomFill `RangeUseful=0.3427`.

## Dense Validation Curve - LR 0.0005

| epoch | validation usefulness | point F1 | entry/exit | crossing | gap | shape | pred std | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5952 | 0.4055 | 0.7346 | 0.4604 | 0.6363 | 0.5312 | 8.2176 | 0.4947 |
| 2 | 0.6457 | 0.4165 | 0.8371 | 0.5373 | 0.6679 | 0.6357 | 17.8888 | 0.9244 |
| 3 | 0.6566 | 0.4169 | 0.8784 | 0.5733 | 0.6619 | 0.6425 | 17.6868 | 2.3500 |
| 4 | 0.6582 | 0.4190 | 0.8872 | 0.5787 | 0.6549 | 0.6368 | 14.7075 | 2.6453 |
| 5 | 0.6595 | 0.4230 | 0.8912 | 0.5758 | 0.6491 | 0.6363 | 12.1207 | 2.0399 |
| 6 | 0.6578 | 0.4262 | 0.8861 | 0.5666 | 0.6479 | 0.6288 | 14.9076 | 2.1084 |
| 7 | 0.6527 | 0.4191 | 0.8822 | 0.5659 | 0.6472 | 0.6182 | 18.1879 | 2.5604 |
| 8 | 0.6414 | 0.3886 | 0.8893 | 0.5698 | 0.6267 | 0.6205 | 15.2670 | 3.0853 |
| 9 | 0.6381 | 0.4152 | 0.8466 | 0.5359 | 0.6440 | 0.6008 | 28.1801 | 3.8038 |
| 10 | 0.6515 | 0.4200 | 0.8703 | 0.5549 | 0.6424 | 0.6289 | 42.9230 | 3.4526 |
| 11 | 0.6256 | 0.4291 | 0.7928 | 0.4912 | 0.6491 | 0.5783 | 58.9923 | 4.6877 |
| 12 | 0.6313 | 0.4299 | 0.7991 | 0.5037 | 0.6571 | 0.5864 | 90.8176 | 5.6772 |
| 13 | 0.6372 | 0.4287 | 0.8300 | 0.5257 | 0.6511 | 0.5920 | 148.5221 | 10.2704 |
| 14 | 0.6508 | 0.4284 | 0.8645 | 0.5500 | 0.6573 | 0.6165 | 143.3566 | 10.2800 |
| 15 | 0.6518 | 0.4291 | 0.8648 | 0.5528 | 0.6523 | 0.6191 | 161.4343 | 16.7995 |
| 16 | 0.6509 | 0.4208 | 0.8771 | 0.5637 | 0.6464 | 0.6162 | 192.4622 | 22.2742 |

Interpretation: validation usefulness effectively peaks at epoch 5. Later
epochs increase point F1 in places, but lose boundary/crossing/shape enough to
hurt `RangeUseful`. The score scale blows up after epoch 9.

## Dense Validation Curve - LR 0.00025

| epoch | validation usefulness | point F1 | entry/exit | crossing | gap | shape | pred std | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6208 | 0.4019 | 0.7997 | 0.5130 | 0.6492 | 0.5835 | 5.6990 | 0.5411 |
| 2 | 0.6143 | 0.4091 | 0.7861 | 0.5017 | 0.6441 | 0.5539 | 6.4143 | 0.4493 |
| 3 | 0.6501 | 0.4172 | 0.8799 | 0.5699 | 0.6392 | 0.6207 | 12.8977 | 0.6593 |
| 4 | 0.6447 | 0.4073 | 0.8646 | 0.5514 | 0.6439 | 0.6244 | 20.5046 | 1.7124 |
| 5 | 0.6479 | 0.4074 | 0.8721 | 0.5581 | 0.6470 | 0.6317 | 19.5216 | 2.4026 |
| 6 | 0.6573 | 0.4180 | 0.8874 | 0.5756 | 0.6491 | 0.6390 | 20.7148 | 2.4323 |
| 7 | 0.6580 | 0.4174 | 0.8864 | 0.5767 | 0.6536 | 0.6424 | 20.1213 | 2.5280 |
| 8 | 0.6567 | 0.4163 | 0.8773 | 0.5683 | 0.6612 | 0.6472 | 18.8863 | 2.6571 |
| 9 | 0.6502 | 0.4037 | 0.8862 | 0.5745 | 0.6417 | 0.6319 | 19.8797 | 2.6791 |
| 10 | 0.6529 | 0.4098 | 0.8857 | 0.5744 | 0.6430 | 0.6334 | 20.2242 | 2.8326 |
| 11 | 0.6564 | 0.4151 | 0.8931 | 0.5832 | 0.6379 | 0.6321 | 20.2753 | 2.9977 |
| 12 | 0.6573 | 0.4213 | 0.8738 | 0.5646 | 0.6589 | 0.6454 | 20.9877 | 3.2041 |
| 13 | 0.6565 | 0.4225 | 0.8804 | 0.5730 | 0.6474 | 0.6358 | 21.5382 | 3.2629 |
| 14 | 0.6578 | 0.4224 | 0.8819 | 0.5745 | 0.6482 | 0.6401 | 21.3751 | 3.3566 |
| 15 | 0.6582 | 0.4226 | 0.8835 | 0.5753 | 0.6525 | 0.6438 | 24.4247 | 3.6018 |
| 16 | 0.6591 | 0.4230 | 0.8871 | 0.5765 | 0.6518 | 0.6428 | 26.1344 | 3.9863 |

Interpretation: half LR prevents the severe score-scale blow-up, but the
validation curve mostly plateaus from epochs 6-16 and never clearly beats the
default-LR epoch-5 checkpoint. Final eval is also worse than the LR 0.0005
dense run.

## Conclusion

Increasing epochs alone is not the right lever.

For `lr=0.0005`, useful learning is front-loaded. Diminishing returns begin
around epoch 5-6, and later epochs mostly add instability. For `lr=0.00025`,
more epochs are needed to reach the same level, but the final model is not
better.

Best artifact in this sweep is `range_aware_t25_e16_pool4_20260513`, but the
reason is denser checkpoint selection and run variance around the same early
peak, not genuine benefit from epochs 9-16. The default 8-epoch profile remains
more practical unless the extra `+0.0032 RangeUseful` is worth about 2.7x
runtime.

Next useful experiments:

- try explicit score-scale regularization or logit normalization after epoch 5
- try LR decay after epoch 4 rather than a globally lower LR
- test whether candidate pool `4` with only 8 epochs captures most of the
  `0.6635` score at lower runtime
- continue attacking MLQDS inference latency; exact validation and inference
  dominate runtime

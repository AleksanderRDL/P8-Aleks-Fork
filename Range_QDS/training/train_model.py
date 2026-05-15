"""Ranking-based model training on trajectory windows. See training/README.md for details."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler

from experiments.experiment_config import ModelConfig
from queries.workload import TypedQueryWorkload
from experiments.torch_runtime import normalize_amp_mode, torch_autocast_context
from models.historical_prior_qds_model import HistoricalPriorRangeQDSModel, HistoricalPriorStudentRangeQDSModel
from models.trajectory_qds_model import TrajectoryQDSModel
from models.turn_aware_qds_model import TurnAwareQDSModel
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model
from models.workload_blind_qds_model import SegmentContextRangeQDSModel, WorkloadBlindRangeQDSModel
from queries.query_types import (
    ID_TO_QUERY_NAME,
    NUM_QUERY_TYPES,
)
from training.checkpoint_selection import (
    CheckpointCandidate,
    record_validation_stats,
    selection_from_stats,
    selection_score,
)
from training.importance_labels import compute_typed_importance_labels
from training.model_features import (
    HISTORICAL_PRIOR_MODEL_TYPES,
    NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES,
    build_model_point_features,
    is_workload_blind_model_type,
)
from training.query_prior_fields import build_train_query_prior_fields, query_prior_field_metadata
from training.query_useful_targets import QUERY_USEFUL_V1_HEAD_NAMES, build_query_useful_v1_targets
from training.inference import windowed_predict
from training.training_diagnostics import (
    _discriminative_sample,
    _kendall_tau,
    _training_target_diagnostics,
    train_target_fit_diagnostics,
)
from training.training_epoch import _train_one_epoch
from training.training_losses import (
    _budget_loss_ratios,
    _effective_budget_loss_ratios,
    _effective_temporal_residual_label_mode,
    _safe_quantile,
    _temporal_base_masks_for_budget_ratios,
)
from training.training_outputs import TrainingOutputs
from training.training_setup import (
    _model_state_on_cpu,
    _pure_query_type_id,
    _query_frequency_workload_map,
    _single_active_type_id,
    _workload_map_tensor,
)
from training.training_targets import _apply_temporal_residual_labels, _scaled_training_target_for_type
from training.training_validation import _validation_checkpoint_scores, _validation_uniform_score
from training.training_windows import _filter_supervised_windows, _trajectory_batch_to_device
from training.scaler import FeatureScaler
from training.trajectory_batching import batch_windows, build_trajectory_windows


def _historical_prior_support_mask(
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    support_ratio: float,
) -> torch.Tensor:
    """Return a per-trajectory top-target support mask for historical priors."""
    ratio = min(1.0, max(0.0, float(support_ratio)))
    support_mask = torch.zeros((int(targets.shape[0]),), dtype=torch.bool, device=targets.device)
    if ratio >= 1.0:
        support_mask[:] = True
        return support_mask
    if ratio <= 0.0:
        return support_mask

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(1, int(math.ceil(ratio * point_count))))
        if keep_count >= point_count:
            support_mask[start:end] = True
            continue
        local_targets = targets[start:end].float()
        local_indices = torch.topk(local_targets, k=keep_count, largest=True).indices
        support_mask[start + local_indices] = True
    return support_mask


def _require_validation_inputs(
    validation_trajectories: list[torch.Tensor] | None,
    validation_boundaries: list[tuple[int, int]] | None,
    validation_workload: TypedQueryWorkload | None,
) -> tuple[list[torch.Tensor], list[tuple[int, int]], TypedQueryWorkload]:
    """Return validation inputs after enforcing the checkpoint-score contract."""
    if validation_trajectories is None or validation_boundaries is None or validation_workload is None:
        raise RuntimeError("Validation scoring requested without complete validation inputs.")
    return validation_trajectories, validation_boundaries, validation_workload


def train_model(
    train_trajectories: list[torch.Tensor],
    train_boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    seed: int,
    train_workload_map: dict[str, float] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    validation_boundaries: list[tuple[int, int]] | None = None,
    validation_workload: TypedQueryWorkload | None = None,
    validation_workload_map: dict[str, float] | None = None,
    precomputed_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
    validation_points: torch.Tensor | None = None,
    precomputed_validation_query_cache: Any | None = None,
    precomputed_validation_geometry_scores: torch.Tensor | None = None,
    train_trajectory_source_ids: list[int] | None = None,
    train_trajectory_mmsis: list[int] | None = None,
    query_prior_workloads: list[TypedQueryWorkload] | None = None,
    query_prior_workload_seeds: list[int] | None = None,
) -> TrainingOutputs:
    """Train one pure-workload model with trajectory-window ranking losses."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    all_points = torch.cat(train_trajectories, dim=0)
    train_point_source_ids: torch.Tensor | None = None
    if train_trajectory_source_ids is not None:
        if len(train_trajectory_source_ids) != len(train_boundaries):
            raise ValueError(
                "train_trajectory_source_ids must match train_boundaries length: "
                f"got {len(train_trajectory_source_ids)} ids for {len(train_boundaries)} boundaries."
            )
        train_point_source_ids = torch.empty((int(all_points.shape[0]),), dtype=torch.long)
        for source_id, (start, end) in zip(train_trajectory_source_ids, train_boundaries, strict=True):
            if int(source_id) < 0:
                raise ValueError("train_trajectory_source_ids must be non-negative.")
            train_point_source_ids[start:end] = int(source_id)
    prior_workloads = list(query_prior_workloads or [workload])
    prior_queries: list[dict[str, Any]] = []
    for prior_workload in prior_workloads:
        prior_queries.extend(prior_workload.typed_queries)

    factorized_targets: torch.Tensor | None = None
    factorized_mask: torch.Tensor | None = None
    factorized_target_diagnostics: dict[str, Any] = {}
    if str(getattr(model_config, "range_training_target_mode", "")).lower() == "query_useful_v1_factorized":
        factorized_bundle = build_query_useful_v1_targets(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=prior_queries,
        )
        labels = factorized_bundle.labels
        labelled_mask = factorized_bundle.labelled_mask
        factorized_targets = factorized_bundle.head_targets
        factorized_mask = factorized_bundle.head_mask
        factorized_target_diagnostics = factorized_bundle.diagnostics
    elif precomputed_labels is None:
        labels, labelled_mask = compute_typed_importance_labels(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=workload.typed_queries,
            range_label_mode=str(getattr(model_config, "range_label_mode", "usefulness")),
            range_boundary_prior_weight=float(getattr(model_config, "range_boundary_prior_weight", 0.0)),
        )
    else:
        labels, labelled_mask = precomputed_labels
        expected_shape = (all_points.shape[0], NUM_QUERY_TYPES)
        if labels.shape != expected_shape or labelled_mask.shape != expected_shape:
            raise ValueError(
                "precomputed_labels must match flattened training points and query type count: "
                f"expected {expected_shape}, got labels={tuple(labels.shape)} mask={tuple(labelled_mask.shape)}"
            )

    query_prior_field: dict[str, Any] | None = None
    if str(model_config.model_type).lower() == "workload_blind_range_v2":
        prior_seed = None
        if query_prior_workload_seeds:
            prior_seed = int(query_prior_workload_seeds[0])
        behavior_prior_values = None
        if factorized_targets is not None:
            try:
                behavior_idx = tuple(QUERY_USEFUL_V1_HEAD_NAMES).index("conditional_behavior_utility")
                behavior_prior_values = factorized_targets[:, behavior_idx]
            except ValueError:
                behavior_prior_values = None
        query_prior_field = build_train_query_prior_fields(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=prior_queries,
            labels=labels,
            behavior_values=behavior_prior_values,
            workload_profile_id=str(
                (workload.generation_diagnostics or {}).get("query_generation", {}).get(
                    "workload_profile_id",
                    "range_workload_v1",
                )
            ),
            train_workload_seed=prior_seed,
        )
    points = build_model_point_features(
        all_points,
        workload,
        model_config.model_type,
        boundaries=train_boundaries,
        trajectory_mmsis=train_trajectory_mmsis,
        query_prior_field=query_prior_field,
    )
    point_dim = int(points.shape[1])
    run_tag = "main"
    requested_temporal_residual_label_mode = str(
        getattr(model_config, "temporal_residual_label_mode", "none")
    ).lower()
    if requested_temporal_residual_label_mode not in {"none", "temporal"}:
        raise ValueError("temporal_residual_label_mode must be 'none' or 'temporal'.")
    temporal_residual_label_mode = _effective_temporal_residual_label_mode(
        model_config,
        requested_temporal_residual_label_mode,
    )
    if requested_temporal_residual_label_mode != temporal_residual_label_mode:
        print(
            f"  [{run_tag}] temporal_residual_label_mode={requested_temporal_residual_label_mode} "
            f"ignored for mlqds_hybrid_mode={getattr(model_config, 'mlqds_hybrid_mode', 'fill')}; "
            f"using {temporal_residual_label_mode}",
            flush=True,
        )
    loss_objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if loss_objective not in {"ranking_bce", "budget_topk", "stratified_budget_topk", "pointwise_bce"}:
        raise ValueError(
            "loss_objective must be 'ranking_bce', 'budget_topk', "
            "'stratified_budget_topk', or 'pointwise_bce'."
        )
    if loss_objective == "stratified_budget_topk" and str(
        getattr(model_config, "mlqds_hybrid_mode", "fill")
    ).lower() != "stratified":
        raise ValueError("loss_objective='stratified_budget_topk' requires mlqds_hybrid_mode='stratified'.")
    configured_budget_ratios = _budget_loss_ratios(model_config)
    budget_ratios = configured_budget_ratios
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...] = ()
    temporal_residual_union_mask: torch.Tensor | None = None
    workload_type_id = _pure_query_type_id(workload.type_ids)
    if temporal_residual_label_mode == "temporal" and loss_objective == "budget_topk":
        budget_ratios = _effective_budget_loss_ratios(model_config, temporal_residual_label_mode)
        temporal_residual_budget_masks = _temporal_base_masks_for_budget_ratios(
            n_points=int(labels.shape[0]),
            boundaries=train_boundaries,
            budget_ratios=configured_budget_ratios,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
            device=labels.device,
        )
        if temporal_residual_budget_masks:
            temporal_residual_union_mask = torch.zeros((labels.shape[0],), dtype=torch.bool, device=labels.device)
            for _total_ratio, _effective_ratio, base_mask in temporal_residual_budget_masks:
                temporal_residual_union_mask |= base_mask
    elif temporal_residual_label_mode == "temporal":
        labels, labelled_mask = _apply_temporal_residual_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=train_boundaries,
            compression_ratio=model_config.compression_ratio,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        )
    training_target = _scaled_training_target_for_type(labels, labelled_mask, workload_type_id)
    training_labelled_mask = labelled_mask[:, workload_type_id]
    base_type_weights_cpu = _workload_map_tensor(
        train_workload_map or _query_frequency_workload_map(workload),
        device=torch.device("cpu"),
    )
    active_type_id = _single_active_type_id(base_type_weights_cpu)
    if active_type_id != workload_type_id:
        raise ValueError("Training workload map and workload query type must refer to the same pure workload.")
    active_type_ids = [active_type_id]
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))
    budget_loss_temperature = float(getattr(model_config, "budget_loss_temperature", 0.10))
    target_diagnostics = _training_target_diagnostics(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=workload_type_id,
        configured_budget_ratios=configured_budget_ratios,
        effective_budget_ratios=budget_ratios,
        temporal_residual_budget_masks=temporal_residual_budget_masks,
        temporal_residual_label_mode=temporal_residual_label_mode,
        loss_objective=loss_objective,
        temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        range_training_target_mode=str(getattr(model_config, "range_training_target_mode", "point_value")),
    )
    if factorized_target_diagnostics:
        target_diagnostics["query_useful_v1_factorized"] = factorized_target_diagnostics
    if query_prior_field is not None:
        target_diagnostics["query_prior_field"] = query_prior_field_metadata(query_prior_field)
    if budget_ratios != configured_budget_ratios:
        print(
            f"  [{run_tag}] effective_budget_loss_ratios={list(budget_ratios)} "
            f"from configured={list(configured_budget_ratios)} "
            f"temporal_residual_label_mode={temporal_residual_label_mode} "
            f"mlqds_temporal_fraction={float(getattr(model_config, 'mlqds_temporal_fraction', 0.0)):.3f}",
            flush=True,
        )
    for row in target_diagnostics.get("budget_rows", []):
        print(
            f"  [{run_tag}] residual_budget total={row['total_budget_ratio']:.4f} "
            f"effective_fill={row['effective_fill_budget_ratio']:.4f} "
            f"base_points={row['temporal_base_point_count']} "
            f"candidates={row['candidate_point_count']} "
            f"residual_pos={row['residual_positive_label_count']}",
            flush=True,
        )

    scaler = FeatureScaler.fit(points, workload.query_features)
    norm_points, norm_queries = scaler.transform(points, workload.query_features)

    uses_historical_prior = model_config.model_type in HISTORICAL_PRIOR_MODEL_TYPES
    if model_config.model_type in NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES:
        model_cls = HistoricalPriorRangeQDSModel
    elif model_config.model_type == "historical_prior_student":
        model_cls = HistoricalPriorStudentRangeQDSModel
    elif model_config.model_type == "segment_context_range":
        model_cls = SegmentContextRangeQDSModel
    elif model_config.model_type == "workload_blind_range_v2":
        model_cls = WorkloadBlindRangeV2Model
    elif is_workload_blind_model_type(model_config.model_type):
        model_cls = WorkloadBlindRangeQDSModel
    elif model_config.model_type == "turn_aware":
        model_cls = TurnAwareQDSModel
    else:
        model_cls = TrajectoryQDSModel
    model_kwargs = {
        "point_dim": point_dim,
        "query_dim": norm_queries.shape[1],
        "embed_dim": model_config.embed_dim,
        "num_heads": model_config.num_heads,
        "num_layers": model_config.num_layers,
        "type_embed_dim": model_config.type_embed_dim,
        "query_chunk_size": model_config.query_chunk_size,
        "dropout": model_config.dropout,
    }
    if uses_historical_prior:
        model_kwargs["historical_prior_k"] = int(getattr(model_config, "historical_prior_k", 32))
        model_kwargs["historical_prior_clock_weight"] = float(
            getattr(model_config, "historical_prior_clock_weight", 0.0)
        )
        model_kwargs["historical_prior_mmsi_weight"] = float(
            getattr(model_config, "historical_prior_mmsi_weight", 1.0)
        )
        model_kwargs["historical_prior_density_weight"] = float(
            getattr(model_config, "historical_prior_density_weight", 1.0)
        )
        model_kwargs["historical_prior_min_target"] = float(
            getattr(model_config, "historical_prior_min_target", 0.0)
        )
        model_kwargs["historical_prior_source_aggregation"] = str(
            getattr(model_config, "historical_prior_source_aggregation", "none")
        )
    model: torch.nn.Module = model_cls(**model_kwargs)
    if query_prior_field is not None:
        setattr(model, "query_prior_field", query_prior_field)
    if uses_historical_prior:
        if not isinstance(model, (HistoricalPriorRangeQDSModel, HistoricalPriorStudentRangeQDSModel)):
            raise TypeError(f"{model_config.model_type} did not build a historical-prior model.")
        historical_model = model
        prior_points = norm_points
        prior_targets = training_target
        support_ratio = min(1.0, max(0.0, float(getattr(model_config, "historical_prior_support_ratio", 1.0))))
        support_mask = _historical_prior_support_mask(
            targets=training_target,
            boundaries=train_boundaries,
            support_ratio=support_ratio,
        )
        if not bool(support_mask.any().item()):
            raise ValueError("historical_prior_support_ratio removed every training point.")
        prior_points = prior_points[support_mask]
        prior_targets = prior_targets[support_mask]
        prior_source_ids = train_point_source_ids[support_mask] if train_point_source_ids is not None else None
        target_diagnostics["historical_prior_support_ratio"] = float(support_ratio)
        target_diagnostics["historical_prior_support_pre_min_count"] = int(prior_targets.shape[0])
        target_diagnostics["historical_prior_support_pre_min_fraction"] = float(
            int(prior_targets.shape[0]) / max(1, int(training_target.shape[0]))
        )
        target_diagnostics["historical_prior_min_target"] = float(
            getattr(model_config, "historical_prior_min_target", 0.0)
        )
        historical_model.set_prior(prior_points, prior_targets, source_ids=prior_source_ids)
        target_diagnostics["historical_prior_stored_support_count"] = int(historical_model.historical_targets.shape[0])
        target_diagnostics["historical_prior_stored_support_fraction"] = float(
            int(historical_model.historical_targets.shape[0]) / max(1, int(training_target.shape[0]))
        )
        stored_sources = torch.unique(historical_model.historical_source_ids).numel()
        target_diagnostics["historical_prior_source_aggregation"] = str(
            getattr(model_config, "historical_prior_source_aggregation", "none")
        )
        target_diagnostics["historical_prior_source_count"] = int(stored_sources)
    if model_config.model_type in NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES:
        history = [
            {
                "epoch": 0.0,
                "loss": 0.0,
                "pred_std": float(training_target.std(unbiased=False).item()),
                f"positive_fraction_t{workload_type_id}": float(
                    target_diagnostics.get("positive_label_fraction", 0.0)
                ),
                f"label_p95_t{workload_type_id}": float(_safe_quantile(training_target[training_labelled_mask], 0.95).item())
                if bool(training_labelled_mask.any().item())
                else 0.0,
                f"kendall_tau_t{workload_type_id}": 1.0,
                "raw_training_window_count": 0.0,
                "trained_training_window_count": 0.0,
                "filtered_zero_window_count": 0.0,
            }
        ]
        fit_t0 = time.perf_counter()
        try:
            with torch.no_grad():
                train_predictions = model(
                    norm_points.unsqueeze(0),
                    queries=None,
                    query_type_ids=None,
                ).squeeze(0)
            fit_diagnostics = train_target_fit_diagnostics(
                predictions=train_predictions,
                target=training_target,
                labelled_mask=training_labelled_mask,
                boundaries=train_boundaries,
                model_config=model_config,
                workload_type=ID_TO_QUERY_NAME.get(workload_type_id, str(workload_type_id)),
                seed=seed,
            )
            fit_diagnostics["seconds"] = float(time.perf_counter() - fit_t0)
            fit_diagnostics["model_fits_stored_train_support"] = True
            matched_delta = fit_diagnostics.get("matched_mlqds_vs_uniform_target_recall")
            low_delta = fit_diagnostics.get("low_budget_mean_mlqds_vs_uniform_target_recall")
            matched_text = f"{float(matched_delta):+.4f}" if matched_delta is not None else "n/a"
            low_text = f"{float(low_delta):+.4f}" if low_delta is not None else "n/a"
            print(
                f"  [{run_tag}] historical_prior_train_target_fit "
                f"tau={fit_diagnostics.get('score_target_kendall_tau', 0.0):+.3f} "
                f"matched_delta={matched_text} low_delta={low_text} "
                f"({fit_diagnostics['seconds']:.2f}s)",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic must not mask fitted prior.
            fit_diagnostics = {
                "enabled": False,
                "error": str(exc),
                "seconds": float(time.perf_counter() - fit_t0),
                "model_fits_stored_train_support": True,
            }
            print(f"  [{run_tag}] historical_prior_train_target_fit failed: {exc}", flush=True)
        return TrainingOutputs(
            model=model.eval(),
            scaler=scaler,
            labels=labels,
            labelled_mask=labelled_mask,
            history=history,
            epochs_trained=0,
            best_epoch=0,
            best_loss=0.0,
            best_selection_score=0.0,
            target_diagnostics=target_diagnostics,
            fit_diagnostics=fit_diagnostics,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = norm_queries.to(device)
    type_ids_dev = workload.type_ids.to(device)
    training_target_dev = training_target.to(device)
    labelled_mask_dev = training_labelled_mask.to(device)
    factorized_targets_dev = factorized_targets.to(device) if factorized_targets is not None else None
    factorized_mask_dev = factorized_mask.to(device) if factorized_mask is not None else None
    if temporal_residual_budget_masks:
        temporal_residual_budget_masks = tuple(
            (total_ratio, effective_ratio, base_mask.to(device=device, non_blocking=True))
            for total_ratio, effective_ratio, base_mask in temporal_residual_budget_masks
        )
    if temporal_residual_union_mask is not None:
        temporal_residual_union_mask = temporal_residual_union_mask.to(device=device, non_blocking=True)

    opt = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    grad_scaler = GradScaler("cuda", enabled=(amp_mode == "fp16" and device.type == "cuda"))
    windows_cpu = build_trajectory_windows(
        points=norm_points,
        boundaries=train_boundaries,
        window_length=model_config.window_length,
        stride=model_config.window_stride,
    )
    raw_window_count = len(windows_cpu)
    windows_cpu, prefiltered_zero_windows = _filter_supervised_windows(
        windows=windows_cpu,
        training_target=training_target,
        labelled_mask=training_labelled_mask,
        active_type_id=active_type_id,
        require_positive=loss_objective != "pointwise_bce",
    )
    if int(prefiltered_zero_windows.sum().item()) > 0:
        filtered_parts = []
        for type_idx in active_type_ids:
            type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
            filtered_parts.append(f"{type_name}={int(prefiltered_zero_windows[type_idx].item())}")
        print(
            f"  [{run_tag}] filtered {raw_window_count - len(windows_cpu)}/{raw_window_count} "
            f"zero-positive training windows before forward ({', '.join(filtered_parts)})",
            flush=True,
        )
    train_batch_size = max(1, int(getattr(model_config, "train_batch_size", 1)))
    windows = batch_windows(windows_cpu, train_batch_size)
    trained_window_count = len(windows_cpu)
    # Keep diagnostics as sampleable single windows, then batch the selected
    # subset before forward so sampled diagnostics still use useful GPU work.
    diag_windows = windows_cpu
    diag_every = max(1, int(getattr(model_config, "diagnostic_every", 1)))
    diag_fraction = float(getattr(model_config, "diagnostic_window_fraction", 1.0))
    diag_fraction = min(1.0, max(0.05, diag_fraction))
    selection_metric = str(getattr(model_config, "checkpoint_selection_metric", "score")).lower()
    if selection_metric not in {"loss", "score", "uniform_gap"}:
        raise ValueError("checkpoint_selection_metric must be 'loss', 'score', or 'uniform_gap'.")
    validation_score_every = int(getattr(model_config, "validation_score_every", 0) or 0)
    has_validation_score = (
        validation_trajectories is not None
        and validation_boundaries is not None
        and validation_workload is not None
        and validation_workload_map is not None
    )
    if selection_metric in {"score", "uniform_gap"} and not has_validation_score:
        print(
            f"  [{run_tag}] WARNING: checkpoint_selection_metric={selection_metric} "
            "requested without validation workload; "
            "falling back to loss selection.",
            flush=True,
        )
        selection_metric = "loss"
    if selection_metric in {"score", "uniform_gap"} and validation_score_every <= 0:
        validation_score_every = diag_every
    validation_points_for_score: torch.Tensor | None = None
    validation_query_cache: Any | None = None
    if has_validation_score:
        from evaluation.query_cache import EvaluationQueryCache

        validation_trajectories, validation_boundaries, validation_workload = _require_validation_inputs(
            validation_trajectories,
            validation_boundaries,
            validation_workload,
        )
        validation_points_for_score = (
            validation_points
            if validation_points is not None
            else torch.cat(validation_trajectories, dim=0)
        )
        if precomputed_validation_query_cache is None:
            validation_query_cache = EvaluationQueryCache.for_workload(
                validation_points_for_score,
                validation_boundaries,
                validation_workload.typed_queries,
            )
        else:
            precomputed_validation_query_cache.validate(
                validation_points_for_score,
                validation_boundaries,
                validation_workload.typed_queries,
            )
            validation_query_cache = precomputed_validation_query_cache
    validation_uniform_result: tuple[float, dict[str, float]] | None = None
    if selection_metric == "uniform_gap" and has_validation_score:
        validation_trajectories, validation_boundaries, validation_workload = _require_validation_inputs(
            validation_trajectories,
            validation_boundaries,
            validation_workload,
        )
        validation_uniform_result = _validation_uniform_score(
            trajectories=validation_trajectories,
            boundaries=validation_boundaries,
            workload=validation_workload,
            workload_map=validation_workload_map or {},
            model_config=model_config,
            validation_points=validation_points_for_score,
            query_cache=validation_query_cache,
        )
        uniform_score, uniform_per_type = validation_uniform_result
        print(
            f"  [{run_tag}] validation uniform_score={uniform_score:.6f}  "
            f"range={uniform_per_type.get('range', 0.0):.6f}",
            flush=True,
        )

    training_sample_generator = torch.Generator().manual_seed(int(seed) + 99)
    # Separate fixed-seed generator for diagnostics so the tau subsample
    # stays consistent across epochs and doesn't oscillate with training state.
    diagnostic_sample_generator = torch.Generator().manual_seed(int(seed) + 777)
    history: list[dict[str, float]] = []

    effective_epochs = max(1, int(model_config.epochs))
    patience = int(getattr(model_config, "early_stopping_patience", 0) or 0)
    smoothing_window = max(1, int(getattr(model_config, "checkpoint_smoothing_window", 1) or 1))
    checkpoint_full_score_every = max(1, int(getattr(model_config, "checkpoint_full_score_every", 1) or 1))
    checkpoint_candidate_pool_size = max(1, int(getattr(model_config, "checkpoint_candidate_pool_size", 1) or 1))
    checkpoint_candidates: list[CheckpointCandidate] = []
    selection_history: list[float] = []
    best_selection = float("-inf")
    best_loss = float("inf")
    best_selection_score = 0.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epoch_label_width = len(str(effective_epochs))
    epochs_trained = 0
    for epoch in range(effective_epochs):
        epoch_t0 = time.perf_counter()
        epoch_result = _train_one_epoch(
            model=model,
            windows=windows,
            opt=opt,
            grad_scaler=grad_scaler,
            model_config=model_config,
            device=device,
            amp_mode=amp_mode,
            norm_queries_dev=norm_queries_dev,
            type_ids_dev=type_ids_dev,
            training_target_dev=training_target_dev,
            labelled_mask_dev=labelled_mask_dev,
            prefiltered_zero_windows=prefiltered_zero_windows,
            active_type_id=active_type_id,
            loss_objective=loss_objective,
            budget_ratios=budget_ratios,
            budget_loss_temperature=budget_loss_temperature,
            temporal_residual_budget_masks=temporal_residual_budget_masks,
            temporal_residual_union_mask=temporal_residual_union_mask,
            training_sample_generator=training_sample_generator,
            factorized_targets_dev=factorized_targets_dev,
            factorized_mask_dev=factorized_mask_dev,
        )
        epoch_timing = {
            "forward_s": float(epoch_result.timing["forward_s"]),
            "loss_s": float(epoch_result.timing["loss_s"]),
            "backward_s": float(epoch_result.timing["backward_s"]),
            "diagnostic_s": 0.0,
            "validation_score_s": 0.0,
        }
        evaluated_checkpoint_candidates: list[CheckpointCandidate] = []
        epoch_loss = epoch_result.loss
        positive_windows = epoch_result.positive_windows
        skipped_zero_windows = epoch_result.skipped_zero_windows
        ranking_pair_counts = epoch_result.ranking_pair_counts

        # Diagnostic pass only on selected epochs (every `diag_every` epochs and
        # the final epoch).  Subsample windows by `diag_fraction` to further cut
        # cost: pred_std and tau are statistical aggregates and noise from a
        # ~20% sample is tiny compared to the training noise we're measuring.
        is_last_epoch = (epoch + 1) == effective_epochs
        is_diag_epoch = ((epoch + 1) % diag_every == 0) or is_last_epoch or epoch == 0
        if is_diag_epoch:
            diagnostic_t0 = time.perf_counter()
            if diag_fraction < 1.0 and len(diag_windows) > 8:
                diagnostic_window_count = max(8, int(len(diag_windows) * diag_fraction))
                sample_indices = torch.randperm(
                    len(diag_windows),
                    generator=diagnostic_sample_generator,
                )[:diagnostic_window_count].tolist()
                diagnostic_windows = [diag_windows[i] for i in sample_indices]
            else:
                diagnostic_windows = diag_windows

            model.eval()
            with torch.no_grad():
                diagnostic_score_sum = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                diagnostic_score_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                diagnostic_batch_size = max(1, int(getattr(model_config, "inference_batch_size", train_batch_size)))
                for diagnostic_batch_cpu in batch_windows(diagnostic_windows, diagnostic_batch_size):
                    diagnostic_batch = _trajectory_batch_to_device(diagnostic_batch_cpu, device)
                    with torch_autocast_context(device, amp_mode):
                        window_scores = model(
                            points=diagnostic_batch.points,
                            queries=norm_queries_dev,
                            query_type_ids=type_ids_dev,
                            padding_mask=diagnostic_batch.padding_mask,
                        )
                    window_scores = window_scores.float()
                    for batch_idx in range(window_scores.shape[0]):
                        point_indices = diagnostic_batch.global_indices[batch_idx]
                        valid_points = point_indices >= 0
                        diagnostic_score_sum[point_indices[valid_points]] = (
                            diagnostic_score_sum[point_indices[valid_points]]
                            + window_scores[batch_idx, valid_points]
                        )
                        diagnostic_score_count[point_indices[valid_points]] = (
                            diagnostic_score_count[point_indices[valid_points]] + 1.0
                        )
                covered_mask = diagnostic_score_count > 0
                diagnostic_score_count = diagnostic_score_count.clamp(min=1.0)
                full_scores = diagnostic_score_sum / diagnostic_score_count

            stats: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
                "pred_std": (
                    float(full_scores[covered_mask].std().item())
                    if bool(covered_mask.any().item())
                    else 0.0
                ),
            }
            for type_idx in range(NUM_QUERY_TYPES):
                stats[f"positive_windows_t{type_idx}"] = float(positive_windows[type_idx].item())
                stats[f"skipped_zero_windows_t{type_idx}"] = float(skipped_zero_windows[type_idx].item())
                stats[f"ranking_pairs_t{type_idx}"] = float(ranking_pair_counts[type_idx].item())
                stats[f"pred_p50_t{type_idx}"] = 0.0
                stats[f"pred_p90_t{type_idx}"] = 0.0
                stats[f"pred_p99_t{type_idx}"] = 0.0
                stats[f"positive_fraction_t{type_idx}"] = 0.0
                stats[f"label_p95_t{type_idx}"] = 0.0
                stats[f"kendall_tau_t{type_idx}"] = 0.0
            for t in range(NUM_QUERY_TYPES):
                if t != active_type_id:
                    continue
                type_scores = full_scores
                stats[f"pred_p50_t{t}"] = float(_safe_quantile(type_scores, 0.50).item())
                stats[f"pred_p90_t{t}"] = float(_safe_quantile(type_scores, 0.90).item())
                stats[f"pred_p99_t{t}"] = float(_safe_quantile(type_scores, 0.99).item())
                labelled_type = labelled_mask_dev
                positive_type = labelled_type & (training_target_dev > 0)
                labelled_count = max(1, int(labelled_type.sum().item()))
                stats[f"positive_fraction_t{t}"] = float(positive_type.sum().item() / labelled_count)
                if bool(positive_type.any().item()):
                    stats[f"label_p95_t{t}"] = float(_safe_quantile(training_target_dev[positive_type], 0.95).item())
                else:
                    stats[f"label_p95_t{t}"] = 0.0
                eval_mask = labelled_mask_dev & covered_mask
                if bool(eval_mask.any().item()):
                    # Reset the diagnostic generator each epoch so the diagnostic
                    # subsample is identical across epochs, giving stable tau trends.
                    diagnostic_sample_generator.manual_seed(int(seed) + 777)
                    pred_sample, target_sample = _discriminative_sample(
                        type_scores[eval_mask].detach().cpu(),
                        training_target_dev[eval_mask].detach().cpu(),
                        n_each=100,
                        generator=diagnostic_sample_generator,
                    )
                    stats[f"kendall_tau_t{t}"] = _kendall_tau(pred_sample, target_sample)
                else:
                    stats[f"kendall_tau_t{t}"] = 0.0

            if stats["pred_std"] < 1e-3:
                stats["collapse_warning"] = 1.0
            epoch_timing["diagnostic_s"] += time.perf_counter() - diagnostic_t0

            candidate_tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            candidate_avg_tau = sum(candidate_tau_vals) / max(1, len(candidate_tau_vals))
            validation_score_due = (
                validation_score_every <= 0
                or (epoch + 1) % validation_score_every == 0
                or is_last_epoch
                or epoch == 0
            )
            full_score_due = validation_score_due and (
                checkpoint_full_score_every <= 1
                or (epoch + 1) % checkpoint_full_score_every == 0
                or is_last_epoch
            )
            use_checkpoint_candidate_pool = (
                has_validation_score
                and validation_score_due
                and selection_metric in {"score", "uniform_gap"}
                and checkpoint_full_score_every > 1
            )
            should_run_validation_score = has_validation_score and full_score_due and (
                selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
            ) and not use_checkpoint_candidate_pool
            if should_run_validation_score:
                score_t0 = time.perf_counter()
                validation_trajectories, validation_boundaries, validation_workload = _require_validation_inputs(
                    validation_trajectories,
                    validation_boundaries,
                    validation_workload,
                )
                validation_score, per_type_score, validation_metrics = _validation_checkpoint_scores(
                    model=model,
                    scaler=scaler,
                    trajectories=validation_trajectories,
                    boundaries=validation_boundaries,
                    workload=validation_workload,
                    workload_map=validation_workload_map or {},
                    model_config=model_config,
                    device=device,
                    validation_points=validation_points_for_score,
                    query_cache=validation_query_cache,
                    range_geometry_scores=precomputed_validation_geometry_scores,
                )
                epoch_timing["validation_score_s"] += time.perf_counter() - score_t0
                record_validation_stats(
                    stats,
                    validation_score=validation_score,
                    per_type_score=per_type_score,
                    validation_metrics=validation_metrics,
                    validation_uniform_result=validation_uniform_result,
                    validation_workload_map=validation_workload_map,
                )
            if has_validation_score and validation_score_due and selection_metric in {"score", "uniform_gap"}:
                stats["checkpoint_score_candidate"] = 1.0
                stats["checkpoint_candidate_cheap_score"] = selection_score(
                    candidate_avg_tau,
                    stats["pred_std"],
                    stats["loss"],
                )
                stats["checkpoint_full_score_due"] = 1.0 if full_score_due else 0.0
                if use_checkpoint_candidate_pool:
                    checkpoint_candidates.append(
                        CheckpointCandidate(
                            epoch_number=epoch + 1,
                            epoch_index=epoch,
                            cheap_score=float(stats["checkpoint_candidate_cheap_score"]),
                            loss=float(stats["loss"]),
                            state_dict=_model_state_on_cpu(model),
                            stats=stats,
                            avg_tau=candidate_avg_tau,
                        )
                    )
                    checkpoint_candidates.sort(key=lambda candidate: candidate.cheap_score, reverse=True)
                    checkpoint_candidates = checkpoint_candidates[:checkpoint_candidate_pool_size]
                    if full_score_due and checkpoint_candidates:
                        score_t0 = time.perf_counter()
                        validation_trajectories, validation_boundaries, validation_workload = _require_validation_inputs(
                            validation_trajectories,
                            validation_boundaries,
                            validation_workload,
                        )
                        current_state_dict = _model_state_on_cpu(model)
                        for candidate in sorted(checkpoint_candidates, key=lambda item: item.epoch_number):
                            candidate_t0 = time.perf_counter()
                            model.load_state_dict(candidate.state_dict)
                            validation_score, per_type_score, validation_metrics = _validation_checkpoint_scores(
                                model=model,
                                scaler=scaler,
                                trajectories=validation_trajectories,
                                boundaries=validation_boundaries,
                                workload=validation_workload,
                                workload_map=validation_workload_map or {},
                                model_config=model_config,
                                device=device,
                                validation_points=validation_points_for_score,
                                query_cache=validation_query_cache,
                                range_geometry_scores=precomputed_validation_geometry_scores,
                            )
                            record_validation_stats(
                                candidate.stats,
                                validation_score=validation_score,
                                per_type_score=per_type_score,
                                validation_metrics=validation_metrics,
                                validation_uniform_result=validation_uniform_result,
                                validation_workload_map=validation_workload_map,
                            )
                            candidate.stats["checkpoint_candidate_evaluated"] = 1.0
                            candidate.stats["checkpoint_full_score_round_epoch"] = float(epoch + 1)
                            candidate.stats["checkpoint_validation_seconds"] = float(time.perf_counter() - candidate_t0)
                            evaluated_checkpoint_candidates.append(candidate)
                        model.load_state_dict(current_state_dict)
                        epoch_timing["validation_score_s"] += time.perf_counter() - score_t0
                        checkpoint_candidates = []
        else:
            # Skip diagnostics this epoch; log only loss.  Patience counters
            # are only updated on diagnostic epochs below.
            stats = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
            }
            for type_idx in range(NUM_QUERY_TYPES):
                stats[f"positive_windows_t{type_idx}"] = float(positive_windows[type_idx].item())
                stats[f"skipped_zero_windows_t{type_idx}"] = float(skipped_zero_windows[type_idx].item())
                stats[f"ranking_pairs_t{type_idx}"] = float(ranking_pair_counts[type_idx].item())

        epoch_dt = time.perf_counter() - epoch_t0
        stats["epoch_seconds"] = float(epoch_dt)
        stats["epoch_forward_seconds"] = float(epoch_timing["forward_s"])
        stats["epoch_loss_seconds"] = float(epoch_timing["loss_s"])
        stats["epoch_backward_seconds"] = float(epoch_timing["backward_s"])
        stats["epoch_diagnostic_seconds"] = float(epoch_timing["diagnostic_s"])
        stats["epoch_validation_score_seconds"] = float(epoch_timing["validation_score_s"])
        stats["epoch_f1_seconds"] = stats["epoch_validation_score_seconds"]
        stats["raw_training_window_count"] = float(raw_window_count)
        stats["trained_training_window_count"] = float(trained_window_count)
        stats["filtered_zero_window_count"] = float(raw_window_count - trained_window_count)
        for type_idx in range(NUM_QUERY_TYPES):
            stats[f"filtered_zero_windows_t{type_idx}"] = float(prefiltered_zero_windows[type_idx].item())
        history.append(stats)

        epochs_trained = epoch + 1

        if is_diag_epoch:
            tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            avg_tau = sum(tau_vals) / max(1, len(tau_vals))
            collapse = "  COLLAPSE" if stats.get("collapse_warning") else ""
            selection: float | None = None
            smoothed_selection: float | None = None
            is_new_best_model = False
            validation_round_had_selection = False
            validation_round_improved = False
            if evaluated_checkpoint_candidates:
                for candidate in sorted(evaluated_checkpoint_candidates, key=lambda item: item.epoch_number):
                    candidate_selection = selection_from_stats(
                        stats=candidate.stats,
                        avg_tau=candidate.avg_tau,
                        selection_metric=selection_metric,
                        validation_uniform_result=validation_uniform_result,
                        validation_workload_map=validation_workload_map,
                        model_config=model_config,
                    )
                    if candidate_selection is None:
                        continue
                    validation_round_had_selection = True
                    candidate.stats["selection_score"] = candidate_selection
                    selection_history.append(float(candidate_selection))
                    window = selection_history[-smoothing_window:]
                    candidate_smoothed = float(sum(window) / len(window))
                    candidate.stats["selection_score_smoothed"] = candidate_smoothed
                    candidate_is_new_best = candidate_smoothed > best_selection + 1e-4 or (
                        abs(candidate_smoothed - best_selection) <= 1e-4 and candidate.loss < best_loss - 1e-8
                    )
                    if candidate_is_new_best:
                        validation_round_improved = True
                        best_selection = candidate_smoothed
                        best_loss = candidate.loss
                        best_selection_score = float(candidate.stats.get("val_selection_score", best_selection_score))
                        best_epoch = candidate.epoch_number
                        best_state_dict = candidate.state_dict
                        candidate.stats["checkpoint_promoted"] = 1.0
                    else:
                        candidate.stats["checkpoint_promoted"] = 0.0
                    if candidate.stats is not stats:
                        status = "promoted" if candidate_is_new_best else "checked"
                        print(
                            f"  [{run_tag}] checkpoint candidate epoch "
                            f"{candidate.epoch_number:0{epoch_label_width}d}/{effective_epochs}  "
                            f"cheap={candidate.cheap_score:+.3f}  "
                            f"select={candidate_selection:+.3f}  "
                            f"smoothed={candidate_smoothed:+.3f}  {status}",
                            flush=True,
                        )
                if "selection_score" in stats:
                    selection = float(stats["selection_score"])
                    smoothed_selection = float(stats["selection_score_smoothed"])
                    is_new_best_model = bool(stats.get("checkpoint_promoted", 0.0))
            else:
                selection = selection_from_stats(
                    stats=stats,
                    avg_tau=avg_tau,
                    selection_metric=selection_metric,
                    validation_uniform_result=validation_uniform_result,
                    validation_workload_map=validation_workload_map,
                    model_config=model_config,
                )
                if selection is not None:
                    validation_round_had_selection = True
                    stats["selection_score"] = selection
                    selection_history.append(float(selection))
                    window = selection_history[-smoothing_window:]
                    smoothed_score = float(sum(window) / len(window))
                    smoothed_selection = smoothed_score
                    stats["selection_score_smoothed"] = smoothed_score
                    # Use the smoothed score for "best" decisions: averages out
                    # epoch-to-epoch validation score noise so we don't lock onto a lucky
                    # spike. Single-epoch loss still tiebreaks on near-equal smoothed.
                    is_new_best_model = smoothed_score > best_selection + 1e-4 or (
                        abs(smoothed_score - best_selection) <= 1e-4 and stats["loss"] < best_loss - 1e-8
                    )
                    validation_round_improved = is_new_best_model
            markers = []
            if epoch > 0 and is_new_best_model:
                markers.append("*** NEW BEST MODEL ***")
            best_marker = ("  " + "  ".join(markers)) if markers else ""
            smoothed_label = (
                f"  smoothed_w{smoothing_window}={smoothed_selection:+.3f}"
                if smoothing_window > 1 and smoothed_selection is not None
                else ""
            )
            selection_text = f"{selection:+.3f}" if selection is not None else "skipped"
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  avg_tau={avg_tau:+.3f}  "
                f"pred_std={stats['pred_std']:.6g}  select={selection_text}{smoothed_label}  "
                f"({epoch_dt:.2f}s){collapse}{best_marker}",
                flush=True,
            )
            if "val_selection_score" in stats:
                print(
                    f"    [{run_tag}] val_selection_score={stats['val_selection_score']:.6f}  "
                    f"range_point_f1={stats.get('val_range_point_f1', 0.0):.6f}  "
                    f"range_usefulness={stats.get('val_range_usefulness', 0.0):.6f}  "
                    f"answer_f1={stats.get('val_answer_f1', 0.0):.6f}  "
                    f"combined_f1={stats.get('val_combined_f1', 0.0):.6f}",
                    flush=True,
                )
            if "val_uniform_score" in stats:
                print(
                    f"    [{run_tag}] val_vs_uniform aggregate={stats['val_selection_uniform_gap']:+.6f}  "
                    f"type_deficit={stats['val_selection_type_deficit']:.6f}  "
                    f"range={stats.get('val_selection_score_gap_range', 0.0):+.6f}",
                    flush=True,
                )
            diag_parts = []
            for type_idx in active_type_ids:
                type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
                diag_parts.append(
                    f"{type_name}:pos={stats[f'positive_fraction_t{type_idx}']:.4f},"
                    f"p95={stats[f'label_p95_t{type_idx}']:.3f},"
                    f"pairs={int(stats[f'ranking_pairs_t{type_idx}'])},"
                    f"skip={int(stats[f'skipped_zero_windows_t{type_idx}'])},"
                    f"filtered={int(stats[f'filtered_zero_windows_t{type_idx}'])}"
                )
            if diag_parts:
                print(f"    [{run_tag}] label_diag  " + "  ".join(diag_parts), flush=True)
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"diagnostic={stats['epoch_diagnostic_seconds']:.2f}s  "
                f"validation_score={stats['epoch_validation_score_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )

            if is_new_best_model:
                best_selection = float(stats["selection_score_smoothed"])
                best_loss = stats["loss"]
                best_selection_score = float(stats.get("val_selection_score", best_selection_score))
                best_epoch = epoch + 1
                best_state_dict = _model_state_on_cpu(model)

            if patience > 0 and validation_round_had_selection:
                if is_new_best_model or validation_round_improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"  [{run_tag}] early stopping at epoch {epoch + 1:0{epoch_label_width}d}: "
                            f"selection score did not improve over {patience} diag epochs "
                            f"(best_selection={best_selection:+.3f}, best_loss={best_loss:.8f})",
                            flush=True,
                        )
                        break
        else:
            # Non-diagnostic epoch: log loss only, no tau / early-stopping update.
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  (no-diag)  ({epoch_dt:.2f}s)",
                flush=True,
            )
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )

    model = model.to("cpu")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"  [{run_tag}] restored best diagnostic epoch {best_epoch}/{epochs_trained} "
            f"(selection={best_selection:+.3f}, loss={best_loss:.8f}, "
            f"val_selection_score={best_selection_score:.6f})",
            flush=True,
        )
    fit_t0 = time.perf_counter()
    fit_diagnostics: dict[str, Any] = {}
    try:
        train_predictions = windowed_predict(
            model=model,
            norm_points=norm_points,
            boundaries=train_boundaries,
            queries=norm_queries,
            query_type_ids=workload.type_ids,
            window_length=model_config.window_length,
            window_stride=model_config.window_stride,
            batch_size=max(1, int(getattr(model_config, "inference_batch_size", train_batch_size))),
            device=device,
            amp_mode=amp_mode,
        )
        fit_diagnostics = train_target_fit_diagnostics(
            predictions=train_predictions,
            target=training_target,
            labelled_mask=training_labelled_mask,
            boundaries=train_boundaries,
            model_config=model_config,
            workload_type=ID_TO_QUERY_NAME.get(workload_type_id, str(workload_type_id)),
            seed=seed,
        )
        fit_diagnostics["seconds"] = float(time.perf_counter() - fit_t0)
        matched_delta = fit_diagnostics.get("matched_mlqds_vs_uniform_target_recall")
        low_delta = fit_diagnostics.get("low_budget_mean_mlqds_vs_uniform_target_recall")
        matched_text = f"{float(matched_delta):+.4f}" if matched_delta is not None else "n/a"
        low_text = f"{float(low_delta):+.4f}" if low_delta is not None else "n/a"
        print(
            f"  [{run_tag}] train_target_fit "
            f"tau={fit_diagnostics.get('score_target_kendall_tau', 0.0):+.3f} "
            f"matched_delta={matched_text} low_delta={low_text} "
            f"({fit_diagnostics['seconds']:.2f}s)",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic must not mask training result.
        fit_diagnostics = {
            "enabled": False,
            "error": str(exc),
            "seconds": float(time.perf_counter() - fit_t0),
        }
        print(f"  [{run_tag}] train_target_fit failed: {exc}", flush=True)
    return TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=labels,
        labelled_mask=labelled_mask,
        history=history,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        target_diagnostics=target_diagnostics,
        fit_diagnostics=fit_diagnostics,
        feature_context={
            "query_prior_field": query_prior_field,
            "query_prior_field_metadata": query_prior_field_metadata(query_prior_field),
        },
    )

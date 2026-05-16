"""AIS-QDS end-to-end experiment entrypoint. See experiments/README.md for details."""

from __future__ import annotations

import time
from pathlib import Path

from data.ais_loader import generate_synthetic_ais_data, load_ais_csv
from data.trajectory_cache import load_or_build_ais_cache
from experiments.cli_utils import normalized_gap_arg, split_csv_path_list
from experiments.experiment_cli import build_parser
from experiments.experiment_config import build_experiment_config
from experiments.experiment_pipeline import run_experiment_pipeline
from experiments.torch_runtime import apply_torch_runtime_settings


def _split_max_segments(args, split: str) -> int | None:
    """Return split-specific segment cap, falling back to the global cap."""
    override = getattr(args, f"{split}_max_segments")
    return args.max_segments if override is None else override


def _cap_loaded_trajectories(
    trajectories,
    mmsis: list[int] | None,
    max_trajectories: int | None,
):
    """Cap loaded trajectories for smoke runs while keeping MMSIs aligned."""
    if max_trajectories is None:
        return trajectories, mmsis
    cap = int(max_trajectories)
    if cap <= 0:
        raise ValueError("--max_trajectories must be positive when provided.")
    if len(trajectories) <= cap:
        return trajectories, mmsis
    capped = trajectories[:cap]
    capped_mmsis = mmsis[:cap] if mmsis is not None else None
    print(f"[load-data] capped trajectories to {cap}", flush=True)
    return capped, capped_mmsis


def _assert_distinct_csv_sources(
    *,
    train_csv_path: str,
    validation_csv_path: str | None,
    eval_csv_path: str,
) -> None:
    """Reject duplicate explicit data splits before loading."""
    seen: dict[Path, str] = {}
    train_paths = split_csv_path_list(train_csv_path)
    validation_paths = split_csv_path_list(validation_csv_path)
    eval_paths = split_csv_path_list(eval_csv_path)
    named_sources = {
        **{
            ("train" if len(train_paths) == 1 else f"train[{idx}]"): value
            for idx, value in enumerate(train_paths)
        },
        **{
            ("validation" if len(validation_paths) == 1 else f"validation[{idx}]"): value
            for idx, value in enumerate(validation_paths)
        },
        **{
            ("eval" if len(eval_paths) == 1 else f"eval[{idx}]"): value
            for idx, value in enumerate(eval_paths)
        },
    }
    for label, value in named_sources.items():
        if value is None:
            continue
        resolved = Path(value).resolve()
        if resolved in seen:
            raise ValueError(f"{label} CSV path must be distinct from {seen[resolved]} CSV path: {value}")
        seen[resolved] = label


def _log_load_audit(label: str, audit) -> None:
    """Print a compact AIS load audit for repeatable run logs."""
    length = audit.segment_length_stats
    gaps = audit.time_gap_stats
    print(
        f"[load-data] {label} audit: rows={audit.rows_loaded} "
        f"dropped_invalid={audit.rows_dropped_invalid} "
        f"duplicates={audit.duplicate_timestamp_rows} "
        f"mmsis={audit.input_mmsi_count} "
        f"segments={audit.output_segment_count} "
        f"points={audit.output_point_count} "
        f"short_segments_dropped={audit.dropped_short_segments} "
        f"gap_splits={audit.time_gap_over_threshold_count} "
        f"segment_len_p50={length.get('p50', 0.0):.1f} "
        f"segment_len_p95={length.get('p95', 0.0):.1f} "
        f"max_gap_s={gaps.get('max', 0.0):.1f}",
        flush=True,
    )


def _load_csv_trajectories(label: str, csv_path: str, args, load_kwargs: dict) -> tuple:
    """Load one CSV either through the Parquet cache or directly from source."""
    if args.cache_dir:
        cache = load_or_build_ais_cache(
            csv_path,
            cache_dir=args.cache_dir,
            refresh_cache=bool(args.refresh_cache),
            **load_kwargs,
        )
        state = "hit" if cache.cache_hit else "built"
        print(f"[load-data] cache {state}: {cache.cache_dir}", flush=True)
        _log_load_audit(label, cache.audit)
        audit_payload = cache.audit.to_dict()
        audit_payload["cache"] = cache.cache_metadata()
        return cache.trajectories, cache.mmsis, cache.audit, audit_payload

    trajectories, mmsis, audit = load_ais_csv(
        csv_path,
        **load_kwargs,
        return_mmsis=True,
        return_audit=True,
    )
    _log_load_audit(label, audit)
    return trajectories, mmsis, audit, audit.to_dict()


def _combined_train_audit_payload(paths: tuple[str, ...], payloads: list[dict]) -> dict:
    """Return a compact audit payload for one or more CSV sources."""
    if len(payloads) == 1:
        return payloads[0]
    output_segment_count = sum(int(payload.get("output_segment_count", 0)) for payload in payloads)
    output_point_count = sum(int(payload.get("output_point_count", 0)) for payload in payloads)
    return {
        "source_path": ",".join(paths),
        "source_count": len(payloads),
        "sources": payloads,
        "output_segment_count": output_segment_count,
        "output_point_count": output_point_count,
    }


def _default_simplified_dir(args) -> str:
    """Build a run-local default directory for simplified CSV output."""
    return str(Path(args.results_dir) / "simplified_eval")


def main() -> None:
    """Parse CLI args and run the AIS-QDS experiment. See experiments/README.md for details."""
    parser = build_parser()
    args = parser.parse_args()

    config = build_experiment_config(
        n_ships=args.n_ships,
        n_points=args.n_points,
        synthetic_route_families=args.synthetic_route_families,
        min_points_per_segment=args.min_points_per_segment,
        max_points_per_segment=args.max_points_per_segment,
        max_time_gap_seconds=normalized_gap_arg(args.max_time_gap_seconds),
        max_segments=args.max_segments,
        train_max_segments=args.train_max_segments,
        validation_max_segments=args.validation_max_segments,
        eval_max_segments=args.eval_max_segments,
        max_trajectories=args.max_trajectories,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        n_queries=args.n_queries,
        query_coverage=args.query_coverage,
        max_queries=args.max_queries,
        range_spatial_fraction=args.range_spatial_fraction,
        range_time_fraction=args.range_time_fraction,
        range_spatial_km=args.range_spatial_km,
        range_time_hours=args.range_time_hours,
        range_footprint_jitter=args.range_footprint_jitter,
        range_time_domain_mode=args.range_time_domain_mode,
        range_anchor_mode=args.range_anchor_mode,
        range_train_anchor_modes=args.range_train_anchor_modes,
        range_train_footprints=args.range_train_footprints,
        range_min_point_hits=args.range_min_point_hits,
        range_max_point_hit_fraction=args.range_max_point_hit_fraction,
        range_min_trajectory_hits=args.range_min_trajectory_hits,
        range_max_trajectory_hit_fraction=args.range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction=args.range_max_box_volume_fraction,
        range_duplicate_iou_threshold=args.range_duplicate_iou_threshold,
        range_acceptance_max_attempts=args.range_acceptance_max_attempts,
        range_max_coverage_overshoot=args.range_max_coverage_overshoot,
        range_train_workload_replicates=args.range_train_workload_replicates,
        workload_profile_id=args.workload_profile_id,
        coverage_calibration_mode=args.coverage_calibration_mode,
        workload_stability_gate_mode=args.workload_stability_gate_mode,
        epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        ranking_pairs_per_type=args.ranking_pairs_per_type,
        ranking_top_quantile=args.ranking_top_quantile,
        pointwise_loss_weight=args.pointwise_loss_weight,
        loss_objective=args.loss_objective,
        budget_loss_ratios=args.budget_loss_ratios,
        budget_loss_temperature=args.budget_loss_temperature,
        temporal_distribution_loss_weight=args.temporal_distribution_loss_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        compression_ratio=args.compression_ratio,
        csv_path=args.csv_path,
        train_csv_path=args.train_csv_path,
        validation_csv_path=args.validation_csv_path,
        eval_csv_path=args.eval_csv_path,
        validation_split_mode=args.validation_split_mode,
        model_type=args.model_type,
        historical_prior_k=args.historical_prior_k,
        historical_prior_clock_weight=args.historical_prior_clock_weight,
        historical_prior_mmsi_weight=args.historical_prior_mmsi_weight,
        historical_prior_density_weight=args.historical_prior_density_weight,
        historical_prior_min_target=args.historical_prior_min_target,
        historical_prior_support_ratio=args.historical_prior_support_ratio,
        historical_prior_source_aggregation=args.historical_prior_source_aggregation,
        workload=args.workload,
        seed=args.seed,
        range_diagnostics_mode=args.range_diagnostics_mode,
        early_stopping_patience=args.early_stopping_patience,
        train_batch_size=args.train_batch_size,
        inference_batch_size=args.inference_batch_size,
        query_chunk_size=args.query_chunk_size,
        diagnostic_every=args.diagnostic_every,
        diagnostic_window_fraction=args.diagnostic_window_fraction,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        validation_score_every=args.validation_score_every,
        checkpoint_uniform_gap_weight=args.checkpoint_uniform_gap_weight,
        checkpoint_type_penalty_weight=args.checkpoint_type_penalty_weight,
        checkpoint_smoothing_window=args.checkpoint_smoothing_window,
        checkpoint_full_score_every=args.checkpoint_full_score_every,
        checkpoint_candidate_pool_size=args.checkpoint_candidate_pool_size,
        checkpoint_score_variant=args.checkpoint_score_variant,
        validation_global_sanity_penalty_enabled=args.validation_global_sanity_penalty,
        validation_global_sanity_penalty_weight=args.validation_global_sanity_penalty_weight,
        validation_sed_penalty_weight=args.validation_sed_penalty_weight,
        validation_endpoint_penalty_weight=args.validation_endpoint_penalty_weight,
        validation_length_preservation_min=args.validation_length_preservation_min,
        mlqds_temporal_fraction=args.mlqds_temporal_fraction,
        mlqds_diversity_bonus=args.mlqds_diversity_bonus,
        mlqds_hybrid_mode=args.mlqds_hybrid_mode,
        selector_type=args.selector_type,
        learned_segment_geometry_gain_weight=args.learned_segment_geometry_gain_weight,
        learned_segment_score_blend_weight=args.learned_segment_score_blend_weight,
        learned_segment_fairness_preallocation=args.learned_segment_fairness_preallocation,
        mlqds_stratified_center_weight=args.mlqds_stratified_center_weight,
        mlqds_min_learned_swaps=args.mlqds_min_learned_swaps,
        mlqds_score_mode=args.mlqds_score_mode,
        mlqds_score_temperature=args.mlqds_score_temperature,
        mlqds_rank_confidence_weight=args.mlqds_rank_confidence_weight,
        mlqds_range_geometry_blend=args.mlqds_range_geometry_blend,
        temporal_residual_label_mode=args.temporal_residual_label_mode,
        range_label_mode=args.range_label_mode,
        range_training_target_mode=args.range_training_target_mode,
        range_target_balance_mode=args.range_target_balance_mode,
        range_replicate_target_aggregation=args.range_replicate_target_aggregation,
        range_component_target_blend=args.range_component_target_blend,
        range_temporal_target_blend=args.range_temporal_target_blend,
        range_structural_target_blend=args.range_structural_target_blend,
        range_structural_target_source_mode=args.range_structural_target_source_mode,
        range_target_budget_weight_power=args.range_target_budget_weight_power,
        range_marginal_target_radius_scale=args.range_marginal_target_radius_scale,
        range_query_spine_fraction=args.range_query_spine_fraction,
        range_query_spine_mass_mode=args.range_query_spine_mass_mode,
        range_query_residual_multiplier=args.range_query_residual_multiplier,
        range_query_residual_mass_mode=args.range_query_residual_mass_mode,
        range_set_utility_multiplier=args.range_set_utility_multiplier,
        range_set_utility_candidate_limit=args.range_set_utility_candidate_limit,
        range_set_utility_mass_mode=args.range_set_utility_mass_mode,
        range_boundary_prior_weight=args.range_boundary_prior_weight,
        range_teacher_distillation_mode=args.range_teacher_distillation_mode,
        range_teacher_epochs=args.range_teacher_epochs,
        range_audit_compression_ratios=args.range_audit_compression_ratios,
        final_metrics_mode=args.final_metrics_mode,
        float32_matmul_precision=args.float32_matmul_precision,
        allow_tf32=args.allow_tf32,
        amp_mode=args.amp_mode,
    )
    runtime_settings = apply_torch_runtime_settings(
        float32_matmul_precision=config.model.float32_matmul_precision,
        allow_tf32=config.model.allow_tf32,
    )

    coverage_msg = (
        f"  query_coverage={args.query_coverage}  max_queries={args.max_queries}"
        if args.query_coverage is not None else ""
    )
    print(
        f"[config] model={args.model_type}  workload={args.workload}  epochs={args.epochs}  "
        f"n_queries={args.n_queries}{coverage_msg}  compression_ratio={args.compression_ratio}  "
        f"lr={args.lr}  "
        f"embed_dim={args.embed_dim}  num_heads={args.num_heads}  "
        f"num_layers={args.num_layers}  dropout={args.dropout}  "
        f"ranking_pairs_per_type={args.ranking_pairs_per_type}  "
        f"ranking_top_quantile={args.ranking_top_quantile}  "
        f"pointwise_loss_weight={args.pointwise_loss_weight}  "
        f"loss_objective={args.loss_objective}  "
        f"budget_loss_ratios={args.budget_loss_ratios}  "
        f"budget_loss_temperature={args.budget_loss_temperature}  "
        f"temporal_distribution_loss_weight={args.temporal_distribution_loss_weight}  "
        f"gradient_clip_norm={args.gradient_clip_norm}  "
        f"train_batch_size={args.train_batch_size}  "
        f"inference_batch_size={args.inference_batch_size}  "
        f"query_chunk_size={args.query_chunk_size}  "
        f"range_diagnostics_mode={args.range_diagnostics_mode}  "
        f"validation_split_mode={args.validation_split_mode}  "
        f"final_metrics_mode={args.final_metrics_mode}  "
        f"synthetic_route_families={args.synthetic_route_families}  "
        f"diagnostic_every={args.diagnostic_every}  "
        f"checkpoint_selection_metric={args.checkpoint_selection_metric}  "
        f"validation_score_every={args.validation_score_every}  "
        f"uniform_gap_weight={args.checkpoint_uniform_gap_weight}  "
        f"type_penalty_weight={args.checkpoint_type_penalty_weight}  "
        f"smoothing_window={args.checkpoint_smoothing_window}  "
        f"full_score_every={args.checkpoint_full_score_every}  "
        f"candidate_pool={args.checkpoint_candidate_pool_size}  "
        f"score_variant={args.checkpoint_score_variant}  "
        f"validation_global_sanity_penalty={args.validation_global_sanity_penalty}  "
        f"range_spatial_fraction={args.range_spatial_fraction}  range_time_fraction={args.range_time_fraction}  "
        f"range_spatial_km={args.range_spatial_km}  range_time_hours={args.range_time_hours}  "
        f"range_footprint_jitter={args.range_footprint_jitter}  "
        f"range_time_domain_mode={args.range_time_domain_mode}  "
        f"range_anchor_mode={args.range_anchor_mode}  "
        f"range_train_anchor_modes={args.range_train_anchor_modes}  "
        f"range_train_footprints={args.range_train_footprints}  "
        f"range_min_point_hits={args.range_min_point_hits}  "
        f"range_max_point_hit_fraction={args.range_max_point_hit_fraction}  "
        f"range_min_trajectory_hits={args.range_min_trajectory_hits}  "
        f"range_max_trajectory_hit_fraction={args.range_max_trajectory_hit_fraction}  "
        f"range_max_box_volume_fraction={args.range_max_box_volume_fraction}  "
        f"range_duplicate_iou_threshold={args.range_duplicate_iou_threshold}  "
        f"range_acceptance_max_attempts={args.range_acceptance_max_attempts}  "
        f"range_max_coverage_overshoot={args.range_max_coverage_overshoot}  "
        f"range_train_workload_replicates={args.range_train_workload_replicates}  "
        f"workload_profile_id={args.workload_profile_id}  "
        f"coverage_calibration_mode={args.coverage_calibration_mode}  "
        f"workload_stability_gate_mode={args.workload_stability_gate_mode}  "
        f"historical_prior_k={args.historical_prior_k}  "
        f"historical_prior_clock_weight={args.historical_prior_clock_weight}  "
        f"historical_prior_mmsi_weight={args.historical_prior_mmsi_weight}  "
        f"historical_prior_density_weight={args.historical_prior_density_weight}  "
        f"historical_prior_min_target={args.historical_prior_min_target}  "
        f"historical_prior_support_ratio={args.historical_prior_support_ratio}  "
        f"historical_prior_source_aggregation={args.historical_prior_source_aggregation}  "
        f"mlqds_temporal_fraction={args.mlqds_temporal_fraction}  "
        f"mlqds_hybrid_mode={args.mlqds_hybrid_mode}  "
        f"selector_type={args.selector_type}  "
        f"learned_segment_geometry_gain_weight={args.learned_segment_geometry_gain_weight}  "
        f"learned_segment_score_blend_weight={args.learned_segment_score_blend_weight}  "
        f"learned_segment_fairness_preallocation={args.learned_segment_fairness_preallocation}  "
        f"mlqds_stratified_center_weight={args.mlqds_stratified_center_weight}  "
        f"mlqds_min_learned_swaps={args.mlqds_min_learned_swaps}  "
        f"mlqds_score_mode={args.mlqds_score_mode}  "
        f"mlqds_score_temperature={args.mlqds_score_temperature}  "
        f"mlqds_rank_confidence_weight={args.mlqds_rank_confidence_weight}  "
        f"mlqds_range_geometry_blend={args.mlqds_range_geometry_blend}  "
        f"temporal_residual_label_mode={args.temporal_residual_label_mode}  "
        f"range_label_mode={args.range_label_mode}  "
        f"range_training_target_mode={args.range_training_target_mode}  "
        f"range_target_balance_mode={args.range_target_balance_mode}  "
        f"range_replicate_target_aggregation={args.range_replicate_target_aggregation}  "
        f"range_component_target_blend={args.range_component_target_blend}  "
        f"range_temporal_target_blend={args.range_temporal_target_blend}  "
        f"range_structural_target_blend={args.range_structural_target_blend}  "
        f"range_structural_target_source_mode={args.range_structural_target_source_mode}  "
        f"range_target_budget_weight_power={args.range_target_budget_weight_power}  "
        f"range_marginal_target_radius_scale={args.range_marginal_target_radius_scale}  "
        f"range_query_spine_fraction={args.range_query_spine_fraction}  "
        f"range_query_spine_mass_mode={args.range_query_spine_mass_mode}  "
        f"range_query_residual_multiplier={args.range_query_residual_multiplier}  "
        f"range_query_residual_mass_mode={args.range_query_residual_mass_mode}  "
        f"range_set_utility_multiplier={args.range_set_utility_multiplier}  "
        f"range_set_utility_candidate_limit={args.range_set_utility_candidate_limit}  "
        f"range_set_utility_mass_mode={args.range_set_utility_mass_mode}  "
        f"range_boundary_prior_weight={args.range_boundary_prior_weight}  "
        f"range_teacher_distillation_mode={args.range_teacher_distillation_mode}  "
        f"range_teacher_epochs={args.range_teacher_epochs}  "
        f"range_audit_compression_ratios={args.range_audit_compression_ratios}  "
        f"min_points_per_segment={args.min_points_per_segment}  "
        f"max_points_per_segment={args.max_points_per_segment}  "
        f"max_time_gap_seconds={normalized_gap_arg(args.max_time_gap_seconds)}  "
        f"max_segments={args.max_segments}  "
        f"train_max_segments={args.train_max_segments}  "
        f"validation_max_segments={args.validation_max_segments}  "
        f"eval_max_segments={args.eval_max_segments}  "
        f"cache_dir={args.cache_dir}  "
        f"refresh_cache={args.refresh_cache}  "
        f"float32_matmul_precision={runtime_settings['float32_matmul_precision']}  "
        f"allow_tf32={runtime_settings['tf32_matmul_allowed']}  "
        f"amp_mode={config.model.amp_mode}",
        flush=True,
    )

    t0 = time.perf_counter()
    mmsis: list[int] | None = None
    validation_trajectories = None
    eval_trajectories = None
    eval_mmsis: list[int] | None = None
    train_source_ids: list[int] | None = None
    data_audit = None
    load_kwargs = {
        "min_points_per_segment": args.min_points_per_segment,
        "max_points_per_segment": args.max_points_per_segment,
        "max_time_gap_seconds": normalized_gap_arg(args.max_time_gap_seconds),
        "max_segments": args.max_segments,
    }
    if args.train_csv_path or args.eval_csv_path or args.validation_csv_path:
        if not args.train_csv_path or not args.eval_csv_path:
            parser.error(
                "--train_csv_path/--train_csv and --eval_csv_path/--eval_csv must be supplied together; "
                "--validation_csv_path is optional but also requires both."
            )
        try:
            _assert_distinct_csv_sources(
                train_csv_path=args.train_csv_path,
                validation_csv_path=args.validation_csv_path,
                eval_csv_path=args.eval_csv_path,
            )
        except ValueError as exc:
            parser.error(str(exc))
        train_paths = split_csv_path_list(args.train_csv_path)
        train_trajectories_parts = []
        train_mmsis_parts: list[int] = []
        train_source_ids_parts: list[int] = []
        train_has_mmsis = True
        train_audit_payloads = []
        for source_index, train_path in enumerate(train_paths):
            train_label = "train" if len(train_paths) == 1 else f"train[{source_index}]"
            print(f"[load-data] reading {train_label} CSV: {train_path}", flush=True)
            train_part, train_part_mmsis, _train_audit, train_audit_payload = _load_csv_trajectories(
                train_label,
                train_path,
                args,
                {**load_kwargs, "max_segments": _split_max_segments(args, "train")},
            )
            train_trajectories_parts.extend(train_part)
            train_source_ids_parts.extend([source_index] * len(train_part))
            if train_part_mmsis is None:
                train_has_mmsis = False
            elif train_has_mmsis:
                train_mmsis_parts.extend(train_part_mmsis)
            train_audit_payload["source_index"] = source_index
            train_audit_payloads.append(train_audit_payload)
        trajectories = train_trajectories_parts
        mmsis = train_mmsis_parts if train_has_mmsis else None
        train_audit_payload = _combined_train_audit_payload(train_paths, train_audit_payloads)
        if len(train_paths) > 1:
            print(
                f"[load-data] combined train CSVs: sources={len(train_paths)} "
                f"segments={len(trajectories)}",
                flush=True,
            )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
        train_source_ids = train_source_ids_parts[: len(trajectories)]
        validation_audit_payload = None
        if args.validation_csv_path:
            validation_paths = split_csv_path_list(args.validation_csv_path)
            validation_trajectories_parts = []
            validation_audit_payloads = []
            for source_index, validation_path in enumerate(validation_paths):
                validation_label = "validation" if len(validation_paths) == 1 else f"validation[{source_index}]"
                print(f"[load-data] reading {validation_label} CSV: {validation_path}", flush=True)
                validation_part, _validation_mmsis, _validation_audit, validation_part_payload = _load_csv_trajectories(
                    validation_label,
                    validation_path,
                    args,
                    {**load_kwargs, "max_segments": _split_max_segments(args, "validation")},
                )
                validation_trajectories_parts.extend(validation_part)
                validation_part_payload["source_index"] = source_index
                validation_audit_payloads.append(validation_part_payload)
            validation_trajectories = validation_trajectories_parts
            validation_audit_payload = _combined_train_audit_payload(
                validation_paths,
                validation_audit_payloads,
            )
            if len(validation_paths) > 1:
                print(
                    f"[load-data] combined validation CSVs: sources={len(validation_paths)} "
                    f"segments={len(validation_trajectories)}",
                    flush=True,
                )
            validation_trajectories, _validation_mmsis = _cap_loaded_trajectories(
                validation_trajectories,
                None,
                args.max_trajectories,
            )
        eval_paths = split_csv_path_list(args.eval_csv_path)
        eval_trajectories_parts = []
        eval_mmsis_parts: list[int] = []
        eval_has_mmsis = True
        eval_audit_payloads = []
        for source_index, eval_path in enumerate(eval_paths):
            eval_label = "eval" if len(eval_paths) == 1 else f"eval[{source_index}]"
            print(f"[load-data] reading {eval_label} CSV: {eval_path}", flush=True)
            eval_part, eval_part_mmsis, _eval_audit, eval_part_payload = _load_csv_trajectories(
                eval_label,
                eval_path,
                args,
                {**load_kwargs, "max_segments": _split_max_segments(args, "eval")},
            )
            eval_trajectories_parts.extend(eval_part)
            if eval_part_mmsis is None:
                eval_has_mmsis = False
            elif eval_has_mmsis:
                eval_mmsis_parts.extend(eval_part_mmsis)
            eval_part_payload["source_index"] = source_index
            eval_audit_payloads.append(eval_part_payload)
        eval_trajectories = eval_trajectories_parts
        eval_mmsis = eval_mmsis_parts if eval_has_mmsis else None
        eval_audit_payload = _combined_train_audit_payload(eval_paths, eval_audit_payloads)
        if len(eval_paths) > 1:
            print(
                f"[load-data] combined eval CSVs: sources={len(eval_paths)} "
                f"segments={len(eval_trajectories)}",
                flush=True,
            )
        eval_trajectories, eval_mmsis = _cap_loaded_trajectories(
            eval_trajectories,
            eval_mmsis,
            args.max_trajectories,
        )
        data_audit = {"train": train_audit_payload, "eval": eval_audit_payload}
        if validation_audit_payload is not None:
            data_audit["validation"] = validation_audit_payload
    elif args.csv_path:
        print(f"[load-data] reading CSV: {args.csv_path}", flush=True)
        trajectories, mmsis, _audit, audit_payload = _load_csv_trajectories(
            "csv",
            args.csv_path,
            args,
            load_kwargs,
        )
        trajectories, mmsis = _cap_loaded_trajectories(trajectories, mmsis, args.max_trajectories)
        data_audit = {"csv": audit_payload}
    else:
        if config.data.n_ships is None or config.data.n_points_per_ship is None:
            raise ValueError("Synthetic data generation requires n_ships and n_points_per_ship.")
        print(
            f"[load-data] generating synthetic data "
            f"(n_ships={config.data.n_ships}, n_points={config.data.n_points_per_ship}, "
            f"route_families={config.data.synthetic_route_families})",
            flush=True,
        )
        trajectories = generate_synthetic_ais_data(
            n_ships=config.data.n_ships,
            n_points_per_ship=config.data.n_points_per_ship,
            seed=config.data.seed,
            route_families=config.data.synthetic_route_families,
        )
    if eval_trajectories is None:
        print(f"[load-data] {len(trajectories)} trajectories loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    else:
        validation_part = (
            f" validation={len(validation_trajectories)}"
            if validation_trajectories is not None
            else ""
        )
        print(
            f"[load-data] train={len(trajectories)}{validation_part} eval={len(eval_trajectories)} trajectories "
            f"loaded in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    save_simplified_dir = args.save_simplified_dir
    if args.eval_csv_path and save_simplified_dir is None:
        save_simplified_dir = _default_simplified_dir(args)
        print(f"[config] auto-saving simplified eval CSV under {save_simplified_dir}", flush=True)

    out = run_experiment_pipeline(
        config=config,
        trajectories=trajectories,
        results_dir=args.results_dir,
        save_model=args.save_model,
        save_queries_dir=args.save_queries_dir,
        save_simplified_dir=save_simplified_dir,
        trajectory_mmsis=mmsis,
        validation_trajectories=validation_trajectories,
        eval_trajectories=eval_trajectories,
        eval_trajectory_mmsis=eval_mmsis,
        trajectory_source_ids=train_source_ids,
        data_audit=data_audit,
    )

    print("\nMatched-workload table")
    print(out.matched_table)
    print("\nGeometric-distortion table (lower is better; SED = time-synchronous, PED = perpendicular, in km)")
    print(out.geometric_table)
    print("\nDistribution-shift table")
    print(out.shift_table)


if __name__ == "__main__":
    main()

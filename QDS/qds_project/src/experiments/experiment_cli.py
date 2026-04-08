"""CLI argument parsing and validation for the AIS experiment."""

from __future__ import annotations

import argparse

POSITIVE_INT_FLAGS = (
    "--n_ships",
    "--n_points",
    "--n_queries",
    "--epochs",
    "--min_points_per_trajectory",
    "--point_batch_size",
    "--importance_chunk_size",
    "--dp_max_points",
    "--max_visualization_points",
    "--max_visualization_ships",
    "--max_points_per_ship_plot",
)

OPTIONAL_POSITIVE_INT_FLAGS = (
    "--max_train_points",
    "--model_max_points",
)

RANGE_VALIDATIONS = (
    ("--compression_ratio", 0.0, 1.0, True, True, False),
    ("--target_ratio", 0.0, 1.0, False, True, True),
    ("--density_ratio", 0.0, 1.0, True, True, False),
    ("--query_spatial_fraction", 0.0, 1.0, False, True, False),
    ("--query_temporal_fraction", 0.0, 1.0, False, True, False),
    ("--query_spatial_lower_quantile", 0.0, 1.0, True, True, False),
    ("--query_spatial_upper_quantile", 0.0, 1.0, True, True, False),
)

RUN_KWARG_NAMES = (
    "n_ships",
    "n_points",
    "n_queries",
    "epochs",
    "threshold",
    "target_ratio",
    "compression_ratio",
    "min_points_per_trajectory",
    "max_train_points",
    "model_max_points",
    "point_batch_size",
    "importance_chunk_size",
    "dp_max_points",
    "skip_baselines",
    "skip_visualizations",
    "max_visualization_points",
    "max_visualization_ships",
    "max_points_per_ship_plot",
    "csv_path",
    "save_csv",
    "workload",
    "density_ratio",
    "query_spatial_fraction",
    "query_temporal_fraction",
    "query_spatial_lower_quantile",
    "query_spatial_upper_quantile",
    "model_type",
    "turn_bias_weight",
    "turn_score_method",
)

ArgumentSpec = tuple[tuple[str, ...], dict[str, object]]

DATA_ARGUMENT_SPECS: tuple[ArgumentSpec, ...] = (
    (("--n_ships",), {"type": int, "default": 10, "help": "Number of ships"}),
    (("--n_points",), {"type": int, "default": 100, "help": "Points per ship"}),
    (("--csv_path",), {"type": str, "default": None, "help": "Path to AIS CSV file"}),
    (
        ("--save_csv",),
        {
            "action": "store_true",
            "help": "Save cleaned retained points CSV when loading data from --csv_path",
        },
    ),
)

QUERY_ARGUMENT_SPECS: tuple[ArgumentSpec, ...] = (
    (("--n_queries",), {"type": int, "default": 100, "help": "Number of queries"}),
    (
        ("--workload",),
        {
            "type": str,
            "default": "density",
            "choices": ["uniform", "density", "mixed", "all"],
            "help": (
                "Query workload type: 'uniform' (bounding-box sampling), "
                "'density' (AIS-point-anchored sampling), "
                "'mixed' (blend of both), "
                "or 'all' (run all three and print a comparison table)."
            ),
        },
    ),
    (
        ("--density_ratio",),
        {
            "type": float,
            "default": 0.7,
            "help": (
                "Fraction of density-biased queries in a 'mixed' workload "
                "(ignored for other workload types). Must be in [0, 1]."
            ),
        },
    ),
    (
        ("--query_spatial_fraction",),
        {
            "type": float,
            "default": 0.03,
            "help": (
                "Maximum spatial query width as a fraction of effective lat/lon "
                "range. Lower values produce tighter query boxes."
            ),
        },
    ),
    (
        ("--query_temporal_fraction",),
        {
            "type": float,
            "default": 0.10,
            "help": (
                "Maximum temporal query width as a fraction of time range. "
                "Lower values produce shorter query windows."
            ),
        },
    ),
    (
        ("--query_spatial_lower_quantile",),
        {
            "type": float,
            "default": 0.01,
            "help": (
                "Lower quantile for robust spatial bounds used by uniform query "
                "placement (and uniform part of mixed workload)."
            ),
        },
    ),
    (
        ("--query_spatial_upper_quantile",),
        {
            "type": float,
            "default": 0.99,
            "help": (
                "Upper quantile for robust spatial bounds used by uniform query "
                "placement (and uniform part of mixed workload)."
            ),
        },
    ),
)

MODEL_ARGUMENT_SPECS: tuple[ArgumentSpec, ...] = (
    (("--epochs",), {"type": int, "default": 50, "help": "Training epochs"}),
    (
        ("--threshold",),
        {"type": float, "default": 0.5, "help": "Compression threshold (global mode only)"},
    ),
    (
        ("--target_ratio",),
        {
            "type": float,
            "default": None,
            "help": "Target retained ratio in (0, 1]; overrides --threshold (global mode only)",
        },
    ),
    (
        ("--compression_ratio",),
        {
            "type": float,
            "default": 0.2,
            "help": (
                "Per-trajectory compression fraction in (0, 1] (default: 0.2). "
                "Each trajectory keeps max(min_points_per_trajectory, "
                "int(compression_ratio * traj_len)) points. "
                "Pass 0 to disable per-trajectory mode and use global --threshold instead."
            ),
        },
    ),
    (
        ("--min_points_per_trajectory",),
        {
            "type": int,
            "default": 5,
            "help": "Minimum number of points to retain per trajectory (default: 5).",
        },
    ),
    (
        ("--max_train_points",),
        {
            "type": int,
            "default": None,
            "help": "Optional cap on number of points used for model training",
        },
    ),
    (
        ("--model_max_points",),
        {
            "type": int,
            "default": 300000,
            "help": "Optional cap for full-set model inference during simplification",
        },
    ),
    (
        ("--point_batch_size",),
        {
            "type": int,
            "default": 50000,
            "help": "Mini-batch size over points during training",
        },
    ),
    (
        ("--importance_chunk_size",),
        {
            "type": int,
            "default": 200000,
            "help": "Chunk size for large-scale importance computation",
        },
    ),
    (
        ("--model_type",),
        {
            "type": str,
            "default": "baseline",
            "choices": ["baseline", "turn_aware", "all"],
            "help": (
                "Model variant to use: 'baseline' (TrajectoryQDSModel, 7 features), "
                "'turn_aware' (TurnAwareQDSModel, 8 features with turn bias), "
                "or 'all' (train and compare both models). Default: 'baseline'."
            ),
        },
    ),
    (
        ("--turn_bias_weight",),
        {
            "type": float,
            "default": 0.1,
            "help": (
                "Additive weight for turn-score bias applied during simplification "
                "when model_type is 'turn_aware'. Default: 0.1."
            ),
        },
    ),
    (
        ("--turn_score_method",),
        {
            "type": str,
            "default": "heading",
            "choices": ["heading", "geometry"],
            "help": (
                "Method used to compute turn_score: 'heading' (default, wrapped "
                "COG/heading deltas) or 'geometry' (turn angle from lat/lon vectors)."
            ),
        },
    ),
)

RUNTIME_ARGUMENT_SPECS: tuple[ArgumentSpec, ...] = (
    (
        ("--dp_max_points",),
        {"type": int, "default": 200000, "help": "Maximum points for Douglas-Peucker baseline"},
    ),
    (
        ("--skip_baselines",),
        {"action": "store_true", "help": "Skip baseline generation/evaluation"},
    ),
    (
        ("--skip_visualizations",),
        {"action": "store_true", "help": "Skip all visualization generation"},
    ),
    (
        ("--max_visualization_points",),
        {
            "type": int,
            "default": 200000,
            "help": "Maximum points used in visualization scatter plots",
        },
    ),
    (
        ("--max_visualization_ships",),
        {
            "type": int,
            "default": 200,
            "help": "Maximum trajectories used in visualization line plots",
        },
    ),
    (
        ("--max_points_per_ship_plot",),
        {
            "type": int,
            "default": 2000,
            "help": "Maximum points per trajectory line in visualization",
        },
    ),
)


def _validate_range(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    value: int | float | None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    allow_none: bool = False,
) -> None:
    """Validate numeric CLI arguments with unified error messaging."""
    if value is None:
        if allow_none:
            return
        parser.error(f"{name} is required.")

    assert value is not None
    if minimum is not None:
        if min_inclusive and value < minimum:
            parser.error(f"{name} must be >= {minimum} (got {value}).")
        if not min_inclusive and value <= minimum:
            parser.error(f"{name} must be > {minimum} (got {value}).")

    if maximum is not None:
        if max_inclusive and value > maximum:
            parser.error(f"{name} must be <= {maximum} (got {value}).")
        if not max_inclusive and value >= maximum:
            parser.error(f"{name} must be < {maximum} (got {value}).")


def _validate_positive_int_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    allow_none: bool,
    flags: tuple[str, ...],
) -> None:
    """Validate positive integer flags by name."""
    for flag_name in flags:
        attr_name = flag_name[2:]
        _validate_range(
            parser,
            name=flag_name,
            value=getattr(args, attr_name),
            minimum=1,
            allow_none=allow_none,
        )


def _normalize_compression_ratio(compression_ratio: float) -> float | None:
    """Map CLI compression mode sentinel to internal representation."""
    return compression_ratio if compression_ratio > 0.0 else None


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments and cross-argument constraints."""
    _validate_positive_int_flags(
        parser,
        args,
        allow_none=False,
        flags=POSITIVE_INT_FLAGS,
    )
    _validate_positive_int_flags(
        parser,
        args,
        allow_none=True,
        flags=OPTIONAL_POSITIVE_INT_FLAGS,
    )

    for flag_name, minimum, maximum, min_inclusive, max_inclusive, allow_none in RANGE_VALIDATIONS:
        _validate_range(
            parser,
            name=flag_name,
            value=getattr(args, flag_name[2:]),
            minimum=minimum,
            maximum=maximum,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
            allow_none=allow_none,
        )

    if args.query_spatial_lower_quantile >= args.query_spatial_upper_quantile:
        parser.error(
            "--query_spatial_lower_quantile must be < "
            "--query_spatial_upper_quantile."
        )

    if args.save_csv and not args.csv_path:
        parser.error("--save_csv requires --csv_path.")

    if args.target_ratio is not None and args.compression_ratio > 0.0:
        parser.error(
            "--target_ratio is only used in global-threshold mode. "
            "Set --compression_ratio 0 to enable it."
        )


def _build_run_kwargs(
    args: argparse.Namespace,
    parsed_compression_ratio: float | None,
) -> dict[str, object]:
    """Build keyword arguments for run_ais_experiment from parsed CLI args."""
    run_kwargs = {name: getattr(args, name) for name in RUN_KWARG_NAMES}
    run_kwargs["compression_ratio"] = parsed_compression_ratio
    return run_kwargs


def _add_argument_specs(parser, specs: tuple[ArgumentSpec, ...]) -> None:
    """Register a sequence of argument definitions on a parser/group."""
    for flags, kwargs in specs:
        parser.add_argument(*flags, **kwargs)


def _add_data_arguments(parser) -> None:
    """Register data-loading and data-generation CLI arguments."""
    _add_argument_specs(parser, DATA_ARGUMENT_SPECS)


def _add_query_arguments(parser) -> None:
    """Register query-workload CLI arguments."""
    _add_argument_specs(parser, QUERY_ARGUMENT_SPECS)


def _add_model_arguments(parser) -> None:
    """Register model-training and simplification CLI arguments."""
    _add_argument_specs(parser, MODEL_ARGUMENT_SPECS)


def _add_runtime_arguments(parser) -> None:
    """Register evaluation and visualization CLI arguments."""
    _add_argument_specs(parser, RUNTIME_ARGUMENT_SPECS)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the experiment entry point."""
    parser = argparse.ArgumentParser(
        description="Run the AIS Query-Driven Simplification experiment"
    )
    _add_data_arguments(parser.add_argument_group("Data"))
    _add_query_arguments(parser.add_argument_group("Query Workload"))
    _add_model_arguments(parser.add_argument_group("Model and Simplification"))
    _add_runtime_arguments(parser.add_argument_group("Evaluation and Visualization"))
    return parser


def parse_and_validate_experiment_args(
    argv: list[str] | None = None,
) -> dict[str, object]:
    """Parse CLI arguments, validate them, and return run kwargs."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    parsed_compression_ratio = _normalize_compression_ratio(args.compression_ratio)
    return _build_run_kwargs(args, parsed_compression_ratio)

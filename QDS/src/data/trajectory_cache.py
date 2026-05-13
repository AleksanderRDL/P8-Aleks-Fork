"""Parquet-backed cache for segmented AIS trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.ais_loader import AISLoadAudit, AISLoadConfig, load_ais_csv

CACHE_SCHEMA_VERSION = 1
FEATURE_COLUMNS = [
    "time",
    "latitude",
    "longitude",
    "speed",
    "heading",
    "is_start",
    "is_end",
    "turn_score",
]
CACHE_COLUMNS = ["segment_id", "point_index", "mmsi", *FEATURE_COLUMNS]


@dataclass
class AISCacheResult:
    """Loaded trajectories plus metadata about cache usage."""

    trajectories: list[torch.Tensor]
    mmsis: list[int]
    audit: AISLoadAudit
    cache_hit: bool
    cache_dir: str
    manifest_path: str
    parquet_path: str

    def cache_metadata(self) -> dict[str, Any]:
        """Return JSON-safe cache metadata for experiment artifacts."""
        return {
            "cache_hit": self.cache_hit,
            "cache_dir": self.cache_dir,
            "manifest_path": self.manifest_path,
            "parquet_path": self.parquet_path,
        }


def _source_info(csv_path: str) -> dict[str, Any]:
    """Return source-file identity used to validate cached trajectory data."""
    path = Path(csv_path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_key(source: dict[str, Any], config: AISLoadConfig) -> str:
    """Build a stable cache key from source identity and loader config."""
    cache_key_json = json.dumps(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source": source,
            "config": config.to_dict(),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(cache_key_json.encode("utf-8")).hexdigest()[:16]
    stem = Path(str(source["path"])).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def _manifest_matches(manifest: dict[str, Any], source: dict[str, Any], config: AISLoadConfig) -> bool:
    """Validate manifest identity before reading cached rows."""
    return (
        int(manifest.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
        and manifest.get("source") == source
        and manifest.get("config") == config.to_dict()
    )


def _audit_from_dict(audit_payload: dict[str, Any]) -> AISLoadAudit:
    """Rehydrate cached audit metadata."""
    return AISLoadAudit(**audit_payload)


def _trajectories_to_frame(trajectories: list[torch.Tensor], mmsis: list[int]) -> pd.DataFrame:
    """Flatten trajectory tensors into a Parquet-friendly point table."""
    columns: dict[str, list[Any]] = {name: [] for name in CACHE_COLUMNS}
    for segment_id, trajectory in enumerate(trajectories):
        n_points = int(trajectory.shape[0])
        mmsi = int(mmsis[segment_id]) if segment_id < len(mmsis) else 0
        columns["segment_id"].extend([segment_id] * n_points)
        columns["point_index"].extend(range(n_points))
        columns["mmsi"].extend([mmsi] * n_points)
        for feature_idx, column_name in enumerate(FEATURE_COLUMNS):
            columns[column_name].extend(trajectory[:, feature_idx].detach().cpu().tolist())

    frame = pd.DataFrame(columns)
    int_columns = ["segment_id", "point_index", "mmsi"]
    for col in int_columns:
        frame[col] = frame[col].astype("int64")
    for col in FEATURE_COLUMNS:
        frame[col] = frame[col].astype("float32")
    return frame


def _frame_to_trajectories(frame: pd.DataFrame) -> tuple[list[torch.Tensor], list[int]]:
    """Convert cached point rows back into trajectory tensors and MMSI IDs."""
    if set(CACHE_COLUMNS) - set(frame.columns):
        missing = sorted(set(CACHE_COLUMNS) - set(frame.columns))
        raise ValueError(f"Cached AIS parquet is missing columns: {missing}")

    frame = frame.sort_values(["segment_id", "point_index"]).reset_index(drop=True)
    trajectories: list[torch.Tensor] = []
    mmsis: list[int] = []
    for _, group in frame.groupby("segment_id", sort=False):
        values = group[FEATURE_COLUMNS].to_numpy(dtype="float32", copy=True)
        trajectories.append(torch.tensor(values, dtype=torch.float32))
        mmsis.append(int(group["mmsi"].iloc[0]))
    return trajectories, mmsis


def _write_cache(
    cache_root: Path,
    source: dict[str, Any],
    config: AISLoadConfig,
    trajectories: list[torch.Tensor],
    mmsis: list[int],
    audit: AISLoadAudit,
) -> tuple[Path, Path]:
    """Write point table and manifest to a source/config-specific cache directory."""
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = cache_root / _cache_key(source, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = run_dir / "points.parquet"
    manifest_path = run_dir / "manifest.json"

    frame = _trajectories_to_frame(trajectories, mmsis)
    frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": source,
        "config": config.to_dict(),
        "parquet_file": parquet_path.name,
        "audit": audit.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, parquet_path


def load_or_build_ais_cache(
    csv_path: str,
    cache_dir: str,
    refresh_cache: bool = False,
    min_points_per_segment: int = 4,
    max_points_per_segment: int | None = None,
    max_time_gap_seconds: float | None = 3600.0,
    max_segments: int | None = None,
) -> AISCacheResult:
    """Load segmented AIS trajectories from cache, or build the cache from CSV."""
    source = _source_info(csv_path)
    config = AISLoadConfig(
        min_points_per_segment=int(min_points_per_segment),
        max_points_per_segment=max_points_per_segment,
        max_time_gap_seconds=max_time_gap_seconds,
        max_segments=max_segments,
    )
    config.validate()
    cache_root = Path(cache_dir)
    run_dir = cache_root / _cache_key(source, config)
    manifest_path = run_dir / "manifest.json"
    parquet_path = run_dir / "points.parquet"

    if not refresh_cache and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate_parquet = run_dir / str(manifest.get("parquet_file", "points.parquet"))
        if _manifest_matches(manifest, source, config) and candidate_parquet.exists():
            frame = pd.read_parquet(candidate_parquet, engine="pyarrow")
            trajectories, mmsis = _frame_to_trajectories(frame)
            return AISCacheResult(
                trajectories=trajectories,
                mmsis=mmsis,
                audit=_audit_from_dict(manifest["audit"]),
                cache_hit=True,
                cache_dir=str(run_dir),
                manifest_path=str(manifest_path),
                parquet_path=str(candidate_parquet),
            )

    trajectories, mmsis, audit = load_ais_csv(
        csv_path,
        min_points_per_segment=min_points_per_segment,
        max_points_per_segment=max_points_per_segment,
        max_time_gap_seconds=max_time_gap_seconds,
        max_segments=max_segments,
        return_mmsis=True,
        return_audit=True,
    )
    manifest_path, parquet_path = _write_cache(cache_root, source, config, trajectories, mmsis, audit)
    return AISCacheResult(
        trajectories=trajectories,
        mmsis=mmsis,
        audit=audit,
        cache_hit=False,
        cache_dir=str(manifest_path.parent),
        manifest_path=str(manifest_path),
        parquet_path=str(parquet_path),
    )

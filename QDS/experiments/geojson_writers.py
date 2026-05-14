"""GeoJSON writers for query workloads and simplified trajectories.

These outputs are designed for inspection in QGIS:
- Queries: one FeatureCollection for range query boxes.
- Simplified trajectories: one FeatureCollection of LineStrings with a Points
  layer for the retained samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _bbox_polygon(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[list[list[float]]]:
    """Build a closed-ring rectangle polygon in GeoJSON [lon, lat] order."""
    return [[
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]]


def _seconds_to_hhmm(seconds: float) -> str:
    """Convert seconds-since-midnight to 'HH:MM' string."""
    total = int(round(seconds)) % 86400
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _query_to_feature(query: dict[str, Any]) -> dict[str, Any]:
    """Convert a typed query dict to a GeoJSON Feature.

    Range queries are rendered as their native axis-aligned lat/lon boxes.
    """
    query_type = str(query["type"]).lower()
    params = query["params"]
    if query_type == "range":
        coords = _bbox_polygon(params["lon_min"], params["lat_min"], params["lon_max"], params["lat_max"])
    else:
        raise ValueError(f"Only range queries are supported for GeoJSON export; got query type: {query_type}")
    geometry = {"type": "Polygon", "coordinates": coords}
    properties: dict[str, Any] = {
        "query_type": query_type,
        **{key: value for key, value in params.items() if isinstance(value, (int, float, str))},
    }
    # Add human-readable time fields alongside the raw seconds values.
    if "t_start" in properties:
        properties["t_start_hm"] = _seconds_to_hhmm(float(properties["t_start"]))
    if "t_end" in properties:
        properties["t_end_hm"] = _seconds_to_hhmm(float(properties["t_end"]))
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }


def write_queries_geojson(out_dir: str, typed_queries: list[dict[str, Any]]) -> None:
    """Write one GeoJSON file per query type into out_dir."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[dict[str, Any]]] = {"range": []}
    for query in typed_queries:
        query_type = str(query["type"]).lower()
        if query_type in by_type:
            by_type[query_type].append(_query_to_feature(query))
    for query_type, features in by_type.items():
        output_path = output_dir / f"queries_{query_type}.geojson"
        payload = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        print(f"  wrote {len(features):>4d} {query_type} queries to {output_path}", flush=True)


def write_simplified_csv(
    out_path: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    trajectory_mmsis: list[int] | None = None,
) -> None:
    """Write retained simplified trajectories as CSV in the AIS preprocessed schema.

    Columns: MMSI, # Timestamp, Latitude, Longitude, SOG, COG.
    Uses the same columns as cleaned AIS files, while callers choose the
    output location for ML-produced data.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points_np = points.detach().cpu().numpy()
    mask_np = retained_mask.detach().cpu().bool().numpy()

    retained_row_count = 0
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MMSI,# Timestamp,Latitude,Longitude,SOG,COG\n")
        for trajectory_idx, (start, end) in enumerate(boundaries):
            trajectory_mask = mask_np[start:end]
            if not trajectory_mask.any():
                continue
            retained_points = points_np[start:end][trajectory_mask]
            mmsi = (
                trajectory_mmsis[trajectory_idx]
                if trajectory_mmsis is not None and trajectory_idx < len(trajectory_mmsis)
                else 100000000 + trajectory_idx
            )
            for retained_point in retained_points:
                # retained_point = [time, lat, lon, speed, heading, ...]
                file.write(
                    f"{mmsi},{float(retained_point[0]):.3f},{float(retained_point[1]):.6f},"
                    f"{float(retained_point[2]):.6f},{float(retained_point[3]):.2f},"
                    f"{float(retained_point[4]):.2f}\n"
                )
                retained_row_count += 1
    print(f"  wrote {retained_row_count} retained points across "
          f"{int(mask_np.reshape(-1).sum())} samples to {output_path}", flush=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon pairs."""
    import math
    earth_radius_km = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat_rad = math.radians(lat2 - lat1)
    delta_lon_rad = math.radians(lon2 - lon1)
    haversine_term = (
        math.sin(delta_lat_rad / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon_rad / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(min(1.0, math.sqrt(haversine_term)))


def _trajectory_length_km(lat_lon: "torch.Tensor") -> float:
    """Sum haversine distances between consecutive (lat, lon) rows."""
    point_count = lat_lon.shape[0]
    if point_count < 2:
        return 0.0
    coordinates = lat_lon.detach().cpu().numpy()
    total = 0.0
    for point_idx in range(1, point_count):
        total += _haversine_km(
            float(coordinates[point_idx - 1, 0]),
            float(coordinates[point_idx - 1, 1]),
            float(coordinates[point_idx, 0]),
            float(coordinates[point_idx, 1]),
        )
    return total


def report_trajectory_length_loss(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    top_k: int = 25,
    min_orig_km: float = 1.0,
    trajectory_mmsis: list[int] | None = None,
) -> None:
    """Print per-trajectory length-loss summary and two top-K rankings.

    For each trajectory:
        orig_len_km   = sum of haversine distances between consecutive original points
        simp_len_km   = same for retained points (preserves order)
        length_loss   = 1 - simp_len_km / orig_len_km      (0 = perfect, 1 = everything collapsed)
        points_kept   = sum(retained_mask[s:e])
        points_removed= (e - s) - points_kept

    Two ranked lists are printed:
        1. Most-distorted  : top-K by highest length_loss (largest shape damage).
        2. Least-distorted : top-K by lowest length_loss (shape best preserved).

    Averages over all non-empty trajectories are also printed.
    """
    mask = retained_mask.detach().cpu().bool()
    length_rows: list[tuple[int, float, float, float, int, int]] = []
    for trajectory_idx, (start, end) in enumerate(boundaries):
        total_points = end - start
        if total_points < 2:
            continue
        trajectory_points = points[start:end]
        original_length_km = _trajectory_length_km(trajectory_points[:, 1:3])
        trajectory_mask = mask[start:end]
        kept = int(trajectory_mask.sum().item())
        removed = total_points - kept
        if kept >= 2:
            simplified_length_km = _trajectory_length_km(trajectory_points[trajectory_mask][:, 1:3])
        else:
            simplified_length_km = 0.0
        length_loss = (
            0.0
            if original_length_km <= 1e-9
            else max(0.0, 1.0 - simplified_length_km / original_length_km)
        )
        display_id = (
            trajectory_mmsis[trajectory_idx]
            if trajectory_mmsis is not None and trajectory_idx < len(trajectory_mmsis)
            else trajectory_idx
        )
        length_rows.append((int(display_id), original_length_km, simplified_length_km, length_loss, kept, removed))

    if not length_rows:
        print("  [length-loss] no trajectories with >=2 points, skipping.", flush=True)
        return

    avg_orig = sum(row[1] for row in length_rows) / len(length_rows)
    avg_simp = sum(row[2] for row in length_rows) / len(length_rows)
    total_orig = sum(row[1] for row in length_rows)
    total_simp = sum(row[2] for row in length_rows)
    length_preserved = total_simp / total_orig if total_orig > 1e-9 else 1.0
    length_preserved = max(0.0, min(1.0, length_preserved))
    avg_removed = sum(row[5] for row in length_rows) / len(length_rows)
    print(
        f"  [length] {len(length_rows)} trajectories  "
        f"avg_orig_km={avg_orig:.2f}  avg_simp_km={avg_simp:.2f}  "
        f"length_preserved={length_preserved:.3f}  avg_points_removed={avg_removed:.1f}",
        flush=True,
    )

    # Filter out near-stationary trajectories so the top-K is meaningful.
    ranked = [row for row in length_rows if row[1] >= min_orig_km]
    dropped = len(length_rows) - len(ranked)
    if dropped:
        print(
            f"  [length-loss] filtered out {dropped} trajectories with orig_km < {min_orig_km:.2f} "
            f"(likely docked/stationary) from top-{top_k} ranking",
            flush=True,
        )
    if not ranked:
        return

    most_distorted = sorted(ranked, key=lambda row: row[3], reverse=True)[:top_k]
    least_distorted = sorted(ranked, key=lambda row: row[3])[:top_k]

    id_label = "mmsi" if trajectory_mmsis is not None else "traj_id"
    header = (
        f"  {'rank':>4}  {id_label:>10}  {'orig_km':>10}  {'simp_km':>10}  "
        f"{'length_loss':>11}  {'kept':>6}  {'removed':>8}"
    )
    print(f"\n  [length-loss] Top {top_k} MOST distorted (highest length_loss):", flush=True)
    print(header, flush=True)
    for rank, row in enumerate(most_distorted, start=1):
        print(
            f"  {rank:>4}  {row[0]:>10d}  {row[1]:>10.2f}  {row[2]:>10.2f}  "
            f"{row[3]:>11.3f}  {row[4]:>6d}  {row[5]:>8d}",
            flush=True,
        )

    print(f"\n  [length-loss] Top {top_k} LEAST distorted (lowest length_loss):", flush=True)
    print(header, flush=True)
    for rank, row in enumerate(least_distorted, start=1):
        print(
            f"  {rank:>4}  {row[0]:>10d}  {row[1]:>10.2f}  {row[2]:>10.2f}  "
            f"{row[3]:>11.3f}  {row[4]:>6d}  {row[5]:>8d}",
            flush=True,
        )

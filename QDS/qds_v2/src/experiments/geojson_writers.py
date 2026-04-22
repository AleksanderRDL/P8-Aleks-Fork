"""GeoJSON writers for query workloads and simplified trajectories.

These outputs are designed for inspection in QGIS:
- Queries: one FeatureCollection per query type (range/knn/similarity/clustering).
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


def _query_to_feature(q: dict[str, Any]) -> dict[str, Any]:
    """Convert a typed query dict to a GeoJSON Feature.

    All query types are rendered as axis-aligned rectangles (Polygons) so
    they're consistently plottable in QGIS next to range/clustering boxes.
    - range / clustering: native lat/lon bbox.
    - knn: square around the anchor sized from `t_half_window`.
    - similarity: square around the centroid sized from `radius`.
    """
    qtype = str(q["type"]).lower()
    p = q["params"]
    if qtype in ("range", "clustering"):
        coords = _bbox_polygon(p["lon_min"], p["lat_min"], p["lon_max"], p["lat_max"])
    elif qtype == "knn":
        half = 0.05  # default half-width in degrees if not derivable
        coords = _bbox_polygon(
            p["lon"] - half, p["lat"] - half, p["lon"] + half, p["lat"] + half
        )
    elif qtype == "similarity":
        r = float(p.get("radius", 0.05))
        coords = _bbox_polygon(
            p["lon_query_centroid"] - r, p["lat_query_centroid"] - r,
            p["lon_query_centroid"] + r, p["lat_query_centroid"] + r,
        )
    else:
        raise ValueError(f"Unsupported query type for GeoJSON export: {qtype}")
    geom = {"type": "Polygon", "coordinates": coords}
    props: dict[str, Any] = {"query_type": qtype, **{k: v for k, v in p.items() if isinstance(v, (int, float, str))}}
    # Add human-readable time fields alongside the raw seconds values.
    if "t_start" in props:
        props["t_start_hm"] = _seconds_to_hhmm(float(props["t_start"]))
    if "t_end" in props:
        props["t_end_hm"] = _seconds_to_hhmm(float(props["t_end"]))
    return {
        "type": "Feature",
        "geometry": geom,
        "properties": props,
    }


def write_queries_geojson(out_dir: str, typed_queries: list[dict[str, Any]]) -> None:
    """Write one GeoJSON file per query type into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[dict[str, Any]]] = {"range": [], "knn": [], "similarity": [], "clustering": []}
    for q in typed_queries:
        qtype = str(q["type"]).lower()
        if qtype in by_type:
            by_type[qtype].append(_query_to_feature(q))
    for qtype, feats in by_type.items():
        path = out / f"queries_{qtype}.geojson"
        payload = {"type": "FeatureCollection", "features": feats}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"  wrote {len(feats):>4d} {qtype} queries to {path}", flush=True)


def write_simplified_csv(
    out_path: str,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    trajectory_mmsis: list[int] | None = None,
) -> None:
    """Write retained simplified trajectories as CSV in the AIS preprocessed schema.

    Columns: MMSI, # Timestamp, Latitude, Longitude, SOG, COG.
    Matches the layout of files in AISDATA/preprocessed_AIS_files so the
    output drops directly into the same downstream tooling.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    points_np = points.detach().cpu().numpy()
    mask_np = retained_mask.detach().cpu().bool().numpy()

    rows = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("MMSI,# Timestamp,Latitude,Longitude,SOG,COG\n")
        for traj_id, (s, e) in enumerate(boundaries):
            sub_mask = mask_np[s:e]
            if not sub_mask.any():
                continue
            sub = points_np[s:e][sub_mask]
            mmsi = trajectory_mmsis[traj_id] if trajectory_mmsis else 100000000 + traj_id
            for row in sub:
                # row = [time, lat, lon, speed, heading, ...]
                f.write(
                    f"{mmsi},{float(row[0]):.3f},{float(row[1]):.6f},"
                    f"{float(row[2]):.6f},{float(row[3]):.2f},{float(row[4]):.2f}\n"
                )
                rows += 1
    print(f"  wrote {rows} retained points across "
          f"{int(mask_np.reshape(-1).sum())} samples to {out}", flush=True)


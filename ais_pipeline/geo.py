"""Shared geospatial helpers for PySpark AIS processing."""

from __future__ import annotations

from pyspark.sql import functions as F

EARTH_RADIUS_KM = 6371.0
KNOTS_TO_KMH = 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    """Return the haversine distance expression in kilometers."""
    d_lat = F.radians(lat2 - lat1)
    d_lon = F.radians(lon2 - lon1)
    a = (
        F.sin(d_lat / 2) ** 2
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * F.atan2(F.sqrt(a), F.sqrt(F.lit(1.0) - a))

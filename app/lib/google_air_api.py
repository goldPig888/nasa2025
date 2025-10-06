"""Helpers for querying Google's Air Quality API for current conditions."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import requests

AIR_QUALITY_CURRENT_ENDPOINT = (
    "https://airquality.googleapis.com/v1/currentConditions:lookup"
)


def _get_google_air_api_key() -> Optional[str]:
    """Resolve the Google Air Quality API key on demand."""

    key = (
        os.getenv("GOOGLE_AIR_QUALITY_API_KEY")
        or os.getenv("GOOGLE_AIR_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if key:
        key = key.strip()
    return key or None


@dataclass
class AirQualityObservation:
    dominant_pollutant: Optional[str]
    aqi: Optional[int]
    category: Optional[str]
    color: Optional[str]
    health_recommendations: Optional[str]
    pollutants: Dict[str, Dict[str, Optional[float]]]


@dataclass
class HeatmapTileResponse:
    status: int
    zoom: int
    x: int
    y: int
    image_bytes: Optional[bytes]
    content_type: Optional[str]
    error_preview: Optional[str] = None


def _normalize_response(payload: Dict[str, object]) -> AirQualityObservation:
    current = payload.get("currentConditions", {}) or {}
    indexes = current.get("indexes", []) or []
    dominant_index = next((idx for idx in indexes if idx.get("dominantPollutant")), None)

    pollutants_blob = {}
    for item in current.get("pollutants", []) or []:
        code = (item.get("code") or item.get("displayName") or "unknown").lower()
        conc = item.get("concentration", {})
        pollutants_blob[code] = {
            "display_name": item.get("displayName"),
            "full_name": item.get("fullName"),
            "value": conc.get("value"),
            "unit": conc.get("unit"),
            "category": item.get("category"),
        }

    health = (dominant_index or {}).get("healthRecommendations") if dominant_index else None
    if isinstance(health, dict):
        health = " ".join(str(text) for text in health.values() if text)
    elif isinstance(health, list):
        health = " ".join(str(text) for text in health if text)
    elif health is not None:
        health = str(health)

    return AirQualityObservation(
        dominant_pollutant=(dominant_index or {}).get("dominantPollutant"),
        aqi=(dominant_index or {}).get("aqi"),
        category=(dominant_index or {}).get("category"),
        color=(dominant_index or {}).get("color"),
        health_recommendations=health,
        pollutants=pollutants_blob,
    )


def fetch_current_air_quality(
    latitude: float,
    longitude: float,
    *,
    language: str = "en",
) -> Optional[AirQualityObservation]:
    """Return the current air quality near a coordinate via Google Air Quality API."""

    api_key = _get_google_air_api_key()
    if not api_key:
        return None

    payload = {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "extraComputations": [
            "POLLUTANT_CONCENTRATION",
            "HEALTH_RECOMMENDATIONS",
        ],
        "languageCode": language,
    }

    try:
        response = requests.post(
            f"{AIR_QUALITY_CURRENT_ENDPOINT}?key={api_key}",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return _normalize_response(data)
    except requests.RequestException as exc:
        print("[google_air] API error:", exc)
        return None


def google_air_heatmap_tile_url(map_type: str = "AQI") -> Optional[str]:
    """Return the tile endpoint for the requested Google air quality map type."""

    api_key = _get_google_air_api_key()
    if not api_key:
        return None
    sanitized = (map_type or "AQI").upper()
    return (
        f"https://airquality.googleapis.com/v1/mapTypes/{sanitized}/heatmapTiles"
        f"/{{z}}/{{x}}/{{y}}?key={api_key}"
    )


SUPPORTED_HEATMAP_TYPES = {
    "US_AQI": "US AQI (Overall)",
    "EU_AQI": "EU AQI (Overall)",
    "PM25": "PM2.5",
    "PM10": "PM10",
    "O3": "Ozone",
    "NO2": "NO₂",
    "SO2": "SO₂",
    "CO": "CO",
}


def fetch_example_heatmap_tile_response(
    map_type: str = "US_AQI",
    *,
    zoom: int = 2,
    tile_x: int = 0,
    tile_y: int = 1,
) -> Optional[int]:
    """Request the documented sample heatmap tile and print the response details."""

    api_key = _get_google_air_api_key()
    if not api_key:
        print("[google_air] Missing API key; cannot fetch heatmap tile example.")
        return None

    sanitized = (map_type or "US_AQI").upper()
    url = (
        f"https://airquality.googleapis.com/v1/mapTypes/{sanitized}/heatmapTiles"
        f"/{zoom}/{tile_x}/{tile_y}?key={api_key}"
    )

    try:
        response = requests.get(url, timeout=15)
    except requests.RequestException as exc:
        print("[google_air] Heatmap tile example request failed:", exc)
        return None

    print(
        f"[google_air] Heatmap tile example ({sanitized} {zoom}/{tile_x}/{tile_y})"
        f" → {response.status_code}"
    )

    if response.status_code == 200:
        print(f"[google_air] Received {len(response.content)} bytes of tile data.")
    else:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type or "text" in content_type:
            preview = response.text[:200]
        else:
            preview = f"{len(response.content)} bytes returned"
        print(f"[google_air] Response preview: {preview}")

    return response.status_code


def _tile_coords_for(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_rad = math.radians(lat)
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_heatmap_tile_for_location(
    *,
    map_type: str = "US_AQI",
    latitude: float,
    longitude: float,
    zoom: int = 7,
) -> Optional[HeatmapTileResponse]:
    """Fetch the Google heatmap tile covering a coordinate."""

    api_key = _get_google_air_api_key()
    if not api_key:
        print("[google_air] Missing API key; cannot fetch heatmap tile.")
        return None

    if zoom < 0 or zoom > 16:
        raise ValueError("Zoom level must be between 0 and 16 for Google heatmap tiles.")

    sanitized = (map_type or "US_AQI").upper()
    xtile, ytile = _tile_coords_for(latitude, longitude, zoom)
    url = (
        f"https://airquality.googleapis.com/v1/mapTypes/{sanitized}/heatmapTiles"
        f"/{zoom}/{xtile}/{ytile}?key={api_key}"
    )

    try:
        response = requests.get(url, timeout=15)
    except requests.RequestException as exc:
        print("[google_air] Heatmap tile request failed:", exc)
        return None

    status = response.status_code
    content_type = response.headers.get("content-type")
    if status == 200:
        print(
            f"[google_air] Heatmap tile {sanitized} {zoom}/{xtile}/{ytile} fetched:"
            f" {len(response.content)} bytes"
        )
        return HeatmapTileResponse(
            status=status,
            zoom=zoom,
            x=xtile,
            y=ytile,
            image_bytes=response.content,
            content_type=content_type,
        )

    if content_type and ("application/json" in content_type or "text" in content_type):
        preview = response.text[:200]
    else:
        preview = f"{len(response.content)} bytes returned"

    print(
        f"[google_air] Heatmap tile {sanitized} {zoom}/{xtile}/{ytile}"
        f" failed with {status}: {preview}"
    )

    return HeatmapTileResponse(
        status=status,
        zoom=zoom,
        x=xtile,
        y=ytile,
        image_bytes=None,
        content_type=content_type,
        error_preview=preview,
    )

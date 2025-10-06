"""Helpers for fetching pollen forecasts using the Google Air Quality API."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional
import requests


# Prefer explicit pollen key but fall back to GOOGLE_API_KEY for backward compatibility
GOOGLE_POLLEN_API_KEY = os.getenv("GOOGLE_POLLEN_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Official Google Pollen Forecast endpoint (HTTP GET with gRPC transcoding).
POLLEN_ENDPOINT = "https://pollen.googleapis.com/v1/forecast:lookup"


def _mock_forecast(latitude: float, longitude: float) -> Dict[str, object]:
    """Fallback data when the real API cannot be reached."""

    today = datetime.utcnow().date()
    mock_days = []
    allergens = ["tree", "grass", "weed"]
    for idx in range(5):
        day = today.fromordinal(today.toordinal() + idx)
        mock_days.append(
            {
                "date": day.isoformat(),
                "pollenType": allergens[idx % len(allergens)],
                "index": 2 + idx % 3,
                "indexDescription": ["Low", "Moderate", "High", "Very High"][(2 + idx % 3) % 4],
            }
        )

    return {
        "source": "mock",
        "latitude": latitude,
        "longitude": longitude,
        "forecasts": mock_days,
        "message": "Mock pollen data (configure GOOGLE_POLLEN_API_KEY for live data).",
    }


def fetch_pollen_forecast(
    latitude: float,
    longitude: float,
    days: int = 5,
    language_code: str = "en",
) -> Dict[str, object]:
    """Fetch pollen forecast for the provided location.

    Returns a dictionary containing the raw API response or fallback data.
    """

    if not GOOGLE_POLLEN_API_KEY:
        print("[pollen] GOOGLE_API_KEY / GOOGLE_POLLEN_API_KEY missing; returning mock data.")
        return _mock_forecast(latitude, longitude)

    params = {
        "key": GOOGLE_POLLEN_API_KEY,
        "location.latitude": latitude,
        "location.longitude": longitude,
        "days": max(1, min(days, 5)),
        "languageCode": language_code,
        "plantsDescription": "false",
    }

    try:
        response = requests.get(
            POLLEN_ENDPOINT,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        # print("[pollen] Google response:", data)
        data["source"] = "google"
        return data
    except requests.RequestException as exc:
        print("[pollen] Google API error:", exc)
        fallback = _mock_forecast(latitude, longitude)
        fallback["error"] = str(exc)
        return fallback


def flatten_pollen_data(data: Dict[str, object]) -> List[Dict[str, object]]:
    """Turn the API response into a row-wise structure for plotting/analysis."""

    rows: List[Dict[str, object]] = []

    if "forecasts" in data:
        for entry in data.get("forecasts", []):
            date_str = entry.get("date")
            try:
                date_val = datetime.fromisoformat(date_str).date() if date_str else None
            except ValueError:
                date_val = None

            pollen_type = entry.get("pollenType") or "unknown"
            idx = entry.get("index")
            idx_desc = entry.get("indexDescription")

            rows.append(
                {
                    "date": date_val,
                    "pollen_type": pollen_type,
                    "index": idx,
                    "index_description": idx_desc,
                    "concentration": entry.get("concentration"),
                }
            )
        return rows

    for day in data.get("dailyInfo", []):
        date_obj = day.get("date") or {}
        date_val = None
        try:
            year = int(date_obj.get("year")) if date_obj.get("year") else None
            month = int(date_obj.get("month")) if date_obj.get("month") else None
            day_num = int(date_obj.get("day")) if date_obj.get("day") else None
            if year and month and day_num:
                date_val = datetime(year=year, month=month, day=day_num).date()
        except Exception:
            date_val = None

        for pollen in day.get("pollenTypeInfo", []):
            display_name = pollen.get("displayName") or pollen.get("code") or "unknown"
            index_info = pollen.get("indexInfo", {}) or {}
            idx_value = index_info.get("value")
            idx_desc = index_info.get("category") or index_info.get("indexDescription")

            rows.append(
                {
                    "date": date_val,
                    "pollen_type": str(display_name).lower(),
                    "index": idx_value,
                    "index_description": idx_desc,
                    "concentration": None,
                }
            )

    return rows

#!/usr/bin/env python3
"""Small helper to print the next-day forecast for a city.

This uses the project's training + forecasting pipeline (train_city_models_auto)
to load cached model bundles (or retrain if needed) and prints the first
predicted row for PM2.5 (or the first available pollutant).

Usage:
  conda run -n nasa2025 python scripts/get_tomorrow_forecast.py --city Houston
"""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import sys
from pathlib import Path

# Make sure the repository root is on sys.path so `app` package can be imported
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the project's forecasting helper (same path used by the app)
from app.lib.model_predict import train_city_models_auto


def spoon_from_pm25(pm25: Optional[float]) -> str:
    """Return the playful spoon description used in the app for a single pm25 value."""
    if pm25 is None or pd.isna(pm25):
        return "No data — your spoon is empty."

    if pm25 < 12:
        texture, desc = "Clear broth", "light and refreshing — clean air."
    elif pm25 < 35.4:
        texture, desc = "Seasoned soup", "a hint of flavor — minor haze detected."
    elif pm25 < 55.4:
        texture, desc = "Creamy stew", "noticeably thick — pollutants mix in."
    elif pm25 < 150.4:
        texture, desc = "Yogurt-thick air", "dense and sticky — unhealthy to breathe."
    elif pm25 < 250.4:
        texture, desc = "Dust paste", "heavy and choking — severe pollution."
    else:
        texture, desc = "Toxic sludge", "hazardous saturation — unsafe exposure."

    return f"In one spoon, it feels like you're tasting {texture}, {desc}"


def get_next_day_forecast(city: str, pollutant: str = "pm25", forecast_steps: int = 7, days: int = 1):
    result = train_city_models_auto(city, forecast_steps=forecast_steps, use_cache=True)
    forecasts = result.get("forecasts") or {}
    if not forecasts:
        print("No forecasts available for this city.")
        return None

    # Prefer the requested pollutant if available
    chosen = pollutant if pollutant in forecasts else next(iter(forecasts.keys()), None)
    if chosen is None:
        print("No pollutant forecasts present.")
        return None

    df: pd.DataFrame = forecasts[chosen]
    if df is None or df.empty:
        print(f"Forecast dataframe for {chosen} is empty.")
        return None

    # Ensure date is datetime and sort ascending
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values("date")

    # Cap days to the available rows
    days = max(1, int(days))
    rows = df.head(days)
    if rows.empty:
        print("No forecast rows available to print.")
        return None

    results = []
    for _, row in rows.iterrows():
        pred_date = row.get("date")
        pred_value = row.get("predicted_value")
        aqi = row.get("aqi")
        category = row.get("category")

        print(f"{chosen.upper()} forecast for {city} → {pred_date.date() if hasattr(pred_date, 'date') else pred_date}: {float(pred_value):.1f} ug/m3")
        if aqi is not None:
            print(f"Estimated AQI: {float(aqi):.0f} ({category})")

        # Spoon description prefers pm25 forecast; if chosen is pm25 use pred_value, else try to read pm25 forecast
        pm25_val = float(pred_value) if chosen == "pm25" else None
        if pm25_val is None and "pm25" in forecasts:
            pm_df = forecasts.get("pm25")
            if isinstance(pm_df, pd.DataFrame) and not pm_df.empty:
                try:
                    pm25_row = pm_df.iloc[rows.index.get_loc(_, 0)] if hasattr(rows.index, 'get_loc') else pm_df.iloc[0]
                except Exception:
                    pm25_row = pm_df.iloc[0]
                pm25_val = pm25_row.get("predicted_value")

        spoon = spoon_from_pm25(pm25_val)
        print("Spoon summary:")
        print(spoon)
        print("---")
        results.append({"date": pred_date, "pollutant": chosen, "value": pred_value, "aqi": aqi, "category": category, "spoon": spoon})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", "-c", default="Houston", help="City name to fetch forecast for")
    parser.add_argument("--pollutant", "-p", default="pm25", help="Pollutant key to prefer (pm25, pm10, etc.)")
    parser.add_argument("--days", "-d", type=int, default=1, help="How many next days to print (1..n)")
    args = parser.parse_args()

    get_next_day_forecast(args.city, pollutant=args.pollutant, days=args.days)


if __name__ == "__main__":
    main()

"""Utility helpers for caching local datasets and model artifacts."""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd


CACHE_DATE_FMT = "%Y%m%d"


def safe_city_name(city: str) -> str:
    """Return a filesystem-safe slug for a city name."""
    return city.lower().strip().replace(" ", "_")


def latest_file(pattern: str) -> Optional[str]:
    """Return the most recently modified file matching a glob pattern."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def load_latest_dataset(
    city: str,
    dataset: str,
    data_dir: str = "data",
    parse_dates: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load the newest `dataset` CSV for `city` if it exists."""
    safe = safe_city_name(city)
    pattern = os.path.join(data_dir, f"{safe}_{dataset}_daily_*\.csv")
    path = latest_file(pattern)
    if not path or not os.path.exists(path):
        return pd.DataFrame(), None

    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    return df, path


def save_dataset_with_stamp(
    city: str,
    dataset: str,
    df: pd.DataFrame,
    data_dir: str = "data",
    date_fmt: str = CACHE_DATE_FMT,
) -> str:
    """Persist `df` to disk using the naming convention for cached datasets."""
    safe = safe_city_name(city)
    os.makedirs(data_dir, exist_ok=True)
    stamp = datetime.now().strftime(date_fmt)
    path = os.path.join(data_dir, f"{safe}_{dataset}_daily_{stamp}.csv")
    df_to_save = df.copy()
    if "date" in df_to_save.columns:
        df_to_save["date"] = pd.to_datetime(df_to_save["date"]).dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(path, index=False)
    return path


def ensure_merged_dataset(
    city: str,
    openaq_df: pd.DataFrame,
    merra_df: Optional[pd.DataFrame] = None,
    data_dir: str = "data",
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Optional[str], bool]:
    """Return a merged OpenAQ+MERRA dataframe, creating a cached CSV if missing."""
    safe = safe_city_name(city)
    pattern = os.path.join(data_dir, f"{safe}_merged_daily_*\.csv")
    existing_path = latest_file(pattern)

    if not force_refresh and existing_path and os.path.exists(existing_path):
        merged = pd.read_csv(existing_path)
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged[merged["date"].notna()].sort_values("date").reset_index(drop=True)
        return merged, existing_path, False

    if openaq_df is None or openaq_df.empty:
        return pd.DataFrame(), None, False

    merged = openaq_df.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged[merged["date"].notna()].sort_values("date").reset_index(drop=True)

    if merra_df is not None and not merra_df.empty and "date" in merra_df.columns:
        merra = merra_df.copy()
        merra["date"] = pd.to_datetime(merra["date"], errors="coerce")
        merra = merra[merra["date"].notna()].sort_values("date")
        merra = merra.drop(columns=[col for col in ("city", "lat", "lon") if col in merra.columns])
        merged = pd.merge(merged, merra, on="date", how="left", suffixes=("", "_merra"))

    for col in merged.columns:
        if col == "date":
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if merged.empty:
        return merged, None, False

    path = save_dataset_with_stamp(city, "merged", merged, data_dir=data_dir)
    return merged, path, True


def dataframe_signature(df: pd.DataFrame) -> str:
    """Compute a stable signature for detecting data changes."""
    if df is None or df.empty:
        return "empty"

    normalized = df.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized = normalized[normalized["date"].notna()].sort_values("date").reset_index(drop=True)

    # Convert to CSV bytes for hashing to keep implementation simple and stable.
    csv_bytes = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(csv_bytes).hexdigest()


@dataclass
class ModelCacheManifest:
    city: str
    horizon: int
    forecast_steps: int
    history_signature: str
    path: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "city": self.city,
            "horizon": self.horizon,
            "forecast_steps": self.forecast_steps,
            "history_signature": self.history_signature,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ModelCacheManifest":
        return cls(
            city=str(payload.get("city", "")),
            horizon=int(payload.get("horizon", 1)),
            forecast_steps=int(payload.get("forecast_steps", 1)),
            history_signature=str(payload.get("history_signature", "")),
            path=str(payload.get("path", "")),
        )

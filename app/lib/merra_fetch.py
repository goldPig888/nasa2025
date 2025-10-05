"""Utilities for fetching MERRA-2 daily air quality variables."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from datetime import datetime, date as date_cls
from typing import Dict, List, Optional, Tuple

import requests

try:  # Optional heavy dependencies; degrade gracefully if unavailable.
    import numpy as np
    import xarray as xr

    XR_AVAILABLE = True
except Exception:  # pragma: no cover - informs caller that extraction is disabled.
    XR_AVAILABLE = False

from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH)

EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

# GES DISC HTTPS archive base
GESDISC_BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2"

# Short collection codes used in testing + production
COLLECTIONS = {
    "AER": "M2T1NXAER.5.12.4",  # Aerosol (PM2.5 etc.)
    "CHM": "M2T1NXCHM.5.12.4",  # Chemistry (SO2, NO2, O3, CO)
    "SLV": "M2T1NXSLV.5.12.4",  # Surface meteorology (temperature, humidity, winds)
}

RUN_CODE = "400"  # Historic model designator used in file names

FNAME_PATTERNS = {
    "AER": f"MERRA2_{RUN_CODE}.tavg1_2d_aer_Nx.{{yyyymmdd}}.nc4",
    "CHM": f"MERRA2_{RUN_CODE}.tavg1_2d_chm_Nx.{{yyyymmdd}}.nc4",
    "SLV": f"MERRA2_{RUN_CODE}.tavg1_2d_slv_Nx.{{yyyymmdd}}.nc4",
}

# Aliases per variable we try to extract from the NetCDF datasets.
WANTED_ALIASES = {
    "so2": ["SO2", "so2", "SO2_column_amount", "SO2_column_density", "SO2CL", "SO2S", "SO2SMASS", "SO2EM"],
    "no2": ["NO2", "no2", "NO2_column_amount", "NO2CL", "NO2S", "NO2SMASS", "NO2EM"],
    "o3": ["O3", "o3", "O3_column_amount", "TO3", "O3S", "O3SMASS"],
    "co": ["CO", "co", "COCL", "COEM", "COPD", "COSC"],
    "pm25": ["PM2_5", "PM2_5_TOT", "PM2_5_Tot", "PM2_5_TOT_Ext", "PM2_5_DRY", "PM25_DRY", "PM2_5_TOTAL_AIR_DRY", "pm25"],
    "temperature": ["T", "temperature", "T2m", "TEMP", "temp", "T2M", "CLDTMP"],
    "humidity": ["RH", "relative_humidity", "QV2M", "RH2m", "specific_humidity", "humidity"],
    "wind_speed": ["WS", "wind_speed", "wind", "U10", "V10", "U10M", "V10M"],
    "pressure": ["PS", "pressure", "press", "PSL", "CLDPRS"],
}

_geolocator = Nominatim(user_agent="nasa-2025-merra")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geocode_city(city_name: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        loc = _geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        return None, None
    return None, None


def _build_urls(dt: datetime) -> Dict[str, str]:
    yyyymmdd = dt.strftime("%Y%m%d")
    month_path = dt.strftime("%m")
    urls = {}
    for key, coll in COLLECTIONS.items():
        fname = FNAME_PATTERNS[key].format(yyyymmdd=yyyymmdd)
        urls[key] = f"{GESDISC_BASE}/{coll}/{dt.year}/{month_path}/{fname}"
    return urls


def _download_file(url: str, auth: Tuple[str, str], out_dir: str, verbose: bool) -> Optional[str]:
    local_path = os.path.join(out_dir, os.path.basename(url))
    try:
        with requests.get(url, auth=auth, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1_048_576):  # 1 MB chunks
                    if chunk:
                        fh.write(chunk)
        if verbose:
            print(f"[✓] Saved {local_path}")
        return local_path
    except requests.HTTPError as exc:
        if verbose:
            print(f"⚠️ HTTP error downloading {url}: {exc}")
    except requests.RequestException as exc:
        if verbose:
            print(f"⚠️ Request error downloading {url}: {exc}")
    except OSError as exc:
        if verbose:
            print(f"⚠️ Filesystem error saving {url}: {exc}")
    return None


def _find_variable(ds, aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        if alias in ds.variables:
            return alias
    lower = {name.lower(): name for name in ds.variables}
    for alias in aliases:
        key = alias.lower()
        if key in lower:
            return lower[key]
    return None


def _extract_nearest(ds, varname: str, lat: float, lon: float):
    if varname is None:
        return None

    lat_names = [coord for coord in ds.coords if "lat" in coord.lower()]
    lon_names = [coord for coord in ds.coords if "lon" in coord.lower()]
    if not lat_names or not lon_names:
        lat_names = [var for var in ds.variables if "lat" in var.lower()]
        lon_names = [var for var in ds.variables if "lon" in var.lower()]
    if not lat_names or not lon_names:
        return None

    lat_name = lat_names[0]
    lon_name = lon_names[0]

    try:
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        if lats.ndim == 2 and lons.ndim == 2:
            dist = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            selection = {lat_name: idx[0], lon_name: idx[1]}
        else:
            lat_idx = int(np.abs(lats - lat).argmin())
            lon_idx = int(np.abs(lons - lon).argmin())
            selection = {lat_name: lat_idx, lon_name: lon_idx}

        var = ds[varname]
        sel = {k: v for k, v in selection.items() if k in var.dims}
        if "time" in var.dims:
            sel = {**{"time": 0}, **sel}
        value = var.isel(**sel).values
        if hasattr(value, "size") and value.size == 1:
            return float(value.item())
        if hasattr(value, "size") and value.size > 1:
            return float(np.nanmean(value))
        return float(value)
    except Exception:
        return None


def _open_dataset(path: str, verbose: bool):
    try:
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=True)
        if verbose:
            vars_preview = list(ds.data_vars)[:8]
            print(f"[✓] Opened {os.path.basename(path)} (vars: {vars_preview}…)")
        return ds
    except Exception as exc:
        if verbose:
            print(f"⚠️ Could not open {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_merra2_day(
    city: str,
    date_value,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Optional[float]]]:
    """Fetch a single day's worth of MERRA-2 data for a city.

    Returns a mapping with keys: date, city, lat, lon, so2, no2, o3, co, pm25,
    temperature, humidity, wind_speed, pressure. Any value that could not be
    determined is set to ``None``. ``None`` is returned if credentials are
    missing, xarray is unavailable, or downloads fail entirely.
    """

    if not XR_AVAILABLE:
        if verbose:
            print("⚠️ xarray/netCDF support unavailable; cannot read MERRA-2 files.")
        return None

    if isinstance(date_value, (datetime, date_cls)):
        dt = datetime.combine(date_value, datetime.min.time()) if isinstance(date_value, date_cls) else date_value
    else:
        dt = datetime.strptime(str(date_value), "%Y-%m-%d")

    username = username or os.getenv("EARTHDATA_USERNAME") or EARTHDATA_USERNAME
    password = password or os.getenv("EARTHDATA_PASSWORD") or EARTHDATA_PASSWORD
    if not username or not password:
        if verbose:
            print("⚠️ Earthdata credentials missing. Set EARTHDATA_USERNAME/PASSWORD.")
        return None

    if lat is None or lon is None:
        geocoded = _geocode_city(city)
        lat, lon = geocoded
        if lat is None or lon is None:
            if verbose:
                print(f"⚠️ Could not geocode {city} for MERRA-2 lookup.")
            return None

    if verbose:
        print(f"[i] MERRA-2 lookup for {city} ({lat:.4f}, {lon:.4f}) on {dt.date():%Y-%m-%d}")

    urls = _build_urls(dt)
    if verbose:
        for key, url in urls.items():
            print(f"  - {key}: {url}")

    tmpdir = tempfile.mkdtemp(prefix="merra2_")
    auth = (username, password)

    downloaded = {}
    for key, url in urls.items():
        if verbose:
            print(f"[i] Downloading {key} …")
        path = _download_file(url, auth, tmpdir, verbose)
        if path:
            downloaded[key] = path

    if not downloaded:
        if verbose:
            print("⚠️ No MERRA-2 files downloaded successfully.")
        return None

    datasets = {key: _open_dataset(path, verbose) for key, path in downloaded.items()}

    result = {
        "date": dt.strftime("%Y-%m-%d"),
        "city": city,
        "lat": lat,
        "lon": lon,
        "so2": None,
        "no2": None,
        "o3": None,
        "co": None,
        "pm25": None,
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "pressure": None,
    }

    def try_extract(dataset_keys: List[str], wanted_key: str):
        aliases = WANTED_ALIASES[wanted_key]
        for key in dataset_keys:
            ds = datasets.get(key)
            if ds is None:
                continue
            varname = _find_variable(ds, aliases)
            if not varname:
                continue
            value = _extract_nearest(ds, varname, lat, lon)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
        return None

    result["so2"] = try_extract(["CHM", "AER", "SLV"], "so2")
    result["no2"] = try_extract(["CHM", "AER", "SLV"], "no2")
    result["o3"] = try_extract(["CHM", "AER", "SLV"], "o3")
    result["co"] = try_extract(["CHM", "AER", "SLV"], "co")
    result["pm25"] = try_extract(["AER", "CHM", "SLV"], "pm25")
    result["temperature"] = try_extract(["SLV", "CHM", "AER"], "temperature")
    result["humidity"] = try_extract(["SLV", "CHM", "AER"], "humidity")

    wind = try_extract(["SLV"], "wind_speed")
    if wind is not None:
        result["wind_speed"] = wind
    else:
        slv_ds = datasets.get("SLV")
        if slv_ds is not None:
            u_name = _find_variable(slv_ds, ["U10", "u10", "U10M", "U", "UGRD"])
            v_name = _find_variable(slv_ds, ["V10", "v10", "V10M", "V", "VGRD"])
            if u_name and v_name:
                u_val = _extract_nearest(slv_ds, u_name, lat, lon)
                v_val = _extract_nearest(slv_ds, v_name, lat, lon)
                if u_val is not None and v_val is not None:
                    result["wind_speed"] = float((u_val ** 2 + v_val ** 2) ** 0.5)

    result["pressure"] = try_extract(["SLV", "CHM", "AER"], "pressure")

    for ds in datasets.values():
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass

    if verbose:
        print(f"Temporary downloads in: {tmpdir}")

    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        if verbose:
            print(f"⚠️ Could not clean temporary directory {tmpdir}")

    return result


__all__ = ["fetch_merra2_day", "XR_AVAILABLE"]

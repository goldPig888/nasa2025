# fetch_past.py
import os, io, gzip, boto3, requests, pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore import UNSIGNED
from botocore.client import Config
from geopy.distance import distance
from dotenv import load_dotenv

from lib.data_cache import ensure_merged_dataset, load_latest_dataset, save_dataset_with_stamp
from lib.merra_fetch import fetch_merra2_day

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE = "https://api.openaq.org/v3"
headers = {"X-API-Key": OPENAQ_API_KEY}
os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------
# 🌎 Geocode any city → (lat, lon)
# ----------------------------------------------------------
def geocode_city(city_name):
    """Use OpenWeatherMap Geocoding API to get city coordinates."""
    key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={key}"
    r = requests.get(url)
    if r.status_code != 200 or not r.json():
        print(f"⚠️ Failed to geocode {city_name}")
        return None, None
    result = r.json()[0]
    return result["lat"], result["lon"]

# ----------------------------------------------------------
# 1️⃣ Nearby OpenAQ locations
# ----------------------------------------------------------
def get_nearby_locations(lat, lon, radius_km=50):
    """Fetch all US stations via pagination and filter locally."""
    print(f"[i] Fetching stations (within {radius_km} km)…")
    all_results = []
    page = 1
    while True:
        url = f"{BASE}/locations?country=US&limit=1000&page={page}"
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"⚠️ Page {page} error: {r.status_code}")
            break
        batch = r.json().get("results", [])
        if not batch:
            break
        all_results.extend(batch)
        if len(batch) < 1000:
            break
        page += 1

    nearby = []
    for loc in all_results:
        coords = loc.get("coordinates")
        if not coords:
            continue
        station_lat = coords.get("latitude")
        station_lon = coords.get("longitude")
        if station_lat is None or station_lon is None:
            continue
        d = distance((lat, lon), (station_lat, station_lon)).km
        if d <= radius_km:
            nearby.append(
                {
                    "id": loc["id"],
                    "name": loc.get("name"),
                    "dist": round(d, 1),
                    "lat": station_lat,
                    "lon": station_lon,
                    "parameters": loc.get("parameters", []),
                }
            )
    print(f"[✓] Found {len(nearby)} nearby locations.")
    return sorted(nearby, key=lambda x: x["dist"])

# ----------------------------------------------------------
# 2️⃣ S3 structure discovery
# ----------------------------------------------------------
def list_year_months(s3, bucket, loc_id, start_year=2021, limit=12):
    """Find recent (year, month) entries after start_year."""
    prefix = f"records/csv.gz/locationid={loc_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    pairs = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "year=" in key and "month=" in key:
                try:
                    parts = key.split("/")
                    y = int([p.split("=")[1] for p in parts if "year=" in p][0])
                    m = int([p.split("=")[1] for p in parts if "month=" in p][0])
                    if y >= start_year:
                        pairs.add((y, m))
                except:
                    pass
    ordered = sorted(pairs)
    if not ordered:
        return []
    if limit is None or limit >= len(ordered):
        return ordered
    return ordered[-limit:]

# ----------------------------------------------------------
# 3️⃣ Parallel S3 downloads
# ----------------------------------------------------------
def download_csv(s3, bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        with gzip.open(io.BytesIO(obj["Body"].read()), "rt") as gz:
            return pd.read_csv(gz)
    except Exception as e:
        print(f"⚠️ {key}: {e}")
        return pd.DataFrame()

def download_station_data(s3, bucket, loc_id, year_months):
    """Fetch all data for given months."""
    all_keys, frames = [], []
    paginator = s3.get_paginator("list_objects_v2")
    for (y, m) in year_months:
        prefix = f"records/csv.gz/locationid={loc_id}/year={y}/month={m:02d}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".csv.gz"):
                    all_keys.append(obj["Key"])
    if not all_keys:
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(download_csv, s3, bucket, k) for k in all_keys]
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                df["location_id"] = loc_id
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ----------------------------------------------------------
# 4️⃣ Daily aggregation
# ----------------------------------------------------------
def process_daily(df):
    if df.empty:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df[df["datetime"].notna()]
    df["date"] = df["datetime"].dt.tz_convert(None).dt.date
    df = df.dropna(subset=["value", "parameter"])

    pivot = (
        df.groupby(["date", "parameter"])["value"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="parameter", values="value")
        .reset_index()
    )
    return pivot

def _openaq_pollutant_columns() -> list[str]:
    return ["pm25", "pm10", "o3", "no2", "so2", "co"]


def fetch_city_openaq(
    city_name,
    radius_km=50,
    start_year=2023,
    station_limit=5,
    months_per_station=1,
    *,
    data_dir: str = "data",
    use_cache: bool = True,
    force_refresh: bool = False,
):
    lat, lon = geocode_city(city_name)
    if not lat or not lon:
        print(f"❌ Could not locate {city_name}")
        return {}

    print(f"[i] {city_name}: ({lat:.4f}, {lon:.4f})")
    nearby = get_nearby_locations(lat, lon, radius_km)

    openaq_df = pd.DataFrame()
    openaq_path = None
    latest_date = None

    if use_cache and not force_refresh:
        openaq_df, openaq_path = load_latest_dataset(city_name, "openaq", data_dir=data_dir)
        if openaq_path and not openaq_df.empty:
            print(f"[cache] Using cached OpenAQ daily dataset → {openaq_path}")
            latest_date = openaq_df["date"].max() if "date" in openaq_df.columns else None
            return {
                "lat": lat,
                "lon": lon,
                "stations": nearby,
                "combined": pd.DataFrame(),
                "openaq_daily": openaq_df,
                "paths": {"openaq": openaq_path},
                "latest_openaq_date": latest_date,
            }
        elif openaq_path and openaq_df.empty:
            print(f"[cache] Cached OpenAQ dataset {openaq_path} is empty; refreshing from remote sources.")

    if not nearby:
        print("⚠️ No nearby OpenAQ stations found.")
        return {
            "lat": lat,
            "lon": lon,
            "stations": [],
            "combined": pd.DataFrame(),
            "openaq_daily": pd.DataFrame(),
            "paths": {},
            "latest_openaq_date": None,
        }

    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    bucket = "openaq-data-archive"

    frames = []
    for loc in nearby[:station_limit]:
        loc_id, name = loc["id"], loc.get("name")
        ym = list_year_months(s3, bucket, loc_id, start_year, months_per_station)
        if not ym:
            continue
        print(f"→ {name} ({loc_id}) months={ym[-3:]} …")
        df = download_station_data(s3, bucket, loc_id, ym)
        if df.empty:
            continue
        daily = process_daily(df)
        daily["station"] = name
        frames.append(daily)
        print(f"   ↳ {len(daily)} daily records ✅")

    if not frames:
        print("⚠️ No data found for any station.")
        empty_cols = _openaq_pollutant_columns()
        empty_openaq = pd.DataFrame(columns=["date"] + empty_cols + ["source"])
        if force_refresh:
            openaq_path = save_dataset_with_stamp(city_name, "openaq", empty_openaq, data_dir=data_dir)
        return {
            "lat": lat,
            "lon": lon,
            "stations": nearby,
            "combined": pd.DataFrame(),
            "openaq_daily": empty_openaq,
            "paths": {"openaq": openaq_path} if openaq_path else {},
            "latest_openaq_date": None,
        }

    combined = pd.concat(frames, ignore_index=True)
    pollutant_cols = _openaq_pollutant_columns()
    for col in pollutant_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined[pollutant_cols] = combined[pollutant_cols].apply(pd.to_numeric, errors="coerce")
    combined["data_source"] = "OpenAQ"

    ordered_cols = ["date", "station", "no2", "o3", "co", "pm25", "so2", "data_source"]
    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[ordered_cols]
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined[combined["date"].notna()].sort_values(["date", "station"]).reset_index(drop=True)

    latest_date = combined["date"].max() if not combined.empty else None

    if combined.empty:
        empty_cols = _openaq_pollutant_columns()
        empty_openaq = pd.DataFrame(columns=["date"] + empty_cols + ["source"])
        openaq_path = save_dataset_with_stamp(city_name, "openaq", empty_openaq, data_dir=data_dir)
        return {
            "lat": lat,
            "lon": lon,
            "stations": nearby,
            "combined": combined,
            "openaq_daily": empty_openaq,
            "paths": {"openaq": openaq_path},
            "latest_openaq_date": None,
        }

    current_month_start = latest_date.replace(day=1)
    start_date = (current_month_start - pd.DateOffset(months=2)).replace(day=1)
    end_date = current_month_start - pd.Timedelta(days=1)

    recent_mask = (combined["date"] >= start_date) & (combined["date"] <= end_date)
    recent_openaq = combined[recent_mask].copy()

    available_cols = [c for c in pollutant_cols if c in recent_openaq.columns]
    if available_cols:
        openaq_daily = (
            recent_openaq.groupby("date")[available_cols]
            .mean()
            .reset_index()
            .sort_values("date")
        )
    else:
        openaq_daily = pd.DataFrame(columns=["date"] + pollutant_cols)

    for col in pollutant_cols:
        if col not in openaq_daily.columns:
            openaq_daily[col] = pd.NA
    openaq_daily["source"] = "OpenAQ"
    openaq_daily = openaq_daily[["date"] + pollutant_cols + ["source"]]
    openaq_daily["date"] = pd.to_datetime(openaq_daily["date"], errors="coerce")

    openaq_path = save_dataset_with_stamp(city_name, "openaq", openaq_daily, data_dir=data_dir)
    print(f"\n💾 Saved OpenAQ daily → {openaq_path} ({len(openaq_daily)} rows)")

    return {
        "lat": lat,
        "lon": lon,
        "stations": nearby,
        "combined": combined,
        "openaq_daily": openaq_daily,
        "paths": {"openaq": openaq_path},
        "latest_openaq_date": latest_date,
    }


def _merra_sample_dates(
    openaq_daily: pd.DataFrame | None,
    *,
    lookback_days: int = 75,
    span_days: int = 35,
    sample_count: int = 8,
) -> list[pd.Timestamp]:
    """Select sparse historical dates for MERRA fetches (1–2 times per week)."""

    if openaq_daily is not None and not openaq_daily.empty and "date" in openaq_daily.columns:
        max_date = pd.to_datetime(openaq_daily["date"], errors="coerce").dropna().max()
    else:
        max_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)

    if pd.isna(max_date):
        max_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)

    reference_end = (max_date - pd.Timedelta(days=lookback_days)).normalize()
    reference_start = (reference_end - pd.Timedelta(days=span_days)).normalize()
    if reference_start >= reference_end:
        reference_start = reference_end - pd.Timedelta(days=span_days or 30)

    all_days = list(pd.date_range(reference_start, reference_end, freq="D"))
    if not all_days:
        return [reference_end]

    sample_count = max(1, sample_count)
    if len(all_days) <= sample_count:
        return [pd.Timestamp(day) for day in all_days]

    indices = np.linspace(0, len(all_days) - 1, sample_count)
    chosen = sorted({int(round(idx)) for idx in indices})
    dates = [pd.Timestamp(all_days[i]) for i in chosen]
    # Ensure coverage spans roughly a month
    if dates and (dates[-1] - dates[0]).days < span_days // 2 and len(all_days) > sample_count:
        extra_idx = min(len(all_days) - 1, chosen[-1] + span_days // 4)
        if extra_idx not in chosen:
            dates.append(pd.Timestamp(all_days[extra_idx]))
    dates = sorted({pd.Timestamp(d) for d in dates})
    return dates


def fetch_city_merra(
    city_name,
    lat,
    lon,
    openaq_daily: pd.DataFrame | None = None,
    *,
    data_dir: str = "data",
    use_cache: bool = True,
    force_refresh: bool = False,
    merra_threads: int = 4,
    merra_verbose: bool = False,
):
    merra_df = pd.DataFrame()
    merra_path = None

    if use_cache and not force_refresh:
        merra_df, merra_path = load_latest_dataset(city_name, "merra", data_dir=data_dir)
        if merra_path and not merra_df.empty:
            print(f"[cache] Using cached MERRA-2 daily dataset → {merra_path}")
            merged_df, merged_path, _ = ensure_merged_dataset(
                city_name,
                openaq_daily if openaq_daily is not None else pd.DataFrame(),
                merra_df,
                data_dir=data_dir,
            )
            return {
                "merra_daily": merra_df,
                "merged_daily": merged_df,
                "paths": {"merra": merra_path, "merged": merged_path},
            }
        elif merra_path and merra_df.empty:
            print(f"[cache] Cached MERRA dataset {merra_path} is empty; refreshing from remote sources.")

    merra_dates = [dt.date() for dt in _merra_sample_dates(openaq_daily)]

    if merra_dates:
        print(
            f"[i] Sampling {len(merra_dates)} MERRA-2 day(s) spanning {merra_dates[0]} → {merra_dates[-1]}"
        )

        def fetch_single(date_obj):
            return fetch_merra2_day(
                city_name,
                date_obj,
                lat=lat,
                lon=lon,
                verbose=merra_verbose,
            )

        results = []
        with ThreadPoolExecutor(max_workers=max(1, merra_threads)) as pool:
            futures = {pool.submit(fetch_single, date_obj): date_obj for date_obj in merra_dates}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    results.append(result)

        merra_df = pd.DataFrame(results)
    else:
        merra_df = pd.DataFrame()

    merra_columns = [
        "date",
        "city",
        "lat",
        "lon",
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
    ]

    if merra_df.empty:
        merra_df = pd.DataFrame(columns=merra_columns)
        merra_df["date"] = pd.to_datetime(merra_df.get("date"), errors="coerce")
    else:
        merra_df["date"] = pd.to_datetime(merra_df["date"], errors="coerce")
        merra_df = merra_df[merra_df["date"].notna()].sort_values("date").reset_index(drop=True)
        for col in merra_columns:
            if col not in merra_df.columns:
                merra_df[col] = pd.NA
        merra_df = merra_df[merra_columns]

    merra_path = save_dataset_with_stamp(city_name, "merra", merra_df, data_dir=data_dir)
    merged_df, merged_path, created = ensure_merged_dataset(
        city_name,
        openaq_daily if openaq_daily is not None else pd.DataFrame(),
        merra_df,
        data_dir=data_dir,
        force_refresh=True,
    )

    print(f"💾 Saved MERRA-2 daily → {merra_path} ({len(merra_df)} rows)")
    if created and merged_path:
        print(f"💾 Saved merged daily → {merged_path} ({len(merged_df)} rows)")

    return {
        "merra_daily": merra_df,
        "merged_daily": merged_df,
        "paths": {"merra": merra_path, "merged": merged_path},
    }


# ----------------------------------------------------------
# 5️⃣ Main orchestrator
# ----------------------------------------------------------
def fetch_city_airquality(
    city_name,
    radius_km=50,
    start_year=2023,
    station_limit=5,
    months_per_station=1,
    merra_threads=4,
    merra_verbose=False,
    *,
    data_dir: str = "data",
    use_cache: bool = True,
    force_refresh: bool = False,
):
    openaq_bundle = fetch_city_openaq(
        city_name,
        radius_km=radius_km,
        start_year=start_year,
        station_limit=station_limit,
        months_per_station=months_per_station,
        data_dir=data_dir,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )

    if not openaq_bundle:
        return {}

    lat = openaq_bundle.get("lat")
    lon = openaq_bundle.get("lon")
    openaq_daily = openaq_bundle.get("openaq_daily")

    merra_bundle = fetch_city_merra(
        city_name,
        lat,
        lon,
        openaq_daily=openaq_daily,
        data_dir=data_dir,
        use_cache=use_cache,
        force_refresh=force_refresh,
        merra_threads=merra_threads,
        merra_verbose=merra_verbose,
    )

    paths = {}
    paths.update(openaq_bundle.get("paths", {}))
    if merra_bundle:
        paths.update(merra_bundle.get("paths", {}))

    return {
        "lat": lat,
        "lon": lon,
        "stations": openaq_bundle.get("stations", []),
        "combined": openaq_bundle.get("combined"),
        "openaq_daily": openaq_daily,
        "merra_daily": merra_bundle.get("merra_daily") if merra_bundle else pd.DataFrame(),
        "merged_daily": merra_bundle.get("merged_daily") if merra_bundle else pd.DataFrame(),
        "paths": paths,
        "latest_openaq_date": openaq_bundle.get("latest_openaq_date"),
    }

# ----------------------------------------------------------
# Run (example)
# ----------------------------------------------------------
if __name__ == "__main__":
    city = input("Enter city name (default: Houston): ").strip() or "Houston"
    bundle = fetch_city_airquality(city, radius_km=50, start_year=2024)
    print(bundle["openaq_daily"].head())

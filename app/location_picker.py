import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os
import math
from datetime import datetime
from typing import Dict, Optional, Sequence
from dotenv import load_dotenv

# ---- Local imports ----
from lib.api_fetch import (
    geocode_city,
    fetch_air_pollution,
    describe_aqi,
    reverse_geocode,
)
from fetch_past import fetch_city_merra, fetch_city_openaq
from lib.model_predict import train_city_models_auto
from lib.pollen_api import fetch_pollen_forecast, flatten_pollen_data
from lib.gpt_advisor import generate_action_plan


def describe_spoon(pm25=None, pm10=None, no2=None, so2=None, co=None, o3=None):
    """Convert pollutant concentrations into a playful 'spoonful' air texture."""

    limits = {"pm25": 12, "pm10": 50, "no2": 53, "so2": 75, "co": 9, "o3": 70}

    ratios = []
    for key, limit in limits.items():
        value = locals()[key]
        if value is not None and pd.notna(value):
            ratios.append(value / limit)

    if not ratios:
        return "No data — your spoon is empty."

    ratio = sum(ratios) / len(ratios)

    if ratio < 0.5:
        texture, desc = "Clear broth", "light and refreshing — clean air."
    elif ratio < 1:
        texture, desc = "Seasoned soup", "a hint of flavor — minor haze detected."
    elif ratio < 2:
        texture, desc = "Creamy stew", "noticeably thick — pollutants mix in."
    elif ratio < 4:
        texture, desc = "Yogurt-thick air", "dense and sticky — unhealthy to breathe."
    elif ratio < 8:
        texture, desc = "Dust paste", "heavy and choking — severe pollution."
    else:
        texture, desc = "Toxic sludge", "hazardous saturation — unsafe exposure."

    return f"In one spoon, it feels like you're tasting **{texture}**, {desc}"


POLLUTANT_LIMITS = {"pm25": 12, "pm10": 50, "no2": 53, "so2": 75, "co": 9, "o3": 70}


def build_rule_based_plan(
    city: str,
    groups: Sequence[str],
    user_notes: str,
    pollutant_payload: Optional[Dict[str, Optional[float]]],
    pollen_df: Optional[pd.DataFrame],
    context_note: Optional[str] = None,
) -> str:
    summary_lines = [f"**Location:** {city or 'Unknown city'}"]
    if groups:
        summary_lines.append("**Sensitive groups:** " + ", ".join(groups))
    if user_notes:
        summary_lines.append("**User notes:** " + user_notes.strip())
    if context_note:
        summary_lines.append("**Context:** " + context_note)

    worst_pollutant = None
    worst_ratio = 0.0
    pollutant_lines = []
    if pollutant_payload:
        for key, value in pollutant_payload.items():
            if value is None or pd.isna(value) or key not in POLLUTANT_LIMITS:
                continue
            ratio = float(value) / POLLUTANT_LIMITS[key]
            pollutant_lines.append(f"{key.upper()}: {value:.1f} (×{ratio:.1f} of limit)")
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_pollutant = key
    if pollutant_lines:
        summary_lines.append("**Key pollutants:** " + "; ".join(pollutant_lines))

    pollen_high = None
    if pollen_df is not None and not pollen_df.empty and "index" in pollen_df.columns:
        try:
            pollen_sorted = pollen_df.dropna(subset=["index"]).sort_values("index", ascending=False)
            if not pollen_sorted.empty:
                row = pollen_sorted.iloc[0]
                pollen_high = (
                    row.get("pollen_type"),
                    row.get("index"),
                    row.get("index_description") or row.get("category"),
                    row.get("date"),
                )
        except Exception:
            pollen_high = None
    if pollen_high:
        pollen_type, pollen_value, pollen_desc, pollen_date = pollen_high
        summary_lines.append(
            f"**Peak pollen:** {str(pollen_type).title()} index {pollen_value} ({pollen_desc}) on {pollen_date}."
        )

    steps = []
    if worst_pollutant:
        if worst_ratio >= 2:
            steps.append(
                "1. **Limit outdoor exposure** during peak pollution hours; shift workouts indoors and keep windows closed."
            )
        else:
            steps.append(
                "1. **Plan outdoor time carefully** and stay indoors when pollution spikes; ventilate with filtered air."
            )
        steps.append(
            "2. **Protect your breathing.** Keep high-filtration masks or respirators ready and follow any inhaler/medication plans."
        )
    else:
        steps.append("1. **Monitor conditions** throughout the day; adjust plans if pollution or symptoms escalate.")

    if pollen_high:
        steps.append(
            "3. **Manage pollen exposure.** Shower/change after outdoor time, run HEPA purifiers, and take antihistamines if prescribed."
        )
    else:
        steps.append("3. **Keep indoor air clean.** Use purifiers and avoid tracking outdoor dust/pollen back inside.")

    if groups:
        steps.append(
            "4. **Shield sensitive members.** Prioritize indoor activities for the listed groups and arrange backup care or transport if needed."
        )
    else:
        steps.append("4. **Hydrate and rest.** Support respiratory recovery with fluids, humidifiers, and short rest breaks.")

    if user_notes:
        steps.append("5. **Track your triggers.** Log how your noted symptoms respond to today's air/pollen so you can adjust routines.")
    else:
        steps.append("5. **Recheck levels mid-day** and reschedule strenuous tasks if conditions worsen.")

    body = "\n".join(summary_lines + ["", *steps])
    body += "\n\n_Using rule-based guidance while AI advice is unavailable._"
    return body


def load_pollen_bundle(city: str, lat: float, lon: float, *, force: bool = False):
    """Load and cache pollen data for a city coordinate."""
    cache = st.session_state.setdefault("pollen_data_cache", {})
    key = f"{city.lower()}_{round(lat,4)}_{round(lon,4)}"

    if force or key not in cache:
        with st.spinner("Fetching pollen forecast..."):
            response = fetch_pollen_forecast(lat, lon)
        flattened = flatten_pollen_data(response)
        df = pd.DataFrame(flattened).sort_values("date")

        if not df.empty:
            summary = ", ".join(
                f"{row['pollen_type']} {row['index_description'] or row['index'] or ''}"
                for _, row in df.head(3).iterrows()
                if row.get("pollen_type")
            )
        else:
            summary = response.get("message") if isinstance(response, dict) else None

        cache[key] = {"response": response, "df": df, "summary": summary}

    return cache[key]


def render_action_plan(
    section_key: str,
    heading: str,
    city: str,
    selected_sensitive_groups,
    symptom_notes: str,
    air_summary: Optional[str],
    pollen_summary: Optional[str],
    pollutant_payload: Optional[Dict[str, Optional[float]]] = None,
    pollen_df: Optional[pd.DataFrame] = None,
    context_note: Optional[str] = None,
    temperature: float = 0.3,
):
    """Generate (with caching) and render a personalized action plan."""

    plan_cache = st.session_state.setdefault("action_plan_cache", {})
    cache_key = (
        section_key,
        city.lower() if city else "unknown",
        tuple(sorted(selected_sensitive_groups or [])),
        (symptom_notes or "").strip(),
        air_summary or "",
        pollen_summary or "",
        context_note or "",
        round(float(temperature), 2),
    )

    plan = plan_cache.get(cache_key)
    if not plan:
        instructions = (
            "Provide 3-5 concise, prioritized action steps tailored to the user. "
            "Consider outdoor versus indoor activity, protective gear, medications, hydration, and rest."
        )
        if context_note:
            instructions += f" Context: {context_note.strip()}"

        with st.spinner("Generating action steps..."):
            plan = generate_action_plan(
                city=city,
                sensitive_groups=selected_sensitive_groups,
                user_notes=symptom_notes,
                pollen_summary=pollen_summary,
                air_quality_summary=air_summary,
                custom_prompt=instructions,
                temperature=temperature,
            )
        plan_cache[cache_key] = plan

    if not plan or plan.get("model") == "fallback" or not plan.get("content"):
        plan = {
            "content": build_rule_based_plan(
                city=city,
                groups=selected_sensitive_groups,
                user_notes=symptom_notes,
                pollutant_payload=pollutant_payload,
                pollen_df=pollen_df,
                context_note=context_note,
            ),
            "model": "rule-based",
        }

    st.markdown(heading)
    st.markdown(plan.get("content", "Action plan unavailable."))
    st.caption(f"Source: {plan.get('model', 'unknown')}")

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
st.set_page_config(page_title="Location & Air Quality", layout="wide")
st.title("🌍 Location and Air Quality Lookup")

# Create output directory
os.makedirs("data", exist_ok=True)

# Default data/model behaviour (controls hidden for now)
use_cache = True
force_refresh = False
force_model_retrain = False

# ----------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------
st.sidebar.title("Navigation")
nav_options = [
    "Location Dashboard",
    "Pollen Dashboard",
    "Social Media",
]
selected_view = st.sidebar.radio("Go to", options=nav_options, index=0)

# ----------------------------------------------------------
# 1️⃣ User input
# ----------------------------------------------------------

default_city = st.session_state.get("city_input", "Houston")
if selected_view == "Location Dashboard":
    st.markdown("### 📍 Choose your city")
    city_input = st.text_input("City name:", value=default_city).strip()
    if city_input:
        st.session_state["city_input"] = city_input
    city = st.session_state.get("city_input", default_city)
else:
    city = st.session_state.get("city_input", default_city)
    st.markdown(f"### 📍 City: {city}")

sensitive_group_options = [
    "Children (under 18)",
    "Older adults (65+)",
    "Pregnant",
    "Asthma or other lung conditions",
    "Heart disease",
    "Compromised immune system",
]

if "sensitive_group_selection" not in st.session_state:
    st.session_state["sensitive_group_selection"] = []

if selected_view == "Location Dashboard":
    st.multiselect(
        "Select any sensitive groups that apply (leave empty if none):",
        sensitive_group_options,
        key="sensitive_group_selection",
    )
else:
    if st.session_state["sensitive_group_selection"]:
        st.caption(
            "Sensitive groups: " + ", ".join(st.session_state["sensitive_group_selection"])
        )

selected_sensitive_groups = st.session_state.get("sensitive_group_selection", [])

st.session_state["sensitive_groups"] = {
    "is_sensitive": len(selected_sensitive_groups) > 0,
    "groups": selected_sensitive_groups,
}

if "symptom_notes" not in st.session_state:
    st.session_state["symptom_notes"] = ""

if selected_view == "Location Dashboard":
    symptom_notes = st.text_area(
        "Anything else we should know about your sensitivities?",
        key="symptom_notes",
        placeholder="I'm allergic to ragweed on dry days, foggy weather triggers headaches, joints flare when it's damp...",
        help="Share symptoms, medications, or patterns that help tailor recommendations.",
    )
else:
    symptom_notes = st.session_state.get("symptom_notes", "")
    if symptom_notes:
        st.caption(f"Symptom notes: {symptom_notes}")

radius = 50
start_year = max(2016, datetime.now().year - 2)
months_per_station = 3

st.markdown("---")
st.write("### ✅ Selected Coordinates and Air Quality")

lat, lon = (None, None)
if city:
    lat, lon = geocode_city(city)
    if lat and lon:
        st.success(f"📍 {city} → ({lat:.4f}, {lon:.4f})")
    else:
        st.warning("Couldn't find that city. Try another name.")

if not (lat and lon):
    st.map(pd.DataFrame([[39.5, -98.35]], columns=["lat", "lon"]))
    if not city:
        st.info("Type a city name to view coordinates and air quality.")
    else:
        st.info("Unable to plot this location yet.")
    st.stop()

location_label = reverse_geocode(lat, lon) or city

pollution_snapshot = fetch_air_pollution(lat, lon)
pollution_info = None
wind_speed = None
if pollution_snapshot and "list" in pollution_snapshot:
    pollution_info = pollution_snapshot["list"][0]
    wind_speed = (pollution_info.get("wind", {}) or {}).get("speed")

city_cache_store = st.session_state.setdefault("city_data_cache", {})
cache_key = (
    city.lower(),
    radius,
    start_year,
    months_per_station,
)
cache_key_str = "_".join(str(part) for part in cache_key)
city_cache = city_cache_store.get(cache_key, {})

if force_refresh or "openaq" not in city_cache:
    with st.spinner(f"Fetching OpenAQ history for {city}..."):
        openaq_bundle = fetch_city_openaq(
            city,
            radius_km=radius,
            start_year=start_year,
            station_limit=5,
            months_per_station=months_per_station,
            data_dir="data",
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
    city_cache["openaq"] = openaq_bundle
    city_cache_store[cache_key] = city_cache
else:
    openaq_bundle = city_cache.get("openaq", {})

if not openaq_bundle:
    st.warning("Unable to assemble OpenAQ history for this location.")
    st.stop()

paths = dict(openaq_bundle.get("paths", {}))
openaq_df = openaq_bundle.get("openaq_daily", pd.DataFrame())
combined_open = openaq_bundle.get("combined", pd.DataFrame())
stations = openaq_bundle.get("stations", [])
merged_df = pd.DataFrame()
merra_df = pd.DataFrame()
merra_bundle = city_cache.get("merra")
if merra_bundle:
    merra_df = merra_bundle.get("merra_daily", pd.DataFrame())
    merged_df = merra_bundle.get("merged_daily", pd.DataFrame())
    paths.update(merra_bundle.get("paths", {}))

air_summary = None
latest_pollutants = {}
if openaq_df is not None and not openaq_df.empty:
    latest_row = openaq_df.sort_values("date").iloc[-1]
    pollutant_cols = [col for col in openaq_df.columns if col not in {"date", "station", "data_source", "source"}]
    air_summary_parts = []
    for col in pollutant_cols:
        val = latest_row.get(col)
        latest_pollutants[col] = float(val) if pd.notna(val) else None
        if pd.notna(val):
            air_summary_parts.append(f"{col.upper()}: {val:.1f}")
    air_summary = ", ".join(air_summary_parts[:6]) if air_summary_parts else None
else:
    pollutant_cols = []
 
pollen_summary = None
model_results = city_cache.get("model_results") if not force_model_retrain else None

if selected_view == "Location Dashboard":
    st.subheader("City Overview")

    station_df = pd.DataFrame(stations)
    station_layer_data = pd.DataFrame(columns=["lon", "lat", "tooltip", "radius", "color", "name"])
    station_metrics: dict[str, dict[str, float | None]] = {}
    name_lookup: dict[str, str] = {}
    metric_cols = list(pollutant_cols)

    if not station_df.empty:
        station_df = station_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if "name" not in station_df.columns:
            station_df["name"] = None
        station_df["original_name"] = station_df["name"]

        if "id" in station_df.columns:
            fallback_names = station_df["id"].astype(str)
        else:
            fallback_names = station_df.index.to_series().apply(lambda idx: f"Station {idx + 1}")
        station_df["name"] = station_df["name"].fillna(fallback_names)
        station_df["name"] = station_df["name"].replace({None: pd.NA}).fillna(fallback_names)

        latest_station_vals = pd.DataFrame()
        if not combined_open.empty and "station" in combined_open.columns:
            latest_station_vals = (
                combined_open.sort_values("date")
                .groupby("station", as_index=False)
                .tail(1)
            )
            candidate_cols = [c for c in pollutant_cols if c in latest_station_vals.columns]
            if candidate_cols:
                metric_cols = candidate_cols

        for _, row in station_df.iterrows():
            display_name = str(row["name"])
            original_name = row.get("original_name")
            if original_name is not None and not pd.isna(original_name):
                name_lookup[str(original_name)] = display_name
            if "id" in row and pd.notna(row["id"]):
                name_lookup[str(row["id"])] = display_name
            name_lookup[display_name] = display_name

        if not latest_station_vals.empty:
            for _, row in latest_station_vals.iterrows():
                source_name = str(row["station"])
                display_name = name_lookup.get(source_name, source_name)
                station_metrics[display_name] = {
                    col: (float(row[col]) if pd.notna(row[col]) else None)
                    for col in metric_cols
                }
                station_metrics[source_name] = dict(station_metrics[display_name])

        def station_tooltip(row):
            display_name = str(row.get("name") or "Station")
            dist = row.get("dist", "?")
            pieces = [f"<b>{display_name}</b>", f"{dist} km away"]
            metrics = station_metrics.get(display_name)
            if metrics:
                vals = [f"{key.upper()}: {value:.1f}" for key, value in metrics.items() if value is not None]
                if vals:
                    pieces.append("<br/>".join(vals))
            return "<br/>".join(pieces)

        station_layer_data = station_df.copy()
        station_layer_data["tooltip"] = station_layer_data.apply(station_tooltip, axis=1)
        station_layer_data["radius"] = 2500
        station_layer_data["color"] = station_layer_data.apply(
            lambda _: [255, 140, 0, 200], axis=1
        )
        station_layer_data = station_layer_data[["lon", "lat", "tooltip", "radius", "color", "name"]]
    else:
        name_lookup = {}

    raw_station_names = station_layer_data["name"].tolist() if "name" in station_layer_data.columns else []
    station_names = list(dict.fromkeys(raw_station_names))

    pollutant_options: list[str] = []
    if station_metrics:
        for col in metric_cols:
            for payload in station_metrics.values():
                if payload and payload.get(col) is not None:
                    pollutant_options.append(col)
                    break
    if not pollutant_options:
        pollutant_options = [k for k, v in latest_pollutants.items() if v is not None]

    selected_pollutant: Optional[str] = None
    if pollutant_options:
        default_idx = pollutant_options.index("pm25") if "pm25" in pollutant_options else 0
        selected_pollutant = st.selectbox(
            "Pollutant to map",
            options=pollutant_options,
            index=default_idx,
            format_func=lambda x: x.upper(),
            key=f"pollutant_map_{cache_key_str}",
        )
    else:
        st.caption("No pollutant readings available to visualize on the map yet.")
    selection_options = ["City average"] + station_names
    selected_station_option = st.selectbox(
        "Focus on",
        options=selection_options,
        index=0,
        help="Choose a station to highlight its latest readings.",
        key=f"focus_station_{cache_key_str}",
    )
    highlight_station = selected_station_option if selected_station_option != "City average" else None

    def resolve_payload(name: str) -> Dict[str, float | None]:
        return station_metrics.get(name) or station_metrics.get(name_lookup.get(name, "")) or {}

    station_plot_records: list[dict[str, object]] = []
    value_map = {name: resolve_payload(name).get(selected_pollutant) if selected_pollutant else None for name in station_names}
    numeric_values = pd.Series([value_map.get(name) for name in station_names], dtype="float64") if station_names else pd.Series(dtype="float64")
    finite_values = numeric_values.dropna()
    if not finite_values.empty:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin = vmax = None

    def norm_value(val: Optional[float]) -> Optional[float]:
        if vmin is None or vmax is None or val is None or pd.isna(val):
            return None
        return max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin)))

    for _, row in station_layer_data.iterrows():
        display_name = row.get("name")
        value = value_map.get(display_name)
        normalized = norm_value(value)
        base_radius = 2500 if normalized is None else 2400 + normalized * 5200
        base_color = (
            [180, 180, 180, 110]
            if normalized is None
            else [
                int(np.interp(normalized, [0, 1], [65, 235])),
                int(np.interp(normalized, [0, 1], [130, 30])),
                int(np.interp(normalized, [0, 1], [255, 60])),
                220,
            ]
        )
        base_elevation = 0.0 if value is None or pd.isna(value) else float(value) * 120.0
        highlight = highlight_station == display_name
        station_plot_records.append(
            {
                "lon": float(row.get("lon", 0.0)),
                "lat": float(row.get("lat", 0.0)),
                "tooltip": row.get("tooltip", display_name),
                "radius": base_radius * (1.25 if highlight else 1.0),
                "color": [230, 57, 70, 230] if highlight else base_color,
                "elevation": base_elevation * (1.2 if highlight else 1.0),
            }
        )

    base_tooltip_lines = [f"<b>{location_label or city}</b>"]
    if latest_pollutants:
        pollutant_lines = [
            f"{key.upper()}: {value:.1f}"
            for key, value in latest_pollutants.items()
            if value is not None
        ]
        if pollutant_lines:
            base_tooltip_lines.append("<br/>".join(pollutant_lines))

    base_point = pd.DataFrame(
        [
            {
                "lat": float(lat),
                "lon": float(lon),
                "tooltip": "<br/>".join(base_tooltip_lines),
            }
        ]
    )
    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=base_point,
        get_position="[lon, lat]",
        get_radius=3000,
        get_fill_color=[0, 122, 255, 200],
        pickable=True,
    )

    merra_color_palette = {
        "pm25": [142, 68, 173, 180],
        "pm10": [46, 204, 113, 180],
        "no2": [52, 152, 219, 180],
        "so2": [241, 196, 15, 180],
        "co": [230, 126, 34, 180],
        "o3": [231, 76, 60, 180],
    }
    merra_records: list[Dict[str, object]] = []
    if "merra_df" in locals() and isinstance(merra_df, pd.DataFrame) and not merra_df.empty:
        recent_merra = merra_df.sort_values("date").tail(42)
        pollutant_columns = [
            col
            for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]
            if col in recent_merra.columns
        ]
        if pollutant_columns:
            base_lat_default = float(lat) if lat else 0.0
            base_lon_default = float(lon) if lon else 0.0
            for idx, row in enumerate(recent_merra.itertuples()):
                base_lat = float(getattr(row, "lat", base_lat_default) or base_lat_default)
                base_lon = float(getattr(row, "lon", base_lon_default) or base_lon_default)
                cos_lat = math.cos(math.radians(base_lat if base_lat else 0.0001)) or 1.0
                for p_idx, pollutant in enumerate(pollutant_columns):
                    value = getattr(row, pollutant, None)
                    if value is None or pd.isna(value):
                        continue
                    value = float(value)
                    date_value = getattr(row, "date", None)
                    if isinstance(date_value, pd.Timestamp):
                        date_label = date_value.strftime("%Y-%m-%d")
                    elif date_value:
                        date_label = str(date_value)
                    else:
                        date_label = "Historical"
                    angle = math.radians((idx * len(pollutant_columns) + p_idx) * 28.0)
                    distance_km = 1.2 + min(abs(value), 80.0) * 0.04
                    delta_lat = (math.cos(angle) * distance_km) / 111.0
                    delta_lon = (math.sin(angle) * distance_km) / (111.0 * cos_lat)
                    point_lat = base_lat + delta_lat
                    point_lon = base_lon + delta_lon
                    radius_m = 900 + min(abs(value), 120.0) * 45
                    merra_records.append(
                        {
                            "lon": point_lon,
                            "lat": point_lat,
                            "pollutant": pollutant.upper(),
                            "value": value,
                            "date": date_label,
                            "tooltip": f"{date_label} — {pollutant.upper()}: {value:.1f}",
                            "color": merra_color_palette.get(pollutant, [127, 127, 127, 160]),
                            "radius": radius_m,
                        }
                    )

    layers = [base_layer]
    if station_plot_records:
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=station_plot_records,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
        )
        layers.append(station_layer)

    if merra_records:
        merra_layer = pdk.Layer(
            "ScatterplotLayer",
            data=merra_records,
            get_position="[lon, lat]",
            get_radius="radius",
            radius_units="meters",
            radius_min_pixels=2,
            radius_max_pixels=30,
            get_fill_color="color",
            pickable=True,
        )
        layers.append(merra_layer)

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8, pitch=0),
        layers=layers,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "{tooltip}"},
    )
    st.pydeck_chart(deck)
    caption_lines = [
        "Blue marker = chosen city | Orange markers = nearby OpenAQ stations"
    ]
    if merra_records:
        legend_entries = ", ".join(
            f"{key.upper()}"
            for key in ["pm25", "pm10", "no2", "so2", "co", "o3"]
            if key in merra_color_palette
        )
        caption_lines.append(
            "MERRA-2 reanalysis scatter colors by pollutant: " + legend_entries
        )
    if station_names:
        caption_lines.append("Use the 'Focus on' selector above to inspect a specific station's readings.")
    st.caption(" \n".join(caption_lines))

    def format_metrics(payload: dict[str, float | None]) -> str:
        return " | ".join(
            f"{k.upper()}: {v:.1f}"
            for k, v in payload.items()
            if v is not None
        )

    def spoon_for(payload: dict[str, float | None]) -> str:
        return describe_spoon(
            pm25=payload.get("pm25"),
            pm10=payload.get("pm10"),
            no2=payload.get("no2"),
            so2=payload.get("so2"),
            co=payload.get("co"),
            o3=payload.get("o3"),
        )

    station_count = len(station_names)
    city_label = location_label or city
    city_metrics_text = format_metrics(latest_pollutants)
    city_spoon = spoon_for(latest_pollutants)

    selected_payload: Dict[str, Optional[float]] = dict(latest_pollutants)
    info_lines: list[str]
    if highlight_station:
        candidate_payload = station_metrics.get(highlight_station) or station_metrics.get(name_lookup.get(highlight_station, ""))
        if candidate_payload:
            selected_payload = dict(candidate_payload)
            selection_label = highlight_station
            selected_metrics_text = format_metrics(selected_payload)
            selected_spoon = spoon_for(selected_payload)

            info_lines = [f"**Selection:** {selection_label}"]
            if station_count:
                info_lines[0] += f" · Nearby stations: {station_count}"
            if selected_metrics_text:
                info_lines.append(selected_metrics_text)
            if city_metrics_text:
                info_lines.append(f"{city_label}: {city_metrics_text}")
            if selected_pollutant:
                info_lines.append(f"Map pollutant: {selected_pollutant.upper()}")
            info_lines.append("")
            info_lines.append(f"{selection_label}: {selected_spoon}")
            info_lines.append(f"{city_label}: {city_spoon}")
        else:
            highlight_station = None

    if not highlight_station:
        info_lines = [f"**Selection:** {city_label}"]
        if station_count:
            info_lines[0] += f" · Nearby stations: {station_count}"
        if city_metrics_text:
            info_lines.append(city_metrics_text)
        if selected_pollutant:
            info_lines.append(f"Map pollutant: {selected_pollutant.upper()}")
        info_lines.append("")
        info_lines.append(city_spoon)

    st.info("\n".join(info_lines))

    air_context_note = " | ".join(
        part
        for part in [
            f"Focus station: {highlight_station or 'city average'}",
            f"Map pollutant: {selected_pollutant.upper()}" if selected_pollutant else None,
        ]
        if part
    )

    render_action_plan(
        section_key=f"air_{highlight_station or 'city'}",
        heading="#### Personalized Action Steps",
        city=city,
        selected_sensitive_groups=selected_sensitive_groups,
        symptom_notes=symptom_notes,
        air_summary=air_summary,
        pollen_summary=pollen_summary,
        pollutant_payload=selected_payload,
        pollen_df=None,
        context_note=air_context_note,
    )

    st.markdown("#### Live Air Snapshot")
    if pollution_info:
        aqi = pollution_info["main"]["aqi"]
        comps = pollution_info["components"]
        payload = {
            "city": city,
            "AQI": f"{aqi} ({describe_aqi(aqi)})",
            "components (µg/m³)": comps,
        }
        if wind_speed is not None:
            payload["wind_speed (m/s)"] = wind_speed
        st.json(payload)
    else:
        st.warning("No live pollution data found for this location.")

    if merra_bundle is None or force_refresh:
        with st.spinner("Building MERRA-2 daily dataset (may take a minute)..."):
            merra_bundle = fetch_city_merra(
                city,
                lat,
                lon,
                openaq_daily=openaq_df,
                data_dir="data",
                use_cache=use_cache,
                force_refresh=force_refresh,
                merra_threads=4,
                merra_verbose=False,
            )
        city_cache["merra"] = merra_bundle
        city_cache_store[cache_key] = city_cache
        merra_df = merra_bundle.get("merra_daily", pd.DataFrame())
        merged_df = merra_bundle.get("merged_daily", pd.DataFrame())
        paths.update(merra_bundle.get("paths", {}))

    openaq_path = paths.get("openaq")
    merra_path = paths.get("merra")
    merged_path = paths.get("merged")
    openaq_rows = len(openaq_df) if openaq_df is not None else 0
    merra_rows = len(merra_df) if merra_df is not None else 0
    merged_rows = len(merged_df) if merged_df is not None else 0

    st.markdown("#### Datasets")
    if openaq_path:
        if openaq_rows:
            st.success(f"💾 OpenAQ daily dataset ({openaq_rows} rows) → `{openaq_path}`")
        else:
            st.warning(f"OpenAQ daily dataset is empty → `{openaq_path}`")
    else:
        st.warning("OpenAQ daily dataset path unavailable.")

    if merra_path:
        if merra_rows:
            st.success(f"💾 MERRA-2 daily dataset ({merra_rows} rows) → `{merra_path}`")
        else:
            st.info(f"MERRA-2 daily dataset is empty → `{merra_path}`")
    else:
        st.info("MERRA-2 daily dataset path unavailable (check credentials or coverage).")

    if merged_path:
        if merged_rows:
            st.success(f"💾 Merged daily dataset ({merged_rows} rows) → `{merged_path}`")
        else:
            st.info(f"Merged daily dataset is empty → `{merged_path}`")
    elif not merged_df.empty:
        st.info("Merged dataset generated in-memory; enable caching to persist.")

    tab_labels = ["OpenAQ (Daily)"]
    if not merra_df.empty:
        tab_labels.append("MERRA-2 (Daily)")
    if not merged_df.empty:
        tab_labels.append("Merged (Daily)")

    dataset_tabs = st.tabs(tab_labels)
    tab_idx = 0
    with dataset_tabs[tab_idx]:
        if not openaq_df.empty:
            openaq_display = openaq_df.copy()
            openaq_display["date"] = openaq_display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(openaq_display, use_container_width=True)
            st.caption(f"Rows: {len(openaq_df)}")
            st.download_button(
                label="⬇️ Download OpenAQ CSV",
                data=openaq_display.to_csv(index=False).encode("utf-8"),
                file_name=(openaq_path or "openaq_daily.csv").split("/")[-1],
                mime="text/csv",
            )
        else:
            st.write("No OpenAQ data to display.")

    tab_idx += 1
    if tab_idx < len(dataset_tabs) and not merra_df.empty:
        with dataset_tabs[tab_idx]:
            merra_display = merra_df.copy()
            merra_display["date"] = merra_display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(merra_display, use_container_width=True)
            st.caption(f"Rows: {len(merra_df)}")
            st.download_button(
                label="⬇️ Download MERRA-2 CSV",
                data=merra_display.to_csv(index=False).encode("utf-8"),
                file_name=(merra_path or "merra_daily.csv").split("/")[-1],
                mime="text/csv",
            )
        tab_idx += 1

    if tab_idx < len(dataset_tabs) and not merged_df.empty:
        with dataset_tabs[tab_idx]:
            merged_display = merged_df.copy()
            merged_display["date"] = merged_display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(merged_display, use_container_width=True)
            st.caption(f"Rows: {len(merged_df)}")
            if merged_path:
                st.download_button(
                    label="⬇️ Download Merged CSV",
                    data=merged_display.to_csv(index=False).encode("utf-8"),
                    file_name=merged_path.split("/")[-1],
                    mime="text/csv",
                )

    st.markdown("#### Predictive Models")
    if not openaq_df.empty:
        if force_model_retrain or model_results is None:
            with st.spinner("Training city-specific models..."):
                try:
                    model_results = train_city_models_auto(
                        city,
                        openaq_df=openaq_df,
                        merra_df=merra_df,
                        data_dir="data",
                        use_cache=use_cache,
                        force_retrain=(force_model_retrain or not use_cache),
                        horizon=1,
                        forecast_steps=7,
                    )
                except Exception as exc:
                    st.warning(f"Could not train models: {exc}")
                    model_results = None
            if model_results:
                city_cache["model_results"] = model_results
                city_cache_store[cache_key] = city_cache
        else:
            st.info("Using cached model bundle for this city.")

        if model_results and not model_results["metrics"].empty:
            metrics_display = model_results["metrics"].copy()
            metrics_display[["rmse", "mae", "r2"]] = metrics_display[["rmse", "mae", "r2"]].applymap(
                lambda x: float(x) if pd.notna(x) else None
            )
            st.dataframe(metrics_display, use_container_width=True)

            if model_results["forecasts"]:
                forecast_tabs = st.tabs([f"Forecast: {name.upper()}" for name in model_results["forecasts"].keys()])
                for tab, (pollutant, forecast_df) in zip(forecast_tabs, model_results["forecasts"].items()):
                    with tab:
                        formatted = forecast_df.copy()
                        formatted["date"] = formatted["date"].dt.strftime("%Y-%m-%d")
                        st.dataframe(formatted, use_container_width=True)
                        model_info = model_results["regression"][pollutant]["model"]
                        total_days = len(formatted)
                        st.caption(
                            f"Predicting {pollutant.upper()} for the next {total_days} day(s) (step size: {model_info.horizon} day)."
                        )
            else:
                st.info("No pollutant forecasts could be generated.")

            clf_pred = model_results.get("classification_prediction")
            clf_report = model_results.get("classification_report")
            if clf_pred and clf_pred.get("predicted_category"):
                st.subheader("AQI Category Outlook")
                st.write(f"Target pollutant: `{clf_pred.get('pollutant', 'pm25').upper()}`")
                st.write(
                    f"Next-day AQI category prediction: **{clf_pred['predicted_category'] or 'Unavailable'}**"
                )
                if clf_report:
                    def _fmt_acc(val: float) -> str:
                        return f"{val:.2f}" if val is not None and not pd.isna(val) else "N/A"

                    st.caption(
                        (
                            f"Training accuracy: {_fmt_acc(clf_report.get('train_accuracy'))}"
                            f" | Validation accuracy: {_fmt_acc(clf_report.get('test_accuracy'))}"
                        )
                    )
            else:
                st.info("Not enough variation to train an AQI category classifier.")
        elif model_results:
            st.info("Not enough clean history to train regression models.")
        else:
            st.info("Model training skipped.")
    else:
        st.info("Insufficient OpenAQ data available for model training.")

    st.markdown("#### Daily Metrics Visualizer")
    datasets_for_plot = {}
    if not openaq_df.empty:
        datasets_for_plot["OpenAQ"] = openaq_df
    if not merra_df.empty:
        datasets_for_plot["MERRA-2"] = merra_df

    if datasets_for_plot:
        dataset_choice = st.radio(
            "Select dataset to plot",
            options=list(datasets_for_plot.keys()),
            horizontal=True,
            key=f"dataset_choice_{cache_key_str}",
        )

        plot_source = datasets_for_plot[dataset_choice].copy()
        plot_source["date"] = pd.to_datetime(plot_source["date"], errors="coerce")
        plot_source = plot_source[plot_source["date"].notna()].sort_values("date")

        ignore_cols = {"date", "source", "station", "data_source"}
        value_cols = [col for col in plot_source.columns if col not in ignore_cols]

        if value_cols:
            melted = plot_source.melt(
                id_vars="date",
                value_vars=value_cols,
                var_name="metric",
                value_name="value",
            )
            melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
            melted = melted.dropna(subset=["value"])
            if not melted.empty:
                melted["abs_value"] = melted["value"].abs()
                chart = (
                    alt.Chart(melted)
                    .mark_circle(opacity=0.7)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("value:Q", title="Measured Value"),
                        size=alt.Size(
                            "abs_value:Q",
                            scale=alt.Scale(range=[30, 800]),
                            legend=alt.Legend(title="Magnitude"),
                        ),
                        color=alt.Color(
                            "metric:N",
                            legend=alt.Legend(title=f"Metrics ({dataset_choice})"),
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("metric:N", title="Metric"),
                            alt.Tooltip("value:Q", title="Value", format=".2f"),
                        ],
                    )
                    .properties(height=360, width="container")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No numeric values available for plotting.")
        else:
            st.write("Selected dataset does not contain numeric metrics to plot.")
    else:
        st.info("No dataset available for visualization.")

elif selected_view == "Pollen Dashboard":
    pollen_bundle = load_pollen_bundle(city, lat, lon, force=force_refresh)
    pollen_df = pollen_bundle["df"]
    pollen_summary = pollen_bundle.get("summary")
    response = pollen_bundle.get("response", {})

    st.markdown("#### Pollen Forecast & Wind")

    if not pollen_df.empty:
        source_name = response.get("source", "unknown").title()
        st.write(f"Forecast source: {source_name}")
        if source_name.lower() != "google":
            st.caption("Using fallback pollen data (Google API key not applied or response unavailable).")
        if response.get("message"):
            st.caption(response["message"])

        bar_chart = (
            alt.Chart(pollen_df.dropna(subset=["index"]))
            .mark_bar(size=40)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("index:Q", title="Pollen Index"),
                color=alt.Color("pollen_type:N", legend=alt.Legend(title="Allergen")),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("pollen_type:N", title="Allergen"),
                    alt.Tooltip("index:Q", title="Index"),
                    alt.Tooltip("index_description:N", title="Category"),
                ],
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)

        map_data = pollen_df.copy()
        map_data["lat"] = float(lat)
        map_data["lon"] = float(lon)
        map_data = map_data.dropna(subset=["index"])

        min_idx = map_data["index"].min()
        max_idx = map_data["index"].max()
        if pd.isna(min_idx) or pd.isna(max_idx) or min_idx == max_idx:
            min_idx, max_idx = 0.0, max(float(map_data["index"].max() or 0), 1.0)

        map_data["index"] = map_data["index"].astype(float)
        map_data["date"] = pd.to_datetime(map_data["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        color_palette = {
            "tree": (34, 139, 34),
            "grass": (60, 179, 113),
            "weed": (154, 205, 50),
        }

        def pollen_color(name: str) -> tuple[int, int, int]:
            base = color_palette.get((name or "").lower())
            if base:
                return base
            idx = abs(hash(name or "unknown")) % 200 + 30
            return (idx, (idx * 2) % 255, (idx * 3) % 255)

        map_data["color"] = map_data["pollen_type"].apply(pollen_color)
        map_data["color_r"] = map_data["color"].apply(lambda c: c[0])
        map_data["color_g"] = map_data["color"].apply(lambda c: c[1])
        map_data["color_b"] = map_data["color"].apply(lambda c: c[2])
        map_data["date_str"] = map_data["date"].astype(str)
        map_data["radius"] = np.interp(
            map_data["index"],
            [min_idx, max_idx],
            [1500, 9000],
        )
        map_data["opacity"] = np.interp(
            map_data["index"],
            [min_idx, max_idx],
            [60, 220],
        )
        map_data = map_data.drop(columns=["color"], errors="ignore")
        map_records = map_data.to_dict("records")

        map_variant = None
        if not map_records:
            st.info("No pollen map data available for this location yet.")
        else:
            map_variant = st.radio(
                "Map style",
                options=["Scatter", "Column"],
                horizontal=True,
                key=f"pollen_map_variant_{cache_key_str}",
            )

            if map_variant == "Scatter":
                pollen_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_records,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="[color_r, color_g, color_b, opacity]",
                    radius_min_pixels=2,
                    radius_max_pixels=25,
                    pickable=True,
                )
            else:
                pollen_layer = pdk.Layer(
                    "ColumnLayer",
                    data=map_records,
                    get_position="[lon, lat]",
                    get_elevation="index",
                    elevation_scale=300,
                    radius=2500,
                    get_fill_color="[color_r, color_g, color_b, opacity]",
                    pickable=True,
                )

            pollen_deck = pdk.Deck(
                layers=[pollen_layer],
                initial_view_state=pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=7.5),
                tooltip={
                    "html": "<b>{pollen_type}</b>: {index}<br/>Category: {index_description}<br/>Date: {date_str}",
                    "style": {"color": "white"},
                },
            )
            st.pydeck_chart(pollen_deck)

        if wind_speed is not None:
            st.caption(f"Current wind speed: {wind_speed} m/s (from live AQ data)")
        elif pollen_df.empty:
            st.info("No pollen forecast data available (showing fallback).")
            pollen_summary = response.get("message") if isinstance(response, dict) else None

    plan_context_note = " | ".join(
        part
        for part in [
            f"Pollen data source: {response.get('source', 'unknown').title()}" if isinstance(response, dict) else None,
            f"Map style: {map_variant}" if map_variant else None,
        ]
        if part
    )

    render_action_plan(
        section_key="pollen_forecast",
        heading="#### Personalized Action Steps",
        city=city,
        selected_sensitive_groups=selected_sensitive_groups,
        symptom_notes=symptom_notes,
        air_summary=air_summary,
        pollen_summary=pollen_summary,
        pollutant_payload=latest_pollutants,
        pollen_df=pollen_df,
        context_note=plan_context_note,
    )

elif selected_view == "Social Media":
    pollen_bundle = load_pollen_bundle(city, lat, lon, force=force_refresh)
    pollen_summary = pollen_bundle.get("summary")

    st.markdown("#### Share & Stay Connected")
    st.write(
        "Craft quick updates for community groups or social feeds using the latest air, pollen, and action insights."
    )

    default_caption = ""
    if city:
        default_caption = (
            f"Air quality check for {city}: {air_summary or 'Monitoring in progress.'} "
            f"Pollen outlook: {pollen_summary or 'Awaiting forecast.'} #AirQuality #Pollen"
        )

    share_caption = st.text_area(
        "Editable caption (copy & paste to share)",
        value=default_caption,
        key=f"share_caption_{cache_key_str}",
    )

    col_share, col_groups = st.columns(2)
    with col_share:
        st.markdown("**Suggested Hashtags**")
        st.code("#AirQuality  #PollenWatch  #HealthFirst")

    with col_groups:
        st.markdown("**Community Channels**")
        st.write("• Neighborhood Slack or Discord\n• Parent-teacher groups\n• Local health department mentions")

    st.markdown("---")
    st.write(
        "Need deeper pollen analytics? Jump to the **Pollen Dashboard** from the sidebar navigation."
    )

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
from streamlit.components.v1 import html as components_html
import re
import json

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
from lib.google_air_api import (
    fetch_current_air_quality,
    google_air_heatmap_tile_url,
    SUPPORTED_HEATMAP_TYPES,
    fetch_heatmap_tile_for_location,
)


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

AQI_SEVERITY_STEPS = [
    {"label": "Good", "color": "#2ecc71"},
    {"label": "Moderate", "color": "#f1c40f"},
    {"label": "Sensitive Groups", "color": "#e67e22"},
    {"label": "Unhealthy", "color": "#e74c3c"},
    {"label": "Very Unhealthy", "color": "#8e44ad"},
    {"label": "Hazardous", "color": "#6c3483"},
]

POLLUTANT_HEALTH_FOCUS = {
    "pm25": "people with asthma or heart disease",
    "pm10": "respiratory sensitivities and older adults",
    "o3": "asthma and active outdoor exercisers",
    "no2": "those with lung conditions and children",
    "so2": "people with asthma and chronic bronchitis",
    "co": "cardiovascular conditions and pregnant individuals",
    "nh3": "sensitive respiratory tract",
}


RAINBOW_GRADIENT = "linear-gradient(90deg, #ff6b6b, #f7d794, #1dd1a1, #54a0ff, #5f27cd)"
BOLD_PATTERN = re.compile(r"\*\*(.*?)\*\*")


def ensure_custom_styles() -> None:
    """Inject reusable CSS styles once per session."""

    flag_key = "_custom_style_injected"
    if st.session_state.get(flag_key):
        return
    st.session_state[flag_key] = True
    st.markdown(
        f"""
        <style>
            .rainbow-heading {{
                font-weight: 800;
                background-image: {RAINBOW_GRADIENT};
                -webkit-background-clip: text;
                color: transparent;
            }}
            .rainbow-divider {{
                background-image: {RAINBOW_GRADIENT};
                border-radius: 999px;
            }}
            .section-gap {{
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
            }}
            .frost-card {{
                background: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(6px);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.45);
                box-shadow: 0 10px 24px rgba(17, 17, 17, 0.08);
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
                gap: 0.45rem;
            }}
            .info-spacer {{
                height: 0.6rem;
            }}
            .spoon-highlight {{
                color: #ffffff; /* improved contrast */
                text-shadow: 0 1px 0 rgba(0,0,0,0.25);
                background: rgba(255, 105, 180, 0.85);
                padding: 0 6px;
                border-radius: 999px;
                font-weight: 700;
            }}
            .spoon-highlight-line {{
                display: inline-block;
                color: #ffffff; /* improved contrast */
                text-shadow: 0 1px 0 rgba(0,0,0,0.28);
                background: linear-gradient(120deg, rgba(255, 105, 180, 0.95), rgba(255, 20, 147, 0.9));
                padding: 6px 12px;
                border-radius: 14px;
                font-weight: 600;
                box-shadow: 0 8px 18px rgba(255, 105, 180, 0.18);
            }}
            .mini-caption {{
                font-size: 0.85rem;
                color: #4a4a4a;
                margin-top: 0.4rem;
            }}
            .mini-caption strong {{
                color: #2c2c2c;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def rainbow_heading(text: str, *, level: int = 2, emoji: Optional[str] = None) -> None:
    """Display a gradient heading with a divider."""

    ensure_custom_styles()
    size_map = {1: "2rem", 2: "1.55rem", 3: "1.2rem"}
    bar_height = {1: "5px", 2: "3.5px", 3: "2.5px"}
    label = f"{emoji} {text}" if emoji else text
    st.markdown(
        f"""
        <div class="section-gap">
            <div class="rainbow-heading" style="font-size: {size_map.get(level, '1.45rem')};">
                {label}
            </div>
            <div class="rainbow-divider" style="height: {bar_height.get(level, '3px')};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def highlight_spoon_text(text: str) -> str:
    """Wrap the 'Creamy stew' phrase with a vivid highlight."""

    highlight_inner = "<span class=\"spoon-highlight\">Creamy stew</span>"
    target = "**Creamy stew**"
    if target in text:
        replaced = text.replace(target, highlight_inner)
        return f"<span class=\"spoon-highlight-line\">{replaced}</span>"
    if "Creamy stew" in text:
        replaced = text.replace("Creamy stew", highlight_inner)
        return f"<span class=\"spoon-highlight-line\">{replaced}</span>"
    return text


def markdown_to_html(text: str) -> str:
    """Convert lightweight markdown snippets into inline HTML."""

    text = highlight_spoon_text(text)

    def _bold_repl(match: re.Match[str]) -> str:
        return f"<strong>{match.group(1)}</strong>"

    text = BOLD_PATTERN.sub(_bold_repl, text)
    return text


def render_info_panel(lines: Sequence[str]) -> None:
    """Render a frosted info card with optional spacing and styling."""

    ensure_custom_styles()
    segments: list[str] = []
    for entry in lines:
        if not entry:
            segments.append("<div class=\"info-spacer\"></div>")
        else:
            segments.append(f"<div>{markdown_to_html(entry)}</div>")
    info_html = "".join(segments)
    st.markdown(f"<div class=\"frost-card\">{info_html}</div>", unsafe_allow_html=True)


def render_mini_caption(text: str) -> None:
    """Display a styled, compact caption with optional highlights."""

    ensure_custom_styles()
    caption_html = markdown_to_html(text)
    st.markdown(f"<div class=\"mini-caption\">{caption_html}</div>", unsafe_allow_html=True)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(max(0, min(255, round(val)))):02x}" for val in rgb)


def _gradient_between(start: str, end: str, steps: int) -> list[str]:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    colors = []
    for idx in range(steps):
        t = idx / max(steps - 1, 1)
        mixed = tuple(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * t for i in range(3))
        colors.append(_rgb_to_hex(mixed))
    return colors


def _severity_gradient(target_label: str, steps: int = 90) -> list[str]:
    labels = [step["label"] for step in AQI_SEVERITY_STEPS]
    index = labels.index(target_label) if target_label in labels else len(labels) - 1
    palette = [AQI_SEVERITY_STEPS[i]["color"] for i in range(index + 1)]
    if len(palette) == 1:
        return _gradient_between("#2ecc71", palette[-1], steps)

    segment_steps = max(1, steps // (len(palette) - 1))
    colors: list[str] = []
    for i in range(len(palette) - 1):
        segment = _gradient_between(palette[i], palette[i + 1], segment_steps)
        if i > 0 and segment:
            segment = segment[1:]
        colors.extend(segment)
    while len(colors) < steps:
        colors.append(palette[-1])
    return colors


def _pollutant_severity(pollutant: str, value: Optional[float]) -> int:
    thresholds = {
        "pm25": [12, 35.4, 55.4, 150.4, 250.4],
        "pm10": [54, 154, 254, 354, 424],
        "o3": [0.054, 0.07, 0.085, 0.105, 0.2],
        "no2": [53, 100, 360, 649, 1249],
        "so2": [35, 75, 185, 304, 604],
        "co": [4.4, 9.4, 12.4, 15.4, 30.4],
        "nh3": [25, 50, 100, 200, 300],
    }
    if value is None:
        return 0
    series = thresholds.get(pollutant.lower())
    if not series:
        return 0
    for idx, limit in enumerate(series):
        if value <= limit:
            return idx
    return len(AQI_SEVERITY_STEPS) - 1


def _adjust_severity_for_user(level: int, has_sensitive_groups: bool, symptom_notes: str) -> int:
    adjusted = level
    symptom_keywords = ["asthma", "bronch", "copd", "pregnan", "heart", "allerg", "sensitive", "lung"]
    if has_sensitive_groups:
        adjusted = min(adjusted + 1, len(AQI_SEVERITY_STEPS) - 1)
    if symptom_notes and any(re.search(word, symptom_notes, re.IGNORECASE) for word in symptom_keywords):
        adjusted = min(adjusted + 1, len(AQI_SEVERITY_STEPS) - 1)
    return adjusted


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
    *,
    heading_level: int = 3,
    heading_emoji: Optional[str] = "🎯",
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

    rainbow_heading(heading, level=heading_level, emoji=heading_emoji)
    st.markdown(plan.get("content", "Action plan unavailable."))
    render_mini_caption(f"Source: {plan.get('model', 'unknown')}")

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
ensure_custom_styles()
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
    rainbow_heading("Choose your city", level=1, emoji="📍")
    city_input = st.text_input("City name:", value=default_city).strip()
    if city_input:
        st.session_state["city_input"] = city_input
    city = st.session_state.get("city_input", default_city)
else:
    city = st.session_state.get("city_input", default_city)
    rainbow_heading(f"City: {city}", level=1, emoji="📍")

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
        placeholder="I'm allergic to ragweed on dry days...",
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
rainbow_heading("Selected Coordinates and Air Quality", level=2, emoji="✅")

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

google_air_obs = fetch_current_air_quality(float(lat), float(lon))
google_heatmap_available = google_air_heatmap_tile_url("US_AQI") is not None
default_google_heatmap = "US_AQI"
air_quality_heatmap_available = google_air_heatmap_tile_url(default_google_heatmap) is not None
if google_air_obs and google_air_obs.dominant_pollutant:
    candidate = google_air_obs.dominant_pollutant.upper().replace("2.5", "25").replace("2_5", "25")
    if candidate in SUPPORTED_HEATMAP_TYPES and google_air_heatmap_tile_url(candidate):
        default_google_heatmap = candidate
    else:
        fallback_candidates = ["US_AQI", "EU_AQI", "PM25"]
        for fallback in fallback_candidates:
            if google_air_heatmap_tile_url(fallback):
                default_google_heatmap = fallback
                break
google_heatmap_available = google_air_heatmap_tile_url(default_google_heatmap) is not None

city_cache_store = st.session_state.setdefault("city_data_cache", {})
merra_cache_store = st.session_state.setdefault("merra_city_cache", {})
cache_key = (
    city.lower(),
    radius,
    start_year,
    months_per_station,
)
cache_key_str = "_".join(str(part) for part in cache_key)
city_cache = city_cache_store.get(cache_key, {})
merra_signature = (
    city.lower(),
    round(float(lat), 4),
    round(float(lon), 4),
)

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
merra_bundle = city_cache.get("merra") or merra_cache_store.get(merra_signature)
if merra_bundle:
    merra_df = merra_bundle.get("merra_daily", pd.DataFrame())
    merged_df = merra_bundle.get("merged_daily", pd.DataFrame())
    paths.update(merra_bundle.get("paths", {}))
    city_cache["merra"] = merra_bundle
    merra_cache_store[merra_signature] = merra_bundle
    city_cache_store[cache_key] = city_cache

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
    rainbow_heading("City Overview", level=2, emoji="🏙️")

    station_df = pd.DataFrame(stations)
    station_layer_data = pd.DataFrame(columns=["lon", "lat", "tooltip", "radius", "color", "name"])
    station_metrics: dict[str, dict[str, float | None]] = {}
    name_lookup: dict[str, str] = {}
    metric_cols = list(pollutant_cols)

    google_heatmap_type = default_google_heatmap
    google_heatmap_tile = None
    heatmap_zoom = 5
    if google_heatmap_available:
        if st.checkbox(
            "Show Google AQ heatmap", value=True, key=f"google_aq_heatmap_toggle_{cache_key_str}"
        ):
            heatmap_zoom = int(
                st.slider(
                    "Google AQ heatmap zoom",
                    min_value=0,
                    max_value=12,
                    value=5,
                    help="Lower values show wider coverage; higher values zoom into street level.",
                    key=f"google_aq_heatmap_zoom_{cache_key_str}",
                )
            )
            try:
                google_heatmap_tile = fetch_heatmap_tile_for_location(
                    map_type=google_heatmap_type,
                    latitude=float(lat),
                    longitude=float(lon),
                    zoom=heatmap_zoom,
                )
            except ValueError as exc:
                st.warning(f"Google heatmap unavailable: {exc}")
            if google_heatmap_tile and google_heatmap_tile.status != 200:
                st.warning(
                    (
                        "Google air quality tile request failed "
                        f"(status {google_heatmap_tile.status})."
                    )
                )

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

    if google_heatmap_tile and google_heatmap_tile.image_bytes:
        label = SUPPORTED_HEATMAP_TYPES.get(google_heatmap_type, google_heatmap_type)
        st.image(
            google_heatmap_tile.image_bytes,
            caption=(
                f"Google {label} heatmap tile · zoom {google_heatmap_tile.zoom} "
                f"(x={google_heatmap_tile.x}, y={google_heatmap_tile.y})"
            ),
            width=360,
        )

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=8, pitch=0),
        layers=layers,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "{tooltip}"},
    )
    st.pydeck_chart(deck)
    caption_lines = [
        "Blue marker = chosen city",
        "Orange markers = nearby OpenAQ stations",
    ]
    if google_heatmap_tile and google_heatmap_tile.image_bytes:
        label = SUPPORTED_HEATMAP_TYPES.get(google_heatmap_type, google_heatmap_type)
        caption_lines.append(f"Google AQ heatmap: {label}")
        caption_lines.append(
            "Heatmap colors (EPA AQI): Green=Good, Yellow=Moderate, Orange=Unhealthy for Sensitive Groups, "
            "Red=Unhealthy, Purple=Very Unhealthy, Maroon=Hazardous"
        )
    # Build a small visual legend (colored swatches for markers and MERRA pollutants)
    merra_legend_html = ""
    if merra_records:
        swatches = []
        for key in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
            if key in merra_color_palette:
                rgba = merra_color_palette[key]
                # ensure we have 4 values (r,g,b,a)
                r, g, b, a = (rgba + [160])[:4]
                color_css = f"rgba({r},{g},{b},{a/255:.2f})"
                label = key.upper()
                swatches.append(
                    f"<span style='display:inline-flex;align-items:center;margin-right:10px;'>"
                    f"<span style='width:14px;height:14px;border-radius:3px;background:{color_css};display:inline-block;margin-right:6px;'></span>"
                    f"<small style='font-size:12px;color:#333'>{label}</small></span>"
                )
        if swatches:
            merra_legend_html = "".join(swatches)
        if station_names:
                caption_lines.append(
                        "Use the 'Focus on' selector above to inspect a specific station's readings."
                )

        # Render a compact legend (HTML) and a short caption below it
        legend_html = """
        <div style="display:flex;flex-direction:row;gap:14px;align-items:center;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:12px;height:12px;border-radius:6px;background:blue;display:inline-block;"></span>
                <small style="color:#333">Chosen city</small>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:12px;height:12px;border-radius:6px;background:orange;display:inline-block;"></span>
                <small style="color:#333">Nearby OpenAQ stations</small>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">"""

        if merra_legend_html:
                legend_html += merra_legend_html

        legend_html += "</div></div>"

        st.markdown(legend_html, unsafe_allow_html=True)
        st.caption("\n\n".join(caption_lines))

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

    render_info_panel(info_lines)

    air_context_note = " | ".join(
        part
        for part in [
            f"Focus station: {highlight_station or 'city average'}",
            f"Map pollutant: {selected_pollutant.upper()}" if selected_pollutant else None,
        ]
        if part
    )

    # add a small spacer line before the action steps
    st.write("")

    render_action_plan(
        section_key=f"air_{highlight_station or 'city'}",
        heading="Pollution Action Steps",
        city=city,
        selected_sensitive_groups=selected_sensitive_groups,
        symptom_notes=symptom_notes,
        air_summary=air_summary,
        pollen_summary=pollen_summary,
        pollutant_payload=selected_payload,
        pollen_df=None,
        context_note=air_context_note,
        heading_level=3,
        heading_emoji="🪄",
    )

    # small spacer before the Live Air Snapshot section
    st.write("")

    rainbow_heading("Live Air Snapshot", level=3, emoji="🌤️")
    if pollution_info:
        aqi = pollution_info["main"]["aqi"]
        comps = pollution_info["components"]
        summary_cards: list[str] = []
        for key, value in comps.items():
            label = key.replace("pm2_5", "PM2.5").replace("pm10", "PM10").upper()
            summary_cards.append(f"{label}: {value}")

        card_html = f"""
<div style='display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;'>
  <div style='padding:14px 18px;border-radius:12px;background:#f5f7fb;'>
    <div style='font-weight:700;color:#2c3e50;font-size:14px;'>City</div>
    <div style='font-size:16px;color:#34495e;'>{city}</div>
  </div>
  <div style='padding:14px 18px;border-radius:12px;background:#f5f7fb;'>
    <div style='font-weight:700;color:#2c3e50;font-size:14px;'>AQI (EPA scale)</div>
    <div style='font-size:16px;color:#34495e;'>{aqi} ({describe_aqi(aqi)})</div>
  </div>
  {f"<div style='padding:14px 18px;border-radius:12px;background:#f5f7fb;'><div style='font-weight:700;color:#2c3e50;font-size:14px;'>Wind speed</div><div style='font-size:16px;color:#34495e;'>{wind_speed:.1f} m/s</div></div>" if wind_speed is not None else ''}
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

        components_df = pd.DataFrame([
            {"pollutant": k.lower(), "value": v}
            for k, v in comps.items()
            if isinstance(v, (int, float))
        ])
        if not components_df.empty:
            has_sensitive_groups = len(selected_sensitive_groups) > 0
            components_df["severity"] = components_df.apply(
                lambda row: _adjust_severity_for_user(
                    _pollutant_severity(row["pollutant"], row["value"]),
                    has_sensitive_groups,
                    symptom_notes or "",
                ),
                axis=1,
            )
            components_df["color"] = components_df["severity"].apply(
                lambda idx: AQI_SEVERITY_STEPS[idx]["color"]
            )
            components_df["label"] = components_df["severity"].apply(
                lambda idx: AQI_SEVERITY_STEPS[idx]["label"]
            )

            display_cards = []
            for _, row in components_df.iterrows():
                display_name = row["pollutant"].replace("pm2_5", "PM2.5").replace("pm10", "PM10").upper()
                display_cards.append(
                    f"<div style='flex:1;min-width:160px;padding:12px 16px;border-radius:12px;background:rgba(236, 240, 241, 0.55);border-left:6px solid {row['color']};'>"
                    f"<div style='font-size:13px;color:#2c3e50;font-weight:700;'>{display_name}</div>"
                    f"<div style='font-size:18px;color:{row['color']};font-weight:700;'>{row['value']:.2f}</div>"
                    f"<div style='font-size:12px;color:#34495e;'>{row['label']}</div></div>"
                )

            if display_cards:
                st.markdown(
                    "<div style='display:flex;flex-wrap:wrap;gap:14px;margin-bottom:16px;'>"
                    + "".join(display_cards)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            severity_legend = "".join(
                f"<div style='flex:1;min-width:150px;padding:10px;border-radius:8px;background:{step['color']};color:#fff;text-align:center;font-weight:600;'>"
                f"{step['label']}"
                "</div>"
                for step in AQI_SEVERITY_STEPS
            )
            st.markdown(
                "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;'>"
                + severity_legend
                + "</div>",
                unsafe_allow_html=True,
            )

            severity_domain = [step["label"] for step in AQI_SEVERITY_STEPS]
            severity_range = [step["color"] for step in AQI_SEVERITY_STEPS]

            history_sources: list[pd.DataFrame] = []
            if openaq_df is not None and not openaq_df.empty:
                openaq_hist = openaq_df.copy()
                if "date" in openaq_hist.columns:
                    openaq_hist["date"] = pd.to_datetime(openaq_hist["date"], errors="coerce")
                    openaq_hist = openaq_hist[openaq_hist["date"].notna()]
                history_sources.append(openaq_hist)
            if merra_df is not None and not merra_df.empty:
                merra_hist = merra_df.copy()
                if "date" in merra_hist.columns:
                    merra_hist["date"] = pd.to_datetime(merra_hist["date"], errors="coerce")
                    merra_hist = merra_hist[merra_hist["date"].notna()]
                history_sources.append(merra_hist)

            summary_records: list[dict[str, float | str]] = []
            if history_sources:
                for poll in components_df["pollutant"].unique():
                    search_names = {
                        poll,
                        poll.replace("pm2_5", "pm25"),
                        poll.replace("pm25", "pm2_5"),
                        poll.upper(),
                        poll.lower(),
                    }
                    collected: list[pd.Series] = []
                    for source in history_sources:
                        frame = source
                        if "date" in frame.columns:
                            frame = frame.sort_values("date").tail(180)
                        for candidate in search_names:
                            if candidate in frame.columns:
                                series = pd.to_numeric(frame[candidate], errors="coerce").dropna()
                                if not series.empty:
                                    collected.append(series)
                                break
                    if collected:
                        combined = pd.concat(collected, ignore_index=True).sort_values()
                        if not combined.empty:
                            summary_records.append(
                                {
                                    "pollutant": poll,
                                    "min": float(combined.min()),
                                    "q1": float(combined.quantile(0.25)),
                                    "median": float(combined.quantile(0.5)),
                                    "q3": float(combined.quantile(0.75)),
                                    "max": float(combined.max()),
                                }
                            )

            summary_df = pd.DataFrame(summary_records)
            components_today = components_df.set_index("pollutant")
            if not summary_df.empty:
                summary_df = summary_df.merge(
                    components_today[["value", "label", "severity", "color"]],
                    left_on="pollutant",
                    right_index=True,
                    how="left",
                )

                if not summary_df.empty:
                    summary_plot = summary_df.copy()
                    summary_plot["pollutant_label"] = summary_plot["pollutant"].str.upper()

                    for _, row in summary_plot.iterrows():
                        min_v, max_v = row["min"], row["max"]
                        if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
                            continue
                        pollutant_label = str(row["pollutant_label"])

                        gradient_colors = _severity_gradient(row.get("label", "Good"), steps=160)
                        values = np.linspace(min_v, max_v, len(gradient_colors) + 1)
                        gradient_df = pd.DataFrame(
                            {
                                "start": values[:-1],
                                "end": values[1:],
                                "color": gradient_colors,
                                "pollutant": [pollutant_label] * len(gradient_colors),
                            }
                        )

                        stats_df = pd.DataFrame(
                            {
                                "pollutant": [row["pollutant_label"]],
                                "min": [min_v],
                                "q1": [row["q1"]],
                                "median": [row["median"]],
                                "q3": [row["q3"]],
                                "max": [max_v],
                                "value": [row["value"]],
                                "label": [row["label"]],
                            }
                        )

                        gradient_layer = (
                            alt.Chart(gradient_df)
                            .mark_rect(opacity=0.75)
                            .encode(
                                x=alt.X(
                                    "start:Q",
                                    title="Concentration",
                                    scale=alt.Scale(domain=[min_v, max_v]),
                                ),
                                x2="end:Q",
                                y=alt.Y("pollutant:N", axis=None),
                                color=alt.Color("color:N", scale=None, legend=None),
                            )
                        )

                        whisker_layer = (
                            alt.Chart(stats_df)
                            .mark_rule(color="#7f8c8d")
                            .encode(x="min:Q", x2="max:Q", y="pollutant:N")
                        )

                        box_layer = (
                            alt.Chart(stats_df)
                            .mark_bar(size=28, color="rgba(127,140,141,0.25)")
                            .encode(x="q1:Q", x2="q3:Q", y="pollutant:N")
                        )

                        median_layer = (
                            alt.Chart(stats_df)
                            .mark_tick(color="#2c3e50", thickness=3, size=32)
                            .encode(x="median:Q", y="pollutant:N")
                        )

                        point_layer = (
                            alt.Chart(stats_df)
                            .mark_point(shape="diamond", size=200, stroke="white", strokeWidth=1.5)
                            .encode(
                                x="value:Q",
                                y="pollutant:N",
                                color=alt.Color(
                                    "label:N",
                                    scale=alt.Scale(domain=severity_domain, range=severity_range),
                                    legend=alt.Legend(title="Today's severity"),
                                ),
                                tooltip=[
                                    alt.Tooltip("pollutant:N", title="Pollutant"),
                                    alt.Tooltip("value:Q", title="Today", format=".2f"),
                                    alt.Tooltip("min:Q", title="Min", format=".2f"),
                                    alt.Tooltip("median:Q", title="Median", format=".2f"),
                                    alt.Tooltip("max:Q", title="Max", format=".2f"),
                                    alt.Tooltip("label:N", title="Severity"),
                                ],
                            )
                        )

                        chart = (
                            gradient_layer
                            + whisker_layer
                            + box_layer
                            + median_layer
                            + point_layer
                        ).properties(width=400, height=160, title=pollutant_label)

                        st.altair_chart(chart, use_container_width=True)

            worst = components_df.sort_values("severity", ascending=False).iloc[0]
            worst_pollutant = worst["pollutant"].upper()
            worst_label = worst["label"]
            audience = POLLUTANT_HEALTH_FOCUS.get(worst["pollutant"], "sensitive individuals")
            st.info(
                f"Today's highest concern: **{worst_pollutant}** at {worst['value']:.2f}. "
                f"Severity level: {worst_label}. Pay extra attention if you are {audience}."
            )
    else:
        st.warning("No live pollution data found for this location.")

    if google_air_obs:
        dom = (google_air_obs.dominant_pollutant or "unknown").upper()
        category = google_air_obs.category or "Unknown"
        aq_value = google_air_obs.aqi
        rec = google_air_obs.health_recommendations
        st.caption(
            f"Google Air Quality · Dominant pollutant: {dom} | AQI: {aq_value or 'N/A'} ({category})"
        )
        if rec:
            st.write(f"**Recommendation:** {rec}")

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
        merra_cache_store[merra_signature] = merra_bundle
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

    with st.expander("Advanced data downloads & viewing", expanded=False):
        preview_tabs = st.tabs(["OpenAQ (Daily)", "MERRA-2 (Daily)"])
        with preview_tabs[0]:
            if not openaq_df.empty:
                openaq_tail = openaq_df.sort_values("date").tail(90)
                value_cols = [
                    col
                    for col in openaq_tail.columns
                    if col not in {"date", "station", "data_source", "source"}
                ]
                if value_cols:
                    melted = openaq_tail.melt(
                        id_vars="date",
                        value_vars=value_cols,
                        var_name="metric",
                        value_name="value",
                    ).dropna()
                    chart = (
                        alt.Chart(melted)
                        .mark_line(opacity=0.8)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("value:Q", title="Value"),
                            color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)
                st.dataframe(openaq_tail.assign(date=openaq_tail["date"].dt.strftime("%Y-%m-%d")))
                st.download_button(
                    label="⬇️ Download OpenAQ CSV",
                    data=openaq_df.to_csv(index=False).encode("utf-8"),
                    file_name=(openaq_path or "openaq_daily.csv").split("/")[-1],
                    mime="text/csv",
                )
            else:
                st.write("No OpenAQ data available.")

        with preview_tabs[1]:
            if not merra_df.empty:
                merra_tail = merra_df.sort_values("date").tail(90).copy()
                if "date" in merra_tail.columns:
                    merra_tail["date"] = pd.to_datetime(merra_tail["date"], errors="coerce")
                    merra_tail = merra_tail[merra_tail["date"].notna()]
                value_cols = [
                    col
                    for col in merra_tail.columns
                    if col not in {"date", "city", "lat", "lon"}
                ]
                if value_cols:
                    melted = merra_tail.melt(
                        id_vars="date",
                        value_vars=value_cols,
                        var_name="metric",
                        value_name="value",
                    ).dropna()
                    chart = (
                        alt.Chart(melted)
                        .mark_line(opacity=0.8)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("value:Q", title="Value"),
                            color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)
                if "date" in merra_tail.columns:
                    display_df = merra_tail.assign(date=merra_tail["date"].dt.strftime("%Y-%m-%d"))
                else:
                    display_df = merra_tail
                st.dataframe(display_df)
                st.download_button(
                    label="⬇️ Download MERRA-2 CSV",
                    data=merra_df.to_csv(index=False).encode("utf-8"),
                    file_name=(merra_path or "merra_daily.csv").split("/")[-1],
                    mime="text/csv",
                )
            else:
                st.write("No MERRA-2 data available.")

        st.markdown("---")
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

        if merged_rows and merged_path:
            st.info(
                "Merged dataset created for experimentation. Timeseries alignment still under review — "
                "prefer OpenAQ/MERRA exports above for now."
            )

    # Lightweight preview tabs retain compatibility with earlier table view.

    rainbow_heading("Predictive Models", level=3, emoji="📈")
    if not openaq_df.empty:
        status_messages: list[str] = []
        forecast_tables: list[tuple[str, pd.DataFrame, int]] = []

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
            status_messages.append("Using cached model bundle for this city.")

        metrics_display = pd.DataFrame()
        if model_results and not model_results.get("metrics", pd.DataFrame()).empty:
            metrics_display = model_results["metrics"].copy()
            metrics_display[["rmse", "mae", "r2"]] = metrics_display[["rmse", "mae", "r2"]].applymap(
                lambda x: float(x) if pd.notna(x) else None
            )

        if model_results and model_results.get("forecasts"):
            for pollutant, forecast_df in model_results["forecasts"].items():
                formatted = forecast_df.copy()
                formatted["date"] = formatted["date"].dt.strftime("%Y-%m-%d")
                forecast_tables.append((pollutant, formatted, len(formatted)))

                history_df = model_results.get("history", pd.DataFrame())
                history_tail = (
                    history_df[["date", pollutant]]
                    .dropna()
                    .sort_values("date")
                    .tail(90)
                    .rename(columns={pollutant: "value"})
                ) if not history_df.empty and pollutant in history_df.columns else pd.DataFrame()

                plot_forecast = forecast_df.copy()
                plot_forecast["lower"] = plot_forecast["predicted_value"]
                plot_forecast["upper"] = plot_forecast["predicted_value"]
                report = model_results["regression"][pollutant]["report"]
                rmse = getattr(report, "rmse", 0.0) or 0.0
                if rmse:
                    plot_forecast["lower"] = plot_forecast["predicted_value"] - rmse
                    plot_forecast["upper"] = plot_forecast["predicted_value"] + rmse

                base = alt.Chart(history_tail).mark_line(color="#1f77b4").encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title=f"{pollutant.upper()}"),
                    tooltip=["date:T", "value:Q"],
                )
                forecast_line = alt.Chart(plot_forecast).mark_line(color="#ff7f0e").encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("predicted_value:Q", title=f"{pollutant.upper()}"),
                    tooltip=["date:T", alt.Tooltip("predicted_value:Q", title="Forecast")],
                )
                error_band = alt.Chart(plot_forecast).mark_area(opacity=0.18, color="#ff7f0e").encode(
                    x="date:T",
                    y="lower:Q",
                    y2="upper:Q",
                )
                forecast_points = alt.Chart(plot_forecast).mark_point(color="#ff7f0e").encode(
                    x="date:T",
                    y="predicted_value:Q",
                )

                chart = (base + error_band + forecast_line + forecast_points).properties(height=280)
                rainbow_heading(f"{pollutant.upper()} forecast", level=3, emoji="🔮")
                st.altair_chart(chart, use_container_width=True)

                sorted_forecast = plot_forecast.sort_values("date") if not plot_forecast.empty else plot_forecast
                if not sorted_forecast.empty and "predicted_value" in sorted_forecast.columns:
                    next_row = sorted_forecast.iloc[0]
                    next_value = next_row.get("predicted_value")
                    if next_value is not None and not pd.isna(next_value):
                        spoon_key = pollutant.lower().replace("2.5", "25").replace("2_5", "25")
                        if spoon_key in {"pm25", "pm10", "no2", "so2", "co", "o3"}:
                            spoon_text = describe_spoon(**{spoon_key: float(next_value)})
                            next_date = next_row.get("date")
                            if isinstance(next_date, pd.Timestamp):
                                next_label = next_date.strftime("%Y-%m-%d")
                            elif isinstance(next_date, datetime):
                                next_label = next_date.strftime("%Y-%m-%d")
                            else:
                                next_label = str(next_date) if next_date is not None else "Next period"
                            render_mini_caption(
                                f"{pollutant.upper()} forecast for {next_label}: {float(next_value):.1f} ug/m3. {spoon_text}"
                            )
                            # Also render a compact 'soup spoon' forecast for tomorrow + next week
                            try:
                                # Use the sorted forecast dataframe to list the next 7 days
                                week_rows = sorted_forecast.head(7) if not sorted_forecast.empty else pd.DataFrame()
                                if not week_rows.empty:
                                    st.markdown("**Spoon forecast — Tomorrow & next 7 days**")
                                    # Build rows for a compact table: date, short texture, value
                                    rows_list = []
                                    forecasts_all = model_results.get("forecasts") if model_results else None
                                    for _, wk_row in week_rows.iterrows():
                                        wk_date = wk_row.get("date")
                                        wk_val = wk_row.get("predicted_value")
                                        pm25_val = None
                                        if forecasts_all and "pm25" in forecasts_all:
                                            try:
                                                pm_df = forecasts_all.get("pm25")
                                                if isinstance(pm_df, pd.DataFrame) and not pm_df.empty and "date" in pm_df.columns:
                                                    match = pm_df[pd.to_datetime(pm_df["date"]) == pd.to_datetime(wk_date)]
                                                    if not match.empty:
                                                        pm25_val = match.iloc[0].get("predicted_value")
                                            except Exception:
                                                pm25_val = None

                                        if pm25_val is None and spoon_key == "pm25":
                                            pm25_val = wk_val

                                        texture_full = describe_spoon(pm25=pm25_val)
                                        if "Clear broth" in texture_full:
                                            texture_short = "Clear broth"
                                        elif "Seasoned soup" in texture_full:
                                            texture_short = "Seasoned soup"
                                        elif "Creamy stew" in texture_full:
                                            texture_short = "Creamy stew"
                                        elif "Yogurt-thick" in texture_full or "Yogurt" in texture_full:
                                            texture_short = "Yogurt-thick"
                                        elif "Dust paste" in texture_full:
                                            texture_short = "Dust paste"
                                        elif "Toxic sludge" in texture_full:
                                            texture_short = "Toxic sludge"
                                        else:
                                            texture_short = texture_full

                                        try:
                                            date_label = wk_date.strftime("%Y-%m-%d") if hasattr(wk_date, "strftime") else str(wk_date)
                                        except Exception:
                                            date_label = str(wk_date)

                                        rows_list.append({
                                            "date": date_label,
                                            "texture": texture_short,
                                            "value": float(wk_val) if wk_val is not None and not pd.isna(wk_val) else None,
                                        })

                                    df_week = pd.DataFrame(rows_list)
                                    # Render an inline table and a small bar chart
                                    st.table(df_week)
                                    try:
                                        bars = (
                                            alt.Chart(df_week)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X("date:N", sort=None, title="Date"),
                                                y=alt.Y("value:Q", title=f"{pollutant.upper()} (ug/m3)"),
                                                color=alt.Color("value:Q", scale=alt.Scale(scheme="oranges")),
                                                tooltip=[alt.Tooltip("date:N"), alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("texture:N")],
                                            )
                                        )
                                        labels = (
                                            alt.Chart(df_week)
                                            .mark_text(dy=-10, fontSize=12)
                                            .encode(
                                                x=alt.X("date:N", sort=None),
                                                y=alt.Y("value:Q"),
                                                text=alt.Text("texture:N"),
                                                color=alt.value("#333333"),
                                            )
                                        )
                                        chart = (bars + labels).properties(height=200)
                                        st.altair_chart(chart, use_container_width=True)
                                    except Exception:
                                        pass
                            except Exception:
                                # Non-fatal: if week rendering fails, skip it silently
                                pass
        else:
            st.info("No pollutant forecasts could be generated.")

        clf_pred = model_results.get("classification_prediction") if model_results else None
        clf_report = model_results.get("classification_report") if model_results else None
        if clf_pred and clf_pred.get("predicted_category"):
            rainbow_heading("AQI Category Outlook", level=3, emoji="📯")
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
        elif model_results:
            # suppressed noisy classifier message (user requested removal)
            pass
    elif model_results:
        st.info("Not enough clean history to train regression models.")
    else:
        st.info("Model training skipped.")

    if not openaq_df.empty and model_results:
        with st.expander("Advanced model details", expanded=False):
            for msg in status_messages:
                st.info(msg)
            if not metrics_display.empty:
                rainbow_heading("Model metrics", level=3, emoji="🧮")
                st.dataframe(metrics_display, use_container_width=True)
            for pollutant, table_df, total_days in forecast_tables:
                rainbow_heading(f"Forecast data · {pollutant.upper()}", level=3, emoji="🗂️")
                st.caption(f"Predicting {pollutant.upper()} for the next {total_days} day(s).")
                st.dataframe(table_df, use_container_width=True)
    else:
        st.info("Insufficient OpenAQ data available for model training.")

    rainbow_heading("Daily Metrics Visualizer", level=3, emoji="📊")
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

    rainbow_heading("Pollen Forecast & Wind", level=3, emoji="🌬️")

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

        google_pollen_key = os.getenv("GOOGLE_POLLEN_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if google_pollen_key:
            map_variant = "Google Heatmap"
        else:
            st.info("Google heatmap requires API key; set GOOGLE_POLLEN_API_KEY to enable the map.")
            map_variant = None

        if map_variant == "Google Heatmap" and google_pollen_key:
            default_heatmap_key = f"pollen_heatmap_default_{cache_key_str}"
            default_heatmap_type = st.session_state.setdefault(default_heatmap_key, "TREE_UPI")
            button_specs = [
                ("TREE_UPI", "Tree", "#009c1a"),
                ("GRASS_UPI", "Grass", "#22b600"),
                ("WEED_UPI", "Weed", "#26cc00"),
            ]
            button_markup = "".join(
                f"<button data-maptype='{code}' style='padding:8px 16px;border:none;border-radius:4px;background:{color};color:#fff;font-weight:500;cursor:pointer;'>" \
                f"{label}</button>"
                for code, label, color in button_specs
            )
            map_dom_id = f"pollen-map-{cache_key_str}".replace(" ", "-")
            button_dom_id = f"pollen-buttons-{cache_key_str}".replace(" ", "-")
            center_lat = float(lat) if lat else 0.0
            center_lon = float(lon) if lon else 0.0

            google_html = f"""
<div style=\"text-align:center;margin-bottom:6px;font-weight:600;\">Choose a heatmap type</div>
<div id=\"{button_dom_id}\" style=\"display:flex;gap:8px;justify-content:center;margin-bottom:12px;flex-wrap:wrap;\">{button_markup}</div>
<div id=\"{map_dom_id}\" style=\"height:460px;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.12);\"></div>
<script>
let activePollenType_{cache_key_str} = "{default_heatmap_type}";

function getNormalizedCoord(coord, zoom) {{
  const tileRange = 1 << zoom;
  if (coord.y < 0 || coord.y >= tileRange) {{
    return null;
  }}
  let x = coord.x;
  if (x < 0 || x >= tileRange) {{
    x = ((x % tileRange) + tileRange) % tileRange;
  }}
  return {{ x, y: coord.y }};
}}

class PollenMapType_{cache_key_str} {{
  constructor(tileSize) {{
    this.tileSize = tileSize;
    this.maxZoom = 16;
    this.minZoom = 3;
    this.name = "Pollen";
  }}

  getTile(coord, zoom, ownerDocument) {{
    const normalized = getNormalizedCoord(coord, zoom);
    if (!normalized) return null;
    const img = ownerDocument.createElement("img");
    img.style.opacity = 0.88;
    img.style.width = this.tileSize.width + "px";
    img.style.height = this.tileSize.height + "px";
    img.src = `https://pollen.googleapis.com/v1/mapTypes/${{activePollenType_{cache_key_str}}}/heatmapTiles/${{zoom}}/${{normalized.x}}/${{normalized.y}}?key={google_pollen_key}`;
    return img;
  }}
  releaseTile(tile) {{}}
}}

function initPollenMap_{cache_key_str}() {{
  const target = document.getElementById("{map_dom_id}");
  if (!target) return;
  const map = new google.maps.Map(target, {{
    center: {{ lat: {center_lat:.6f}, lng: {center_lon:.6f} }},
    zoom: 8,
    minZoom: 3,
    maxZoom: 14,
    mapId: "ffcdd6091fa9fb03",
    streetViewControl: false,
    fullscreenControl: true,
  }});

  const pollenMapType = new PollenMapType_{cache_key_str}(new google.maps.Size(256, 256));
  map.overlayMapTypes.insertAt(0, pollenMapType);

  const buttons = document.querySelectorAll('#{button_dom_id} button');
  buttons.forEach((btn) => {{
    if (btn.dataset.maptype === activePollenType_{cache_key_str}) {{
      btn.style.filter = "brightness(0.85)";
    }}
    btn.addEventListener("click", () => {{
      activePollenType_{cache_key_str} = btn.dataset.maptype;
      map.overlayMapTypes.clear();
      map.overlayMapTypes.insertAt(0, new PollenMapType_{cache_key_str}(new google.maps.Size(256, 256)));
      buttons.forEach((other) => (other.style.filter = ""));
      btn.style.filter = "brightness(0.85)";
    }});
  }});
}}

window.initPollenMap_{cache_key_str} = initPollenMap_{cache_key_str};
</script>
<script defer src=\"https://maps.googleapis.com/maps/api/js?callback=initPollenMap_{cache_key_str}&v=weekly&key={google_pollen_key}&language=en\"></script>
"""
            components_html(google_html, height=520)
            st.caption(
                "Google Pollen heatmap (interactive). Tap the buttons above to switch allergen layers."
            )

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
        heading="Personalized Action Steps",
        city=city,
        selected_sensitive_groups=selected_sensitive_groups,
        symptom_notes=symptom_notes,
        air_summary=air_summary,
        pollen_summary=pollen_summary,
        pollutant_payload=latest_pollutants,
        pollen_df=pollen_df,
        context_note=plan_context_note,
        heading_level=3,
        heading_emoji="🌈",
    )

elif selected_view == "Social Media":
    pollen_bundle = load_pollen_bundle(city, lat, lon, force=force_refresh)
    pollen_summary = pollen_bundle.get("summary")

    rainbow_heading("Share & Stay Connected", level=3, emoji="🤳")
    st.write(
        "Craft quick updates for community groups or social feeds using the latest air, pollen, and action insights."
    )

    default_caption = ""
    if city:
        default_caption = (
            f"Air quality check for {city}: {air_summary or 'Monitoring in progress.'} "
            f"Pollen outlook: {pollen_summary or 'Awaiting forecast.'} #AirQuality #Pollen"
        )

    # Ensure the caption session key exists before creating the widget to avoid
    # setting a widget default value while also manipulating the same key via
    # the Session State API later (Streamlit raises an exception for that).
    caption_key = f"share_caption_{cache_key_str}"
    if caption_key not in st.session_state:
        st.session_state[caption_key] = default_caption
    share_caption = st.text_area(
        "Editable caption (copy & paste to share)",
        key=caption_key,
    )

    def _safe_initiative_prompt(city_name: str) -> str:
        """Ask the GPT advisor for structured local initiatives (name + handle) and return a chosen name."""
        prompt = (
            f"Provide a JSON array of local clean-air initiatives or community groups focused on air quality in {city_name}. "
            "Return an array of objects with keys `name` and `handle` (handle may be empty). "
            "Example: [{\"name\": \"Clean Air Initiative\", \"handle\": \"@cleanair\"}]. "
            "If you are unsure, return an empty array."
        )
        try:
            resp = generate_action_plan(
                city=city_name,
                sensitive_groups=[],
                user_notes="",
                pollen_summary=pollen_summary,
                air_quality_summary=air_summary,
                custom_prompt=prompt,
                temperature=0.2,
            )
            if isinstance(resp, dict):
                content = resp.get("content") or resp.get("result") or ""
            else:
                content = str(resp)

            # Try to extract JSON substring
            json_text = content.strip()
            try:
                data = json.loads(json_text)
            except Exception:
                # Attempt to find first JSON array in text
                m = re.search(r"(\[.*\])", json_text, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group(1))
                    except Exception:
                        data = []
                else:
                    data = []

            if isinstance(data, list) and data:
                for item in data:
                    name = (item.get("name") if isinstance(item, dict) else None) or ""
                    handle = (item.get("handle") if isinstance(item, dict) else None) or ""
                    if name:
                        # prefer handle if present
                        return name if not handle else f"{name} {handle}"
            # fallback: simple extraction
            first_line = (content.splitlines() or [""])[0].strip()
            first_line = re.sub(r'^[\'"\s]+|[\'"\s]+$', "", first_line)
            if not first_line:
                return "Unknown Initiative"
            return first_line[:120]
        except Exception:
            return "Unknown Initiative"

    def _build_social_post(
        initiative_name: str,
        city_name: str,
        pollutant_payload: dict,
        google_obs: object | None = None,
        merra_frame: pd.DataFrame | None = None,
    ) -> str:
        """Format the exact message requested by the user.

        Use GPT only to get the initiative name (passed in). Build the rest from data.
        Desired wording:
        i looked xxx up and you are responsible for the clean air in my region @xxx
        currently we have levels of x x x according to nasa, google, and openaqs data
        lets have cleaner air!
        #projectkoleidascope
        """
        # Use the initiative_name verbatim. It may already contain handles or multiple names.
        tag = ""

        # Gather OpenAQ (local) PM2.5
        try:
            openaq_pm25 = pollutant_payload.get("pm25")
            openaq_str = f"OpenAQ: {float(openaq_pm25):.1f} µg/m3" if openaq_pm25 is not None and not pd.isna(openaq_pm25) else "OpenAQ: N/A"
        except Exception:
            openaq_str = "OpenAQ: N/A"

        # Gather Google Air (AQI)
        try:
            if google_obs and getattr(google_obs, "aqi", None) is not None:
                google_str = f"Google AQI: {getattr(google_obs, 'aqi')} ({getattr(google_obs, 'category', 'N/A')})"
            else:
                google_str = "Google AQI: N/A"
        except Exception:
            google_str = "Google AQI: N/A"

        # Gather NASA / MERRA estimate (use recent merra_frame pm25 if available)
        try:
            if merra_frame is not None and not merra_frame.empty:
                # find most recent pm25 in merra_frame
                candidate_cols = [c for c in merra_frame.columns if c.lower() in {"pm25", "pm2_5", "pm2.5"}]
                if candidate_cols:
                    col = candidate_cols[0]
                    recent = merra_frame.sort_values("date").dropna(subset=[col]).tail(1)
                    if not recent.empty:
                        val = float(recent.iloc[0][col])
                        nasa_str = f"NASA (MERRA): {val:.1f} µg/m3"
                    else:
                        nasa_str = "NASA (MERRA): N/A"
                else:
                    nasa_str = "NASA (MERRA): N/A"
            else:
                nasa_str = "NASA (MERRA): N/A"
        except Exception:
            nasa_str = "NASA (MERRA): N/A"

        # Compose the exact message
        lines = [
            f"i looked {initiative_name} up and you are responsible for the clean air in my region{tag}",
            "",
            f"currently we have levels of {nasa_str}, {google_str}, {openaq_str} — data we have from the daily summary",
            "",
            "lets have cleaner air!",
            "",
            "#projectkoleidascope",
        ]

        return "\n".join(lines)


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

    # --- Social post generator ---
    st.markdown("---")
    st.markdown("### Auto-generate a social post")
    # Try to fetch parsed initiative candidates using the helper module
    candidates = []
    try:
        from app.lib.initiative_lookup import find_initiatives
        candidates = find_initiatives(city or "your city")
    except Exception:
        candidates = []

    # Present user with a candidate chooser if we found any
    candidate_display = None
    selected_candidate = None
    if candidates:
        options = [f"{c.get('name')} {('@'+c.get('handle')) if c.get('handle') else ''}".strip() for c in candidates]
        sel_idx = st.selectbox("Choose initiative to tag", options=options, key=f"initiative_select_{cache_key_str}")
        # resolve selected candidate dict
        try:
            sel_index = options.index(sel_idx)
            selected_candidate = candidates[sel_index]
        except Exception:
            selected_candidate = None

    def _gen_and_set(city_name=city, key=f"share_caption_{cache_key_str}", pollutants=None, google_obs=None, merra_frame=None, chosen=None, candidates_list=None):
        # Generate initiative name and formatted post, then set the caption in session_state.
        try:
            # If we have a candidates list, prefer joining handles; otherwise join names
            if candidates_list:
                handles = []
                names = []
                for c in candidates_list:
                    if not isinstance(c, dict):
                        continue
                    name = (c.get("name") or "").strip()
                    handle = (c.get("handle") or "").strip()
                    if handle:
                        # ensure it starts with @
                        if not handle.startswith("@"):
                            handle_val = "@" + handle
                        else:
                            handle_val = handle
                        handles.append(handle_val)
                    if name:
                        names.append(name)
                if handles:
                    initiative = ", ".join(handles)
                elif names:
                    initiative = ", ".join(names)
                else:
                    initiative = _safe_initiative_prompt(city_name or "your city")
            elif chosen:
                name = chosen.get("name") or "Unknown Initiative"
                handle = chosen.get("handle") or ""
                initiative = f"{name} {'@'+handle if handle else ''}".strip()
            else:
                initiative = _safe_initiative_prompt(city_name or "your city")
            post_text = _build_social_post(initiative, city_name or "Unknown city", pollutants or {}, google_obs, merra_frame)
            st.session_state[key] = post_text
            st.success("Generated post added to the caption box above. You can edit or copy it.")
            try:
                # Force a rerun so the caption widget refreshes and shows the generated text immediately
                st.experimental_rerun()
            except Exception:
                # If rerun isn't allowed in this context, silently continue — user can still see caption.
                pass
        except Exception as exc:
            st.error(f"Could not generate post: {exc}")

    st.button(
        "Generate post using GPT",
        key=f"gen_post_{cache_key_str}",
        on_click=_gen_and_set,
        args=(city, f"share_caption_{cache_key_str}", latest_pollutants, google_air_obs, merra_df, selected_candidate, candidates),
    )

    st.caption("Tips: Review tags before posting; GPT may guess initiative names. Edit text if needed.")

    # --- Show raw GPT response for debugging / review ---
    raw_key = f"raw_gpt_resp_{cache_key_str}"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""

    def _fetch_raw_initiatives(city_name: str):
        prompt = (
            f"Provide a JSON array of local clean-air initiatives or community groups focused on air quality in {city_name}. "
            "Return an array of objects with keys `name` and `handle` (handle may be empty). If unsure, return an empty array."
        )
        try:
            # Run the GPT query inside a spinner so the UI waits and then updates the caption in one go.
            with st.spinner("Searching initiatives and generating post..."):
                resp = generate_action_plan(
                    city=city_name,
                    sensitive_groups=[],
                    user_notes="",
                    pollen_summary=pollen_summary,
                    air_quality_summary=air_summary,
                    custom_prompt=prompt,
                    temperature=0.2,
                )
                if isinstance(resp, dict):
                    content = resp.get("content") or resp.get("result") or ""
                else:
                    content = str(resp)

                # Store raw content internally (not displayed), for debugging if needed.
                st.session_state[raw_key] = content

                # Try to parse a JSON array of candidates. If found, prefer handles when building the initiative string
                candidates_data = []
                try:
                    candidates_data = json.loads(content)
                except Exception:
                    m = re.search(r"(\[.*\])", content, re.DOTALL)
                    if m:
                        try:
                            candidates_data = json.loads(m.group(1))
                        except Exception:
                            candidates_data = []

                # If candidates parsed, generate the post using the same generation function to ensure consistency
                if isinstance(candidates_data, list) and candidates_data:
                    _gen_and_set(city_name, caption_key, latest_pollutants, google_air_obs, merra_df, None, candidates_data)
                else:
                    # No structured candidates found — fall back to generating via the safe prompt and single-call generator
                    # This will call _safe_initiative_prompt internally via _gen_and_set when chosen and candidates_list are None.
                    _gen_and_set(city_name, caption_key, latest_pollutants, google_air_obs, merra_df, None, None)
        except Exception as exc:
            st.session_state[raw_key] = f"Error: {exc}"

    # Trigger a GPT-backed search for local initiatives; the response is parsed and used to
    # populate the caption automatically. The raw GPT text is stored internally but not shown.
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        st.button("Autofind initiative for your city", key=f"search_initiatives_{cache_key_str}", on_click=_fetch_raw_initiatives, args=(city,))
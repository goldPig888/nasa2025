# lib/api_fetch.py
import requests
import os
import streamlit as st
import pandas as pd
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

geolocator = Nominatim(user_agent="kolidascope")

# -------------------------
# Basic Geocoding
# -------------------------
def geocode_city(city_name):
    """Convert city name → (lat, lon)."""
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        pass
    return None, None


def reverse_geocode(lat, lon):
    """Convert coordinates → nearest city/state name."""
    try:
        location = geolocator.reverse((lat, lon), timeout=10, language="en")
        if location and "address" in location.raw:
            address = location.raw["address"]
            return (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("state")
            )
    except GeocoderTimedOut:
        pass
    return "Unknown location"


# -------------------------
# OpenWeatherMap Air Pollution
# -------------------------
def fetch_air_pollution(lat, lon):
    """Fetch AQI and pollutant concentrations from OpenWeatherMap."""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        st.warning("Could not fetch pollution data.")
        return None
    return r.json()


def describe_aqi(aqi_value):
    """Convert AQI value (1–5) to human-readable description."""
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }.get(aqi_value, "Unknown")


# -------------------------
# OpenAQ Sensor & Historical Data
# -------------------------
from pprint import pprint

def get_nearest_sensor(lat, lon, parameter="pm25"):
    """
    Find the nearest OpenAQ sensor to a coordinate.
    Tries progressively broader searches if nothing is found.
    Logs the raw API responses for debugging.
    """
    headers = {"X-API-Key": OPENAQ_API_KEY}

    def _query(radius_m, param=None):
        url = f"{BASE_URL}/sensors?coordinates={lat},{lon}&radius={radius_m}&limit=5"
        if param:
            url += f"&parameter={param}"
        r = requests.get(url, headers=headers)
        try:
            r.raise_for_status()
        except Exception as e:
            print(f"[!] OpenAQ sensors query failed ({r.status_code}): {e}")
            return []
        data = r.json().get("results", [])
        if not data:
            print(f"[i] No sensors found within {radius_m/1000:.0f} km (param={param})")
        return data

    # --- tier 1: narrow search ---
    results = _query(25000, parameter)
    if not results:
        # --- tier 2: wider radius ---
        results = _query(100000, parameter)
    if not results and parameter:
        # --- tier 3: drop parameter filter entirely ---
        results = _query(100000, None)

    if results:
        print(f"[✓] Found {len(results)} sensors near ({lat:.3f},{lon:.3f}):")
        pprint([{"id": s["id"], "name": s.get("name"), "parameter": s.get("parameter")} for s in results[:5]])
        return results[0]["id"], results[0]["name"]

    print("[x] No nearby OpenAQ sensors found after all attempts.")
    return None

def fetch_sensor_days(sensor_id, limit=365):
    """Fetch daily-average data for a given sensor (up to a year)."""
    url = f"{BASE_URL}/sensors/{sensor_id}/days?limit={limit}"
    headers = {"X-API-Key": OPENAQ_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Error fetching daily averages:", r.status_code)
        return pd.DataFrame()
    data = r.json().get("results", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"value": "avg_value", "parameter": "parameter_name"})
    return df[["datetime", "avg_value", "parameter_name"]]

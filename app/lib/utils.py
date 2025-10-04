import streamlit as st
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
if not API_KEY:
    st.error("⚠️ Missing OPENWEATHERMAP_API_KEY in .env file.")
    st.stop()

geolocator = Nominatim(user_agent="kolidascope")

def geocode_city(city_name):
    """Convert city name → (lat, lon)."""
    try:
        loc = geolocator.geocode(city_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        pass
    return None, None


def fetch_air_pollution(lat, lon):
    """Fetch AQI and pollutant concentrations from OpenWeatherMap."""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        st.warning("❌ Could not fetch pollution data.")
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


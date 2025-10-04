# location_picker.py
import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv
from lib.utils import *
import os

st.set_page_config(page_title="Location & Air Quality", layout="wide")
st.title("🌍 Location and Air Quality Lookup")

# Load environment variable (OpenWeatherMap API key)
load_dotenv()
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
if not API_KEY:
    st.error("⚠️ Missing OPENWEATHERMAP_API_KEY in .env file.")
    st.stop()

geolocator = Nominatim(user_agent="kolidascope")


city = st.text_input("Enter city name:", "Houston")

lat, lon = (None, None)
if city:
    lat, lon = geocode_city(city)
    if lat and lon:
        st.success(f"📍 {city} → ({lat:.4f}, {lon:.4f})")
    else:
        st.warning("Couldn't find that city. Try another name.")

# Map
if lat and lon:
    st.map(pd.DataFrame([[lat, lon]], columns=["lat", "lon"]))
else:
    st.map(pd.DataFrame([[39.5, -98.35]], columns=["lat", "lon"]))

# ---------------------------
# Display Air Quality
# ---------------------------
st.markdown("---")
st.write("### ✅ Selected Coordinates and Air Quality")

if lat and lon:
    pollution = fetch_air_pollution(lat, lon)
    if pollution and "list" in pollution:
        info = pollution["list"][0]
        aqi = info["main"]["aqi"]
        comps = info["components"]

        st.json(
            {
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "AQI": f"{aqi} ({describe_aqi(aqi)})",
                "components (µg/m³)": comps,
            }
        )
    else:
        st.warning("No pollution data found for this location.")
else:
    st.info("Enter a city to view its coordinates and air quality.")

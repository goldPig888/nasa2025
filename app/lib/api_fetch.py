# lib/api_fetch.py
import requests
import os
import pandas as pd
from geopy.distance import distance
from dotenv import load_dotenv

load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

def get_locations(country="US", city=None, limit=100):
    """Fetch list of locations from OpenAQ."""
    params = {"country": country, "limit": limit}
    if city:
        params["city"] = city

    headers = {"X-API-Key": OPENAQ_API_KEY}
    r = requests.get(f"{BASE_URL}/locations", params=params, headers=headers)

    if r.status_code != 200:
        print("Error:", r.status_code, r.text)
        return pd.DataFrame()

    data = r.json()
    if "results" not in data:
        print("No 'results' key in response.")
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df["lat"] = df["coordinates"].apply(lambda x: x["latitude"])
    df["lon"] = df["coordinates"].apply(lambda x: x["longitude"])
    return df

def find_nearest(lat, lon, df):
    """Return the nearest OpenAQ station from a DataFrame."""
    if df.empty:
        return None
    df["dist_km"] = df.apply(lambda x: distance((lat, lon), (x.lat, x.lon)).km, axis=1)
    return df.sort_values("dist_km").iloc[0]

# config/api_config.py
import os
from dotenv import load_dotenv
load_dotenv()

OPENAQ_BASE = "https://api.openaq.org/v3"
NWS_BASE = "https://api.weather.gov"

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

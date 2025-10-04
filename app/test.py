from lib.api_fetch import get_openaq_data
import requests, os
import pprint

headers = {"X-API-Key": os.getenv("OPENAQ_API_KEY")}
r = requests.get(
    "https://api.openaq.org/v3/locations?country=US&city=Houston&limit=5",
    headers=headers
)
pprint.pprint(r.json())

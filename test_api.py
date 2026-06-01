import requests
from config import OPENAQ_API_KEY, COLOMBO_LAT, COLOMBO_LON

url = "https://api.openaq.org/v3/locations"
params = {
    "coordinates": f"{COLOMBO_LAT},{COLOMBO_LON}",
    "radius": 25000,
    "limit": 100
}
headers = {"X-API-Key": OPENAQ_API_KEY}

r = requests.get(url, params=params, headers=headers)
data = r.json()
results = data.get("results", [])

print(f"Found {len(results)} real sensor locations:\n")
for loc in results:
    print(f"  ID       : {loc['id']}")
    print(f"  Name     : {loc.get('name', '?')}")
    print(f"  Locality : {loc.get('locality', '?')}")
    print(f"  Provider : {loc.get('provider', {}).get('name', '?')}")
    coords = loc.get("coordinates", {})
    print(f"  Lat/Lon  : {coords.get('latitude')}, {coords.get('longitude')}")
    sensors = loc.get("sensors", [])
    for s in sensors:
        print(f"  Sensor   : id={s['id']} | {s['name']}")
    print()
# ============================================================
#  step1_collect_data.py
#  Step 1 — Collect air quality + weather data
# ============================================================
#
#  What this script does:
#  1. Finds all air quality sensor locations near Colombo (OpenAQ v3 API)
#  2. Downloads PM2.5 measurements from those sensors
#  3. Downloads weather data for the same time period (OpenWeatherMap)
#  4. Saves everything as CSV files in the data/ folder
#
#  Run this once to collect raw data. After this, run step2_preprocess.py
# ============================================================

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from config import (
    OPENAQ_API_KEY,
    OPENWEATHER_API_KEY,
    COLOMBO_LAT,
    COLOMBO_LON,
    SEARCH_RADIUS_M,
    DATE_FROM,
    DATE_TO,
    DATA_DIR,
)

# Make sure the data folder exists
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
#  PART A — OpenAQ v3: Find sensor locations near Colombo
# ============================================================

def find_colombo_locations():
    """
    Searches the OpenAQ v3 API for all air quality monitoring
    stations within SEARCH_RADIUS_M meters of Colombo city center.

    Returns a list of location dicts, each with:
      - id         : the location's unique ID (you need this to get measurements)
      - name       : station name
      - lat / lon  : coordinates
      - sensors    : list of sensor IDs attached to this location
    """

    print("\n=== PART A: Finding sensor locations near Colombo ===")

    url = "https://api.openaq.org/v3/locations"

    # These parameters tell the API:
    #   coordinates = center of Colombo
    #   radius = search within 50km
    #   limit = return up to 100 results per page
    params = {
        "coordinates": f"{COLOMBO_LAT},{COLOMBO_LON}",
        "radius":      SEARCH_RADIUS_M,
        "limit":       100,
        "page":        1,
    }

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    response = requests.get(url, params=params, headers=headers)

    # If the API returns an error, print what went wrong
    if response.status_code != 200:
        print(f"[ERROR] OpenAQ API returned status {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    locations = data.get("results", [])

    print(f"Found {len(locations)} sensor locations near Colombo")

    # Print a summary of what we found
    location_records = []
    for loc in locations:
        loc_id   = loc["id"]
        loc_name = loc.get("name", "Unknown")
        lat      = loc.get("coordinates", {}).get("latitude", None)
        lon      = loc.get("coordinates", {}).get("longitude", None)

        # Each location has multiple sensors (PM2.5, PM10, NO2 etc.)
        # We extract all sensor IDs here
        sensors = loc.get("sensors", [])
        sensor_ids = [s["id"] for s in sensors]

        print(f"  → Location {loc_id}: {loc_name} | lat={lat}, lon={lon} | sensors={sensor_ids}")

        location_records.append({
            "location_id":  loc_id,
            "name":         loc_name,
            "latitude":     lat,
            "longitude":    lon,
            "sensor_ids":   str(sensor_ids),  # save as string for CSV
        })

    # Save locations to CSV
    df_locations = pd.DataFrame(location_records)
    save_path = os.path.join(DATA_DIR, "colombo_locations.csv")
    df_locations.to_csv(save_path, index=False)
    print(f"\nSaved {len(df_locations)} locations to: {save_path}")

    return locations


# ============================================================
#  PART B — OpenAQ v3: Get PM2.5 measurements from sensors
# ============================================================

def get_sensor_measurements(sensor_id, date_from, date_to):
    """
    Fetches hourly PM2.5 measurements from a single sensor.

    In OpenAQ v3, the flow is:
      Location → has many Sensors → each Sensor has Measurements

    This function calls:
      GET /v3/sensors/{sensor_id}/measurements/hourly

    Parameters:
      sensor_id  : the sensor's numeric ID from OpenAQ
      date_from  : start date string "YYYY-MM-DD"
      date_to    : end date string "YYYY-MM-DD"

    Returns a list of measurement dicts
    """

    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    all_measurements = []
    page = 1

    while True:
        params = {
            "datetime_from": f"{date_from}T00:00:00Z",
            "datetime_to":   f"{date_to}T23:59:59Z",
            "limit":         1000,   # max per page
            "page":          page,
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"    [WARNING] Sensor {sensor_id} returned {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            break  # no more pages

        all_measurements.extend(results)

        # Check if there are more pages
        meta = data.get("meta", {})
        found = meta.get("found", 0)
        limit = meta.get("limit", 1000)
        found_clean = str(found).replace(">", "").replace("<", "").strip()
        if page * limit >= int(found_clean):
            break

        page += 1
        time.sleep(0.3)  # be polite — don't hammer the API

    return all_measurements


def collect_all_pm25_data(locations):
    """
    Loops through all locations, finds PM2.5 sensors,
    and downloads their measurements.

    PM2.5 is parameter ID = 2 in OpenAQ.
    We identify PM2.5 sensors by checking the parameter name.
    """

    print("\n=== PART B: Downloading PM2.5 measurements ===")

    all_rows = []

    for loc in locations:
        loc_id   = loc["id"]
        loc_name = loc.get("name", "Unknown")
        lat      = loc.get("coordinates", {}).get("latitude")
        lon      = loc.get("coordinates", {}).get("longitude")
        sensors  = loc.get("sensors", [])

        for sensor in sensors:
            sensor_id   = sensor["id"]
            param       = sensor.get("parameter", {})
            param_name  = param.get("name", "").lower()

            # Only download PM2.5 sensors
            if "pm25" not in param_name and "pm2.5" not in param_name:
                continue

            print(f"  Downloading: {loc_name} | sensor_id={sensor_id} | param={param_name}")

            measurements = get_sensor_measurements(sensor_id, DATE_FROM, DATE_TO)
            print(f"    → Got {len(measurements)} hourly records")

            for m in measurements:
                # Extract the timestamp and value
                dt_info  = m.get("period", {}).get("datetimeTo", {})
                dt_local = dt_info.get("local", None)
                value    = m.get("value", None)

                all_rows.append({
                    "datetime":    dt_local,
                    "location_id": loc_id,
                    "location":    loc_name,
                    "latitude":    lat,
                    "longitude":   lon,
                    "sensor_id":   sensor_id,
                    "pm25":        value,
                })

            time.sleep(0.5)  # pause between sensors

    # Build DataFrame and save
    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\n[WARNING] No PM2.5 data was collected.")
        print("This can happen if:")
        print("  - There are no active sensors in Colombo right now")
        print("  - Your API key is incorrect")
        print("  - The date range has no data")
        print("\nUsing DEMO DATA instead so you can continue learning...")
        df = generate_demo_data()
    else:
        print(f"\nTotal PM2.5 records collected: {len(df)}")

    save_path = os.path.join(DATA_DIR, "pm25_raw.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved PM2.5 data to: {save_path}")

    return df


# ============================================================
#  PART C — OpenWeatherMap: Get historical weather data
# ============================================================

def get_weather_for_date_range(date_from, date_to):
    """
    Downloads historical weather data from OpenWeatherMap
    for Colombo, for every day in the date range.

    OpenWeatherMap free tier doesn't give hourly historical data,
    so we use the "history" endpoint which gives daily summaries.

    What we collect: temperature, humidity, wind speed, weather description.

    NOTE: For historical data beyond 5 days, OpenWeatherMap requires
    the "History" paid plan. For free tier, we use a clever workaround:
    We call the "onecall/timemachine" endpoint which gives 1 day at a time,
    going back up to 5 days for free. For a full year, you either need
    the paid plan OR we generate realistic synthetic weather below.
    """

    print("\n=== PART C: Downloading weather data ===")

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    # Parse date range
    start = datetime.strptime(date_from, "%Y-%m-%d")
    end   = datetime.strptime(date_to,   "%Y-%m-%d")

    weather_rows = []
    current = start

    print(f"Fetching current weather for Colombo as a sample...")

    # For the free API key, we just collect current conditions
    # then generate synthetic historical data based on Colombo's climate
    params = {
        "lat":   COLOMBO_LAT,
        "lon":   COLOMBO_LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # Celsius
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        w = response.json()
        print(f"  Current weather in Colombo:")
        print(f"    Temperature : {w['main']['temp']} °C")
        print(f"    Humidity    : {w['main']['humidity']} %")
        print(f"    Wind speed  : {w['wind']['speed']} m/s")
        print(f"    Description : {w['weather'][0]['description']}")
        print()

    # Since free API doesn't give full historical data,
    # generate realistic synthetic data based on Colombo's actual climate
    print("Generating realistic Colombo weather data for the full date range...")
    weather_rows = generate_colombo_weather(start, end)

    df_weather = pd.DataFrame(weather_rows)
    save_path = os.path.join(DATA_DIR, "weather_raw.csv")
    df_weather.to_csv(save_path, index=False)
    print(f"Saved weather data ({len(df_weather)} rows) to: {save_path}")

    return df_weather


def generate_colombo_weather(start_date, end_date):
    """
    Generates realistic hourly weather data for Colombo based on
    actual climate patterns:
      - Hot and humid year-round (26–32°C)
      - Two monsoon seasons:
          SW Monsoon: May–September (high humidity, more rain)
          NE Monsoon: November–January
      - Wind picks up during monsoons
    """
    import numpy as np

    rows = []
    current = start_date

    while current <= end_date:
        month = current.month

        # Base temperature by month (Colombo averages)
        base_temp = {
            1: 27.5, 2: 28.2, 3: 29.1, 4: 29.5,
            5: 29.0, 6: 28.2, 7: 27.9, 8: 28.0,
            9: 28.2, 10: 28.0, 11: 27.5, 12: 27.2
        }[month]

        # Humidity by month (higher during monsoons)
        base_humidity = {
            1: 75, 2: 74, 3: 73, 4: 77,
            5: 82, 6: 85, 7: 84, 8: 84,
            9: 82, 10: 81, 11: 79, 12: 77
        }[month]

        # Wind speed by month (m/s)
        base_wind = {
            1: 2.8, 2: 2.5, 3: 2.2, 4: 2.5,
            5: 4.0, 6: 5.2, 7: 5.5, 8: 5.0,
            9: 3.5, 10: 2.8, 11: 2.5, 12: 2.8
        }[month]

        # Generate 24 hourly records per day
        for hour in range(24):
            # Temperature peaks at 2–3pm, lowest at 6am
            hour_offset = 2.0 * np.sin((hour - 6) * np.pi / 12)
            temp = base_temp + hour_offset + np.random.normal(0, 0.5)

            # Humidity inverse of temperature (higher at night)
            humidity = base_humidity - hour_offset * 3 + np.random.normal(0, 2)
            humidity = max(60, min(98, humidity))

            # Wind varies randomly around base
            wind = base_wind + np.random.exponential(0.5)

            rows.append({
                "datetime":    current.strftime(f"%Y-%m-%dT{hour:02d}:00:00+05:30"),
                "date":        current.strftime("%Y-%m-%d"),
                "hour":        hour,
                "temperature": round(temp, 2),
                "humidity":    round(humidity, 2),
                "wind_speed":  round(wind, 2),
                "month":       month,
                "season":      get_colombo_season(month),
            })

        current += timedelta(days=1)

    return rows


def get_colombo_season(month):
    """Maps month number to Sri Lanka's climate season."""
    if month in [12, 1, 2]:
        return "Northeast Monsoon"
    elif month in [3, 4]:
        return "First Inter-monsoon"
    elif month in [5, 6, 7, 8, 9]:
        return "Southwest Monsoon"
    else:
        return "Second Inter-monsoon"


# ============================================================
#  DEMO DATA — used when API keys aren't set up yet
# ============================================================

def generate_demo_data():
    """
    Generates realistic demo PM2.5 data for Colombo.
    Used when you haven't set up API keys yet, so you can
    still run and learn from the rest of the pipeline.

    Based on real patterns:
      - Morning rush hour (7–9am): higher PM2.5
      - Evening rush hour (5–7pm): higher PM2.5
      - Monsoon season: lower PM2.5 (rain washes the air)
      - Dry season: higher PM2.5
    """
    import numpy as np

    print("Generating realistic demo PM2.5 data for Colombo...")

    # Three fictional but realistic sensor locations in Colombo
    locations = [
        {"location_id": 1001, "location": "Colombo Fort",       "latitude": 6.9344, "longitude": 79.8428},
        {"location_id": 1002, "location": "Nugegoda Junction",  "latitude": 6.8728, "longitude": 79.8894},
        {"location_id": 1003, "location": "Dehiwala",           "latitude": 6.8514, "longitude": 79.8653},
    ]

    rows = []
    start = datetime.strptime(DATE_FROM, "%Y-%m-%d")
    end   = datetime.strptime(DATE_TO,   "%Y-%m-%d")
    current = start

    while current <= end:
        month = current.month

        # Seasonal base PM2.5 (µg/m³) — higher in dry season
        seasonal_base = {
            1: 45, 2: 50, 3: 48, 4: 40,
            5: 30, 6: 25, 7: 22, 8: 24,
            9: 28, 10: 35, 11: 42, 12: 48
        }[month]

        for loc in locations:
            # Each location has slightly different pollution level
            loc_offset = np.random.uniform(-5, 10)

            for hour in range(24):
                # Rush hour spikes
                if 7 <= hour <= 9:
                    hour_factor = 1.6
                elif 17 <= hour <= 19:
                    hour_factor = 1.5
                elif 0 <= hour <= 5:
                    hour_factor = 0.6   # low traffic at night
                else:
                    hour_factor = 1.0

                pm25 = (seasonal_base + loc_offset) * hour_factor
                pm25 += np.random.normal(0, 5)
                pm25 = max(0, pm25)  # can't be negative

                rows.append({
                    "datetime":    current.strftime(f"%Y-%m-%dT{hour:02d}:00:00+05:30"),
                    "location_id": loc["location_id"],
                    "location":    loc["location"],
                    "latitude":    loc["latitude"],
                    "longitude":   loc["longitude"],
                    "sensor_id":   loc["location_id"] * 10,
                    "pm25":        round(pm25, 2),
                })

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} demo PM2.5 records across {len(locations)} locations")
    return df


# ============================================================
#  MAIN — runs everything in order
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 1: Data Collection")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    # Check if API keys are set
    using_demo = False
    if OPENAQ_API_KEY == "your_openaq_api_key_here":
        print("\n[INFO] OpenAQ API key not set — will use demo data")
        print("       Edit config.py to add your real API key\n")
        using_demo = True

    if using_demo:
        # Skip API calls, generate realistic demo data directly
        df_pm25    = generate_demo_data()
        save_path  = os.path.join(DATA_DIR, "pm25_raw.csv")
        df_pm25.to_csv(save_path, index=False)
        print(f"Saved demo PM2.5 data to: {save_path}")
    else:
        # Real API calls
        locations = find_colombo_locations()
        df_pm25   = collect_all_pm25_data(locations)

    # Weather data (generates synthetic Colombo climate data either way)
    df_weather = get_weather_for_date_range(DATE_FROM, DATE_TO)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nPM2.5 data: {len(df_pm25)} rows")
    print(f"Columns: {list(df_pm25.columns)}")
    print(f"\nWeather data: {len(df_weather)} rows")
    print(f"Columns: {list(df_weather.columns)}")
    print(f"\nFiles saved in: {DATA_DIR}")
    print("\n→ Next step: run   python step2_preprocess.py")

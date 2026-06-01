# ============================================================
#  step2_preprocess.py
#  Step 2 — Clean, merge, and engineer features
# ============================================================
#
#  What this script does:
#  1. Loads pm25_raw.csv and weather_raw.csv
#  2. Cleans both datasets (fix types, handle missing values)
#  3. Merges them by matching date + hour
#  4. Calculates AQI from PM2.5 values
#  5. Engineers new features (rush_hour, season, lag values etc.)
#  6. Saves the final clean dataset: data/processed_data.csv
#
#  Run: python step2_preprocess.py
# ============================================================

import pandas as pd
import numpy as np
import os
from config import DATA_DIR

# ============================================================
#  PART A — Load the raw CSV files
# ============================================================

def load_raw_data():
    """
    Loads both CSV files saved by step1_collect_data.py
    and does basic type fixing.
    """

    print("\n=== PART A: Loading raw data ===")

    pm25_path    = os.path.join(DATA_DIR, "pm25_raw.csv")
    weather_path = os.path.join(DATA_DIR, "weather_raw.csv")

    # Check files exist
    if not os.path.exists(pm25_path):
        print(f"[ERROR] File not found: {pm25_path}")
        print("Please run step1_collect_data.py first!")
        exit()

    df_pm25    = pd.read_csv(pm25_path)
    df_weather = pd.read_csv(weather_path)

    print(f"PM2.5 data loaded   : {df_pm25.shape[0]} rows, {df_pm25.shape[1]} columns")
    print(f"Weather data loaded : {df_weather.shape[0]} rows, {df_weather.shape[1]} columns")

    # Convert datetime columns to proper datetime type
    # pd.to_datetime understands most formats automatically
    df_pm25["datetime"]    = pd.to_datetime(df_pm25["datetime"], utc=False, errors="coerce")
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], utc=False, errors="coerce")

    print("\nPM2.5 columns   :", list(df_pm25.columns))
    print("Weather columns :", list(df_weather.columns))

    return df_pm25, df_weather


# ============================================================
#  PART B — Clean PM2.5 data
# ============================================================

def clean_pm25(df):
    """
    Cleans the PM2.5 dataset:
    - Removes rows where PM2.5 value is missing
    - Removes impossible values (negative or > 500 µg/m³)
    - Extracts date and hour from the datetime column
    - Averages multiple sensors at the same location/time into one value
    """

    print("\n=== PART B: Cleaning PM2.5 data ===")
    print(f"Rows before cleaning: {len(df)}")

    # Drop rows with missing PM2.5 values
    df = df.dropna(subset=["pm25"])

    # Remove physically impossible values
    # WHO says anything above 500 is sensor error
    df = df[df["pm25"] >= 0]
    df = df[df["pm25"] <= 500]

    # Extract date and hour — we need these to merge with weather data
    # The datetime might have timezone info (+05:30) — we strip it for simplicity
    df["datetime"] = df["datetime"].dt.tz_localize(None) if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"]     = df["datetime"].dt.date.astype(str)
    df["hour"]     = df["datetime"].dt.hour

    # If multiple sensors reported at the same location + date + hour,
    # average them into one value
    df = df.groupby(
        ["date", "hour", "location_id", "location", "latitude", "longitude"],
        as_index=False
    ).agg(pm25=("pm25", "mean"))

    print(f"Rows after cleaning : {len(df)}")
    print(f"PM2.5 range         : {df['pm25'].min():.1f} – {df['pm25'].max():.1f} µg/m³")
    print(f"PM2.5 average       : {df['pm25'].mean():.1f} µg/m³")

    return df


# ============================================================
#  PART C — Clean weather data
# ============================================================

def clean_weather(df):
    """
    Cleans the weather dataset:
    - Removes rows with missing values
    - Removes impossible temperature/humidity values
    - Keeps only the columns we need
    """

    print("\n=== PART C: Cleaning weather data ===")
    print(f"Rows before cleaning: {len(df)}")

    # Drop rows where any key weather value is missing
    df = df.dropna(subset=["temperature", "humidity", "wind_speed"])

    # Remove impossible values
    df = df[df["temperature"].between(-10, 50)]   # Colombo never goes outside this
    df = df[df["humidity"].between(0, 100)]
    df = df[df["wind_speed"] >= 0]

    # Make sure date and hour columns exist
    if "date" not in df.columns:
        df["datetime"] = df["datetime"].dt.tz_localize(None) if df["datetime"].dt.tz is not None else df["datetime"]
        df["date"] = df["datetime"].dt.date.astype(str)
        df["hour"] = df["datetime"].dt.hour

    # Keep only what we need
    df = df[["date", "hour", "temperature", "humidity", "wind_speed", "season"]].copy()

    print(f"Rows after cleaning : {len(df)}")
    print(f"Temperature range   : {df['temperature'].min():.1f} – {df['temperature'].max():.1f} °C")
    print(f"Humidity range      : {df['humidity'].min():.1f} – {df['humidity'].max():.1f} %")

    return df


# ============================================================
#  PART D — Merge PM2.5 + Weather
# ============================================================

def merge_datasets(df_pm25, df_weather):
    """
    Merges the two datasets by matching date + hour.

    Think of it like this:
      PM2.5 row:   2023-06-15, hour=8, location=Colombo Fort, pm25=45.2
      Weather row: 2023-06-15, hour=8, temp=28.5, humidity=82, wind=3.1
      Merged row:  2023-06-15, hour=8, location=Colombo Fort,
                   pm25=45.2, temp=28.5, humidity=82, wind=3.1

    We use a LEFT JOIN — every PM2.5 row is kept even if
    there's no matching weather row (weather will be NaN for those).
    """

    print("\n=== PART D: Merging datasets ===")

    df_merged = pd.merge(
        df_pm25,
        df_weather,
        on=["date", "hour"],
        how="left"
    )

    print(f"Merged dataset shape: {df_merged.shape}")

    # How many rows have missing weather after merge?
    missing_weather = df_merged["temperature"].isna().sum()
    print(f"Rows with missing weather: {missing_weather}")

    # Fill any remaining missing weather with column median
    # (median is better than mean when there are outliers)
    for col in ["temperature", "humidity", "wind_speed"]:
        if df_merged[col].isna().any():
            median_val = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(median_val)
            print(f"  Filled missing {col} with median: {median_val:.2f}")

    return df_merged


# ============================================================
#  PART E — Calculate AQI from PM2.5
# ============================================================

def pm25_to_aqi(pm25):
    """
    Converts PM2.5 concentration (µg/m³) to AQI score.

    This uses the US EPA breakpoints — the same standard
    used by most air quality apps worldwide.

    AQI Breakpoints:
    ┌─────────────┬────────────────────┬─────────────────────────┐
    │ AQI Range   │ PM2.5 Range        │ Health Category         │
    ├─────────────┼────────────────────┼─────────────────────────┤
    │   0 – 50    │  0.0 –  12.0       │ Good                    │
    │  51 – 100   │ 12.1 –  35.4       │ Moderate                │
    │ 101 – 150   │ 35.5 –  55.4       │ Unhealthy for sensitive │
    │ 151 – 200   │ 55.5 –  150.4      │ Unhealthy               │
    │ 201 – 300   │ 150.5 – 250.4      │ Very Unhealthy          │
    │ 301 – 500   │ 250.5 – 500.4      │ Hazardous               │
    └─────────────┴────────────────────┴─────────────────────────┘

    Formula:
    AQI = ((AQI_high - AQI_low) / (PM_high - PM_low))
          * (PM2.5 - PM_low) + AQI_low
    """

    # Each tuple: (pm_low, pm_high, aqi_low, aqi_high)
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    if pd.isna(pm25) or pm25 < 0:
        return np.nan

    for (pm_lo, pm_hi, aqi_lo, aqi_hi) in breakpoints:
        if pm_lo <= pm25 <= pm_hi:
            aqi = ((aqi_hi - aqi_lo) / (pm_hi - pm_lo)) * (pm25 - pm_lo) + aqi_lo
            return round(aqi)

    # Above 500 — off the chart
    return 500


def classify_health_risk(aqi):
    """
    Classifies AQI into health risk categories.
    This is what we'll predict in the classification model (Step 4).

    Returns a numeric label (easier for ML models):
      0 = Good
      1 = Moderate
      2 = Unhealthy for Sensitive Groups
      3 = Unhealthy
      4 = Very Unhealthy
      5 = Hazardous
    """
    if pd.isna(aqi):
        return np.nan
    elif aqi <= 50:
        return 0   # Good
    elif aqi <= 100:
        return 1   # Moderate
    elif aqi <= 150:
        return 2   # Unhealthy for Sensitive Groups
    elif aqi <= 200:
        return 3   # Unhealthy
    elif aqi <= 300:
        return 4   # Very Unhealthy
    else:
        return 5   # Hazardous


RISK_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy (Sensitive)",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}


def add_aqi_columns(df):
    """Adds AQI score and health risk category columns to the dataframe."""

    print("\n=== PART E: Calculating AQI and health risk ===")

    df["aqi"]         = df["pm25"].apply(pm25_to_aqi)
    df["risk_level"]  = df["aqi"].apply(classify_health_risk)
    df["risk_label"]  = df["risk_level"].map(RISK_LABELS)

    print("AQI distribution:")
    print(df["risk_label"].value_counts().to_string())

    return df


# ============================================================
#  PART F — Feature Engineering
# ============================================================

def engineer_features(df):
    """
    Creates new columns (features) that help the ML model
    learn patterns better.

    Features we create:
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Feature          │ Why it helps the model                         │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ hour             │ Rush hours (7-9am, 5-7pm) have higher pollution│
    │ day_of_week      │ Weekends have less traffic → less pollution     │
    │ month            │ Seasonal patterns (monsoon cleans the air)      │
    │ is_weekend       │ 0 or 1 — simpler for model than day name        │
    │ is_rush_hour     │ 1 if hour is 7-9 or 17-19, else 0              │
    │ season_encoded   │ Monsoon/dry season as a number                  │
    │ lag_pm25_1h      │ PM2.5 from 1 hour ago (air pollution persists) │
    │ lag_pm25_24h     │ PM2.5 from 24 hours ago (same time yesterday)  │
    │ lag_pm25_48h     │ PM2.5 from 48 hours ago                        │
    │ rolling_mean_24h │ Average PM2.5 over last 24 hours               │
    │ rolling_mean_7d  │ Average PM2.5 over last 7 days                 │
    │ temp_humidity    │ Temperature × humidity interaction              │
    └──────────────────┴────────────────────────────────────────────────┘
    """

    print("\n=== PART F: Engineering features ===")

    # Make sure data is sorted by location and time
    # (important for lag features — we need the previous rows to exist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location_id", "date", "hour"]).reset_index(drop=True)

    # ── Time features ──────────────────────────────────────
    df["hour"]        = df["date"].dt.hour if "hour" not in df.columns else df["hour"]
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)  # 1 if Sat or Sun

    # Rush hour: morning 7–9am and evening 5–7pm
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )

    # ── Season encoding ────────────────────────────────────
    # Convert season name to number for the ML model
    season_map = {
        "Northeast Monsoon":      0,
        "First Inter-monsoon":    1,
        "Southwest Monsoon":      2,
        "Second Inter-monsoon":   3,
    }
    if "season" in df.columns:
        df["season_encoded"] = df["season"].map(season_map).fillna(0).astype(int)
    else:
        # Derive from month if season column missing
        df["season_encoded"] = df["month"].apply(lambda m:
            0 if m in [12, 1, 2] else
            1 if m in [3, 4] else
            2 if m in [5, 6, 7, 8, 9] else 3
        )

    # ── Lag features ───────────────────────────────────────
    # These are the MOST IMPORTANT features for predicting air quality.
    # If PM2.5 was high an hour ago, it's likely still high now.
    # We calculate lags separately per location (using groupby).

    for lag_hours in [1, 24, 48]:
        col_name = f"lag_pm25_{lag_hours}h"
        df[col_name] = df.groupby("location_id")["pm25"].shift(lag_hours)

    # ── Rolling averages ───────────────────────────────────
    # Average PM2.5 over a window — smooths out spikes
    df["rolling_mean_24h"] = (
        df.groupby("location_id")["pm25"]
        .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    )

    df["rolling_mean_7d"] = (
        df.groupby("location_id")["pm25"]
        .transform(lambda x: x.rolling(window=24*7, min_periods=1).mean())
    )

    # ── Interaction feature ────────────────────────────────
    # High temperature + high humidity together = stagnant air = more pollution
    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100

    # ── Next-day AQI (regression TARGET) ──────────────────
    # This is what the regression model will PREDICT.
    # We shift the AQI back by 24 hours so each row has
    # "what will the AQI be 24 hours from now?"
    df["next_day_aqi"] = df.groupby("location_id")["aqi"].shift(-24)

    print(f"Features engineered. Dataset now has {df.shape[1]} columns.")
    print("\nNew feature columns:")
    new_cols = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
                "season_encoded", "lag_pm25_1h", "lag_pm25_24h", "lag_pm25_48h",
                "rolling_mean_24h", "rolling_mean_7d", "temp_humidity", "next_day_aqi"]
    for c in new_cols:
        if c in df.columns:
            non_null = df[c].notna().sum()
            print(f"  {c:<22} : {non_null} non-null values")

    return df


# ============================================================
#  PART G — Final cleanup and save
# ============================================================

def final_cleanup_and_save(df):
    """
    Does a final cleanup:
    - Drops rows where lag features are NaN
      (the first few rows of each location won't have lag values)
    - Reorders columns logically
    - Saves to data/processed_data.csv
    """

    print("\n=== PART G: Final cleanup and save ===")
    print(f"Rows before final cleanup: {len(df)}")

    # Drop rows where the most important lag feature is missing
    # (these are the first 24 hours of each location's data)
    df = df.dropna(subset=["lag_pm25_24h"])

    # Also drop rows where the regression target is missing
    # (the last 24 hours won't have a "next day" value)
    df = df.dropna(subset=["next_day_aqi"])

    print(f"Rows after final cleanup : {len(df)}")

    # Reorder columns — put identifiers first, then features, then targets
    id_cols       = ["date", "hour", "location_id", "location", "latitude", "longitude"]
    weather_cols  = ["temperature", "humidity", "wind_speed"]
    time_features = ["day_of_week", "month", "is_weekend", "is_rush_hour", "season_encoded"]
    lag_features  = ["lag_pm25_1h", "lag_pm25_24h", "lag_pm25_48h",
                     "rolling_mean_24h", "rolling_mean_7d", "temp_humidity"]
    target_cols   = ["pm25", "aqi", "risk_level", "risk_label", "next_day_aqi"]

    all_cols = id_cols + weather_cols + time_features + lag_features + target_cols

    # Only keep columns that exist
    all_cols = [c for c in all_cols if c in df.columns]
    df = df[all_cols]

    # Save
    save_path = os.path.join(DATA_DIR, "processed_data.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved processed data to: {save_path}")

    return df


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 2: Preprocessing")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    # Load
    df_pm25, df_weather = load_raw_data()

    # Clean each dataset separately
    df_pm25    = clean_pm25(df_pm25)
    df_weather = clean_weather(df_weather)

    # Merge into one table
    df = merge_datasets(df_pm25, df_weather)

    # Add AQI and health risk columns
    df = add_aqi_columns(df)

    # Create all the features the model will learn from
    df = engineer_features(df)

    # Final cleanup and save
    df = final_cleanup_and_save(df)

    # ── Preview ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))

    print("\nColumn summary:")
    print(df.describe(include="all").loc[["mean","min","max"]].to_string())

    print("\n→ Next step: run   python step3_eda.py")

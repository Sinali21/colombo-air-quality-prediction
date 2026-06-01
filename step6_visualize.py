# ============================================================
#  step6_visualize.py
#  Step 6 — Geo Visualization (Interactive Colombo Map)
# ============================================================
#
#  What this script does:
#  Creates an interactive HTML map of Colombo showing:
#    - Each sensor location as a colour-coded circle
#    - Colour = health risk zone (green/yellow/orange/red)
#    - Click any marker → popup with AQI, PM2.5, risk label
#    - Heatmap layer showing pollution intensity across the city
#    - A legend explaining the colour codes
#    - A time-of-day slider (morning / afternoon / evening / night)
#
#  Output saved to:
#    outputs/colombo_air_quality_map.html  ← open this in any browser!
#    outputs/map_static_snapshot.png       ← static image version
#
#  Run: python step6_visualize.py
#  Then: open outputs/colombo_air_quality_map.html in Chrome/Firefox
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from config import DATA_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── AQI colour scheme (standard international colours) ──────
RISK_CONFIG = {
    #  risk_level : (label,                   hex_color,  text_color)
    0: ("Good",                               "#00e400",  "#000000"),
    1: ("Moderate",                           "#ffff00",  "#000000"),
    2: ("Unhealthy for Sensitive Groups",     "#ff7e00",  "#000000"),
    3: ("Unhealthy",                          "#ff0000",  "#ffffff"),
    4: ("Very Unhealthy",                     "#8f3f97",  "#ffffff"),
    5: ("Hazardous",                          "#7e0023",  "#ffffff"),
}


# ============================================================
#  PART A — Load data and models
# ============================================================

def load_data():
    print("\n=== PART A: Loading data ===")

    data_path = os.path.join(DATA_DIR, "processed_data.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows")

    # Load trained models
    reg_path  = os.path.join(OUTPUT_DIR, "model_regression.pkl")
    clf_path  = os.path.join(OUTPUT_DIR, "model_classification.pkl")
    feat_path = os.path.join(OUTPUT_DIR, "model_feature_names.pkl")

    with open(reg_path,  "rb") as f: reg_model    = pickle.load(f)
    with open(clf_path,  "rb") as f: clf_model    = pickle.load(f)
    with open(feat_path, "rb") as f: feature_cols = pickle.load(f)

    print("Models loaded.")
    return df, reg_model, clf_model, feature_cols


# ============================================================
#  PART B — Aggregate stats per location
# ============================================================

def get_location_stats(df, reg_model, clf_model, feature_cols):
    """
    For each sensor location, compute:
      - Average PM2.5 and AQI across all records
      - Most common risk level
      - Time-of-day breakdown (morning/afternoon/evening/night averages)
      - Model prediction for a "typical" current reading
    """
    print("\n=== PART B: Aggregating per-location stats ===")

    locations = []

    for loc_name, group in df.groupby("location"):
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]

        avg_pm25 = group["pm25"].mean()
        avg_aqi  = group["aqi"].mean()
        max_aqi  = group["aqi"].max()

        # Most common risk level
        risk_mode = int(group["risk_level"].mode().iloc[0])

        # Time-of-day averages
        morning   = group[group["hour"].between(6, 11)]["pm25"].mean()
        afternoon = group[group["hour"].between(12, 17)]["pm25"].mean()
        evening   = group[group["hour"].between(17, 21)]["pm25"].mean()
        night     = group[(group["hour"] >= 22) | (group["hour"] <= 5)]["pm25"].mean()

        # Monthly averages for sparkline
        monthly   = group.groupby("month")["pm25"].mean().to_dict()

        # Predict AQI for "right now" using most recent available data
        latest = group.dropna(subset=feature_cols).tail(1)
        if len(latest) > 0:
            pred_aqi  = reg_model.predict(latest[feature_cols])[0]
            pred_risk = int(clf_model.predict(latest[feature_cols])[0])
        else:
            pred_aqi  = avg_aqi
            pred_risk = risk_mode

        locations.append({
            "name":       loc_name,
            "lat":        lat,
            "lon":        lon,
            "avg_pm25":   round(avg_pm25, 1),
            "avg_aqi":    round(avg_aqi, 1),
            "max_aqi":    round(max_aqi, 1),
            "risk_level": risk_mode,
            "pred_aqi":   round(pred_aqi, 1),
            "pred_risk":  pred_risk,
            "morning":    round(morning,   1),
            "afternoon":  round(afternoon, 1),
            "evening":    round(evening,   1),
            "night":      round(night,     1),
            "monthly":    monthly,
        })

        label, color, _ = RISK_CONFIG[risk_mode]
        print(f"  {loc_name:<25} avg AQI={avg_aqi:.0f}  risk={label}  pred_AQI={pred_aqi:.0f}")

    return locations


# ============================================================
#  PART C — Build the interactive Folium map
# ============================================================

def build_map(locations, df):
    """
    Builds a full interactive HTML map using Folium.

    Layers:
      1. Base tile layer (OpenStreetMap)
      2. Heatmap layer (PM2.5 intensity across city)
      3. Circle markers (one per sensor, colour = risk level)
      4. Legend (colour → risk level explanation)
    """
    print("\n=== PART C: Building interactive map ===")

    # Centre map on Colombo
    m = folium.Map(
        location=[6.9271, 79.8612],
        zoom_start=12,
        tiles="CartoDB positron",   # clean light basemap
        control_scale=True,
    )

    # ── Layer 1: Heatmap ──────────────────────────────────
    # Each row in df gives one point on the heatmap
    # weight = pm25 value (higher = more intense heat colour)
    heat_data = df[["latitude", "longitude", "pm25"]].dropna().values.tolist()

    HeatMap(
        heat_data,
        name="PM2.5 Heatmap",
        min_opacity=0.3,
        max_zoom=15,
        radius=25,
        blur=20,
        gradient={
            0.0:  "#00e400",   # good
            0.25: "#ffff00",   # moderate
            0.5:  "#ff7e00",   # unhealthy sensitive
            0.75: "#ff0000",   # unhealthy
            1.0:  "#7e0023",   # hazardous
        }
    ).add_to(m)

    # ── Layer 2: Sensor markers ───────────────────────────
    marker_group = folium.FeatureGroup(name="Sensor Locations", show=True)

    for loc in locations:
        risk       = loc["risk_level"]
        pred_risk  = loc["pred_risk"]
        label, color, text_color = RISK_CONFIG[risk]
        _, pred_color, pred_text = RISK_CONFIG[pred_risk]
        pred_label = RISK_CONFIG[pred_risk][0]

        # ── Popup HTML ───────────────────────────────────
        # This is what appears when you CLICK a marker
        popup_html = f"""
        <div style="
            font-family: Arial, sans-serif;
            font-size: 13px;
            min-width: 240px;
            padding: 4px;
        ">
            <h3 style="margin:0 0 8px 0; color:#2c3e50; border-bottom:2px solid {color}; padding-bottom:4px;">
                📍 {loc['name']}
            </h3>

            <!-- Current / Historical stats -->
            <table style="width:100%; border-collapse:collapse; margin-bottom:10px;">
                <tr>
                    <td style="padding:3px 6px; color:#555;">Average PM2.5</td>
                    <td style="padding:3px 6px; font-weight:bold;">{loc['avg_pm25']} µg/m³</td>
                </tr>
                <tr style="background:#f8f8f8;">
                    <td style="padding:3px 6px; color:#555;">Average AQI</td>
                    <td style="padding:3px 6px; font-weight:bold;">{loc['avg_aqi']}</td>
                </tr>
                <tr>
                    <td style="padding:3px 6px; color:#555;">Worst recorded AQI</td>
                    <td style="padding:3px 6px; font-weight:bold; color:#e74c3c;">{loc['max_aqi']}</td>
                </tr>
                <tr style="background:#f8f8f8;">
                    <td style="padding:3px 6px; color:#555;">Typical risk level</td>
                    <td style="padding:3px 6px;">
                        <span style="background:{color}; color:{text_color};
                                     padding:2px 8px; border-radius:4px; font-size:11px;">
                            {label}
                        </span>
                    </td>
                </tr>
            </table>

            <!-- Model prediction -->
            <div style="background:#eaf4fb; border-left:3px solid #3498db;
                        padding:6px 10px; border-radius:0 4px 4px 0; margin-bottom:10px;">
                <div style="color:#2980b9; font-size:11px; font-weight:bold; margin-bottom:3px;">
                    🤖 MODEL PREDICTION (next day)
                </div>
                <div>Predicted AQI: <strong>{loc['pred_aqi']}</strong></div>
                <div>Risk zone:
                    <span style="background:{pred_color}; color:{pred_text};
                                 padding:1px 8px; border-radius:4px; font-size:11px;">
                        {pred_label}
                    </span>
                </div>
            </div>

            <!-- Time of day breakdown -->
            <div style="font-size:11px; color:#555; margin-bottom:4px; font-weight:bold;">
                PM2.5 by time of day:
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:12px;">
                <tr>
                    <td>🌅 Morning (6am–12pm)</td>
                    <td style="text-align:right; font-weight:bold;">{loc['morning']} µg/m³</td>
                </tr>
                <tr style="background:#f8f8f8;">
                    <td>☀️ Afternoon (12–5pm)</td>
                    <td style="text-align:right; font-weight:bold;">{loc['afternoon']} µg/m³</td>
                </tr>
                <tr>
                    <td>🌆 Evening (5–9pm)</td>
                    <td style="text-align:right; font-weight:bold;">{loc['evening']} µg/m³</td>
                </tr>
                <tr style="background:#f8f8f8;">
                    <td>🌙 Night (10pm–5am)</td>
                    <td style="text-align:right; font-weight:bold;">{loc['night']} µg/m³</td>
                </tr>
            </table>

            <div style="font-size:10px; color:#aaa; margin-top:8px; text-align:right;">
                Air Quality Project · Colombo, Sri Lanka
            </div>
        </div>
        """

        # Tooltip (shown on hover — before clicking)
        tooltip_text = f"{loc['name']} | AQI: {loc['avg_aqi']} | {label}"

        # Circle radius scales with pollution level
        radius = 300 + (loc["avg_aqi"] / 500) * 400

        folium.Circle(
            location=[loc["lat"], loc["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip_text,
        ).add_to(marker_group)

        # Small white label with location name
        folium.Marker(
            location=[loc["lat"] + 0.003, loc["lon"]],
            icon=folium.DivIcon(
                html=f"""<div style="
                    font-family:Arial; font-size:10px; font-weight:bold;
                    color:#2c3e50; white-space:nowrap;
                    text-shadow: 1px 1px 2px white, -1px -1px 2px white;
                ">{loc['name']}</div>""",
                icon_size=(150, 20),
                icon_anchor=(75, 0),
            )
        ).add_to(marker_group)

    marker_group.add_to(m)

    # ── Layer control (toggle heatmap / markers) ──────────
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legend ────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 15px;
        z-index: 1000;
        background: white;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
        min-width: 200px;
    ">
        <div style="font-weight:bold; font-size:13px; margin-bottom:8px;
                    color:#2c3e50; border-bottom:1px solid #eee; padding-bottom:4px;">
            AQI Health Risk Levels
        </div>
    """
    for level, (label, color, text_color) in RISK_CONFIG.items():
        aqi_ranges = {0:"0–50", 1:"51–100", 2:"101–150",
                      3:"151–200", 4:"201–300", 5:"301–500"}
        legend_html += f"""
        <div style="display:flex; align-items:center; margin:4px 0;">
            <div style="width:18px; height:18px; background:{color};
                        border-radius:3px; margin-right:8px; flex-shrink:0;
                        border:1px solid rgba(0,0,0,0.1);"></div>
            <span style="color:#333;">{aqi_ranges[level]} — {label}</span>
        </div>"""

    legend_html += """
        <div style="margin-top:10px; padding-top:6px; border-top:1px solid #eee;
                    font-size:10px; color:#888;">
            Click any circle for details.<br>
            Circle size = relative pollution level.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title banner ──────────────────────────────────────
    title_html = """
    <div style="
        position: fixed;
        top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000;
        background: rgba(255,255,255,0.95);
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px 20px;
        font-family: Arial, sans-serif;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <div style="font-size:15px; font-weight:bold; color:#2c3e50;">
            🌬️ Colombo Air Quality — Health Risk Map
        </div>
        <div style="font-size:11px; color:#888; margin-top:2px;">
            PM2.5 monitoring · ML-predicted AQI · Click markers for details
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# ============================================================
#  PART D — Static snapshot chart (for reports / README)
# ============================================================

def save_static_snapshot(locations):
    """
    Creates a static matplotlib chart showing all locations
    on a simple coordinate plot — useful for reports and README.
    """
    print("\n=== PART D: Static snapshot chart ===")

    fig, ax = plt.subplots(figsize=(9, 8))

    # Background colour
    ax.set_facecolor("#e8f4f8")
    fig.patch.set_facecolor("#f0f4f8")

    for loc in locations:
        risk  = loc["risk_level"]
        label, color, text_color = RISK_CONFIG[risk]
        size = 800 + (loc["avg_aqi"] / 500) * 1200

        ax.scatter(loc["lon"], loc["lat"],
                   s=size, c=color, alpha=0.75,
                   edgecolors="#333", linewidths=1.2, zorder=3)

        ax.annotate(
            f"{loc['name']}\nAQI: {loc['avg_aqi']:.0f}",
            (loc["lon"], loc["lat"]),
            textcoords="offset points", xytext=(0, 18),
            ha="center", fontsize=9, fontweight="500",
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="#ccc")
        )

    # Legend patches
    patches = [
        mpatches.Patch(color=v[1], label=f"AQI {aqi} — {v[0]}")
        for aqi, (k, v) in zip(
            ["0–50","51–100","101–150","151–200","201–300","301+"],
            RISK_CONFIG.items()
        )
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#ccc")

    ax.set_title("Colombo Air Quality — Sensor Locations & Risk Levels\n"
                 "(circle size = relative pollution level)",
                 fontsize=13, pad=14, color="#2c3e50")
    ax.set_xlabel("Longitude", fontsize=10, color="#555")
    ax.set_ylabel("Latitude",  fontsize=10, color="#555")
    ax.grid(True, alpha=0.3, color="white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_static_snapshot.png")
    plt.savefig(path)
    plt.close()
    print(f"  Static snapshot saved → {path}")


# ============================================================
#  PART E — Monthly trend chart per location
# ============================================================

def save_monthly_trend_chart(df):
    """
    Bar chart showing average monthly PM2.5 per location side by side.
    Good for showing the seasonal pattern per area.
    """
    print("\n=== PART E: Monthly trend chart ===")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    monthly = (
        df.groupby(["location", "month"])["pm25"]
        .mean()
        .reset_index()
    )
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])

    locs   = monthly["location"].unique()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"][:len(locs)]

    fig, ax = plt.subplots(figsize=(12, 5))
    x       = np.arange(12)
    width   = 0.8 / len(locs)

    for i, (loc, color) in enumerate(zip(locs, colors)):
        data   = monthly[monthly["location"] == loc].sort_values("month")
        offset = (i - len(locs)/2 + 0.5) * width
        ax.bar(x + offset, data["pm25"].values,
               width=width * 0.9, color=color,
               alpha=0.82, label=loc, edgecolor="white")

    ax.axhline(35.4, color="black", linestyle="--",
               linewidth=1.2, label="WHO guideline (35.4 µg/m³)")

    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly average PM2.5 by location — seasonal pattern comparison",
                 fontsize=12, pad=12)
    ax.set_ylabel("Average PM2.5 (µg/m³)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_monthly_trend.png")
    plt.savefig(path)
    plt.close()
    print(f"  Monthly trend chart saved → {path}")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 6: Geo Visualization")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    # Load everything
    df, reg_model, clf_model, feature_cols = load_data()

    # Get per-location stats + predictions
    locations = get_location_stats(df, reg_model, clf_model, feature_cols)

    # Build interactive Folium map
    folium_map = build_map(locations, df)

    # Save as HTML
    map_path = os.path.join(OUTPUT_DIR, "colombo_air_quality_map.html")
    folium_map.save(map_path)
    print(f"\nInteractive map saved → {map_path}")
    print("Open this file in Chrome or Firefox to see the map!")

    # Static charts
    save_static_snapshot(locations)
    save_monthly_trend_chart(df)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GEO VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nFiles saved to outputs/:")
    print("  colombo_air_quality_map.html  ← OPEN THIS in a browser")
    print("  map_static_snapshot.png")
    print("  map_monthly_trend.png")

    print("\nLocation summary:")
    for loc in sorted(locations, key=lambda x: x["avg_aqi"], reverse=True):
        risk_label = RISK_CONFIG[loc["risk_level"]][0]
        print(f"  {loc['name']:<28} avg AQI={loc['avg_aqi']:<6} → {risk_label}")

    print("\n" + "=" * 60)
    print("  CORE PIPELINE COMPLETE!")
    print("=" * 60)
    print("""
  ✅ Step 1 — Data collection
  ✅ Step 2 — Preprocessing
  ✅ Step 3 — EDA (8 charts)
  ✅ Step 4 — Model training
  ✅ Step 5 — Evaluation
  ✅ Step 6 — Geo visualization
  ⬜ Step 7 — Frontend dashboard (next!)

  The ML pipeline is fully complete.
  Next: build the frontend web app to show
  everything in a nice dashboard!
    """)

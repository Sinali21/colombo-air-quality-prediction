# ============================================================
#  step3_eda.py
#  Step 3 — Exploratory Data Analysis (EDA)
# ============================================================
#
#  What this script does:
#  Loads processed_data.csv and creates 8 charts that reveal
#  the patterns in Colombo's air quality data.
#
#  Charts saved to outputs/ folder:
#  1. eda_01_aqi_distribution.png     — How often is air Good/Moderate/Hazardous?
#  2. eda_02_hourly_pattern.png       — Which hours of day are most polluted?
#  3. eda_03_monthly_pattern.png      — Which months are worst? (monsoon effect)
#  4. eda_04_weekly_pattern.png       — Weekdays vs weekends
#  5. eda_05_location_comparison.png  — Which area of Colombo is most polluted?
#  6. eda_06_weather_correlation.png  — How does weather affect PM2.5?
#  7. eda_07_time_series.png          — PM2.5 over time (the full picture)
#  8. eda_08_rush_hour_vs_normal.png  — Rush hour spike clearly shown
#
#  Run: python step3_eda.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from config import DATA_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global plot style ──────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"]      = 130
plt.rcParams["savefig.dpi"]     = 150
plt.rcParams["savefig.bbox"]    = "tight"
plt.rcParams["font.family"]     = "sans-serif"

# Colour palette — maps each risk level to a colour
RISK_COLORS = {
    "Good":                    "#2ecc71",
    "Moderate":                "#f1c40f",
    "Unhealthy (Sensitive)":   "#e67e22",
    "Unhealthy":               "#e74c3c",
    "Very Unhealthy":          "#9b59b6",
    "Hazardous":               "#7f0000",
}

RISK_ORDER = [
    "Good", "Moderate", "Unhealthy (Sensitive)",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]


# ============================================================
#  Load data
# ============================================================

def load_data():
    path = os.path.join(DATA_DIR, "processed_data.csv")
    if not os.path.exists(path):
        print("[ERROR] processed_data.csv not found.")
        print("Please run step2_preprocess.py first!")
        exit()

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows from processed_data.csv")
    return df


# ============================================================
#  Chart 1 — AQI distribution (how often is air safe?)
# ============================================================

def chart_aqi_distribution(df):
    print("  Chart 1: AQI distribution...")

    counts = df["risk_label"].value_counts()
    # Keep only categories that exist in the data, in correct order
    order  = [r for r in RISK_ORDER if r in counts.index]
    counts = counts[order]
    colors = [RISK_COLORS[r] for r in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    # Add count + percentage labels on each bar
    total = len(df)
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9.5, fontweight="500"
        )

    ax.set_title("How often is Colombo's air quality at each risk level?", fontsize=13, pad=14)
    ax.set_xlabel("Health risk category", fontsize=11)
    ax.set_ylabel("Number of hourly readings", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.2)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_01_aqi_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 2 — Hourly pattern (rush hour effect)
# ============================================================

def chart_hourly_pattern(df):
    print("  Chart 2: Hourly PM2.5 pattern...")

    hourly = df.groupby("hour")["pm25"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour bars by rush hour
    bar_colors = [
        "#e74c3c" if (7 <= h <= 9 or 17 <= h <= 19) else "#3498db"
        for h in hourly["hour"]
    ]
    bars = ax.bar(hourly["hour"], hourly["pm25"], color=bar_colors,
                  edgecolor="white", linewidth=0.6)

    # Shade rush hour zones
    ax.axvspan(6.5,  9.5, alpha=0.08, color="red",   label="Morning rush")
    ax.axvspan(16.5, 19.5, alpha=0.08, color="orange", label="Evening rush")

    ax.set_title("Average PM2.5 by hour of day — rush hour spikes clearly visible",
                 fontsize=13, pad=14)
    ax.set_xlabel("Hour of day (0 = midnight)", fontsize=11)
    ax.set_ylabel("Average PM2.5 (µg/m³)", fontsize=11)
    ax.set_xticks(range(0, 24))

    # Legend
    rush_patch  = mpatches.Patch(color="#e74c3c", label="Rush hour")
    other_patch = mpatches.Patch(color="#3498db", label="Other hours")
    ax.legend(handles=[rush_patch, other_patch], fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_02_hourly_pattern.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 3 — Monthly pattern (monsoon effect)
# ============================================================

def chart_monthly_pattern(df):
    print("  Chart 3: Monthly PM2.5 pattern...")

    monthly = df.groupby("month")["pm25"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour by season
    season_colors = {
        "NE Monsoon":        "#3498db",
        "Inter-monsoon 1":   "#2ecc71",
        "SW Monsoon":        "#1abc9c",
        "Inter-monsoon 2":   "#e67e22",
    }

    def get_season(m):
        if m in [12, 1, 2]:   return "NE Monsoon"
        elif m in [3, 4]:     return "Inter-monsoon 1"
        elif m in [5,6,7,8,9]: return "SW Monsoon"
        else:                  return "Inter-monsoon 2"

    bar_colors = [
        {"NE Monsoon": "#3498db", "Inter-monsoon 1": "#2ecc71",
         "SW Monsoon": "#1abc9c", "Inter-monsoon 2": "#e67e22"}[get_season(m)]
        for m in monthly["month"]
    ]

    bars = ax.bar(monthly["month_name"], monthly["pm25"],
                  color=bar_colors, edgecolor="white", linewidth=0.8)

    # PM2.5 guideline line
    ax.axhline(35.4, color="#e74c3c", linestyle="--", linewidth=1.3,
               label="WHO 24h guideline (35.4 µg/m³)")

    ax.set_title("Average PM2.5 by month — monsoon season cleans the air",
                 fontsize=13, pad=14)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Average PM2.5 (µg/m³)", fontsize=11)

    # Season legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in season_colors.items()]
    patches.append(mpatches.Patch(color="#e74c3c", label="WHO guideline"))
    ax.legend(handles=patches, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_03_monthly_pattern.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 4 — Weekday vs weekend
# ============================================================

def chart_weekly_pattern(df):
    print("  Chart 4: Weekday vs weekend pattern...")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("day_of_week")["pm25"].mean().reset_index()
    daily["day_name"] = daily["day_of_week"].apply(lambda d: day_names[d])

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#e74c3c" if d >= 5 else "#3498db" for d in daily["day_of_week"]]
    ax.bar(daily["day_name"], daily["pm25"], color=colors,
           edgecolor="white", linewidth=0.8)

    ax.set_title("Average PM2.5 by day of week — weekends vs weekdays",
                 fontsize=13, pad=14)
    ax.set_xlabel("Day of week", fontsize=11)
    ax.set_ylabel("Average PM2.5 (µg/m³)", fontsize=11)

    weekday_patch = mpatches.Patch(color="#3498db", label="Weekday")
    weekend_patch = mpatches.Patch(color="#e74c3c", label="Weekend")
    ax.legend(handles=[weekday_patch, weekend_patch], fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_04_weekly_pattern.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 5 — Location comparison
# ============================================================

def chart_location_comparison(df):
    print("  Chart 5: Location comparison...")

    loc_stats = df.groupby("location")["pm25"].agg(["mean","median","max"]).reset_index()
    loc_stats = loc_stats.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    y = range(len(loc_stats))
    ax.barh(y, loc_stats["mean"],   color="#e74c3c", alpha=0.85, label="Mean PM2.5")
    ax.barh(y, loc_stats["median"], color="#3498db", alpha=0.60, label="Median PM2.5")

    ax.set_yticks(list(y))
    ax.set_yticklabels(loc_stats["location"], fontsize=11)
    ax.set_title("PM2.5 by location — which area of Colombo is most polluted?",
                 fontsize=13, pad=14)
    ax.set_xlabel("PM2.5 (µg/m³)", fontsize=11)
    ax.legend(fontsize=10)

    # Add mean value label
    for i, (_, row) in enumerate(loc_stats.iterrows()):
        ax.text(row["mean"] + 0.5, i, f'{row["mean"]:.1f}',
                va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_05_location_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 6 — Weather correlation heatmap
# ============================================================

def chart_weather_correlation(df):
    print("  Chart 6: Weather correlation heatmap...")

    cols = ["pm25", "aqi", "temperature", "humidity",
            "wind_speed", "is_rush_hour", "is_weekend",
            "lag_pm25_24h", "rolling_mean_24h"]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 9}
    )

    ax.set_title("Correlation between PM2.5/AQI and weather features\n"
                 "(red = strong positive, green = strong negative)",
                 fontsize=12, pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_06_weather_correlation.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 7 — Time series (PM2.5 over full year)
# ============================================================

def chart_time_series(df):
    print("  Chart 7: Time series...")

    # Use one location only (cleaner chart)
    loc = df["location"].iloc[0]
    df_loc = df[df["location"] == loc].copy()

    # Daily average to keep chart readable
    daily = df_loc.groupby("date")["pm25"].mean().reset_index()

    # 30-day rolling average (trend line)
    daily["trend"] = daily["pm25"].rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(daily["date"], daily["pm25"],
                    alpha=0.25, color="#3498db", label="Daily avg PM2.5")
    ax.plot(daily["date"], daily["pm25"],
            color="#3498db", linewidth=0.6, alpha=0.6)
    ax.plot(daily["date"], daily["trend"],
            color="#e74c3c", linewidth=2.2, label="30-day trend")

    # WHO guideline
    ax.axhline(35.4, color="#e67e22", linestyle="--",
               linewidth=1.2, label="WHO guideline (35.4 µg/m³)")

    ax.set_title(f"PM2.5 over time — {loc} (daily average + 30-day trend)",
                 fontsize=13, pad=14)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_07_time_series.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Chart 8 — Rush hour vs normal hours box plot
# ============================================================

def chart_rush_hour_boxplot(df):
    print("  Chart 8: Rush hour vs normal boxplot...")

    df_plot = df.copy()
    df_plot["Period"] = df_plot["is_rush_hour"].map(
        {1: "Rush hour\n(7–9am, 5–7pm)", 0: "Normal hours"}
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=df_plot, x="Period", y="pm25",
        palette={"Rush hour\n(7–9am, 5–7pm)": "#e74c3c",
                 "Normal hours": "#3498db"},
        width=0.45, linewidth=1.2, ax=ax,
        order=["Normal hours", "Rush hour\n(7–9am, 5–7pm)"]
    )

    ax.set_title("PM2.5 during rush hour vs normal hours",
                 fontsize=13, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)

    # Show median values
    for i, period in enumerate(["Normal hours", "Rush hour\n(7–9am, 5–7pm)"]):
        median = df_plot[df_plot["Period"] == period]["pm25"].median()
        ax.text(i, median + 1, f"Median: {median:.1f}",
                ha="center", fontsize=10, fontweight="500", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="grey", alpha=0.6))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_08_rush_hour_vs_normal.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved → {path}")


# ============================================================
#  Print key insights as text
# ============================================================

def print_key_insights(df):
    print("\n" + "=" * 60)
    print("  KEY INSIGHTS FROM YOUR DATA")
    print("=" * 60)

    total = len(df)
    good  = (df["risk_label"] == "Good").sum()
    mod   = (df["risk_label"] == "Moderate").sum()
    unh   = df["risk_label"].isin(["Unhealthy (Sensitive)","Unhealthy","Very Unhealthy","Hazardous"]).sum()

    print(f"\n Air quality breakdown:")
    print(f"   Good          : {good/total*100:.1f}% of hours")
    print(f"   Moderate      : {mod/total*100:.1f}% of hours")
    print(f"   Unhealthy+    : {unh/total*100:.1f}% of hours")

    rush     = df[df["is_rush_hour"] == 1]["pm25"].mean()
    non_rush = df[df["is_rush_hour"] == 0]["pm25"].mean()
    print(f"\n Rush hour effect:")
    print(f"   Rush hour avg PM2.5   : {rush:.1f} µg/m³")
    print(f"   Non-rush hour avg     : {non_rush:.1f} µg/m³")
    print(f"   Rush hour is {(rush/non_rush - 1)*100:.0f}% worse")

    worst_month = df.groupby("month")["pm25"].mean().idxmax()
    best_month  = df.groupby("month")["pm25"].mean().idxmin()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    print(f"\n Seasonal pattern:")
    print(f"   Worst month : {month_names[worst_month]} (dry season)")
    print(f"   Best month  : {month_names[best_month]} (monsoon cleans the air)")

    worst_loc = df.groupby("location")["pm25"].mean().idxmax()
    best_loc  = df.groupby("location")["pm25"].mean().idxmin()
    print(f"\n Location comparison:")
    print(f"   Most polluted  : {worst_loc}")
    print(f"   Least polluted : {best_loc}")

    corr_wind = df["pm25"].corr(df["wind_speed"])
    corr_hum  = df["pm25"].corr(df["humidity"])
    print(f"\n Weather correlations:")
    print(f"   Wind speed vs PM2.5 : {corr_wind:.3f}  (negative = wind disperses pollution)")
    print(f"   Humidity vs PM2.5   : {corr_hum:.3f}  (monsoon humidity = cleaner air)")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 3: Exploratory Data Analysis (EDA)")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    df = load_data()

    print("\nGenerating charts...")
    chart_aqi_distribution(df)
    chart_hourly_pattern(df)
    chart_monthly_pattern(df)
    chart_weekly_pattern(df)
    chart_location_comparison(df)
    chart_weather_correlation(df)
    chart_time_series(df)
    chart_rush_hour_boxplot(df)

    print_key_insights(df)

    print("\n" + "=" * 60)
    print("  EDA COMPLETE — 8 charts saved to outputs/ folder")
    print("=" * 60)
    print("\nOpen the outputs/ folder in VS Code or File Explorer")
    print("to view all the charts.")
    print("\n→ Next step: run   python step4_model.py")

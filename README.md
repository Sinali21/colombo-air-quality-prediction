# 🌬️ Colombo Air Quality Prediction & Health Risk Analysis

> A full end-to-end Machine Learning project that predicts next-day AQI levels and classifies health risk zones across Colombo, Sri Lanka — using real government sensor data.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat&logo=scikit-learn)
![Folium](https://img.shields.io/badge/Folium-0.15-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 📌 Project Overview

Colombo's air pollution is a growing public health concern — driven by traffic congestion, industrial activity, and seasonal weather patterns. Yet most residents have no easy way to know whether tomorrow's air will be safe to breathe.

This project builds a data-driven solution:

- **Fetches real PM2.5 data** from government sensors via the OpenAQ v3 API
- **Merges with weather data** (temperature, humidity, wind speed) from OpenWeatherMap
- **Trains two ML models** — a regressor to predict next-day AQI, and a classifier to label health risk zones
- **Visualises results** on an interactive geo map of Colombo with colour-coded risk zones

---

## 🎯 Purpose & Real-World Impact

| Problem | This Project's Solution |
|---------|------------------------|
| No next-day AQI forecast for Colombo | Random Forest model predicts next-day AQI with ±12.4 point accuracy |
| Hard to know if your area is safe | Classification model labels zones: Good / Moderate / Unhealthy / Hazardous |
| No visual city-wide overview | Interactive Folium map with heatmap + clickable sensor markers |
| Limited awareness of pollution patterns | EDA reveals rush hour spikes, monsoon effects, seasonal trends |

---

## 🗂️ Project Structure

```
colombo-air-quality-prediction/
│
├── step1_collect_data.py     ← Fetch real data from OpenAQ + OpenWeatherMap APIs
├── step2_preprocess.py       ← Clean, merge, engineer 13 ML features
├── step3_eda.py              ← Exploratory analysis — 8 charts
├── step4_model.py            ← Train regression + classification models
├── step5_evaluate.py         ← Deep evaluation — residuals, cross-validation
├── step6_visualize.py        ← Interactive Colombo map (Folium)
│
├── config.py                 ← API keys and settings (add your own keys)
├── requirements.txt          ← All Python dependencies
├── README.md                 ← You are here
│
├── data/                     ← Raw + processed CSV files (auto-generated)
└── outputs/                  ← Charts, maps, trained model .pkl files
```

---

## 🔬 Machine Learning Pipeline

```
OpenAQ API          OpenWeatherMap API
     │                      │
     ▼                      ▼
  PM2.5 Data  ──────  Weather Data
           │
           ▼
      Preprocessing
  (clean + merge + feature engineering)
           │
           ▼
     EDA & Insights
  (patterns by hour, month, location)
           │
      ┌────┴────┐
      ▼         ▼
 Regression  Classification
  (next-day    (risk zone:
    AQI)       Good/Moderate/
               Unhealthy/Hazardous)
      │         │
      └────┬────┘
           ▼
      Evaluation
  (MAE, R², accuracy,
   cross-validation)
           │
           ▼
    Geo Visualization
  (Interactive Colombo map)
```

---

## 📊 Model Performance

### Regression Model — Predicting Next-Day AQI
| Metric | Score | Meaning |
|--------|-------|---------|
| MAE | **±12.4 AQI points** | Average prediction error |
| RMSE | 24.78 | Root mean squared error |
| R² | **0.607** | 60.7% of variance explained |
| CV R² (5-fold) | 0.550 ± 0.038 | Consistent across all folds |

### Classification Model — Health Risk Zone
| Metric | Score |
|--------|-------|
| Accuracy | **79.7%** |
| CV Accuracy (5-fold) | 79.9% ± 0.3% |
| Moderate F1 | 0.83 |
| Unhealthy F1 | 0.85 |

> The model correctly predicts the health risk zone **8 out of every 10 times** on unseen data.

---

## 🔑 Key Findings from EDA

- **Rush hour effect** — PM2.5 is **79% higher** during 7–9am and 5–7pm compared to other hours
- **Monsoon effect** — SW Monsoon (May–September) significantly reduces PM2.5; July is the cleanest month
- **Dry season** — February has the highest average PM2.5 due to stagnant air conditions
- **Most important feature** — `lag_pm25_24h` (yesterday's reading) explains 50% of next-day predictions
- **Wind is the best natural cleaner** — strong correlation (-0.46) between wind speed and PM2.5

---

## 🗺️ Interactive Map Features

The Folium map (`outputs/colombo_air_quality_map.html`) includes:

- **PM2.5 heatmap** — colour gradient showing pollution intensity across the city
- **Sensor markers** — circle size scales with pollution level, colour = risk zone
- **Click popups** — shows average PM2.5, AQI, model prediction, time-of-day breakdown
- **Layer toggle** — switch heatmap / markers on and off
- **AQI legend** — standard international colour coding

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/colombo-air-quality-prediction.git
cd colombo-air-quality-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys to `config.py`
```python
OPENAQ_API_KEY      = "your_openaq_api_key_here"      # Free at openaq.org
OPENWEATHER_API_KEY = "your_openweather_key_here"     # Free at openweathermap.org
```

### 4. Run each step in order
```bash
python step1_collect_data.py    # Fetch data from APIs
python step2_preprocess.py      # Clean and engineer features
python step3_eda.py             # Generate EDA charts
python step4_model.py           # Train ML models
python step5_evaluate.py        # Evaluate model performance
python step6_visualize.py       # Build interactive map
```

### 5. View the map
Open `outputs/colombo_air_quality_map.html` in any browser.

> **Note:** If you don't have API keys yet, the project runs in demo mode automatically using realistic synthetic Colombo data — so you can explore the full pipeline without signing up.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **pandas / numpy** | Data manipulation |
| **scikit-learn** | ML models (Random Forest) |
| **XGBoost** | Gradient boosting (available as alternative) |
| **matplotlib / seaborn** | EDA charts |
| **Folium** | Interactive geo map |
| **requests** | API calls |
| **OpenAQ v3 API** | Real PM2.5 sensor data |
| **OpenWeatherMap API** | Weather data |

---

## 📡 Data Sources

- **[OpenAQ](https://openaq.org)** — Open-source air quality data platform. Free API with real government sensor data. Colombo has 2 active PM2.5 sensors (AirNow + US Embassy StateAir).
- **[OpenWeatherMap](https://openweathermap.org)** — Weather API. Free tier provides current and forecast data.

---

## 🌍 Why This Matters for Sri Lanka

Sri Lanka currently has very limited public air quality monitoring infrastructure. This project demonstrates how open data sources and machine learning can be combined to:

- Provide **next-day health risk forecasts** for the public
- Help **vulnerable populations** (elderly, children, asthma patients) plan outdoor activities
- Give **policymakers** data-driven insights into pollution hotspots and peak hours
- Serve as a foundation for a **city-wide early warning system**

---

## 📁 Output Files

After running all steps, the `outputs/` folder contains:

```
outputs/
├── eda_01_aqi_distribution.png
├── eda_02_hourly_pattern.png
├── eda_03_monthly_pattern.png
├── eda_04_weekly_pattern.png
├── eda_05_location_comparison.png
├── eda_06_weather_correlation.png
├── eda_07_time_series.png
├── eda_08_rush_hour_vs_normal.png
├── model_regression.pkl
├── model_classification.pkl
├── model_feature_names.pkl
├── model_reg_feature_importance.png
├── model_reg_actual_vs_predicted.png
├── model_clf_confusion_matrix.png
├── eval_01_residuals.png
├── eval_02_error_by_season.png
├── eval_03_error_by_hour.png
├── eval_04_error_by_location.png
├── eval_05_crossval_scores.png
├── eval_06_prediction_timeline.png
├── colombo_air_quality_map.html   ← Open in browser
├── map_static_snapshot.png
└── map_monthly_trend.png
```

---

## 👩‍💻 Author

**Sinali**
- 📍 Sri Lanka
- 🔗 GitHub: https://github.com/Sinali21

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

*Built as part of a data science learning journey — combining real-world local impact with end-to-end ML engineering.*

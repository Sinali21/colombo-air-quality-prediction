# ============================================================
#  config.py  —  Your API keys and project settings
#  Edit this file before running anything else!
# ============================================================

# ----------------------------------------------------------
# STEP 1: Get your API keys
# ----------------------------------------------------------
# OpenAQ API key:
#   → Go to https://openaq.org/register
#   → Create a free account
#   → Copy your API key and paste it below
#
# OpenWeatherMap API key:
#   → Go to https://openweathermap.org/api
#   → Sign up for free
#   → Go to "API Keys" tab in your account
#   → Copy your key and paste it below
# ----------------------------------------------------------

OPENAQ_API_KEY       = "API_KEY_GOES_HERE"
OPENWEATHER_API_KEY  = "API_KEY_GOES_HERE"

# ----------------------------------------------------------
# Colombo coordinates — we use these to find nearby sensors
# ----------------------------------------------------------
COLOMBO_LAT  = 6.9271
COLOMBO_LON  = 79.8612

# Search radius around Colombo (in meters)
# 25000 = 25 km — catches all sensors in the Colombo metro area
SEARCH_RADIUS_M = 25000

# ----------------------------------------------------------
# How much historical data to fetch
# ----------------------------------------------------------
# Format: "YYYY-MM-DD"
DATE_FROM = "2023-01-01"
DATE_TO   = "2024-12-31"

# ----------------------------------------------------------
# Where to save data files
# ----------------------------------------------------------
DATA_DIR    = "data/"
OUTPUT_DIR  = "outputs/"

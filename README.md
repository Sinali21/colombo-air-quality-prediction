# Air Quality Prediction & Health Risk Analysis — Sri Lanka

## Project Structure
```
air_quality_project/
│
├── data/                  ← Raw & processed CSV files saved here
├── notebooks/             ← Jupyter notebooks (step by step)
├── src/                   ← Python scripts (reusable functions)
├── outputs/               ← Charts, maps, model outputs
│
├── step1_collect_data.py  ← OpenAQ + OpenWeatherMap data collection
├── step2_preprocess.py    ← Cleaning, merging, feature engineering
├── step3_eda.py           ← Exploratory data analysis + plots
├── step4_model.py         ← Train regression + classification models
├── step5_evaluate.py      ← Evaluate models, metrics, confusion matrix
├── step6_visualize.py     ← Folium map, geo risk zones
│
├── config.py              ← API keys and settings (edit this first!)
└── requirements.txt       ← All Python packages to install
```

## How to Run
1. Install packages:  `pip install -r requirements.txt`
2. Edit `config.py` with your API keys
3. Run each step file in order: `python step1_collect_data.py`

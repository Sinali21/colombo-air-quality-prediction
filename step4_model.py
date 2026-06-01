# ============================================================
#  step4_model.py
#  Step 4 — Train ML Models
# ============================================================
#
#  What this script does:
#  Trains TWO models on the processed data:
#
#  MODEL A — Regression (Random Forest)
#    Predicts: next_day_aqi (a number, e.g. 112.5)
#    Question: "What will the AQI be tomorrow?"
#
#  MODEL B — Classification (Random Forest)
#    Predicts: risk_level (0=Good, 1=Moderate, 2=Unhealthy...)
#    Question: "What health risk zone will this area be in?"
#
#  What gets saved to outputs/:
#    model_regression.pkl      ← trained regression model
#    model_classification.pkl  ← trained classification model
#    model_feature_names.pkl   ← list of features used (needed in step5)
#
#  Run: python step4_model.py
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, accuracy_score
)
from config import DATA_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
#  PART A — Load and prepare data
# ============================================================

def load_and_prepare():
    """
    Loads processed_data.csv and splits it into:
      X  — feature matrix (the inputs the model learns from)
      y_reg  — regression target   (next_day_aqi)
      y_clf  — classification target (risk_level)

    Then splits into train set (80%) and test set (20%).

    WHY 80/20 split?
      We train the model on 80% of the data.
      We test it on the remaining 20% — data the model has
      NEVER seen. This tells us how well it generalises
      to new, real-world data.
    """

    print("\n=== PART A: Loading and preparing data ===")

    path = os.path.join(DATA_DIR, "processed_data.csv")
    df   = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows")

    # ── Define features (X) ───────────────────────────────
    # These are the columns the model uses as INPUT to make predictions.
    # We chose these based on what the EDA showed matters most.
    feature_cols = [
        # Time features
        "hour", "day_of_week", "month", "is_weekend",
        "is_rush_hour", "season_encoded",
        # Weather
        "temperature", "humidity", "wind_speed", "temp_humidity",
        # Lag features (most important!)
        "lag_pm25_1h", "lag_pm25_24h", "lag_pm25_48h",
        # Rolling averages (trend)
        "rolling_mean_24h", "rolling_mean_7d",
    ]

    # Make sure all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        print("Please run step2_preprocess.py again.")
        exit()

    # Drop any rows with NaN in features or targets
    df_clean = df.dropna(subset=feature_cols + ["next_day_aqi", "risk_level"])
    print(f"Rows after dropping NaN: {len(df_clean):,}")

    X     = df_clean[feature_cols]
    y_reg = df_clean["next_day_aqi"]    # regression target
    y_clf = df_clean["risk_level"].astype(int)  # classification target

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Features used        : {feature_cols}")
    print(f"\nRegression target (next_day_aqi):")
    print(f"  Min: {y_reg.min():.1f}  Max: {y_reg.max():.1f}  Mean: {y_reg.mean():.1f}")
    print(f"\nClassification target (risk_level):")
    print(dict(y_clf.value_counts().sort_index()))

    # ── Train / Test split ────────────────────────────────
    # random_state=42 means the split is the same every time you run this.
    # This is important for reproducibility.
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)
    )

    print(f"\nTrain set size : {len(X_train):,} rows ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Test set size  : {len(X_test):,} rows  ({len(X_test)/len(X)*100:.0f}%)")

    return (X_train, X_test,
            y_reg_train, y_reg_test,
            y_clf_train, y_clf_test,
            feature_cols)


# ============================================================
#  PART B — Train Regression Model
# ============================================================

def train_regression_model(X_train, X_test, y_train, y_test, feature_cols):
    """
    Trains a Random Forest Regressor to predict next-day AQI.

    WHY Random Forest?
    ─────────────────
    A Random Forest builds MANY decision trees (n_estimators=200),
    each trained on a random subset of the data and features.
    The final prediction is the AVERAGE of all trees.

    This is better than a single decision tree because:
      - Less overfitting (doesn't memorise the training data)
      - Handles non-linear patterns well (AQI isn't just a straight line)
      - Works well without much tuning

    Key parameters explained:
      n_estimators=200   → build 200 trees (more = more accurate, slower)
      max_depth=15       → each tree can go 15 levels deep
      min_samples_leaf=5 → each leaf needs at least 5 data points
      n_jobs=-1          → use all CPU cores (faster training)
      random_state=42    → same result every run
    """

    print("\n=== PART B: Training Regression Model ===")
    print("Predicting: next-day AQI value")
    print("Training... (this takes about 30-60 seconds)")

    model = RandomForestRegressor(
        n_estimators   = 200,
        max_depth      = 15,
        min_samples_leaf = 5,
        n_jobs         = -1,
        random_state   = 42,
        verbose        = 0
    )

    model.fit(X_train, y_train)
    print("Training complete!")

    # ── Evaluate on test set ──────────────────────────────
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\nRegression Results (on unseen test data):")
    print(f"  MAE  (Mean Absolute Error)  : {mae:.2f} AQI points")
    print(f"  RMSE (Root Mean Sq. Error)  : {rmse:.2f} AQI points")
    print(f"  R²   (Explained variance)   : {r2:.4f}  ({r2*100:.1f}% of variance explained)")
    print(f"\n  Interpretation:")
    print(f"  → On average, predictions are off by ±{mae:.1f} AQI points")
    print(f"  → The model explains {r2*100:.1f}% of the variation in next-day AQI")

    # ── Feature importance ────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 5 most important features:")
    for _, row in importance_df.head(5).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<22} {bar} {row['importance']:.4f}")

    # ── Save feature importance chart ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ["#e74c3c" if i < 3 else "#3498db"
               for i in range(len(importance_df))]
    ax.barh(importance_df["feature"][::-1],
            importance_df["importance"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_title("Regression model — feature importance\n(which inputs matter most for predicting next-day AQI)",
                 fontsize=12, pad=12)
    ax.set_xlabel("Importance score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_reg_feature_importance.png"))
    plt.close()

    # ── Save actual vs predicted chart ────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.2, s=8, color="#3498db")
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"Regression: actual vs predicted AQI\nR² = {r2:.3f}  |  MAE = {mae:.1f}",
                 fontsize=12, pad=12)
    ax.set_xlabel("Actual next-day AQI",    fontsize=11)
    ax.set_ylabel("Predicted next-day AQI", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_reg_actual_vs_predicted.png"))
    plt.close()

    return model, y_pred


# ============================================================
#  PART C — Train Classification Model
# ============================================================

def train_classification_model(X_train, X_test, y_train, y_test, feature_cols):
    """
    Trains a Random Forest Classifier to predict health risk zone.

    The output is one of these classes:
      0 = Good
      1 = Moderate
      2 = Unhealthy for Sensitive Groups
      3 = Unhealthy
      4 = Very Unhealthy
      5 = Hazardous

    class_weight="balanced" tells the model to pay more attention
    to rare classes (like Hazardous) so it doesn't just predict
    "Moderate" for everything.
    """

    print("\n=== PART C: Training Classification Model ===")
    print("Predicting: health risk zone (Good/Moderate/Unhealthy/Hazardous)")
    print("Training... (this takes about 30-60 seconds)")

    model = RandomForestClassifier(
        n_estimators    = 200,
        max_depth       = 15,
        min_samples_leaf = 5,
        class_weight    = "balanced",
        n_jobs          = -1,
        random_state    = 42,
        verbose         = 0
    )

    model.fit(X_train, y_train)
    print("Training complete!")

    # ── Evaluate on test set ──────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nClassification Results (on unseen test data):")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"\n  Interpretation: the model correctly classifies")
    print(f"  the health risk zone {acc*100:.1f}% of the time")

    # Detailed report
    risk_names = {
        0: "Good", 1: "Moderate", 2: "Unhealthy(Sens)",
        3: "Unhealthy", 4: "Very Unhealthy", 5: "Hazardous"
    }
    labels_present = sorted(y_test.unique())
    target_names   = [risk_names[l] for l in labels_present]

    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred,
                                labels=labels_present,
                                target_names=target_names,
                                zero_division=0))

    # ── Confusion matrix chart ────────────────────────────
    cm     = confusion_matrix(y_test, y_pred, labels=labels_present)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of actual class"})
    ax.set_title(f"Classification confusion matrix — accuracy {acc*100:.1f}%\n"
                 "Rows = actual class, Columns = predicted class",
                 fontsize=11, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_clf_confusion_matrix.png"))
    plt.close()

    # ── Feature importance chart ──────────────────────────
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ["#9b59b6" if i < 3 else "#8e44ad"
               for i in range(len(importance_df))]
    ax.barh(importance_df["feature"][::-1],
            importance_df["importance"][::-1],
            color="#9b59b6", edgecolor="white")
    ax.set_title("Classification model — feature importance\n"
                 "(which inputs matter most for predicting health risk zone)",
                 fontsize=12, pad=12)
    ax.set_xlabel("Importance score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_clf_feature_importance.png"))
    plt.close()

    return model, y_pred


# ============================================================
#  PART D — Save models to disk
# ============================================================

def save_models(reg_model, clf_model, feature_cols):
    """
    Saves the trained models as .pkl (pickle) files.

    Pickle is Python's way of saving any object to a file.
    Later in step5 and the frontend, we load these files
    instead of retraining from scratch every time.

    Think of it like saving a Word document — you do the work
    once and reopen the saved file when you need it.
    """

    print("\n=== PART D: Saving models ===")

    reg_path   = os.path.join(OUTPUT_DIR, "model_regression.pkl")
    clf_path   = os.path.join(OUTPUT_DIR, "model_classification.pkl")
    feat_path  = os.path.join(OUTPUT_DIR, "model_feature_names.pkl")

    with open(reg_path,  "wb") as f:
        pickle.dump(reg_model, f)
    print(f"  Regression model saved    → {reg_path}")

    with open(clf_path,  "wb") as f:
        pickle.dump(clf_model, f)
    print(f"  Classification model saved → {clf_path}")

    with open(feat_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"  Feature names saved        → {feat_path}")

    print("\n  To load a model later:")
    print("    import pickle")
    print("    with open('outputs/model_regression.pkl', 'rb') as f:")
    print("        model = pickle.load(f)")
    print("    prediction = model.predict(X_new)")


# ============================================================
#  PART E — Quick prediction demo
# ============================================================

def prediction_demo(reg_model, clf_model, feature_cols):
    """
    Shows a simple example of using the trained models
    to make a real prediction.

    This is exactly what the frontend will do:
    1. Take current conditions as input
    2. Feed them to the model
    3. Get a prediction back
    """

    print("\n=== PART E: Prediction demo ===")
    print("Example: predicting AQI for tomorrow morning in Colombo Fort")

    RISK_LABELS = {
        0: "Good", 1: "Moderate", 2: "Unhealthy (Sensitive)",
        3: "Unhealthy", 4: "Very Unhealthy", 5: "Hazardous"
    }

    # Scenario 1: Rush hour, dry season
    scenario_1 = {
        "hour":             8,      # 8am
        "day_of_week":      1,      # Tuesday
        "month":            2,      # February (dry season)
        "is_weekend":       0,      # weekday
        "is_rush_hour":     1,      # morning rush
        "season_encoded":   0,      # NE monsoon / dry
        "temperature":      29.5,
        "humidity":         78.0,
        "wind_speed":       2.1,    # low wind = pollution trapped
        "temp_humidity":    29.5 * 78.0 / 100,
        "lag_pm25_1h":      65.0,   # PM2.5 was 65 an hour ago
        "lag_pm25_24h":     58.0,   # PM2.5 was 58 yesterday same time
        "lag_pm25_48h":     52.0,
        "rolling_mean_24h": 55.0,
        "rolling_mean_7d":  50.0,
    }

    # Scenario 2: Monsoon, weekend, windy
    scenario_2 = {
        "hour":             14,     # 2pm
        "day_of_week":      6,      # Sunday
        "month":            7,      # July (SW monsoon)
        "is_weekend":       1,      # weekend
        "is_rush_hour":     0,      # not rush hour
        "season_encoded":   2,      # SW monsoon
        "temperature":      27.2,
        "humidity":         88.0,   # high humidity from monsoon
        "wind_speed":       5.8,    # strong wind
        "temp_humidity":    27.2 * 88.0 / 100,
        "lag_pm25_1h":      22.0,
        "lag_pm25_24h":     18.0,
        "lag_pm25_48h":     20.0,
        "rolling_mean_24h": 20.0,
        "rolling_mean_7d":  24.0,
    }

    for i, scenario in enumerate([scenario_1, scenario_2], 1):
        label = "Rush hour, dry season (Feb)" if i == 1 else "Monsoon, weekend, windy (Jul)"
        print(f"\n  Scenario {i}: {label}")

        X_new = pd.DataFrame([scenario])[feature_cols]

        pred_aqi      = reg_model.predict(X_new)[0]
        pred_risk     = clf_model.predict(X_new)[0]
        pred_risk_lbl = RISK_LABELS.get(pred_risk, "Unknown")

        print(f"    Predicted next-day AQI  : {pred_aqi:.0f}")
        print(f"    Predicted risk zone     : {pred_risk_lbl}")
        print(f"    Recommendation          : ", end="")
        if pred_risk <= 1:
            print("Safe for all activities")
        elif pred_risk == 2:
            print("Sensitive groups should limit outdoor exposure")
        elif pred_risk == 3:
            print("Everyone should reduce prolonged outdoor activity")
        else:
            print("Stay indoors. Wear a mask if going outside.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 4: Model Training")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    # Load and prepare
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_clf_train, y_clf_test,
     feature_cols) = load_and_prepare()

    # Train both models
    reg_model, reg_preds = train_regression_model(
        X_train, X_test, y_reg_train, y_reg_test, feature_cols
    )

    clf_model, clf_preds = train_classification_model(
        X_train, X_test, y_clf_train, y_clf_test, feature_cols
    )

    # Save models to disk
    save_models(reg_model, clf_model, feature_cols)

    # Demo prediction
    prediction_demo(reg_model, clf_model, feature_cols)

    print("\n" + "=" * 60)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 60)
    print("\nFiles saved to outputs/:")
    print("  model_regression.pkl")
    print("  model_classification.pkl")
    print("  model_feature_names.pkl")
    print("  model_reg_feature_importance.png")
    print("  model_reg_actual_vs_predicted.png")
    print("  model_clf_confusion_matrix.png")
    print("  model_clf_feature_importance.png")
    print("\n→ Next step: run   python step5_evaluate.py")

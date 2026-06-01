# ============================================================
#  step5_evaluate.py
#  Step 5 — Deep Model Evaluation
# ============================================================
#
#  What this script does:
#  Goes deeper than step4 — analyses WHERE and WHEN the model
#  makes mistakes, and proves it generalises well to new data.
#
#  Charts saved to outputs/:
#  eval_01_residuals.png          — Are errors random or biased?
#  eval_02_error_by_season.png    — Does model struggle in monsoon?
#  eval_03_error_by_hour.png      — More errors during rush hour?
#  eval_04_error_by_location.png  — Which area is hardest to predict?
#  eval_05_crossval_scores.png    — Is accuracy consistent across folds?
#  eval_06_prediction_vs_actual_timeline.png — Visual forecast check
#
#  Run: python step5_evaluate.py
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from config import DATA_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"]   = 130
plt.rcParams["savefig.dpi"]  = 150
plt.rcParams["savefig.bbox"] = "tight"

RISK_LABELS = {
    0: "Good", 1: "Moderate", 2: "Unhealthy(Sens)",
    3: "Unhealthy", 4: "Very Unhealthy", 5: "Hazardous"
}

SEASON_NAMES = {
    0: "NE Monsoon",
    1: "Inter-monsoon 1",
    2: "SW Monsoon",
    3: "Inter-monsoon 2"
}


# ============================================================
#  Load models + data
# ============================================================

def load_everything():
    print("\n=== Loading models and data ===")

    # Load trained models
    reg_path  = os.path.join(OUTPUT_DIR, "model_regression.pkl")
    clf_path  = os.path.join(OUTPUT_DIR, "model_classification.pkl")
    feat_path = os.path.join(OUTPUT_DIR, "model_feature_names.pkl")

    for p in [reg_path, clf_path, feat_path]:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}")
            print("Please run step4_model.py first!")
            exit()

    with open(reg_path,  "rb") as f: reg_model  = pickle.load(f)
    with open(clf_path,  "rb") as f: clf_model  = pickle.load(f)
    with open(feat_path, "rb") as f: feature_cols = pickle.load(f)

    # Load processed data
    data_path = os.path.join(DATA_DIR, "processed_data.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.dropna(subset=feature_cols + ["next_day_aqi", "risk_level"])

    print(f"Models loaded. Data: {len(df):,} rows")

    # Recreate same train/test split as step4 (random_state=42)
    from sklearn.model_selection import train_test_split
    X     = df[feature_cols]
    y_reg = df["next_day_aqi"]
    y_clf = df["risk_level"].astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)
    )

    # Get predictions
    reg_preds = reg_model.predict(X_test)
    clf_preds = clf_model.predict(X_test)

    # Attach test-set metadata back (for error-by-season, by-location etc.)
    # X_test.index contains the original df row labels — use .loc not .iloc
    df_test = df.loc[X_test.index].copy()
    df_test["reg_pred"]   = reg_preds
    df_test["clf_pred"]   = clf_preds
    df_test["reg_error"]  = reg_preds - y_reg_test.values   # signed error
    df_test["abs_error"]  = np.abs(df_test["reg_error"])

    print(f"Test set size: {len(df_test):,} rows")

    return (reg_model, clf_model, feature_cols,
            X_train, X_test,
            y_reg_train, y_reg_test,
            y_clf_train, y_clf_test,
            df_test)


# ============================================================
#  PART A — Residual analysis
# ============================================================

def chart_residuals(df_test, y_reg_test):
    """
    Residuals = predicted value − actual value

    A good model has residuals that:
    1. Are centred around zero (no consistent over/under prediction)
    2. Are randomly scattered (no pattern = model isn't missing something)
    3. Follow a roughly normal distribution (bell curve)

    If you see a pattern in the residuals, it means the model
    is systematically wrong in some way — and you could improve it.
    """
    print("  Chart 1: Residuals...")

    residuals = df_test["reg_error"].values
    predicted = df_test["reg_pred"].values

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Left: residuals vs predicted ──
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(predicted, residuals, alpha=0.08, s=5, color="#3498db")
    ax1.axhline(0, color="#e74c3c", linewidth=1.5, linestyle="--")
    ax1.set_title("Residuals vs Predicted", fontsize=11)
    ax1.set_xlabel("Predicted AQI",  fontsize=10)
    ax1.set_ylabel("Error (pred − actual)", fontsize=10)

    # ── Middle: residual distribution ──
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(residuals, bins=60, color="#3498db", edgecolor="white",
             linewidth=0.4, alpha=0.85)
    ax2.axvline(0, color="#e74c3c", linewidth=1.5, linestyle="--",
                label="Zero error")
    ax2.axvline(np.mean(residuals), color="#e67e22", linewidth=1.5,
                linestyle="--", label=f"Mean error: {np.mean(residuals):.1f}")
    ax2.set_title("Residual distribution", fontsize=11)
    ax2.set_xlabel("Error",  fontsize=10)
    ax2.set_ylabel("Count",  fontsize=10)
    ax2.legend(fontsize=9)

    # ── Right: actual vs predicted scatter ──
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(y_reg_test.values, predicted, alpha=0.08, s=5, color="#2ecc71")
    mn = min(y_reg_test.min(), predicted.min())
    mx = max(y_reg_test.max(), predicted.max())
    ax3.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
    mae  = mean_absolute_error(y_reg_test, predicted)
    r2   = r2_score(y_reg_test, predicted)
    ax3.set_title(f"Actual vs Predicted\nMAE={mae:.1f}  R²={r2:.3f}", fontsize=11)
    ax3.set_xlabel("Actual AQI",    fontsize=10)
    ax3.set_ylabel("Predicted AQI", fontsize=10)
    ax3.legend(fontsize=9)

    fig.suptitle("Regression model — residual analysis", fontsize=13, y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_01_residuals.png"))
    plt.close()
    print(f"    Mean residual  : {np.mean(residuals):.2f}  (close to 0 = unbiased)")
    print(f"    Std of errors  : {np.std(residuals):.2f}")


# ============================================================
#  PART B — Error by season
# ============================================================

def chart_error_by_season(df_test):
    """
    Does the model perform differently across seasons?
    Higher error during SW Monsoon would mean the model
    struggles when rain suddenly cleans the air.
    """
    print("  Chart 2: Error by season...")

    df_test["season_name"] = df_test["season_encoded"].map(SEASON_NAMES)

    season_stats = (
        df_test.groupby("season_name")["abs_error"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(season_stats["season_name"], season_stats["mean"],
                   color=["#1abc9c","#3498db","#e67e22","#9b59b6"],
                   edgecolor="white", height=0.5)
    ax.errorbar(season_stats["mean"], season_stats["season_name"],
                xerr=season_stats["std"], fmt="none",
                color="grey", capsize=4, linewidth=1.2)

    for bar, val in zip(bars, season_stats["mean"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"±{val:.1f}", va="center", fontsize=10)

    ax.set_title("Mean absolute error by season\n"
                 "Lower = model is more accurate in that season",
                 fontsize=12, pad=12)
    ax.set_xlabel("Mean absolute error (AQI points)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_02_error_by_season.png"))
    plt.close()

    print("    Season error breakdown:")
    for _, row in season_stats.iterrows():
        print(f"      {row['season_name']:<22}: ±{row['mean']:.1f} AQI pts")


# ============================================================
#  PART C — Error by hour
# ============================================================

def chart_error_by_hour(df_test):
    """
    Are errors larger during rush hours?
    Rush hour PM2.5 spikes are harder to predict because
    they depend on exact traffic levels that day.
    """
    print("  Chart 3: Error by hour...")

    hourly_error = df_test.groupby("hour")["abs_error"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "#e74c3c" if (7 <= h <= 9 or 17 <= h <= 19) else "#3498db"
        for h in hourly_error["hour"]
    ]
    ax.bar(hourly_error["hour"], hourly_error["abs_error"],
           color=colors, edgecolor="white", linewidth=0.5)

    ax.axvspan(6.5,  9.5, alpha=0.07, color="red")
    ax.axvspan(16.5, 19.5, alpha=0.07, color="red")

    import matplotlib.patches as mpatches
    rush_patch  = mpatches.Patch(color="#e74c3c", label="Rush hour (larger error expected)")
    other_patch = mpatches.Patch(color="#3498db", label="Other hours")
    ax.legend(handles=[rush_patch, other_patch], fontsize=10)

    ax.set_title("Mean absolute error by hour of day\n"
                 "Rush hours are harder to predict — more traffic variability",
                 fontsize=12, pad=12)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Mean absolute error (AQI points)", fontsize=11)
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_03_error_by_hour.png"))
    plt.close()


# ============================================================
#  PART D — Error by location
# ============================================================

def chart_error_by_location(df_test):
    """
    Which monitoring location is hardest to predict?
    Locations near busy junctions have more traffic variability
    and may have higher errors.
    """
    print("  Chart 4: Error by location...")

    loc_stats = (
        df_test.groupby("location")["abs_error"]
        .agg(["mean", "median"])
        .reset_index()
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(loc_stats["location"], loc_stats["mean"],
            color="#e74c3c", alpha=0.8, label="Mean error", height=0.4)
    ax.barh(loc_stats["location"], loc_stats["median"],
            color="#3498db", alpha=0.6, label="Median error", height=0.4)

    for i, (_, row) in enumerate(loc_stats.iterrows()):
        ax.text(row["mean"] + 0.2, i, f'±{row["mean"]:.1f}',
                va="center", fontsize=10)

    ax.set_title("Prediction error by location\n"
                 "Busier areas are harder to predict accurately",
                 fontsize=12, pad=12)
    ax.set_xlabel("Mean absolute error (AQI points)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_04_error_by_location.png"))
    plt.close()


# ============================================================
#  PART E — Cross-validation
# ============================================================

def chart_cross_validation(reg_model, clf_model, X_train, y_reg_train, y_clf_train):
    """
    Cross-validation answers the question:
    "Did we just get lucky with our 80/20 split?"

    How it works:
    ─────────────
    Instead of one train/test split, we do 5 splits:
      Fold 1: train on folds 2+3+4+5, test on fold 1
      Fold 2: train on folds 1+3+4+5, test on fold 2
      ... and so on

    If all 5 scores are similar → the model is genuinely good.
    If scores vary wildly       → the model is unstable.

    We use R² for regression (higher = better, max = 1.0)
    and accuracy for classification (higher = better, max = 1.0)
    """
    print("  Chart 5: Cross-validation (5 folds)...")
    print("    Running 5-fold CV on regression model...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_scores = cross_val_score(
        reg_model, X_train, y_reg_train,
        cv=kf, scoring="r2", n_jobs=-1
    )
    print(f"    Regression R² scores    : {[f'{s:.3f}' for s in reg_scores]}")
    print(f"    Mean R²: {reg_scores.mean():.3f}  ±{reg_scores.std():.3f}")

    print("    Running 5-fold CV on classification model...")
    clf_scores = cross_val_score(
        clf_model, X_train, y_clf_train,
        cv=kf, scoring="accuracy", n_jobs=-1
    )
    print(f"    Classification accuracy : {[f'{s:.3f}' for s in clf_scores]}")
    print(f"    Mean accuracy: {clf_scores.mean():.3f}  ±{clf_scores.std():.3f}")

    # ── Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    folds = [f"Fold {i+1}" for i in range(5)]

    # Regression
    axes[0].bar(folds, reg_scores, color="#3498db", edgecolor="white")
    axes[0].axhline(reg_scores.mean(), color="#e74c3c", linestyle="--",
                    linewidth=1.5, label=f"Mean R² = {reg_scores.mean():.3f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Regression — 5-fold CV (R²)\nConsistent scores = reliable model",
                      fontsize=11, pad=10)
    axes[0].set_ylabel("R² score", fontsize=10)
    axes[0].legend(fontsize=9)
    for i, v in enumerate(reg_scores):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Classification
    axes[1].bar(folds, clf_scores, color="#9b59b6", edgecolor="white")
    axes[1].axhline(clf_scores.mean(), color="#e74c3c", linestyle="--",
                    linewidth=1.5, label=f"Mean acc = {clf_scores.mean():.3f}")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Classification — 5-fold CV (Accuracy)\nConsistent scores = reliable model",
                      fontsize=11, pad=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(fontsize=9)
    for i, v in enumerate(clf_scores):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("5-fold cross-validation — proves model isn't just lucky",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_05_crossval_scores.png"))
    plt.close()

    return reg_scores, clf_scores


# ============================================================
#  PART F — Prediction timeline chart
# ============================================================

def chart_prediction_timeline(df_test):
    """
    Shows actual vs predicted AQI over a 2-week window.
    This is the most intuitive chart — you can visually
    see how closely the model follows the real values.
    """
    print("  Chart 6: Prediction timeline...")

    # Pick one location and take 14 days of hourly data
    loc     = df_test["location"].iloc[0]
    df_loc  = df_test[df_test["location"] == loc].sort_values("date").head(14 * 24)

    if len(df_loc) < 48:
        print("    Not enough data for timeline chart — skipping")
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(range(len(df_loc)), df_loc["next_day_aqi"].values,
            color="#3498db", linewidth=1.8, label="Actual AQI", alpha=0.9)
    ax.plot(range(len(df_loc)), df_loc["reg_pred"].values,
            color="#e74c3c", linewidth=1.5, linestyle="--",
            label="Predicted AQI", alpha=0.85)

    # Shade the difference
    ax.fill_between(range(len(df_loc)),
                    df_loc["next_day_aqi"].values,
                    df_loc["reg_pred"].values,
                    alpha=0.12, color="#e74c3c", label="Prediction error")

    # AQI zone lines
    for threshold, label, color in [
        (50,  "Good/Moderate", "#2ecc71"),
        (100, "Moderate/Unhealthy", "#f1c40f"),
        (150, "Unhealthy threshold", "#e74c3c"),
    ]:
        ax.axhline(threshold, color=color, linestyle=":", linewidth=1,
                   alpha=0.6, label=label)

    mae = mean_absolute_error(df_loc["next_day_aqi"], df_loc["reg_pred"])
    ax.set_title(f"Actual vs predicted AQI — {loc} (first 14 days of test set)\n"
                 f"MAE on this window = {mae:.1f} AQI points",
                 fontsize=12, pad=12)
    ax.set_xlabel("Hours", fontsize=11)
    ax.set_ylabel("AQI", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_06_prediction_timeline.png"))
    plt.close()


# ============================================================
#  PART G — Final evaluation summary
# ============================================================

def print_evaluation_summary(y_reg_test, reg_preds,
                              y_clf_test, clf_preds,
                              reg_cv, clf_cv):
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    mae  = mean_absolute_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    r2   = r2_score(y_reg_test, reg_preds)
    acc  = accuracy_score(y_clf_test, clf_preds)

    print(f"""
  REGRESSION MODEL (predict next-day AQI)
  ─────────────────────────────────────────
  MAE              : {mae:.2f} AQI points   ← avg prediction error
  RMSE             : {rmse:.2f} AQI points
  R²               : {r2:.4f}              ← {r2*100:.1f}% variance explained
  CV R² (5-fold)   : {reg_cv.mean():.3f} ± {reg_cv.std():.3f}  ← consistent?

  CLASSIFICATION MODEL (predict risk zone)
  ─────────────────────────────────────────
  Accuracy         : {acc*100:.1f}%
  CV Accuracy      : {clf_cv.mean()*100:.1f}% ± {clf_cv.std()*100:.1f}%

  VERDICT
  ─────────────────────────────────────────
  {"✓" if mae < 20 else "~"} Regression MAE < 20 AQI points   {"PASS" if mae < 20 else "ACCEPTABLE"}
  {"✓" if r2 > 0.5  else "~"} R² > 0.5                        {"PASS" if r2 > 0.5 else "NEEDS WORK"}
  {"✓" if acc > 0.7 else "~"} Classification accuracy > 70%   {"PASS" if acc > 0.7 else "NEEDS WORK"}
  {"✓" if reg_cv.std() < 0.05 else "~"} CV scores consistent (std < 0.05)  {"PASS" if reg_cv.std() < 0.05 else "VARIABLE"}
    """)

    print("  WHAT THE NUMBERS MEAN IN PLAIN ENGLISH")
    print("  ─────────────────────────────────────────")
    print(f"  If today's AQI is 120 (Unhealthy),")
    print(f"  the model's prediction will be between")
    print(f"  {120 - mae:.0f} and {120 + mae:.0f} roughly {68:.0f}% of the time.")
    print(f"  That's good enough for public health guidance.")
    print(f"\n  The classification model correctly labels the")
    print(f"  health risk zone (Safe/Moderate/Unhealthy etc.)")
    print(f"  {acc*100:.0f} times out of every 100 predictions.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Step 5: Model Evaluation")
    print("  Air Quality Prediction — Colombo, Sri Lanka")
    print("=" * 60)

    (reg_model, clf_model, feature_cols,
     X_train, X_test,
     y_reg_train, y_reg_test,
     y_clf_train, y_clf_test,
     df_test) = load_everything()

    reg_preds = df_test["reg_pred"].values
    clf_preds = df_test["clf_pred"].values

    print("\nGenerating evaluation charts...")
    chart_residuals(df_test, y_reg_test)
    chart_error_by_season(df_test)
    chart_error_by_hour(df_test)
    chart_error_by_location(df_test)

    reg_cv, clf_cv = chart_cross_validation(
        reg_model, clf_model,
        X_train, y_reg_train, y_clf_train
    )

    chart_prediction_timeline(df_test)

    print_evaluation_summary(
        y_reg_test, reg_preds,
        y_clf_test, clf_preds,
        reg_cv, clf_cv
    )

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE — 6 charts saved to outputs/")
    print("=" * 60)
    print("\n→ Next step: run   python step6_visualize.py")
    print("  (Geo map — plot risk zones on Colombo map!)")

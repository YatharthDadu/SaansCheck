"""Run the complete Delhi AQI next-day daily & hourly forecasting pipeline on cached data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"

AQ_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "uv_index_clear_sky",
    "methane",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]

IST = timezone(timedelta(hours=5, minutes=30))
NOW_IST = datetime.fromtimestamp(time.time(), tz=IST)
TODAY_IST = pd.Timestamp(NOW_IST.date())
TOMORROW_IST = TODAY_IST + pd.Timedelta(days=1)
EVAL_END = min(TODAY_IST - pd.Timedelta(days=1), pd.Timestamp("2026-11-15"))

TRAIN_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")


def compute_india_naqi_pm25(pm25: float) -> tuple[float, str]:
    """Compute India CPCB PM2.5 sub-index and category."""
    if np.isnan(pm25):
        return np.nan, "Unknown"
    breakpoints = [
        (0.0, 30.0, 0.0, 50.0, "Good"),
        (30.0, 60.0, 51.0, 100.0, "Satisfactory"),
        (60.0, 90.0, 101.0, 200.0, "Moderate"),
        (90.0, 120.0, 201.0, 300.0, "Poor"),
        (120.0, 250.0, 301.0, 400.0, "Very Poor"),
        (250.0, 500.0, 401.0, 500.0, "Severe"),
    ]
    for bp_lo, bp_hi, i_lo, i_hi, cat in breakpoints:
        if pm25 <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + i_lo
            return float(round(aqi, 1)), cat
    return 500.0, "Severe"


def binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """Compute precision, recall, F1, false negatives, and false positives."""
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "positives": int(np.sum(y_true_bin == 1)),
        "tp": tp,
        "fp_false_alarms": fp,
        "fn_missed": fn,
        "tn": tn,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


def load_hourly_dataframe() -> tuple[pd.DataFrame, dict]:
    """Load Open-Meteo AQ + Weather JSONs and compute exact variable coverage audit."""
    aq_raw = json.loads((DATA_DIR / "openmeteo_aq_hourly.json").read_text(encoding="utf-8"))
    w_raw = json.loads((DATA_DIR / "openmeteo_weather_hourly.json").read_text(encoding="utf-8"))

    df_aq = pd.DataFrame(aq_raw["hourly"])
    df_w = pd.DataFrame(w_raw["hourly"])
    df = df_aq.merge(df_w, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour

    total_rows = len(df)
    coverage_audit = {}
    for col in AQ_VARIABLES + WEATHER_VARIABLES:
        non_null = int(df[col].notna().sum())
        first_valid = df.loc[df[col].notna(), "time"].min()
        coverage_audit[col] = {
            "source": "air_quality" if col in AQ_VARIABLES else "weather",
            "non_null_hours": non_null,
            "total_hours": total_rows,
            "coverage_pct": round(100.0 * non_null / total_rows, 2),
            "first_valid": str(first_valid) if pd.notna(first_valid) else None,
        }
    return df, coverage_audit


def build_daily_features(df_hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Aggregate hourly Delhi records into daily features for forecasting Day t+1."""
    df = df_hourly.copy()
    # Northwest wind sector (270 to 360 deg): upwind from Punjab/Haryana
    df["is_nw_wind"] = ((df["wind_direction_10m"] >= 270.0) & (df["wind_direction_10m"] <= 360.0)).astype(float)
    df["is_calm"] = (df["wind_speed_10m"] < 6.0).astype(float)
    df["wind_rad"] = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(df["wind_rad"])
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(df["wind_rad"])

    evening_mask = df["hour"] >= 20
    morning_mask = (df["hour"] >= 6) & (df["hour"] <= 9)

    daily = df.groupby("date").agg(
        pm25_mean=("pm2_5", "mean"),
        pm25_max=("pm2_5", "max"),
        pm25_min=("pm2_5", "min"),
        pm10_mean=("pm10", "mean"),
        co_mean=("carbon_monoxide", "mean"),
        co2_mean=("carbon_dioxide", "mean"),
        no2_mean=("nitrogen_dioxide", "mean"),
        so2_mean=("sulphur_dioxide", "mean"),
        o3_mean=("ozone", "mean"),
        aod_mean=("aerosol_optical_depth", "mean"),
        dust_mean=("dust", "mean"),
        uv_max=("uv_index", "max"),
        uv_clear_max=("uv_index_clear_sky", "max"),
        ch4_mean=("methane", "mean"),
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        rh_mean=("relative_humidity_2m", "mean"),
        precip_sum=("precipitation", "sum"),
        pressure_mean=("surface_pressure", "mean"),
        wind_mean=("wind_speed_10m", "mean"),
        wind_min=("wind_speed_10m", "min"),
        wind_u_mean=("wind_u", "mean"),
        wind_v_mean=("wind_v", "mean"),
        nw_wind_share=("is_nw_wind", "mean"),
        calm_share=("is_calm", "mean"),
        blh_mean=("boundary_layer_height", "mean"),
        blh_min=("boundary_layer_height", "min"),
    )

    eve_pm25 = df.loc[evening_mask].groupby("date")["pm2_5"].mean().rename("pm25_evening")
    eve_no2 = df.loc[evening_mask].groupby("date")["nitrogen_dioxide"].mean().rename("no2_evening")
    eve_wind = df.loc[evening_mask].groupby("date")["wind_speed_10m"].mean().rename("wind_evening")
    morn_pm25 = df.loc[morning_mask].groupby("date")["pm2_5"].mean().rename("pm25_morning")
    daily = daily.join([eve_pm25, eve_no2, eve_wind, morn_pm25]).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    # Engineered domain features (strictly <= Day t)
    daily["temp_diurnal_range"] = daily["temp_max"] - daily["temp_min"]
    daily["uv_attenuation"] = daily["uv_clear_max"] - daily["uv_max"]
    daily["smoke_ratio"] = daily["pm25_mean"] / (daily["pm10_mean"] + 1.0)
    daily["evening_to_day_ratio"] = daily["pm25_evening"] / (daily["pm25_mean"] + 1.0)
    daily["morning_to_day_ratio"] = daily["pm25_morning"] / (daily["pm25_mean"] + 1.0)
    daily["stagnation_index"] = daily["pm25_mean"] / (daily["wind_mean"] * daily["blh_mean"].fillna(500.0) / 1000.0 + 0.5)
    daily["co2_available"] = daily["co2_mean"].notna().astype(float)
    daily["ch4_available"] = daily["ch4_mean"].notna().astype(float)
    daily["blh_available"] = daily["blh_mean"].notna().astype(float)

    # Lags & rolling memory (strictly <= Day t)
    daily["pm25_lag1"] = daily["pm25_mean"].shift(1)
    daily["pm25_lag2"] = daily["pm25_mean"].shift(2)
    daily["pm25_diff1"] = daily["pm25_mean"] - daily["pm25_lag1"]
    daily["pm25_roll3"] = daily["pm25_mean"].rolling(3).mean()
    daily["pm25_roll7"] = daily["pm25_mean"].rolling(7).mean()
    daily["pm25_std7"] = daily["pm25_mean"].rolling(7).std()
    daily["wind_lag1"] = daily["wind_mean"].shift(1)
    daily["aod_diff1"] = daily["aod_mean"] - daily["aod_mean"].shift(1)

    # Calendar cyclical features
    doy = daily["date"].dt.day_of_year
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    daily["is_smog_season"] = daily["date"].dt.month.isin([10, 11, 12, 1]).astype(float)

    # Day t+1 target & Day t+1 operational weather forecast proxies
    daily["target_pm25_next"] = daily["pm25_mean"].shift(-1)
    daily["target_date"] = daily["date"].shift(-1)
    next_weather_cols = [
        "temp_mean",
        "temp_diurnal_range",
        "rh_mean",
        "precip_sum",
        "pressure_mean",
        "wind_mean",
        "wind_min",
        "wind_u_mean",
        "wind_v_mean",
        "nw_wind_share",
        "calm_share",
        "blh_mean",
        "blh_min",
    ]
    next_weather_features = []
    for col in next_weather_cols:
        ncol = f"next_{col}"
        daily[ncol] = daily[col].shift(-1)
        next_weather_features.append(ncol)

    day_t_features = [
        "pm25_mean",
        "pm25_max",
        "pm25_min",
        "pm25_evening",
        "pm25_morning",
        "pm10_mean",
        "co_mean",
        "co2_mean",
        "no2_mean",
        "no2_evening",
        "so2_mean",
        "o3_mean",
        "aod_mean",
        "dust_mean",
        "uv_max",
        "uv_clear_max",
        "uv_attenuation",
        "ch4_mean",
        "temp_mean",
        "temp_diurnal_range",
        "rh_mean",
        "precip_sum",
        "pressure_mean",
        "wind_mean",
        "wind_min",
        "wind_evening",
        "wind_u_mean",
        "wind_v_mean",
        "nw_wind_share",
        "calm_share",
        "blh_mean",
        "blh_min",
        "smoke_ratio",
        "evening_to_day_ratio",
        "morning_to_day_ratio",
        "stagnation_index",
        "co2_available",
        "ch4_available",
        "blh_available",
        "pm25_lag1",
        "pm25_lag2",
        "pm25_diff1",
        "pm25_roll3",
        "pm25_roll7",
        "pm25_std7",
        "wind_lag1",
        "aod_diff1",
        "doy_sin",
        "doy_cos",
        "is_smog_season",
    ]
    all_features = day_t_features + next_weather_features
    daily = daily.dropna(subset=["target_pm25_next", "pm25_roll7", "pm25_lag2"]).reset_index(drop=True)
    return daily, day_t_features, all_features


def evaluate_daily_predictions(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    quantile_preds: np.ndarray | None = None,
) -> dict:
    """Evaluate regression and threshold/spike metrics on held-out test set."""
    y_true = test_df["target_pm25_next"].to_numpy()
    y_today = test_df["pm25_mean"].to_numpy()
    is_smog = test_df["target_date"].dt.month.isin([10, 11, 12, 1, 2]).to_numpy()
    is_transition = np.abs(y_true - y_today) >= 20.0

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))
    smog_mae = float(mean_absolute_error(y_true[is_smog], preds[is_smog]))
    trans_mae = float(mean_absolute_error(y_true[is_transition], preds[is_transition]))

    # Use quantile upper bound for safety alerts if supplied, else point forecast
    alert_score = quantile_preds if quantile_preds is not None else preds

    epa_35 = binary_metrics((y_true > 35.0).astype(int), (alert_score > 35.0).astype(int))
    poor_90 = binary_metrics((y_true > 90.0).astype(int), (alert_score > 90.0).astype(int))
    very_poor_120 = binary_metrics((y_true > 120.0).astype(int), (alert_score > 120.0).astype(int))

    # Sudden overnight spike >= +20 ug/m3
    true_spike = ((y_true - y_today) >= 20.0).astype(int)
    pred_spike = ((alert_score - y_today) >= 15.0).astype(int) if quantile_preds is not None else ((preds - y_today) >= 20.0).astype(int)
    spike_20 = binary_metrics(true_spike, pred_spike)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "smog_season_mae": round(smog_mae, 3),
        "transition_day_mae": round(trans_mae, 3),
        "n_test_days": int(len(y_true)),
        "n_transition_days": int(np.sum(is_transition)),
        "epa_gt35": epa_35,
        "cpcb_poor_gt90": poor_90,
        "cpcb_verypoor_gt120": very_poor_120,
        "sudden_spike_ge20": spike_20,
    }


def run_hourly_safe_window(df_hourly: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Train hourly model for Day t+1 diurnal PM2.5 curve and evaluate Safest 2-Hour Window."""
    h = df_hourly.copy()
    h["target_date"] = h["date"]
    # Merge Day t-1 daily state onto target_date = t+1 (strictly previous day summary)
    prev_daily = daily_df[["target_date", "pm25_mean", "pm25_evening", "pm10_mean", "co_mean", "no2_mean", "aod_mean", "wind_mean"]].rename(
        columns={
            "pm25_mean": "prev_pm25_mean",
            "pm25_evening": "prev_pm25_evening",
            "pm10_mean": "prev_pm10_mean",
            "co_mean": "prev_co_mean",
            "no2_mean": "prev_no2_mean",
            "aod_mean": "prev_aod_mean",
            "wind_mean": "prev_wind_mean",
        }
    )
    h["prev_pm25_same_hour"] = h["pm2_5"].shift(24)
    h = h.merge(prev_daily, on="target_date", how="inner")

    h["hour_sin"] = np.sin(2 * np.pi * h["hour"] / 24.0)
    h["hour_cos"] = np.cos(2 * np.pi * h["hour"] / 24.0)
    doy = h["target_date"].dt.day_of_year
    h["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    h["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(np.deg2rad(h["wind_direction_10m"]))
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(np.deg2rad(h["wind_direction_10m"]))

    hourly_features = [
        "prev_pm25_mean",
        "prev_pm25_evening",
        "prev_pm25_same_hour",
        "prev_pm10_mean",
        "prev_co_mean",
        "prev_no2_mean",
        "prev_aod_mean",
        "prev_wind_mean",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_u",
        "wind_v",
        "boundary_layer_height",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    h = h.dropna(subset=["pm2_5", "prev_pm25_same_hour", "prev_pm25_mean"]).reset_index(drop=True)
    h_train = h[h["target_date"] <= TRAIN_END]
    h_test = h[(h["target_date"] >= TEST_START) & (h["target_date"] <= EVAL_END)].copy()
    h_live = h[h["target_date"] >= TEST_START].copy()

    h_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=250,
        random_state=42,
    )
    h_model.fit(h_train[hourly_features], h_train["pm2_5"])
    h_test["pred_pm25"] = np.clip(h_model.predict(h_test[hourly_features]), 0.0, None)
    h_live["pred_pm25"] = np.clip(h_model.predict(h_live[hourly_features]), 0.0, None)

    pers_mae = float(mean_absolute_error(h_test["pm2_5"], h_test["prev_pm25_same_hour"]))
    model_mae = float(mean_absolute_error(h_test["pm2_5"], h_test["pred_pm25"]))

    # Evaluate Safest 2-hour daytime outdoor window (06:00 to 18:00) on test days
    daytime = h_test[(h_test["hour"] >= 6) & (h_test["hour"] <= 18)].copy()
    rec_pm25_list = []
    morning_pm25_list = []
    true_best_pm25_list = []
    within_10_hits = 0
    total_eval_days = 0
    smog_rec_list = []
    smog_morning_list = []

    for dt, grp in daytime.groupby("target_date"):
        if len(grp) < 13:
            continue
        grp = grp.sort_values("hour")
        hours = grp["hour"].to_numpy()
        actual_vals = grp["pm2_5"].to_numpy()
        pred_vals = grp["pred_pm25"].to_numpy()

        # 2-hour rolling means across 06..18
        pred_2h = (pred_vals[:-1] + pred_vals[1:]) / 2.0
        act_2h = (actual_vals[:-1] + actual_vals[1:]) / 2.0
        best_pred_idx = int(np.argmin(pred_2h))
        best_true_idx = int(np.argmin(act_2h))

        chosen_actual = float(act_2h[best_pred_idx])
        true_best = float(act_2h[best_true_idx])
        # Morning school assembly / commute window 07:00-09:00 (hours 7 and 8)
        morn_idx = int(np.where(hours[:-1] == 7)[0][0])
        morning_actual = float(act_2h[morn_idx])

        rec_pm25_list.append(chosen_actual)
        true_best_pm25_list.append(true_best)
        morning_pm25_list.append(morning_actual)
        total_eval_days += 1
        if abs(chosen_actual - true_best) <= 10.0:
            within_10_hits += 1
        if np.mean(actual_vals) > 90.0:
            smog_rec_list.append(chosen_actual)
            smog_morning_list.append(morning_actual)

    mean_morn = float(np.mean(morning_pm25_list))
    mean_rec = float(np.mean(rec_pm25_list))
    mean_true_best = float(np.mean(true_best_pm25_list))
    smog_morn = float(np.mean(smog_morning_list))
    smog_rec = float(np.mean(smog_rec_list))

    hourly_metrics = {
        "train_hours": int(len(h_train)),
        "test_hours": int(len(h_test)),
        "hourly_persistence_mae": round(pers_mae, 3),
        "hourly_model_mae": round(model_mae, 3),
        "test_days_evaluated": total_eval_days,
        "within_10ug_of_true_best_pct": round(100.0 * within_10_hits / total_eval_days, 1),
        "avg_morning_07_09_pm25": round(mean_morn, 2),
        "avg_recommended_2h_pm25": round(mean_rec, 2),
        "avg_oracle_best_2h_pm25": round(mean_true_best, 2),
        "avg_exposure_reduction_pct": round(100.0 * (mean_morn - mean_rec) / mean_morn, 1),
        "smog_days_count": len(smog_rec_list),
        "smog_days_morning_07_09_pm25": round(smog_morn, 2),
        "smog_days_recommended_2h_pm25": round(smog_rec, 2),
        "smog_days_exposure_reduction_pct": round(100.0 * (smog_morn - smog_rec) / smog_morn, 1),
    }
    return hourly_metrics, h_test, h_live


def compute_cpcb_comparison(daily_df: pd.DataFrame) -> dict:
    """Compare Open-Meteo CAMS PM2.5 against real Delhi CPCB ground monitors in Oct-Nov 2024."""
    cpcb_path = DATA_DIR / "xkdr_cpcb_delhi_oct_nov_2024.csv"
    if not cpcb_path.exists():
        return {"available": False}
    cpcb = pd.read_csv(cpcb_path)
    cpcb["date"] = pd.to_datetime(cpcb["period_start"]).dt.normalize()
    cpcb_daily = (
        cpcb.groupby("date")
        .agg(
            cpcb_mean_pm25=("mean", "mean"),
            cpcb_median_pm25=("mean", "median"),
            cpcb_p90_station_pm25=("mean", lambda s: float(np.quantile(s, 0.9))),
            stations_reporting=("station_id", "nunique"),
        )
        .reset_index()
    )
    merged = daily_df[["date", "pm25_mean"]].merge(cpcb_daily, on="date", how="inner")
    merged["ratio_cpcb_to_cams"] = merged["cpcb_mean_pm25"] / merged["pm25_mean"]
    return {
        "available": True,
        "overlap_days": int(len(merged)),
        "pearson_r": round(float(merged["pm25_mean"].corr(merged["cpcb_mean_pm25"])), 3),
        "cams_mean_pm25": round(float(merged["pm25_mean"].mean()), 2),
        "cpcb_city_mean_pm25": round(float(merged["cpcb_mean_pm25"].mean()), 2),
        "cams_max_pm25": round(float(merged["pm25_mean"].max()), 2),
        "cpcb_city_max_pm25": round(float(merged["cpcb_mean_pm25"].max()), 2),
        "median_cpcb_to_cams_ratio": round(float(merged["ratio_cpcb_to_cams"].median()), 2),
        "records": [
            {
                "date": str(r["date"].date()),
                "cams_pm25": round(float(r["pm25_mean"]), 1),
                "cpcb_pm25": round(float(r["cpcb_mean_pm25"]), 1),
            }
            for _, r in merged.iterrows()
        ],
    }


def make_figures(
    test_df: pd.DataFrame,
    preds_hgbr: np.ndarray,
    preds_q72: np.ndarray,
    daily_metrics: dict,
    perm_importance_list: list[dict],
    cpcb_info: dict,
    h_test: pd.DataFrame,
) -> None:
    """Create 4 crisp Matplotlib charts in Code/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: Held-out test forecast timeline (Oct 2025 - Nov 2026)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    dates = test_df["target_date"]
    ax.plot(dates, test_df["target_pm25_next"], color="#111827", linewidth=1.6, label="Actual PM2.5 (Open-Meteo CAMS)")
    ax.plot(dates, test_df["pm25_mean"], color="#9CA3AF", linewidth=1.0, linestyle="--", alpha=0.8, label="Persistence Baseline (Tomorrow = Today)")
    ax.plot(dates, preds_hgbr, color="#0D9488", linewidth=1.4, label="HistGBR Forecast")
    ax.fill_between(dates, preds_hgbr, preds_q72, color="#F59E0B", alpha=0.22, label="Safety Buffer (Q72 Upper Bound)")
    ax.axhline(120, color="#DC2626", linestyle=":", linewidth=1.2, label="CPCB Very Poor Threshold (120 ug/m3)")
    ax.set_title("Unseen Test Period (Oct 2025 – Nov 2026): Delhi Next-Day PM2.5 Forecast", fontsize=12, fontweight="bold")
    ax.set_ylabel("Daily Mean PM2.5 (ug/m3)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_test_forecast_and_baselines.png")
    plt.close(fig)

    # Fig 2: Model Progression (MAE + Very Poor Recall vs False Alarms)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["Persistence\nBaseline", "Ridge\nRegression", "Decision Tree\n(Rejected)", "HistGBR\n(Full Model)"]
    overall_maes = [
        daily_metrics["persistence"]["mae"],
        daily_metrics["ridge"]["mae"],
        daily_metrics["decision_tree_rejected"]["mae"],
        daily_metrics["hgbr_full"]["mae"],
    ]
    trans_maes = [
        daily_metrics["persistence"]["transition_day_mae"],
        daily_metrics["ridge"]["transition_day_mae"],
        daily_metrics["decision_tree_rejected"]["transition_day_mae"],
        daily_metrics["hgbr_full"]["transition_day_mae"],
    ]
    x = np.arange(len(labels))
    w = 0.36
    ax1.bar(x - w / 2, overall_maes, width=w, color="#64748B", label=f"All Test Days (n={len(test_df)})")
    ax1.bar(x + w / 2, trans_maes, width=w, color="#0D9488", label="High-Change Days (|dPM2.5| >= 20)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("MAE (ug/m3, lower is better)")
    ax1.set_title("Forecast Error Across Models", fontweight="bold")
    ax1.legend(fontsize=8)
    for i, (v1, v2) in enumerate(zip(overall_maes, trans_maes)):
        ax1.text(i - w / 2, v1 + 0.6, f"{v1:.1f}", ha="center", fontsize=8)
        ax1.text(i + w / 2, v2 + 0.6, f"{v2:.1f}", ha="center", fontsize=8)

    # Spike recall comparison
    spike_labels = ["Persistence", "Ridge", "HistGBR", "HistGBR + Q72\nSafety Alert"]
    spike_recalls = [
        daily_metrics["persistence"]["sudden_spike_ge20"]["recall"] * 100,
        daily_metrics["ridge"]["sudden_spike_ge20"]["recall"] * 100,
        daily_metrics["hgbr_full"]["sudden_spike_ge20"]["recall"] * 100,
        daily_metrics["hgbr_safety_q72"]["sudden_spike_ge20"]["recall"] * 100,
    ]
    vp_recalls = [
        daily_metrics["persistence"]["cpcb_verypoor_gt120"]["recall"] * 100,
        daily_metrics["ridge"]["cpcb_verypoor_gt120"]["recall"] * 100,
        daily_metrics["hgbr_full"]["cpcb_verypoor_gt120"]["recall"] * 100,
        daily_metrics["hgbr_safety_q72"]["cpcb_verypoor_gt120"]["recall"] * 100,
    ]
    x2 = np.arange(len(spike_labels))
    ax2.bar(x2 - w / 2, vp_recalls, width=w, color="#2563EB", label="Very Poor Day Recall (>120 ug/m3)")
    ax2.bar(x2 + w / 2, spike_recalls, width=w, color="#DC2626", label="Sudden Spike Recall (+20 ug/m3)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(spike_labels, fontsize=8.5)
    ax2.set_ylabel("Recall (%, higher catches more hazards)")
    ax2.set_ylim(0, 108)
    ax2.set_title("Catching Hazardous & Sudden Spike Days", fontweight="bold")
    ax2.legend(fontsize=8)
    for i, (v1, v2) in enumerate(zip(vp_recalls, spike_recalls)):
        ax2.text(i - w / 2, v1 + 2, f"{v1:.0f}%", ha="center", fontsize=8)
        ax2.text(i + w / 2, v2 + 2, f"{v2:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_model_comparison_and_spikes.png")
    plt.close(fig)

    # Fig 3: Top Physical Drivers + CAMS vs Real CPCB Ground Sensors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    top_feats = perm_importance_list[:10][::-1]
    ax1.barh([f["feature"] for f in top_feats], [f["importance_mean"] for f in top_feats], color="#0F766E")
    ax1.set_xlabel("Permutation Importance (MAE increase if shuffled)")
    ax1.set_title("Top Drivers of Next-Day Delhi PM2.5", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=8.5)

    if cpcb_info.get("available"):
        rec_df = pd.DataFrame(cpcb_info["records"])
        rec_df["date"] = pd.to_datetime(rec_df["date"])
        ax2.plot(rec_df["date"], rec_df["cpcb_pm25"], color="#DC2626", linewidth=1.6, label="Real CPCB Ground Monitors (City Mean)")
        ax2.plot(rec_df["date"], rec_df["cams_pm25"], color="#2563EB", linewidth=1.6, linestyle="--", label="Open-Meteo CAMS Estimate")
        ax2.set_title(f"CAMS vs. Real CPCB Sensors\n(Oct–Nov 2024, r={cpcb_info['pearson_r']})", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Daily Mean PM2.5 (ug/m3)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.tick_params(axis="x", labelsize=8.5)
        ax2.legend(fontsize=8.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_drivers_and_cpcb_gap.png")
    plt.close(fig)

    # Fig 4: Hourly Diurnal Curve & Safest 2-Hour Window (Smog Season Oct-Feb)
    smog_h = h_test[h_test["target_date"].dt.month.isin([10, 11, 12, 1, 2])]
    diurnal = smog_h.groupby("hour")[["pm2_5", "pred_pm25"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.plot(diurnal["hour"], diurnal["pm2_5"], color="#111827", linewidth=2.0, marker="o", markersize=4, label="Actual Mean PM2.5 (Smog Season Test)")
    ax.plot(diurnal["hour"], diurnal["pred_pm25"], color="#0D9488", linewidth=1.8, linestyle="--", marker="s", markersize=4, label="Hourly Model Prediction")
    ax.axvspan(7, 9, color="#DC2626", alpha=0.14, label="Morning School Arrival / Assembly (07:00–09:00)")
    ax.axvspan(13, 15, color="#16A34A", alpha=0.16, label="Typical Cleanest Afternoon Window (13:00–15:00)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of Day (IST)")
    ax.set_ylabel("Mean PM2.5 (ug/m3)")
    ax.set_title("When Is Delhi's Air Cleanest? Diurnal Mixing & Safe 2-Hour Outdoor Window", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_hourly_safe_window.png")
    plt.close(fig)


def main() -> None:
    df_hourly, coverage_audit = load_hourly_dataframe()
    daily_df, day_t_features, all_features = build_daily_features(df_hourly)

    train_df = daily_df[daily_df["target_date"] <= TRAIN_END].copy()
    test_df = daily_df[(daily_df["target_date"] >= TEST_START) & (daily_df["target_date"] <= EVAL_END)].copy()
    live_df = daily_df[(daily_df["target_date"] >= TEST_START) & (daily_df["target_date"] <= TOMORROW_IST)].copy()

    y_train = train_df["target_pm25_next"].to_numpy()
    y_test = test_df["target_pm25_next"].to_numpy()

    # 1. Baseline: Persistence
    preds_pers = test_df["pm25_mean"].to_numpy()

    # 2. Model 1: Ridge Regression (all features)
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=25.0)),
        ]
    )
    ridge.fit(train_df[all_features], y_train)
    preds_ridge = np.clip(ridge.predict(test_df[all_features]), 0.0, None)

    # 3. Rejected Alternative: Unregularized Decision Tree
    dt_rejected = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", DecisionTreeRegressor(random_state=42)),
        ]
    )
    dt_rejected.fit(train_df[all_features], y_train)
    preds_dt = np.clip(dt_rejected.predict(test_df[all_features]), 0.0, None)

    # 4. Ablation: Day-t-only HistGBR (without next-day weather forecast)
    hgbr_day_t_only = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1.0,
        random_state=42,
    )
    hgbr_day_t_only.fit(train_df[day_t_features], y_train)
    preds_day_t_only = np.clip(hgbr_day_t_only.predict(test_df[day_t_features]), 0.0, None)

    # 5. Final Model: HistGradientBoostingRegressor (Day-t pollutants + next-day meteorology)
    hgbr_full = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=250,
        l2_regularization=1.0,
        random_state=42,
    )
    hgbr_full.fit(train_df[all_features], y_train)
    preds_hgbr = np.clip(hgbr_full.predict(test_df[all_features]), 0.0, None)

    # 6. Safety Risk Bound (Quantile 0.72) for asymmetric school/asthma spike alerts
    hgbr_q72 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.72,
        max_depth=4,
        learning_rate=0.05,
        max_iter=250,
        l2_regularization=1.0,
        random_state=42,
    )
    hgbr_q72.fit(train_df[all_features], y_train)
    preds_q72 = np.clip(hgbr_q72.predict(test_df[all_features]), 0.0, None)

    daily_metrics = {
        "persistence": evaluate_daily_predictions(test_df, preds_pers),
        "ridge": evaluate_daily_predictions(test_df, preds_ridge),
        "decision_tree_rejected": evaluate_daily_predictions(test_df, preds_dt),
        "hgbr_day_t_only": evaluate_daily_predictions(test_df, preds_day_t_only),
        "hgbr_full": evaluate_daily_predictions(test_df, preds_hgbr),
        "hgbr_safety_q72": evaluate_daily_predictions(test_df, preds_hgbr, quantile_preds=preds_q72),
    }

    # Permutation importance on held-out test set for explainability
    perm = permutation_importance(
        hgbr_full,
        test_df[all_features],
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
    )
    perm_idx = np.argsort(perm.importances_mean)[::-1]
    perm_list = [
        {
            "feature": all_features[i],
            "importance_mean": round(float(perm.importances_mean[i]), 3),
            "importance_std": round(float(perm.importances_std[i]), 3),
        }
        for i in perm_idx[:15]
    ]

    # Top 5 failure cases on held-out test set
    err_df = test_df[["date", "target_date", "pm25_mean", "target_pm25_next", "wind_mean", "next_wind_mean", "next_precip_sum", "next_blh_mean"]].copy()
    err_df["pred_hgbr"] = preds_hgbr
    err_df["pred_q72"] = preds_q72
    err_df["abs_err"] = np.abs(err_df["target_pm25_next"] - err_df["pred_hgbr"])
    err_df["signed_err"] = err_df["pred_hgbr"] - err_df["target_pm25_next"]
    top_failures = []
    for _, row in err_df.sort_values("abs_err", ascending=False).head(6).iterrows():
        top_failures.append(
            {
                "forecast_made_on": str(row["date"].date()),
                "target_date": str(row["target_date"].date()),
                "today_pm25": round(float(row["pm25_mean"]), 1),
                "actual_next_pm25": round(float(row["target_pm25_next"]), 1),
                "pred_hgbr": round(float(row["pred_hgbr"]), 1),
                "pred_q72": round(float(row["pred_q72"]), 1),
                "abs_err": round(float(row["abs_err"]), 1),
                "next_wind_kmh": round(float(row["next_wind_mean"]), 1),
                "next_precip_mm": round(float(row["next_precip_sum"]), 1),
                "next_blh_m": round(float(row["next_blh_mean"]), 1) if pd.notna(row["next_blh_mean"]) else None,
            }
        )

    hourly_metrics, h_test, h_live = run_hourly_safe_window(df_hourly, daily_df)
    cpcb_info = compute_cpcb_comparison(daily_df)

    # Sample concrete live + historical demo dates in India Standard Time
    live_df["pred_hgbr"] = np.clip(hgbr_full.predict(live_df[all_features]), 0.0, None)
    live_df["pred_q72"] = np.clip(hgbr_q72.predict(live_df[all_features]), 0.0, None)
    target_demo_dates = pd.to_datetime([str(TOMORROW_IST.date()), str(TODAY_IST.date()), "2026-11-08"])
    sample_dates_rows = live_df[live_df["target_date"].isin(target_demo_dates)].sort_values("target_date", ascending=False).copy()
    demo_examples = []
    for _, row in sample_dates_rows.iterrows():
        pred_pm = float(row["pred_hgbr"])
        q72_pm = float(row["pred_q72"])
        today_pm = float(row["pm25_mean"])
        aqi_val, aqi_cat = compute_india_naqi_pm25(pred_pm)
        q72_aqi, q72_cat = compute_india_naqi_pm25(q72_pm)
        today_aqi, today_cat = compute_india_naqi_pm25(today_pm)
        day_h = h_live[(h_live["target_date"] == row["target_date"]) & (h_live["hour"] >= 6) & (h_live["hour"] <= 18)].sort_values("hour")
        best_win = "1:00 PM - 3:00 PM"
        best_aqi = aqi_val
        if len(day_h) >= 13:
            pvals = day_h["pred_pm25"].to_numpy()
            hrs = day_h["hour"].to_numpy()
            p2h = (pvals[:-1] + pvals[1:]) / 2.0
            idx_b = int(np.argmin(p2h))
            sh, eh = int(hrs[idx_b]), int(hrs[idx_b]) + 2
            fmt_h = lambda hr: f"{12 if hr % 12 == 0 else hr % 12}:00 {'AM' if hr < 12 else 'PM'}"
            best_win = f"{fmt_h(sh)} - {fmt_h(eh)}"
            best_aqi, _ = compute_india_naqi_pm25(float(p2h[idx_b]))
        demo_examples.append(
            {
                "decision_night": str(row["date"].date()),
                "target_date": str(row["target_date"].date()),
                "today_aqi": round(today_aqi),
                "pred_aqi": round(aqi_val),
                "pred_category": aqi_cat,
                "safety_aqi": round(q72_aqi),
                "safety_category": q72_cat,
                "best_window": best_win,
                "best_window_aqi": round(best_aqi),
                "next_wind_kmh": round(float(row["next_wind_mean"]), 1),
                "next_precip_mm": round(float(row["next_precip_sum"]), 1),
            }
        )

    make_figures(test_df, preds_hgbr, preds_q72, daily_metrics, perm_list, cpcb_info, h_test)

    summary = {
        "ist_clock": {
            "now_ist": NOW_IST.strftime("%Y-%m-%d %H:%M IST"),
            "today_ist": str(TODAY_IST.date()),
            "tomorrow_ist": str(TOMORROW_IST.date()),
            "eval_end": str(EVAL_END.date()),
        },
        "dataset_span": {
            "hourly_rows": int(len(df_hourly)),
            "daily_rows_usable": int(len(daily_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_range": [str(train_df["target_date"].min().date()), str(train_df["target_date"].max().date())],
            "test_range": [str(test_df["target_date"].min().date()), str(test_df["target_date"].max().date())],
        },
        "coverage_audit": coverage_audit,
        "daily_metrics": daily_metrics,
        "hourly_safe_window_metrics": hourly_metrics,
        "top_15_features_perm_importance": perm_list,
        "top_failures": top_failures,
        "demo_examples": demo_examples,
        "cpcb_comparison_oct_nov_2024": {k: v for k, v in cpcb_info.items() if k != "records"},
    }
    (DATA_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

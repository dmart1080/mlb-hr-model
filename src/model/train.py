from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.frozen import FrozenEstimator
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def f3(x: float) -> str:
    return f"{x:.3f}"

def f2(x: float) -> str:
    return f"{x:.2f}"

def fmt_int(x: int) -> str:
    return f"{x:,}"


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # --- Batter combined (14d) ---
    "b_pa_14", "b_hr_rate_14", "b_barrel_rate_14",
    "b_ev_mean_14", "b_la_mean_14", "b_hardhit_rate_14", "b_fb_rate_14",
    "b_k_rate_14", "b_bb_rate_14",

    # --- Batter platoon splits (14d) ---
    "b_pa_14_vsL", "b_hr_rate_14_vsL", "b_barrel_rate_14_vsL",
    "b_hardhit_rate_14_vsL", "b_fb_rate_14_vsL",
    "b_k_rate_14_vsL", "b_bb_rate_14_vsL",

    "b_pa_14_vsR", "b_hr_rate_14_vsR", "b_barrel_rate_14_vsR",
    "b_hardhit_rate_14_vsR", "b_fb_rate_14_vsR",
    "b_k_rate_14_vsR", "b_bb_rate_14_vsR",

    # --- Batter combined (season) ---
    "b_pa_szn", "b_hr_rate_szn", "b_barrel_rate_szn",
    "b_ev_mean_szn", "b_la_mean_szn", "b_hardhit_rate_szn", "b_fb_rate_szn",
    "b_k_rate_szn", "b_bb_rate_szn",

    # --- Batter platoon splits (season) ---
    "b_pa_szn_vsL", "b_hr_rate_szn_vsL", "b_barrel_rate_szn_vsL",
    "b_hardhit_rate_szn_vsL", "b_fb_rate_szn_vsL",
    "b_k_rate_szn_vsL", "b_bb_rate_szn_vsL",

    "b_pa_szn_vsR", "b_hr_rate_szn_vsR", "b_barrel_rate_szn_vsR",
    "b_hardhit_rate_szn_vsR", "b_fb_rate_szn_vsR",
    "b_k_rate_szn_vsR", "b_bb_rate_szn_vsR",

    # --- Batter EV trend (7d vs 8–14d) ---
    "b_ev_mean_7", "b_hardhit_rate_7",
    "b_ev_trend", "b_hardhit_trend", "b_barrel_trend", "b_hr_trend",

    # --- Batter home/away splits (season) ---
    "b_hr_rate_home", "b_hr_rate_away",
    "b_hardhit_rate_home", "b_hardhit_rate_away",
    "b_barrel_rate_home", "b_barrel_rate_away",
    "b_hr_rate_home_edge",
    "is_home_game",

    # --- Batter rest ---
    "b_days_rest",

    # --- Pitcher combined (30d) ---
    "p_pa_30", "p_hr_allowed_rate_30",
    "p_ev_allowed_mean_30", "p_hardhit_allowed_rate_30",
    "p_fb_allowed_rate_30", "p_barrel_allowed_rate_30",
    "p_k_rate_30", "p_bb_rate_30",

    # --- Pitcher platoon splits (30d) ---
    "p_pa_30_vsL", "p_hr_allowed_rate_30_vsL",
    "p_hardhit_allowed_rate_30_vsL", "p_fb_allowed_rate_30_vsL",
    "p_barrel_allowed_rate_30_vsL", "p_k_rate_30_vsL", "p_bb_rate_30_vsL",

    "p_pa_30_vsR", "p_hr_allowed_rate_30_vsR",
    "p_hardhit_allowed_rate_30_vsR", "p_fb_allowed_rate_30_vsR",
    "p_barrel_allowed_rate_30_vsR", "p_k_rate_30_vsR", "p_bb_rate_30_vsR",

    # --- Pitcher combined (season) ---
    "p_pa_szn", "p_hr_allowed_rate_szn",
    "p_ev_allowed_mean_szn", "p_hardhit_allowed_rate_szn",
    "p_fb_allowed_rate_szn", "p_barrel_allowed_rate_szn",

    # --- Pitcher platoon splits (season) ---
    "p_pa_szn_vsL", "p_hr_allowed_rate_szn_vsL",
    "p_hardhit_allowed_rate_szn_vsL", "p_barrel_allowed_rate_szn_vsL",

    "p_pa_szn_vsR", "p_hr_allowed_rate_szn_vsR",
    "p_hardhit_allowed_rate_szn_vsR", "p_barrel_allowed_rate_szn_vsR",

    # --- Pitcher rest + velo ---
    "p_days_rest", "p_is_short_rest",
    "p_fb_velo_30", "p_fb_pct_30", "p_offspeed_pct_30", "p_fb_velo_trend",

    # --- Combined edge features ---
    "ev_edge_14_30", "hardhit_edge_14_30", "fb_edge_14_30",
    "barrel_edge_14_30", "hr_rate_edge_14_30",
    "k_rate_edge_14_30", "bb_rate_edge_14_30",
    "k_rate_interaction_14_30", "bb_rate_interaction_14_30",
    "contact_pressure_14_30", "discipline_balance_14_30",

    # --- Platoon edge features ---
    "hr_rate_edge_14_30_vsL", "hardhit_edge_14_30_vsL", "barrel_edge_14_30_vsL",
    "hr_rate_edge_14_30_vsR", "hardhit_edge_14_30_vsR", "barrel_edge_14_30_vsR",

    # --- Matchup ---
    "same_hand_matchup",

    # --- Context ---
    "park_factor_hr",
]


# ---------------------------------------------------------------------------
# Summary + CSV logging
# ---------------------------------------------------------------------------

def print_summary(
    train_path: Path,
    model_path: Path,
    feature_cols: list[str],
    metrics: dict,
    extra: dict,
) -> None:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 50)
    print("MLB HR MODEL — TRAIN SUMMARY")
    print("=" * 50)
    print(f"Run time:       {run_time}")
    print(f"Training table: {train_path.name}\n")

    print(f"Date range:  train={metrics['train_start']}→{metrics['train_end']}  "
          f"test={metrics['test_start']}→{metrics['test_end']}\n")

    print(f"Rows:  train={fmt_int(metrics['train_rows'])}  test={fmt_int(metrics['test_rows'])}")
    print(f"Test HR rate (baseline): {pct(metrics['test_hr_rate'])}\n")

    print("Performance (test):")
    print(f"  ROC-AUC:   {f3(metrics['roc_auc'])}")
    print(f"  Log loss:  {f3(metrics['log_loss'])}")
    print(f"  Avg pred:  {pct(metrics['avg_pred_prob'])}")
    print(f"  Max pred:  {f3(extra['max_pred_prob'])}\n")

    print("Lift checks (test):")
    print(f"  Top 10% HR rate: {pct(extra['top10_hr_rate'])}  ({extra['top10_lift']:.2f}x baseline)")
    print(f"  Top 1%  HR rate: {pct(extra['top1_hr_rate'])}  ({extra['top1_lift']:.2f}x baseline)  "
          f"(n={fmt_int(extra['top1_count'])})")
    print(f"  Top 1%  avg b_pa_14: {f2(extra['top1_avg_b_pa_14'])}")
    print(f"  Top 1%  avg p_pa_30: {f2(extra['top1_avg_p_pa_30'])}\n")

    print(f"Calibration delta: {extra['avg_minus_base_pp']:+.2f} pp")
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Model saved: {model_path.name}")
    print("=" * 50 + "\n")

    log_file = MODELS_DIR / "train_runs.csv"
    write_header = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_time", "train_table", "roc_auc", "log_loss",
                "baseline_hr_rate", "top10_hr_rate", "top1_hr_rate",
                "top10_lift", "top1_lift", "features",
            ])
        writer.writerow([
            run_time, train_path.name,
            metrics["roc_auc"], metrics["log_loss"],
            metrics["test_hr_rate"],
            extra["top10_hr_rate"], extra["top1_hr_rate"],
            extra["top10_lift"], extra["top1_lift"],
            len(feature_cols),
        ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def latest_train_table() -> Path:
    for pattern in (
        "train_table_*_combined.parquet",
        "train_table_*_full_season.parquet",
        "train_table_*.parquet",
    ):
        files = sorted(
            PROCESSED_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    raise FileNotFoundError("No train_table_*.parquet found in data/processed/")


def time_split(df: pd.DataFrame, test_size: float = 0.2):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]
    extra: dict


def train_baseline(train_path: Path) -> TrainResult:
    df = pd.read_parquet(train_path)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        print(f"  Note: {len(missing_features)} feature(s) not in table, skipping: "
              f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

    for col in ["hr_hit", "game_date"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    feature_cols = available_features

    train_df, test_df = time_split(df, test_size=0.2)

    train_start = str(train_df["game_date"].min().date())
    train_end   = str(train_df["game_date"].max().date())
    test_start  = str(test_df["game_date"].min().date())
    test_end    = str(test_df["game_date"].max().date())

    train_core_df, calib_df = time_split(train_df, test_size=0.2)

    X_train_core = train_core_df[feature_cols].fillna(0.0)
    y_train_core = train_core_df["hr_hit"].astype(int)
    X_calib      = calib_df[feature_cols].fillna(0.0)
    y_calib      = calib_df["hr_hit"].astype(int)
    X_test       = test_df[feature_cols].fillna(0.0)
    y_test       = test_df["hr_hit"].astype(int)

    # Logistic Regression
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    base_pipeline.fit(X_train_core, y_train_core)
    roc_lr_raw = float(roc_auc_score(y_test, base_pipeline.predict_proba(X_test)[:, 1]))
    print(f"LogReg  (raw) ROC-AUC: {roc_lr_raw:.3f}")

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        random_state=42,
        verbosity=-1,
    )
    lgbm.fit(X_train_core, y_train_core)
    roc_lgbm_raw = float(roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1]))
    print(f"LightGBM (raw) ROC-AUC: {roc_lgbm_raw:.3f}")

    # Calibrate both, pick winner
    calibrated_lr = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline), method="sigmoid", cv=None,
    )
    calibrated_lr.fit(X_calib, y_calib)

    calibrated_lgbm = CalibratedClassifierCV(
        estimator=FrozenEstimator(lgbm), method="isotonic", cv=None,
    )
    calibrated_lgbm.fit(X_calib, y_calib)

    if roc_lgbm_raw >= roc_lr_raw:
        print("Chosen model: LightGBM")
        chosen_model = calibrated_lgbm
        chosen_name  = "lightgbm_calibrated"
        p_test = calibrated_lgbm.predict_proba(X_test)[:, 1]
    else:
        print("Chosen model: LogReg")
        chosen_model = calibrated_lr
        chosen_name  = "logreg_calibrated"
        p_test = calibrated_lr.predict_proba(X_test)[:, 1]

    baseline = float(y_test.mean())
    avg_pred = float(p_test.mean())

    q90 = np.quantile(p_test, 0.90)
    top10_mask    = p_test >= q90
    top10_hr_rate = float(y_test[top10_mask].mean())

    q99 = np.quantile(p_test, 0.99)
    top1_mask    = p_test >= q99
    top1_hr_rate = float(y_test[top1_mask].mean())

    extra = {
        "top10_hr_rate":     top10_hr_rate,
        "top1_hr_rate":      top1_hr_rate,
        "top1_count":        int(top1_mask.sum()),
        "top1_avg_b_pa_14":  float(X_test.loc[top1_mask, "b_pa_14"].mean()),
        "top1_avg_p_pa_30":  float(X_test.loc[top1_mask, "p_pa_30"].mean()),
        "max_pred_prob":     float(p_test.max()),
        "top10_lift":        (top10_hr_rate / baseline) if baseline > 0 else float("nan"),
        "top1_lift":         (top1_hr_rate  / baseline) if baseline > 0 else float("nan"),
        "avg_minus_base_pp": (avg_pred - baseline) * 100.0,
    }

    run_metrics = {
        "train_rows":    int(len(train_df)),
        "test_rows":     int(len(test_df)),
        "train_start":   train_start,
        "train_end":     train_end,
        "test_start":    test_start,
        "test_end":      test_end,
        "test_hr_rate":  baseline,
        "avg_pred_prob": avg_pred,
        "log_loss":      float(log_loss(y_test, p_test, labels=[0, 1])),
        "roc_auc":       float(roc_auc_score(y_test, p_test)),
    }

    year_tag   = f"{train_df['game_date'].min().year}_{train_df['game_date'].max().year}"
    model_path = MODELS_DIR / f"hr_model_{chosen_name}_{year_tag}.joblib"
    joblib.dump({"model": chosen_model, "feature_cols": feature_cols}, model_path)

    return TrainResult(
        model_path=model_path,
        metrics=run_metrics,
        feature_cols=feature_cols,
        extra=extra,
    )


if __name__ == "__main__":
    train_path = latest_train_table()
    print(f"Training from: {train_path.name}")
    result = train_baseline(train_path)
    print_summary(train_path, result.model_path, result.feature_cols, result.metrics, result.extra)

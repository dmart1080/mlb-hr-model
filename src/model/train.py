from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
from xml.parsers.expat import model
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.frozen import FrozenEstimator
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hard calendar cut-point: train on everything before this date, test on/after
# ---------------------------------------------------------------------------
TEST_START_DATE = "2025-03-27"


def pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def f3(x: float) -> str:
    return f"{x:.3f}"

def f2(x: float) -> str:
    return f"{x:.2f}"

def fmt_int(x: int) -> str:
    return f"{x:,}"

from datetime import datetime
import csv

def print_summary(train_path: Path, model_path: Path, feature_cols: list[str], metrics: dict, extra: dict):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = model_path.name

    print("\n" + "=" * 30)
    print("MLB HR MODEL — TRAIN SUMMARY")
    print("=" * 30)
    print(f"Run time: {run_time}")
    print(f"Training table: {train_path.name}\n")

    print(f"Date range: train={metrics['train_start']}→{metrics['train_end']}  "
          f"test={metrics['test_start']}→{metrics['test_end']}\n")

    print(f"Rows: train={fmt_int(metrics['train_rows'])}  "
          f"test={fmt_int(metrics['test_rows'])}")
    print(f"Test HR rate (baseline): {pct(metrics['test_hr_rate'])}\n")

    print("Performance (test):")
    print(f"  ROC-AUC:   {f3(metrics['roc_auc'])}")
    print(f"  Log loss:  {f3(metrics['log_loss'])}")
    print(f"  Avg pred:  {pct(metrics['avg_pred_prob'])}")
    print(f"  Max pred:  {f3(extra['max_pred_prob'])}\n")

    print("Lift checks (test):")
    print(f"  Top 10% HR rate: {pct(extra['top10_hr_rate'])}  "
          f"({extra['top10_lift']:.2f}x baseline)")
    print(f"  Top 1%  HR rate: {pct(extra['top1_hr_rate'])}  "
          f"({extra['top1_lift']:.2f}x baseline)  "
          f"(n={fmt_int(extra['top1_count'])})")
    print(f"  Top 1%  avg b_pa_14: {f2(extra['top1_avg_b_pa_14'])}")
    print(f"  Top 1%  avg p_pa_30: {f2(extra['top1_avg_p_pa_30'])}\n")

    print(f"Calibration delta: {extra['avg_minus_base_pp']:+.2f} pp")

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Model saved: {model_name}")
    print("=" * 30 + "\n")

    # -----------------------
    # CSV Run Logging
    # -----------------------
    log_file = MODELS_DIR / "train_runs.csv"
    write_header = not log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "run_time",
                "train_table",
                "roc_auc",
                "log_loss",
                "baseline_hr_rate",
                "top10_hr_rate",
                "top1_hr_rate",
                "top10_lift",
                "top1_lift",
                "features",
            ])

        writer.writerow([
            run_time,
            train_path.name,
            metrics["roc_auc"],
            metrics["log_loss"],
            metrics["test_hr_rate"],
            extra["top10_hr_rate"],
            extra["top1_hr_rate"],
            extra["top10_lift"],
            extra["top1_lift"],
            len(feature_cols),
        ])

@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]
    extra: dict


def latest_train_table() -> Path:
    """
    Preference order:
      1. 2021-2025 multi-season file (produced by build_features_multi_season.py)
      2. 2024 full-season file (legacy)
      3. Most-recently-modified train_table_*.parquet
    """
    multi = PROCESSED_DIR / "train_table_2021_2025_full.parquet"
    if multi.exists():
        return multi

    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season

    files = sorted(
        PROCESSED_DIR.glob("train_table_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed/")
    return files[0]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def calendar_split(
    df: pd.DataFrame,
    test_start: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard calendar-based train/test split.

    All rows with game_date < test_start  → train
    All rows with game_date >= test_start → test

    This is the correct split for a multi-year dataset: train on 2021-2024,
    test on 2025 (out-of-time validation).
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cut = pd.Timestamp(test_start)
    train = df[df["game_date"] < cut].sort_values("game_date").reset_index(drop=True)
    test  = df[df["game_date"] >= cut].sort_values("game_date").reset_index(drop=True)
    return train, test


def pct_time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Percentage-based time split (kept for internal calibration sub-split).
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_baseline(train_path: Path) -> TrainResult:
    df = pd.read_parquet(train_path)

    feature_cols = [
        # ---- Batter (14d) -----------------------------------------------
        "b_pa_14",
        "b_hr_rate_14",
        "b_barrel_rate_14",
        "b_ev_mean_14",
        "b_la_mean_14",
        "b_hardhit_rate_14",
        "b_fb_rate_14",

        # ---- Batter (season) --------------------------------------------
        "b_pa_szn",
        "b_hr_rate_szn",
        "b_barrel_rate_szn",
        "b_ev_mean_szn",
        "b_la_mean_szn",
        "b_hardhit_rate_szn",
        "b_fb_rate_szn",

        # ---- Pitcher allowed (30d) — starter only -----------------------
        "p_pa_30",
        "p_hr_allowed_rate_30",
        "p_ev_allowed_mean_30",
        "p_hardhit_allowed_rate_30",
        "p_fb_allowed_rate_30",
        "p_barrel_allowed_rate_30",

        # ---- Pitcher allowed (season) — starter only --------------------
        "p_pa_szn",
        "p_hr_allowed_rate_szn",
        "p_ev_allowed_mean_szn",
        "p_hardhit_allowed_rate_szn",
        "p_fb_allowed_rate_szn",
        "p_barrel_allowed_rate_szn",

        # ---- Edge (batter 14d vs starter 30d) ---------------------------
        "ev_edge_14_30",
        "hardhit_edge_14_30",
        "fb_edge_14_30",
        "barrel_edge_14_30",
        "hr_rate_edge_14_30",

        # ---- K/BB discipline --------------------------------------------
        "b_k_rate_14",
        "b_bb_rate_14",
        "p_k_rate_30",
        "p_bb_rate_30",
        "k_rate_edge_14_30",
        "bb_rate_edge_14_30",
        "k_rate_interaction_14_30",
        "bb_rate_interaction_14_30",
        "contact_pressure_14_30",
        "discipline_balance_14_30",

        # ---- Context ----------------------------------------------------
        "park_factor_hr",

        # ---- Pitcher assignment quality ---------------------------------
        # Fraction of in-game PAs the batter took against the starter.
        # High relief_pa_pct means the starter-based matchup features are
        # noisier (batter actually faced a lot of relievers).
        "relief_pa_pct",

        # ---- Lineup context (NEW) ---------------------------------------
        # batting_order_pos: 1–9 raw slot; 0 when unknown.
        # is_top_of_order:   1 if slots 1–4 (more PA, more HR opportunity).
        # expected_pa_today: continuous proxy for plate-appearance volume
        #                    derived from historical PA-by-lineup-slot averages.
        "batting_order_pos",
        "is_top_of_order",
        "expected_pa_today",

        # ---- Lineup protection (NEW) ------------------------------------
        # OPS-proxy of the hitter directly ahead/behind in the order.
        # Higher protection_ops_ahead → pitcher more likely to throw strikes
        # to avoid putting runners on ahead of a dangerous hitter.
        # lineup_ops_context → overall quality of the lineup surrounding batter.
        "protection_ops_ahead",
        "protection_ops_behind",
        "lineup_ops_context",

        # ---- Bullpen quality (NEW) --------------------------------------
        # Rolling 30d stats for the opposing team's bullpen (non-starters).
        # bp_hr_rate_x_relief_pct: interaction — bullpen quality only matters
        # when the batter actually faces relievers (high relief_pa_pct).
        "bp_hr_allowed_rate_30",
        "bp_hardhit_allowed_rate_30",
        "bp_bb_rate_30",
        "bp_era_proxy_30",
        "bp_hr_rate_x_relief_pct",
    ]

    # Gracefully drop any feature that isn't in the table yet
    # (allows reuse of legacy train tables built without new features).
    available = set(df.columns)
    missing_features = [c for c in feature_cols if c not in available]
    if missing_features:
        print(
            f"⚠️  The following features are absent from the training table and "
            f"will be excluded this run:\n    {missing_features}\n"
            f"   Rebuild the feature table to include them."
        )
    feature_cols = [c for c in feature_cols if c in available]

    missing_required = [c for c in ["hr_hit", "game_date"] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in training table: {missing_required}")

    # ------------------------------------------------------------------
    # Calendar split: train=2021-2024, test=2025
    # ------------------------------------------------------------------
    train_df, test_df = calendar_split(df, test_start=TEST_START_DATE)

    if len(test_df) == 0:
        print(
            f"⚠️  No rows found on/after {TEST_START_DATE}. "
            "Falling back to percentage-based split (last 20%)."
        )
        train_df, test_df = pct_time_split(df, test_size=0.2)
    else:
        print(
            f"Calendar split: train={len(train_df):,} rows "
            f"({train_df['game_date'].min().date()} → {train_df['game_date'].max().date()})  "
            f"test={len(test_df):,} rows "
            f"({test_df['game_date'].min().date()} → {test_df['game_date'].max().date()})"
        )

    train_start = str(train_df["game_date"].min().date())
    train_end   = str(train_df["game_date"].max().date())
    test_start  = str(test_df["game_date"].min().date())
    test_end    = str(test_df["game_date"].max().date())

    # Split train into core + calibration (time-aware, percentage-based within train)
    train_core_df, calib_df = pct_time_split(train_df, test_size=0.2)

    X_train_core = train_core_df[feature_cols].fillna(0.0)
    y_train_core = train_core_df["hr_hit"].astype(int)

    X_calib = calib_df[feature_cols].fillna(0.0)
    y_calib = calib_df["hr_hit"].astype(int)

    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["hr_hit"].astype(int)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    base_pipeline.fit(X_train_core, y_train_core)
    p_test_lr_raw = base_pipeline.predict_proba(X_test)[:, 1]
    roc_lr_raw = float(roc_auc_score(y_test, p_test_lr_raw))
    print(f"LogReg (raw) ROC-AUC: {roc_lr_raw:.3f}")

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
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
    p_test_lgbm_raw = lgbm.predict_proba(X_test)[:, 1]
    roc_lgbm_raw = float(roc_auc_score(y_test, p_test_lgbm_raw))
    print(f"LightGBM (raw) ROC-AUC: {roc_lgbm_raw:.3f}")

    # ------------------------------------------------------------------
    # Calibrate both
    # ------------------------------------------------------------------
    calibrated_lr = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method="sigmoid",
        cv=None,
    )
    calibrated_lr.fit(X_calib, y_calib)

    calibrated_lgbm = CalibratedClassifierCV(
        estimator=FrozenEstimator(lgbm),
        method="sigmoid",
        cv=None,
    )
    calibrated_lgbm.fit(X_calib, y_calib)

    # ------------------------------------------------------------------
    # Choose best model by raw ROC-AUC
    # ------------------------------------------------------------------
    if roc_lgbm_raw > roc_lr_raw:
        print("Chosen model: LightGBM")
        model = calibrated_lgbm
        p_test = calibrated_lgbm.predict_proba(X_test)[:, 1]
        chosen_name = "lightgbm_calibrated"
    else:
        print("Chosen model: LogReg")
        model = calibrated_lr
        p_test = calibrated_lr.predict_proba(X_test)[:, 1]
        chosen_name = "logreg_calibrated"

    # ------------------------------------------------------------------
    # Lift / bucket stats
    # ------------------------------------------------------------------
    q90 = np.quantile(p_test, 0.90)
    top10_mask = p_test >= q90
    top10_hr_rate = float(y_test[top10_mask].mean())

    q99 = np.quantile(p_test, 0.99)
    top1_mask = p_test >= q99
    top1_hr_rate = float(y_test[top1_mask].mean())

    baseline = float(y_test.mean())
    avg_pred = float(p_test.mean())

    extra = {
        "top10_hr_rate": top10_hr_rate,
        "top1_hr_rate": top1_hr_rate,
        "top1_count": int(top1_mask.sum()),
        "top1_avg_b_pa_14": float(X_test.loc[top1_mask, "b_pa_14"].mean()),
        "top1_avg_p_pa_30": float(X_test.loc[top1_mask, "p_pa_30"].mean()),
        "max_pred_prob": float(p_test.max()),
        "top10_lift": (top10_hr_rate / baseline) if baseline > 0 else float("nan"),
        "top1_lift":  (top1_hr_rate  / baseline) if baseline > 0 else float("nan"),
        "avg_minus_base_pp": (avg_pred - baseline) * 100.0,
    }

    metrics_dict = {
        "train_rows":    int(len(train_df)),
        "test_rows":     int(len(test_df)),
        "train_start":   train_start,
        "train_end":     train_end,
        "test_start":    test_start,
        "test_end":      test_end,
        "test_hr_rate":  float(y_test.mean()),
        "avg_pred_prob": float(p_test.mean()),
        "log_loss":      float(log_loss(y_test, p_test, labels=[0, 1])),
        "roc_auc":       float(roc_auc_score(y_test, p_test)),
    }

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / f"hr_model_{chosen_name}_2021_2025.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    return TrainResult(
        model_path=model_path,
        metrics=metrics_dict,
        feature_cols=feature_cols,
        extra=extra,
    )


if __name__ == "__main__":
    train_path = latest_train_table()
    result = train_baseline(train_path)
    print_summary(train_path, result.model_path, result.feature_cols, result.metrics, result.extra)

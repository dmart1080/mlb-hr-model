from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def f3(x: float) -> str:
    return f"{x:.3f}"

def f2(x: float) -> str:
    return f"{x:.2f}"

def fmt_int(x: int) -> str:
    return f"{x:,}"

def print_summary(train_path: Path, model_path: Path, feature_cols: list[str], metrics: dict, extra: dict):
    print("\n" + "=" * 30)
    print("MLB HR MODEL — TRAIN SUMMARY")
    print("=" * 30)
    print(f"Training table: {train_path.name}\n")
    
    print(f"Date range: train={metrics['train_start']}→{metrics['train_end']}  test={metrics['test_start']}→{metrics['test_end']}\n")
    print(f"Rows: train={fmt_int(metrics['train_rows'])}  test={fmt_int(metrics['test_rows'])}")
    print(f"Test HR rate (baseline): {pct(metrics['test_hr_rate'])}\n")

    print("Performance (test):")
    print(f"  ROC-AUC:   {f3(metrics['roc_auc'])}")
    print(f"  Log loss:  {f3(metrics['log_loss'])}")
    print(f"  Avg pred:  {pct(metrics['avg_pred_prob'])}")
    print(f"  Max pred:  {f3(extra['max_pred_prob'])}\n")

    print("Lift checks (test):")
    print(f"  Top 10% HR rate: {pct(extra['top10_hr_rate'])}  ({extra['top10_lift']:.2f}x baseline)")
    print(
        f"  Top 1%  HR rate: {pct(extra['top1_hr_rate'])}  ({extra['top1_lift']:.2f}x baseline)"
        f"  (n={fmt_int(extra['top1_count'])})"
    )
    print(f"  Top 1%  avg b_pa_14: {f2(extra['top1_avg_b_pa_14'])}")
    print(f"  Top 1%  avg p_pa_30: {f2(extra['top1_avg_p_pa_30'])}\n")

    print(f"Calibration delta: {extra['avg_minus_base_pp']:+.2f} pp")

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Model saved: {model_path}")

    print("=" * 30 + "\n")


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]
    extra: dict


def latest_train_table() -> Path:
    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season
    files = sorted(PROCESSED_DIR.glob("train_table_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found.")
    return files[0]



def time_split(df: pd.DataFrame, test_size: float = 0.2):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_baseline(train_path: Path) -> TrainResult:
    df = pd.read_parquet(train_path)

    # MVP features you already built in build_features.py
    feature_cols = [
        # Batter (14d)
        "b_pa_14",
        "b_hr_rate_14",
        "b_barrel_rate_14",
        "b_ev_mean_14",
        "b_la_mean_14",
        "b_hardhit_rate_14",
        "b_fb_rate_14",

        # Batter (season)
        "b_pa_szn",
        "b_hr_rate_szn",
        "b_barrel_rate_szn",
        "b_ev_mean_szn",
        "b_la_mean_szn",
        "b_hardhit_rate_szn",
        "b_fb_rate_szn",

        # Pitcher allowed (30d)
        "p_pa_30",
        "p_hr_allowed_rate_30",
        "p_ev_allowed_mean_30",
        "p_hardhit_allowed_rate_30",
        "p_fb_allowed_rate_30",
        "p_barrel_allowed_rate_30",

        # Pitcher allowed (season)
        "p_pa_szn",
        "p_hr_allowed_rate_szn",
        "p_ev_allowed_mean_szn",
        "p_hardhit_allowed_rate_szn",
        "p_fb_allowed_rate_szn",
        "p_barrel_allowed_rate_szn",

        "ev_edge_14_30",
        "hardhit_edge_14_30",
        "fb_edge_14_30",
        "barrel_edge_14_30",
        "hr_rate_edge_14_30",
        # Context
        "park_factor_hr",
    ]


    missing = [c for c in (feature_cols + ["hr_hit", "game_date"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training table: {missing}")

    train_df, test_df = time_split(df, test_size=0.2)
    
    train_start = str(train_df["game_date"].min().date())
    train_end   = str(train_df["game_date"].max().date())
    test_start  = str(test_df["game_date"].min().date())
    test_end    = str(test_df["game_date"].max().date())

    # Split train into core + calibration (time-aware)
    train_core_df, calib_df = time_split(train_df, test_size=0.2)

    X_train_core = train_core_df[feature_cols].fillna(0.0)
    y_train_core = train_core_df["hr_hit"].astype(int)

    X_calib = calib_df[feature_cols].fillna(0.0)
    y_calib = calib_df["hr_hit"].astype(int)

    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["hr_hit"].astype(int)

    # Logistic regression (needs scaling) + class balancing
    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    # Fit on early training period
    base_pipeline.fit(X_train_core, y_train_core)

    # Calibrate using later training period (sigmoid = Platt scaling)
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method="sigmoid",
        cv=None,
    )
    calibrated_model.fit(X_calib, y_calib)

    p_test = calibrated_model.predict_proba(X_test)[:, 1]

    # top bucket stats
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
        # useful add-ons
        "top10_lift": (top10_hr_rate / baseline) if baseline > 0 else float("nan"),
        "top1_lift": (top1_hr_rate / baseline) if baseline > 0 else float("nan"),
        "avg_minus_base_pp": (avg_pred - baseline) * 100.0,
    }

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "test_hr_rate": float(y_test.mean()),
        "avg_pred_prob": float(p_test.mean()),
        "log_loss": float(log_loss(y_test, p_test, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
    }


    model_path = MODELS_DIR / "hr_model_logreg_edges_calibrated_2024.joblib"
    joblib.dump({"model": calibrated_model, "feature_cols": feature_cols}, model_path)

    return TrainResult(model_path=model_path, metrics=metrics, feature_cols=feature_cols, extra=extra)

if __name__ == "__main__":
    train_path = latest_train_table()
    result = train_baseline(train_path)
    print_summary(train_path, result.model_path, result.feature_cols, result.metrics, result.extra)
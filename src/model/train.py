from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]


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
        "b_pa_14",
        "b_hr_rate_14",
        "b_pa_szn",
        "b_hr_rate_szn",
        "p_pa_30",
        "p_hr_allowed_rate_30",
        "p_pa_szn",
        "p_hr_allowed_rate_szn",
        "park_factor_hr",
    ]

    missing = [c for c in (feature_cols + ["hr_hit", "game_date"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training table: {missing}")

    train_df, test_df = time_split(df, test_size=0.2)

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["hr_hit"].astype(int)

    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["hr_hit"].astype(int)

    model = LogisticRegression(max_iter=400, class_weight="balanced")
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_hr_rate": float(y_test.mean()),
        "avg_pred_prob": float(p_test.mean()),
        "log_loss": float(log_loss(y_test, p_test, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
    }

    bundle = {"model": model, "feature_cols": feature_cols}
    model_path = MODELS_DIR / "hr_model_logreg_baseline.joblib"
    joblib.dump(bundle, model_path)

    return TrainResult(model_path=model_path, metrics=metrics, feature_cols=feature_cols)


if __name__ == "__main__":
    train_path = latest_train_table()
    print(f"Using training table: {train_path.name}")

    result = train_baseline(train_path)

    print(f"Saved model: {result.model_path}")
    print("Features:", result.feature_cols)
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

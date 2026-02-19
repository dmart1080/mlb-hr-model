from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def latest_train_table() -> Path:
    # Prefer combined multi-season, then full-season, then any chunk
    for pattern in (
        "train_table_*_combined.parquet",
        "train_table_*_full_season.parquet",
        "train_table_*.parquet",
    ):
        files = sorted(PROCESSED_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    raise FileNotFoundError("No train_table_*.parquet found in data/processed/")


def load_model():
    """Load the most recently saved calibrated model."""
    candidates = sorted(
        MODELS_DIR.glob("hr_model_*_calibrated_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No calibrated model found in models/. Run train.py first."
        )
    model_path = candidates[0]
    print(f"Loading model: {model_path.name}")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


def add_player_names(df: pd.DataFrame, id_col: str, out_col: str) -> pd.DataFrame:
    try:
        from pybaseball.playerid_lookup import playerid_reverse_lookup
        ids = df[id_col].dropna().astype(int).unique().tolist()
        if not ids:
            df[out_col] = None
            return df
        look = playerid_reverse_lookup(ids, key_type="mlbam")
        look["full_name"] = (
            look["name_first"].fillna("") + " " + look["name_last"].fillna("")
        ).str.strip()
        mapping = dict(zip(look["key_mlbam"].astype(int), look["full_name"]))
        df[out_col] = df[id_col].astype("Int64").map(
            lambda x: mapping.get(int(x)) if pd.notna(x) else None
        )
    except Exception:
        # Fallback: just show the numeric ID
        df[out_col] = df[id_col].astype(str)
    return df

if __name__ == "__main__":
    model, feature_cols = load_model()
    train_path = latest_train_table()

    df = pd.read_parquet(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    target_date = df["game_date"].max()
    today_df = df[df["game_date"] == target_date].copy()

    X = today_df[feature_cols].fillna(0.0)
    today_df["hr_prob"] = model.predict_proba(X)[:, 1]

    today_df = add_player_names(today_df, "batter", "batter_name")
    if "pitcher" in today_df.columns:
        today_df = add_player_names(today_df, "pitcher", "pitcher_name")

    ranked = today_df.sort_values("hr_prob", ascending=False)

    cols = ["batter_name", "pitcher_name", "hr_prob", "batter", "pitcher"]
    cols = [c for c in cols if c in ranked.columns]

    print(f"\nTop HR candidates for {target_date.date()}:\n")
    print(ranked[cols].head(15).to_string(index=False))

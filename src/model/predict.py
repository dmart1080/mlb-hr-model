from __future__ import annotations

from pathlib import Path
import glob
import joblib
import pandas as pd

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def latest_train_table() -> Path:
    files = sorted(glob.glob(str(PROCESSED_DIR / "train_table_*.parquet")))
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed.")
    files = sorted(files, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(files[0])


def load_model():
    model_path = MODELS_DIR / "hr_model_logreg_edges_calibrated_2024.joblib"
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


def add_player_names(df: pd.DataFrame, id_col: str, out_col: str) -> pd.DataFrame:
    ids = df[id_col].dropna().astype(int).unique().tolist()
    if not ids:
        df[out_col] = None
        return df

    # reverse lookup returns key_mlbam + name_first/name_last
    look = playerid_reverse_lookup(ids, key_type="mlbam")
    look["full_name"] = look["name_first"].fillna("") + " " + look["name_last"].fillna("")
    mapping = dict(zip(look["key_mlbam"].astype(int), look["full_name"].str.strip()))

    df[out_col] = df[id_col].astype("Int64").map(lambda x: mapping.get(int(x)) if pd.notna(x) else None)
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

    # Add names
    today_df = add_player_names(today_df, "batter", "batter_name")
    if "pitcher" in today_df.columns:
        today_df = add_player_names(today_df, "pitcher", "pitcher_name")

    ranked = today_df.sort_values("hr_prob", ascending=False)

    cols = ["batter_name", "pitcher_name", "hr_prob", "batter", "pitcher"]
    cols = [c for c in cols if c in ranked.columns]

    print(f"\nTop HR candidates for {target_date.date()}:\n")
    print(ranked[cols].head(15).to_string(index=False))

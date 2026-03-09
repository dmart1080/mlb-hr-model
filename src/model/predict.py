from __future__ import annotations

from pathlib import Path
import glob
import joblib
import pandas as pd

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Maximum IDs to resolve in a single playerid_reverse_lookup call.
_LOOKUP_CHUNK_SIZE = 500


def latest_train_table() -> Path:
    """
    Preference order:
      1. 2021-2025 multi-season file
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
        glob.glob(str(PROCESSED_DIR / "train_table_*.parquet")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed.")
    return Path(files[0])


def load_model():
    """
    Load the best available model.  Preference order:
      1. 2021-2025 LightGBM calibrated
      2. 2021-2025 LogReg calibrated
      3. Legacy 2024 models
    """
    candidates = [
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_lightgbm_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_edges_calibrated_2024.joblib",
    ]
    for path in candidates:
        if path.exists():
            print(f"Loading model: {path.name}")
            bundle = joblib.load(path)
            return bundle["model"], bundle["feature_cols"], bundle.get("apply_shrinkage", False)

    raise FileNotFoundError(
        "No trained model found in models/. Run src/model/train.py first."
    )


def _build_id_name_map(ids: list[int]) -> dict[int, str]:
    """
    Resolve a list of MLBAM IDs to full names using playerid_reverse_lookup.
    De-dupes across both batter and pitcher columns to minimise HTTP calls.
    """
    if not ids:
        return {}

    unique_ids = list(set(ids))
    mapping: dict[int, str] = {}

    for start in range(0, len(unique_ids), _LOOKUP_CHUNK_SIZE):
        chunk = unique_ids[start : start + _LOOKUP_CHUNK_SIZE]
        try:
            look = playerid_reverse_lookup(chunk, key_type="mlbam")
            look["full_name"] = (
                look["name_first"].fillna("") + " " + look["name_last"].fillna("")
            ).str.strip()
            mapping.update(
                dict(zip(look["key_mlbam"].astype(int), look["full_name"]))
            )
        except Exception as exc:
            print(f"  ⚠️  playerid_reverse_lookup failed for chunk: {exc}")

    return mapping


def add_player_names(
    df: pd.DataFrame,
    id_name_map: dict[int, str],
    id_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Map MLBAM IDs to display names using a pre-built lookup dict."""
    df[out_col] = df[id_col].astype("Int64").map(
        lambda x: id_name_map.get(int(x)) if pd.notna(x) else None
    )
    return df


if __name__ == "__main__":
    model, feature_cols, apply_shrinkage = load_model()
    train_path = latest_train_table()

    df = pd.read_parquet(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Apply the same empirical Bayes shrinkage used at training time.
    # This ensures the feature distribution at inference matches what
    # the model was calibrated on.  Without this, cold-start players
    # would appear with inflated raw rates and get over-scored.
    if apply_shrinkage:
        from src.model.train import apply_shrinkage as _apply_shrinkage
        print("Applying empirical Bayes shrinkage ...")
        df = _apply_shrinkage(df)

    target_date = df["game_date"].max()
    today_df = df[df["game_date"] == target_date].copy()

    # Keep only feature columns that exist in the current table
    available_features = [c for c in feature_cols if c in today_df.columns]
    missing = [c for c in feature_cols if c not in today_df.columns]
    if missing:
        print(f"⚠️  {len(missing)} feature(s) missing at inference — filling with 0:")
        for c in missing:
            print(f"     - {c}")
            today_df[c] = 0.0

    X = today_df[feature_cols].fillna(0.0)
    today_df["hr_prob"] = model.predict_proba(X)[:, 1]

    # Collect ALL unique IDs from both columns in one pass
    all_ids: list[int] = []
    for col in ("batter", "pitcher"):
        if col in today_df.columns:
            all_ids.extend(
                today_df[col].dropna().astype(int).unique().tolist()
            )
    id_name_map = _build_id_name_map(all_ids)

    today_df = add_player_names(today_df, id_name_map, "batter",  "batter_name")
    if "pitcher" in today_df.columns:
        today_df = add_player_names(today_df, id_name_map, "pitcher", "pitcher_name")

    ranked = today_df.sort_values("hr_prob", ascending=False)

    # Show lineup position alongside predictions so the impact of the
    # batting_order_pos feature is transparent in the output.
    display_cols = [
        "batter_name", "pitcher_name", "hr_prob",
        "batting_order_pos", "expected_pa_today",
        "batter", "pitcher",
    ]
    cols = [c for c in display_cols if c in ranked.columns]

    print(f"\nTop HR candidates for {target_date.date()}:\n")
    print(ranked[cols].head(15).to_string(index=False))

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
# The pybaseball implementation fetches a static CSV from Chadwick Bureau;
# splitting into chunks avoids hitting any implicit request-size limits while
# still keeping the total number of HTTP round-trips low.
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
      1. 2021-2025 LightGBM calibrated  (produced by new train.py)
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
            return bundle["model"], bundle["feature_cols"]

    raise FileNotFoundError(
        "No trained model found in models/. Run src/model/train.py first."
    )


def _build_id_name_map(ids: list[int]) -> dict[int, str]:
    """
    Resolve a list of MLBAM IDs to full names using playerid_reverse_lookup.

    FIX: the original code called playerid_reverse_lookup once per column,
    which is fine, but the underlying pybaseball function downloads the full
    Chadwick register CSV on every cold call (~3 MB).  This helper de-dupes
    across both batter and pitcher columns and chunks large ID lists so a
    single large slate (~300 matchups) doesn't trigger multiple full downloads.
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
            # Non-fatal — prediction still works without display names
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
    model, feature_cols = load_model()
    train_path = latest_train_table()

    df = pd.read_parquet(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    target_date = df["game_date"].max()
    today_df = df[df["game_date"] == target_date].copy()

    X = today_df[feature_cols].fillna(0.0)
    today_df["hr_prob"] = model.predict_proba(X)[:, 1]

    # FIX: collect ALL unique IDs from both columns in one pass, then build
    # the name map once — avoids downloading the Chadwick CSV twice.
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

    cols = ["batter_name", "pitcher_name", "hr_prob", "batter", "pitcher"]
    cols = [c for c in cols if c in ranked.columns]

    print(f"\nTop HR candidates for {target_date.date()}:\n")
    print(ranked[cols].head(15).to_string(index=False))

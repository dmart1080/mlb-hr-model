from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events, REGULAR_SEASON_GAME_TYPE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LabelsBuildResult:
    labels_df: pd.DataFrame
    output_path: Path


def build_batter_game_labels(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Statcast event-level rows into batter-game labels.

    Output: one row per (game_date, game_pk, batter) with:
      - hr_hit       : 1 if the batter hit at least one HR in that game
      - starter_id   : pitcher_id of the first pitcher faced (true starter proxy)
      - pitcher_mode : pitcher faced most often (kept for back-compat)
      - pa_count     : number of PAs observed for that batter in that game
    """
    required = {"game_date", "game_pk", "batter", "pitcher", "events"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = events_df.copy()

    if "is_hr" not in df.columns:
        df["is_hr"] = (df["events"] == "home_run").fillna(False).astype("int8")
    df["events"] = df["events"].astype("string")

    # --- pitcher_mode (legacy) ---
    pitcher_mode = (
        df.groupby(["game_date", "game_pk", "batter"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .rename("pitcher_mode")
        .reset_index()
    )

    # --- starter_id (new, preferred) ---
    # Use sort + groupby().first() instead of apply() — this avoids a pandas
    # version inconsistency where groupby().apply() can return a DataFrame
    # instead of a Series (causing TypeError on .rename("starter_id")).
    # Sort by at_bat_number when available so we pick the first pitcher faced.
    _sort_cols = (
        ["game_date", "game_pk", "batter", "at_bat_number"]
        if "at_bat_number" in df.columns
        else ["game_date", "game_pk", "batter"]
    )
    starter_id = (
        df.sort_values(_sort_cols, na_position="last")
        .groupby(["game_date", "game_pk", "batter"], sort=False)["pitcher"]
        .first()
        .rename("starter_id")
        .reset_index()
    )

    # --- aggregate to batter-game level ---
    labels = (
        df.groupby(["game_date", "game_pk", "batter"], as_index=False)
        .agg(
            hr_hit=("is_hr", "max"),
            pa_count=("pitcher", "size"),
        )
        .merge(pitcher_mode, on=["game_date", "game_pk", "batter"], how="left")
        .merge(starter_id,   on=["game_date", "game_pk", "batter"], how="left")
    )

    return labels


def run_build_labels(start_date: str, end_date: str) -> LabelsBuildResult:
    events = fetch_statcast_events(
        start_date=start_date,
        end_date=end_date,
        columns=[
            "game_date", "game_pk", "batter", "pitcher",
            "events", "at_bat_number",
            "game_type",          # required for regular_season_only filter
        ],
        regular_season_only=True,
    ).df

    labels_df = build_batter_game_labels(events)

    output_path = PROCESSED_DIR / f"labels_{start_date}_to_{end_date}.parquet"
    labels_df.to_parquet(output_path, index=False)

    return LabelsBuildResult(labels_df=labels_df, output_path=output_path)


if __name__ == "__main__":
    result = run_build_labels("2024-06-01", "2024-06-03")
    print(f"Saved labels to: {result.output_path}")
    print(f"Rows (batter-games): {len(result.labels_df):,}")
    print(result.labels_df.head(15))
    print("\nHR rate in this slice:", result.labels_df["hr_hit"].mean())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events

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
      - hr_hit: 1 if the batter hit at least one HR in that game else 0
      - pitcher_mode: the pitcher faced most often in that game (simple MVP matchup proxy)
      - pa_count: number of PAs observed for that batter in that game (rough proxy)
    """
    required = {"game_date", "game_pk", "batter", "pitcher", "events"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = events_df.copy()

    # A plate appearance is roughly each row here; some rows may not be true PAs,
    # but for MVP it's fine as a count proxy.
    if "is_hr" not in df.columns:
        df["is_hr"] = (df["events"] == "home_run").fillna(False).astype("int8")
    df = df.copy()
    df["events"] = df["events"].astype("string")


    # Find the most common pitcher faced per batter-game (mode)
    pitcher_mode = (
        df.groupby(["game_date", "game_pk", "batter"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .rename("pitcher_mode")
        .reset_index()
    )

    # Aggregate to one row per batter-game
    labels = (
        df.groupby(["game_date", "game_pk", "batter"], as_index=False)
        .agg(
            hr_hit=("is_hr", "max"),
            pa_count=("pitcher", "size"),
        )
        .merge(pitcher_mode, on=["game_date", "game_pk", "batter"], how="left")
    )

    return labels


def run_build_labels(start_date: str, end_date: str) -> LabelsBuildResult:
    # Pull events (cached)
    events = fetch_statcast_events(
        start_date=start_date,
        end_date=end_date,
        columns=["game_date", "game_pk", "batter", "pitcher", "events"],
    ).df

    labels_df = build_batter_game_labels(events)

    output_path = PROCESSED_DIR / f"labels_{start_date}_to_{end_date}.parquet"
    labels_df.to_parquet(output_path, index=False)

    return LabelsBuildResult(labels_df=labels_df, output_path=output_path)


if __name__ == "__main__":
    # Quick test build for the same window you already pulled
    result = run_build_labels("2024-06-01", "2024-06-03")
    print(f"Saved labels to: {result.output_path}")
    print(f"Rows (batter-games): {len(result.labels_df):,}")
    print(result.labels_df.head(15))
    print("\nHR rate in this slice:", result.labels_df["hr_hit"].mean())

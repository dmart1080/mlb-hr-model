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
      - hr_hit          : 1 if batter hit at least one HR in that game
      - pitcher_mode    : most-common pitcher faced (legacy fallback proxy)
      - pa_count        : number of PAs observed for that batter in that game
      - relief_pa_pct   : fraction of PAs faced against non-starter pitchers.
                          Requires pitcher_is_starter to be set downstream
                          (via enrich_labels_with_roster); defaults to 0.0
                          until then and is recomputed in build_features.py.
    """
    required = {"game_date", "game_pk", "batter", "pitcher", "events"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = events_df.copy()

    if "is_hr" not in df.columns:
        df["is_hr"] = (df["events"] == "home_run").fillna(False).astype("int8")
    df["events"] = df["events"].astype("string")

    # Most common pitcher faced per batter-game (kept as fallback)
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

    # relief_pa_pct placeholder — will be filled properly in build_features.py
    # once we know which pitcher is the starter for each game
    labels["relief_pa_pct"] = 0.0

    return labels


def compute_relief_pa_pct(
    labels: pd.DataFrame,
    pa_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each batter-game, compute the fraction of plate appearances taken
    against pitchers who are NOT the game's starting pitcher.

    Parameters
    ----------
    labels : DataFrame
        Must contain game_pk, batter, starter_pitcher_id columns.
    pa_df  : DataFrame
        Plate-appearance level data with game_pk, batter, pitcher columns.

    Returns
    -------
    labels with relief_pa_pct filled in (0.0 where data is unavailable).
    """
    required_label_cols = {"game_pk", "batter", "starter_pitcher_id"}
    if not required_label_cols.issubset(labels.columns):
        return labels  # can't compute yet

    pa = pa_df[["game_pk", "batter", "pitcher"]].copy()
    pa["game_pk"] = pa["game_pk"].astype(int)
    pa["batter"]  = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)

    merged = pa.merge(
        labels[["game_pk", "batter", "starter_pitcher_id"]].dropna(
            subset=["starter_pitcher_id"]
        ).assign(
            game_pk=lambda d: d["game_pk"].astype(int),
            batter=lambda d: d["batter"].astype(int),
            starter_pitcher_id=lambda d: d["starter_pitcher_id"].astype(int),
        ),
        on=["game_pk", "batter"],
        how="inner",
    )

    merged["is_vs_starter"] = (
        merged["pitcher"] == merged["starter_pitcher_id"]
    ).astype(int)

    relief_pct = (
        merged.groupby(["game_pk", "batter"])
        .agg(
            total_pa=("pitcher", "size"),
            starter_pa=("is_vs_starter", "sum"),
        )
        .assign(relief_pa_pct=lambda d: 1.0 - d["starter_pa"] / d["total_pa"])
        .reset_index()[["game_pk", "batter", "relief_pa_pct"]]
    )

    labels = labels.copy()
    labels["game_pk"] = labels["game_pk"].astype(int)
    labels["batter"]  = labels["batter"].astype(int)

    # Drop old placeholder and replace
    if "relief_pa_pct" in labels.columns:
        labels = labels.drop(columns=["relief_pa_pct"])

    labels = labels.merge(relief_pct, on=["game_pk", "batter"], how="left")
    labels["relief_pa_pct"] = labels["relief_pa_pct"].fillna(0.0)

    return labels


def run_build_labels(start_date: str, end_date: str) -> LabelsBuildResult:
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
    result = run_build_labels("2024-06-01", "2024-06-03")
    print(f"Saved labels to: {result.output_path}")
    print(f"Rows (batter-games): {len(result.labels_df):,}")
    print(result.labels_df.head(15))
    print("\nHR rate in this slice:", result.labels_df["hr_hit"].mean())

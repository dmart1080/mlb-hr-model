from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events
from src.features.build_labels import build_batter_game_labels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeaturesBuildResult:
    features_df: pd.DataFrame
    output_path: Path


def _to_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()


def _date_minus_days(d: pd.Timestamp, days: int) -> pd.Timestamp:
    return d - pd.Timedelta(days=days)


def build_features_for_range(start_date: str, end_date: str) -> FeaturesBuildResult:
    """
    Builds a training table for batter-games in [start_date, end_date],
    using ONLY historical events prior to each game date (as-of date = game_date - 1).

    Features (MVP):
      Batter last 14 days:
        - b_pa_14
        - b_hr_14
        - b_hr_rate_14

      Pitcher last 30 days:
        - p_pa_30
        - p_hr_allowed_30
        - p_hr_allowed_rate_30

      Season-to-date fallbacks (as-of D-1):
        - b_pa_szn, b_hr_szn, b_hr_rate_szn
        - p_pa_szn, p_hr_allowed_szn, p_hr_allowed_rate_szn

    Label:
      - hr_hit
    """
    start_dt = _to_date(start_date)
    end_dt = _to_date(end_date)

    # We need earlier data to compute rolling windows
    history_start = _date_minus_days(start_dt, 60)  # enough for 30-day pitcher + buffer

    # Pull event-level data for history + target window
    events = fetch_statcast_events(
        start_date=history_start.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        columns=["game_date", "game_pk", "batter", "pitcher", "events"],
    ).df.copy()

    # Convert game_date to Timestamp for filtering
    events["game_date"] = pd.to_datetime(events["game_date"])

    # Create labels only for the target window
    target_events = events[(events["game_date"] >= start_dt) & (events["game_date"] <= end_dt)]
    labels = build_batter_game_labels(target_events)

    # Make labels game_date Timestamp too
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    # Precompute HR flags
    events["is_hr"] = (events["events"] == "home_run").astype(int)

    feature_rows = []

    # Iterate through batter-games in the label table
    for row in labels.itertuples(index=False):
        game_date = row.game_date
        batter_id = row.batter
        pitcher_id = row.pitcher_mode  # MVP proxy
        hr_hit = row.hr_hit
        game_pk = row.game_pk

        as_of = game_date - pd.Timedelta(days=1)

        # --- Batter windows ---
        b_14_start = as_of - pd.Timedelta(days=14)
        batter_14 = events[
            (events["batter"] == batter_id)
            & (events["game_date"] > b_14_start)
            & (events["game_date"] <= as_of)
        ]

        b_pa_14 = len(batter_14)
        b_hr_14 = int(batter_14["is_hr"].sum())
        b_hr_rate_14 = (b_hr_14 / b_pa_14) if b_pa_14 > 0 else 0.0

        batter_szn = events[
            (events["batter"] == batter_id)
            & (events["game_date"] >= pd.Timestamp(as_of.year, 3, 1))  # season-ish start
            & (events["game_date"] <= as_of)
        ]
        b_pa_szn = len(batter_szn)
        b_hr_szn = int(batter_szn["is_hr"].sum())
        b_hr_rate_szn = (b_hr_szn / b_pa_szn) if b_pa_szn > 0 else 0.0

        # --- Pitcher windows (HR allowed) ---
        p_30_start = as_of - pd.Timedelta(days=30)
        pitcher_30 = events[
            (events["pitcher"] == pitcher_id)
            & (events["game_date"] > p_30_start)
            & (events["game_date"] <= as_of)
        ]

        p_pa_30 = len(pitcher_30)
        p_hr_allowed_30 = int(pitcher_30["is_hr"].sum())
        p_hr_allowed_rate_30 = (p_hr_allowed_30 / p_pa_30) if p_pa_30 > 0 else 0.0

        pitcher_szn = events[
            (events["pitcher"] == pitcher_id)
            & (events["game_date"] >= pd.Timestamp(as_of.year, 3, 1))
            & (events["game_date"] <= as_of)
        ]
        p_pa_szn = len(pitcher_szn)
        p_hr_allowed_szn = int(pitcher_szn["is_hr"].sum())
        p_hr_allowed_rate_szn = (p_hr_allowed_szn / p_pa_szn) if p_pa_szn > 0 else 0.0

        feature_rows.append(
            {
                "game_date": game_date.date(),
                "game_pk": game_pk,
                "batter": batter_id,
                "pitcher": int(pitcher_id) if pd.notna(pitcher_id) else None,
                "hr_hit": int(hr_hit),
                "b_pa_14": b_pa_14,
                "b_hr_14": b_hr_14,
                "b_hr_rate_14": b_hr_rate_14,
                "b_pa_szn": b_pa_szn,
                "b_hr_szn": b_hr_szn,
                "b_hr_rate_szn": b_hr_rate_szn,
                "p_pa_30": p_pa_30,
                "p_hr_allowed_30": p_hr_allowed_30,
                "p_hr_allowed_rate_30": p_hr_allowed_rate_30,
                "p_pa_szn": p_pa_szn,
                "p_hr_allowed_szn": p_hr_allowed_szn,
                "p_hr_allowed_rate_szn": p_hr_allowed_rate_szn,
            }
        )

    features_df = pd.DataFrame(feature_rows)

    out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
    features_df.to_parquet(out_path, index=False)

    return FeaturesBuildResult(features_df=features_df, output_path=out_path)


if __name__ == "__main__":
    # Start small first (3 days) to verify speed and correctness
    result = build_features_for_range("2024-06-01", "2024-06-03")
    print(f"Saved features to: {result.output_path}")
    print(f"Rows: {len(result.features_df):,}")
    print(result.features_df.head(10))
    print("\nLabel HR rate:", result.features_df["hr_hit"].mean())

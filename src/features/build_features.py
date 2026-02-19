from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events
from src.features.build_labels import build_batter_game_labels
from src.features.park_factors import HR_PARK_FACTOR, DEFAULT_PARK_FACTOR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeaturesBuildResult:
    features_df: pd.DataFrame
    output_path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()


def _date_minus_days(d: pd.Timestamp, days: int) -> pd.Timestamp:
    return d - pd.Timedelta(days=days)


def _load_and_clean_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Download (or load from cache) and collapse to one row per plate appearance."""
    events = fetch_statcast_events(
        start_date=start_date,
        end_date=end_date,
        columns=[
            "game_date",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "events",
            "home_team",
            "launch_speed",
            "launch_angle",
        ],
    ).df.copy()

    # Use numpy-backed dtypes — avoids slow pyarrow boolean indexing
    events = events.convert_dtypes(dtype_backend="numpy_nullable")
    events["game_date"] = pd.to_datetime(events["game_date"])

    events["is_hr"] = (events["events"] == "home_run").fillna(False).astype("int8")

    # Collapse pitch-level rows → one row per plate appearance
    events = (
        events
        .sort_values(["game_pk", "batter", "game_date"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .agg({
            "game_date":    "first",
            "batter":       "last",
            "pitcher":      "last",
            "home_team":    "first",
            "events":       "last",
            "is_hr":        "max",
            "launch_speed": "max",
            "launch_angle": "max",
        })
    )

    ev_str = events["events"].astype("string")
    events["is_so"] = ev_str.str.contains("strikeout", na=False).astype("int8")
    events["is_bb"] = (ev_str == "walk").fillna(False).astype("int8")

    # Barrel approximation (EV ≥ 95 + LA 20–35)
    ev = pd.to_numeric(events["launch_speed"], errors="coerce")
    la = pd.to_numeric(events["launch_angle"], errors="coerce")
    events["is_barrel"] = ((ev >= 95) & (la.between(20, 35))).fillna(False).astype("int8")

    events["launch_speed"] = pd.to_numeric(events["launch_speed"], errors="coerce")
    events["launch_angle"] = pd.to_numeric(events["launch_angle"], errors="coerce")

    events = (
        events
        .sort_values("game_date")
        .dropna(subset=["game_date"])
        .reset_index(drop=True)
    )
    events["game_date"] = events["game_date"].astype("datetime64[ns]")
    return events


# ---------------------------------------------------------------------------
# Vectorised window aggregation
# ---------------------------------------------------------------------------

def _batter_stats_for_window(grp: pd.DataFrame, suffix: str) -> dict:
    """Aggregate a pre-filtered window of batter PA rows into a flat stat dict."""
    pa = len(grp)
    if pa == 0:
        return {
            f"b_pa_{suffix}":           0,
            f"b_hr_{suffix}":           0,
            f"b_hr_rate_{suffix}":      0.0,
            f"b_barrel_rate_{suffix}":  0.0,
            f"b_ev_mean_{suffix}":      0.0,
            f"b_la_mean_{suffix}":      0.0,
            f"b_hardhit_rate_{suffix}": 0.0,
            f"b_fb_rate_{suffix}":      0.0,
            f"b_k_rate_{suffix}":       0.0,
            f"b_bb_rate_{suffix}":      0.0,
        }
    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr  = int(grp["is_hr"].sum())
    return {
        f"b_pa_{suffix}":           pa,
        f"b_hr_{suffix}":           hr,
        f"b_hr_rate_{suffix}":      hr / pa,
        f"b_barrel_rate_{suffix}":  float(grp["is_barrel"].sum()) / pa,
        f"b_ev_mean_{suffix}":      float(ev.mean()) if ev.notna().any() else 0.0,
        f"b_la_mean_{suffix}":      float(la.mean()) if la.notna().any() else 0.0,
        f"b_hardhit_rate_{suffix}": float((ev >= 95).mean()) if ev.notna().any() else 0.0,
        f"b_fb_rate_{suffix}":      float(la.between(20, 40).mean()) if la.notna().any() else 0.0,
        f"b_k_rate_{suffix}":       float(grp["is_so"].sum()) / pa,
        f"b_bb_rate_{suffix}":      float(grp["is_bb"].sum()) / pa,
    }


def _pitcher_stats_for_window(grp: pd.DataFrame, suffix: str) -> dict:
    """Aggregate a pre-filtered window of pitcher PA rows into a flat stat dict."""
    pa = len(grp)
    if pa == 0:
        return {
            f"p_pa_{suffix}":                   0,
            f"p_hr_allowed_{suffix}":            0,
            f"p_hr_allowed_rate_{suffix}":       0.0,
            f"p_ev_allowed_mean_{suffix}":       0.0,
            f"p_hardhit_allowed_rate_{suffix}":  0.0,
            f"p_fb_allowed_rate_{suffix}":       0.0,
            f"p_barrel_allowed_rate_{suffix}":   0.0,
            f"p_k_rate_{suffix}":                0.0,
            f"p_bb_rate_{suffix}":               0.0,
        }
    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr = int(grp["is_hr"].sum())
    return {
        f"p_pa_{suffix}":                   pa,
        f"p_hr_allowed_{suffix}":           hr,
        f"p_hr_allowed_rate_{suffix}":      hr / pa,
        f"p_ev_allowed_mean_{suffix}":      float(ev.mean()) if ev.notna().any() else 0.0,
        f"p_hardhit_allowed_rate_{suffix}": float((ev >= 95).mean()) if ev.notna().any() else 0.0,
        f"p_fb_allowed_rate_{suffix}":      float(la.between(20, 40).mean()) if la.notna().any() else 0.0,
        f"p_barrel_allowed_rate_{suffix}":  float(grp["is_barrel"].sum()) / pa,
        f"p_k_rate_{suffix}":               float(grp["is_so"].sum()) / pa,
        f"p_bb_rate_{suffix}":              float(grp["is_bb"].sum()) / pa,
    }


def _precompute_batter_windows(
    events: pd.DataFrame,
    target_dates: pd.Series,
) -> pd.DataFrame:
    """
    For every (batter, game_date) pair present in target_dates,
    compute 14-day and season-to-date rolling stats from events
    (which must already include the history buffer).

    Returns a DataFrame with one row per (batter, game_date).
    """
    # Unique (batter, game_date) combos we actually need
    need = target_dates.drop_duplicates().reset_index(drop=True)  # expects cols: batter, game_date

    events_by_batter = {k: g for k, g in events.groupby("batter", sort=False)}

    rows = []
    for _, r in need.iterrows():
        batter_id = r["batter"]
        game_date  = r["game_date"]
        as_of      = game_date - pd.Timedelta(days=1)

        grp = events_by_batter.get(batter_id)

        if grp is None:
            stats_14  = _batter_stats_for_window(events.iloc[0:0], "14")
            stats_szn = _batter_stats_for_window(events.iloc[0:0], "szn")
        else:
            w14 = grp[grp["game_date"].between(
                as_of - pd.Timedelta(days=13), as_of  # 14-day inclusive
            )]
            szn_start = pd.Timestamp(as_of.year, 3, 1)
            wszn = grp[grp["game_date"].between(szn_start, as_of)]
            stats_14  = _batter_stats_for_window(w14, "14")
            stats_szn = _batter_stats_for_window(wszn, "szn")

        rows.append({"batter": batter_id, "game_date": game_date, **stats_14, **stats_szn})

    return pd.DataFrame(rows)


def _precompute_pitcher_windows(
    events: pd.DataFrame,
    target_dates: pd.Series,
) -> pd.DataFrame:
    """
    For every (pitcher, game_date) pair, compute 30-day and season-to-date
    stats allowed. Returns one row per (pitcher, game_date).
    """
    need = target_dates.drop_duplicates().reset_index(drop=True)  # cols: pitcher, game_date

    events_by_pitcher = {k: g for k, g in events.groupby("pitcher", sort=False)}

    rows = []
    for _, r in need.iterrows():
        pitcher_id = r["pitcher"]
        game_date  = r["game_date"]
        as_of      = game_date - pd.Timedelta(days=1)

        grp = events_by_pitcher.get(pitcher_id)

        if grp is None:
            stats_30  = _pitcher_stats_for_window(events.iloc[0:0], "30")
            stats_szn = _pitcher_stats_for_window(events.iloc[0:0], "szn")
        else:
            w30 = grp[grp["game_date"].between(
                as_of - pd.Timedelta(days=29), as_of  # 30-day inclusive
            )]
            szn_start = pd.Timestamp(as_of.year, 3, 1)
            wszn = grp[grp["game_date"].between(szn_start, as_of)]
            stats_30  = _pitcher_stats_for_window(w30, "30")
            stats_szn = _pitcher_stats_for_window(wszn, "szn")

        rows.append({"pitcher": pitcher_id, "game_date": game_date, **stats_30, **stats_szn})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interaction / edge features (fully vectorised on the merged DataFrame)
# ---------------------------------------------------------------------------

def _add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ev_edge_14_30"]             = df["b_ev_mean_14"]      - df["p_ev_allowed_mean_30"]
    df["hardhit_edge_14_30"]        = df["b_hardhit_rate_14"] - df["p_hardhit_allowed_rate_30"]
    df["fb_edge_14_30"]             = df["b_fb_rate_14"]      - df["p_fb_allowed_rate_30"]
    df["barrel_edge_14_30"]         = df["b_barrel_rate_14"]  - df["p_barrel_allowed_rate_30"]
    df["hr_rate_edge_14_30"]        = df["b_hr_rate_14"]      - df["p_hr_allowed_rate_30"]
    df["k_rate_edge_14_30"]         = df["b_k_rate_14"]       - df["p_k_rate_30"]
    df["bb_rate_edge_14_30"]        = df["b_bb_rate_14"]      - df["p_bb_rate_30"]
    df["k_rate_interaction_14_30"]  = df["b_k_rate_14"]       * df["p_k_rate_30"]
    df["bb_rate_interaction_14_30"] = df["b_bb_rate_14"]      * df["p_bb_rate_30"]
    df["contact_pressure_14_30"]    = (1 - df["b_k_rate_14"]) * (1 - df["p_k_rate_30"])
    df["discipline_balance_14_30"]  = (
        (df["b_bb_rate_14"] - df["b_k_rate_14"]) -
        (df["p_bb_rate_30"] - df["p_k_rate_30"])
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features_for_range(start_date: str, end_date: str) -> FeaturesBuildResult:
    start_dt = _to_date(start_date)
    end_dt   = _to_date(end_date)

    # Pull enough history for rolling windows (60 days back from start)
    history_start = _date_minus_days(start_dt, 60)

    print(f"  Loading events {history_start.date()} → {end_dt.date()} ...")
    events = _load_and_clean_events(
        start_date=history_start.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )

    # Park factor lookup: game_pk → home_team
    game_pk_to_home = (
        events.dropna(subset=["home_team"])
        .drop_duplicates(subset=["game_pk"])
        .set_index("game_pk")["home_team"]
        .to_dict()
    )

    # Build labels (one row per batter-game in the target window)
    target_events = events[
        events["game_date"].between(start_dt, end_dt)
    ]
    labels = build_batter_game_labels(target_events)
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    # Resolve pitcher_mode safely
    labels["pitcher_id"] = (
        pd.to_numeric(labels["pitcher_mode"], errors="coerce")
        .astype("Int64")
    )

    print(f"  Precomputing batter windows for {labels[['batter','game_date']].drop_duplicates().shape[0]:,} (batter, date) pairs ...")
    batter_stats = _precompute_batter_windows(
        events,
        labels[["batter", "game_date"]],
    )

    print(f"  Precomputing pitcher windows for {labels[['pitcher_id','game_date']].drop_duplicates().shape[0]:,} (pitcher, date) pairs ...")
    pitcher_need = (
        labels[["pitcher_id", "game_date"]]
        .rename(columns={"pitcher_id": "pitcher"})
        .dropna(subset=["pitcher"])
    )
    pitcher_need["pitcher"] = pitcher_need["pitcher"].astype(int)
    pitcher_stats = _precompute_pitcher_windows(events, pitcher_need)

    # -----------------------------------------------------------------------
    # Merge everything — no Python loop over rows
    # -----------------------------------------------------------------------
    features_df = (
        labels
        .merge(batter_stats, on=["batter", "game_date"], how="left")
        .merge(
            pitcher_stats.rename(columns={"pitcher": "pitcher_id"}),
            on=["pitcher_id", "game_date"],
            how="left",
        )
    )

    # Park factor
    features_df["home_team"]    = features_df["game_pk"].map(game_pk_to_home)
    features_df["park_factor_hr"] = (
        features_df["home_team"]
        .map(HR_PARK_FACTOR)
        .fillna(DEFAULT_PARK_FACTOR)
        / 100.0
    )

    # Edge / interaction features (vectorised)
    features_df = _add_edge_features(features_df)

    # Tidy up columns
    features_df = features_df.rename(columns={"pitcher_id": "pitcher"})
    features_df["game_date"] = features_df["game_date"].dt.date
    features_df["hr_hit"]    = features_df["hr_hit"].astype(int)

    # Fill any remaining NaNs (players with zero history)
    stat_cols = [c for c in features_df.columns if c not in
                 ("game_date", "game_pk", "batter", "pitcher", "home_team", "hr_hit")]
    features_df[stat_cols] = features_df[stat_cols].fillna(0.0)

    out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
    features_df.to_parquet(out_path, index=False)

    return FeaturesBuildResult(features_df=features_df, output_path=out_path)


if __name__ == "__main__":
    result = build_features_for_range("2024-03-20", "2024-10-01")
    full_path = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    result.features_df.to_parquet(full_path, index=False)

    print(f"\nSaved features to: {full_path}")
    print(f"Rows: {len(result.features_df):,}")
    print(result.features_df.head(5))
    print("\nLabel HR rate:", result.features_df["hr_hit"].mean())

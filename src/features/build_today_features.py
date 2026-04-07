"""
build_today_features.py
========================
Builds today's prediction feature rows from live MLB API data and carries
forward rolling Statcast statistics from the historical train table.

This module is the bridge between the historical training pipeline and the
live daily prediction pipeline. It does NOT re-compute rolling statistics
from raw Statcast data (that would take too long). Instead, it:

  1. Fetches today's game schedule and probable/confirmed starters.
  2. Fetches batting orders for each game.
  3. Builds one row per (game, batter, opposing_starter) matchup.
  4. Carries forward each batter's and pitcher's most recent rolling
     statistics from the historical train table (last known game).
  5. Resolves handedness and same_hand_matchup.
  6. Fetches weather for each ballpark.
  7. Applies park factors.
  8. Recomputes edge/interaction features.
  9. Fills any remaining NaN stat columns with league-average priors.
 10. Writes the result to data/processed/train_table_{date}_today.parquet
     AND appends/overwrites today's rows in the combined train table so
     predict.py finds them automatically.

Carrying them forward to today is correct because
no new games have been played today yet.

Edge cases handled
------------------
- Batter not in train table (debut / new player) → row filled with league
  averages (same as cold-start shrinkage in train.py).
- Pitcher not in train table → pitcher features filled with league averages.
- Game not yet started → weather uses neutral values (72°F, 0 mph wind);
  will be refreshed on the final pass once the game feed is live.
- Indoor parks → wind/temp forced to neutral regardless.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.logging_config import configure_logging
from src.data_sources.mlb_schedule import (
    fetch_probable_starters_for_date,
    fetch_rosters_for_games,
    _EXPECTED_PA,
    _EXPECTED_PA_DEFAULT,
)
from src.data_sources.weather import INDOOR_PARKS
from src.features.park_factors import get_park_factors, DEFAULT_PARK_FACTOR
from src.features.build_features import fetch_weather_for_games_fast, _compute_park_direction_factor

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")

# League-average fill values used when a player has no history in the
# train table.  Mirrors the priors in train.py / build_features_common.py.
_LEAGUE_FILL = {
    "b_hr_rate_14":               0.033,
    "b_barrel_rate_14":           0.073,
    "b_hardhit_rate_14":          0.380,
    "b_fb_rate_14":               0.350,
    "b_k_rate_14":                0.227,
    "b_bb_rate_14":               0.083,
    "b_ev_mean_14":               88.5,
    "b_la_mean_14":               12.0,
    "b_hr_rate_szn":              0.033,
    "b_barrel_rate_szn":          0.073,
    "b_hardhit_rate_szn":         0.380,
    "b_fb_rate_szn":              0.350,
    "b_k_rate_szn":               0.227,
    "b_bb_rate_szn":              0.083,
    "b_ev_mean_szn":              88.5,
    "b_la_mean_szn":              12.0,
    "p_hr_allowed_rate_30":       0.033,
    "p_barrel_allowed_rate_30":   0.073,
    "p_hardhit_allowed_rate_30":  0.380,
    "p_fb_allowed_rate_30":       0.350,
    "p_k_rate_30":                0.227,
    "p_bb_rate_30":               0.083,
    "p_ev_allowed_mean_30":       88.5,
    "p_hr_allowed_rate_szn":      0.033,
    "p_barrel_allowed_rate_szn":  0.073,
    "p_hardhit_allowed_rate_szn": 0.380,
    "p_fb_allowed_rate_szn":      0.350,
    "p_fb_velo_30":               93.5,
    "p_fb_pct_30":                0.55,
    "p_offspeed_pct_30":          0.35,
    "park_factor_hr":             1.0,
    "temp_f":                     72.0,
    "wind_hr_impact":             0.0,
    "wind_speed_mph":             0.0,
    "wind_hr_factor":             0.0,
    "b_days_rest":                4.0,
    "p_days_rest":                4.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now(tz=ET).strftime("%Y-%m-%d")


def _load_latest_train_table() -> pd.DataFrame:
    """Load the best available train table, filtered to last 60 days only.
    We only need recent rows for carry-forward (latest row per player).
    """
    candidates = [
        PROCESSED_DIR / "train_table_2021_2026_full.parquet",
        PROCESSED_DIR / "train_table_2021_2025_full.parquet",
        PROCESSED_DIR / "train_table_2024_full_season.parquet",
    ]
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=60)
    for c in candidates:
        if c.exists() and c.stat().st_size > 10_000:
            logger.info("Loading train table: %s (last 60 days only)", c.name)
            df = pd.read_parquet(c, filters=[("game_date", ">=", cutoff)])
            df["game_date"] = pd.to_datetime(df["game_date"])
            if not df.empty:
                return df
            # 60-day window is empty (season start / offseason gap) —
            # fall back to the last 60 days of actual data in the file.
            logger.warning(
                "60-day window empty (season start?) — loading last 60 days of available data"
            )
            df = pd.read_parquet(c)
            df["game_date"] = pd.to_datetime(df["game_date"])
            max_date = df["game_date"].max()
            df = df[df["game_date"] >= max_date - pd.Timedelta(days=60)]
            return df

    # Glob fallback
    files = sorted(
        [p for p in PROCESSED_DIR.glob("train_table_*.parquet") if p.stat().st_size > 10_000],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            "No train_table_*.parquet found in data/processed/.\n"
            "Run build_features_multi_season.py first."
        )
    logger.info("Loading train table (fallback): %s (last 60 days only)", files[0].name)
    df = pd.read_parquet(files[0], filters=[("game_date", ">=", cutoff)])
    df["game_date"] = pd.to_datetime(df["game_date"])
    if not df.empty:
        return df
    # Same season-start fallback for glob path
    df = pd.read_parquet(files[0])
    df["game_date"] = pd.to_datetime(df["game_date"])
    max_date = df["game_date"].max()
    df = df[df["game_date"] >= max_date - pd.Timedelta(days=60)]
    return df


def _latest_batter_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each batter, return their most recent feature row.
    This represents rolling windows calculated through their last game.
    """
    train_df = train_df.copy()
    train_df["game_date"] = pd.to_datetime(train_df["game_date"])
    return (
        train_df
        .sort_values("game_date")
        .groupby("batter", sort=False)
        .last()
        .reset_index()
    )


def _latest_pitcher_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pitcher, return their most recent feature row (as pitcher).
    """
    train_df = train_df.copy()
    train_df["game_date"] = pd.to_datetime(train_df["game_date"])
    pitcher_cols = ["pitcher"] + [c for c in train_df.columns if c.startswith("p_")]
    avail = [c for c in pitcher_cols if c in train_df.columns]
    return (
        train_df[avail]
        .sort_values("game_date") if "game_date" in avail
        else train_df[avail]
    )


def _recompute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute edge/interaction features after merging batter+pitcher."""
    df = df.copy()

    def _e(a, b):
        if a in df.columns and b in df.columns:
            return df[a] - df[b]
        return pd.Series(np.nan, index=df.index)

    df["ev_edge_14_30"]             = _e("b_ev_mean_14",       "p_ev_allowed_mean_30")
    df["hardhit_edge_14_30"]        = _e("b_hardhit_rate_14",  "p_hardhit_allowed_rate_30")
    df["fb_edge_14_30"]             = _e("b_fb_rate_14",       "p_fb_allowed_rate_30")
    df["barrel_edge_14_30"]         = _e("b_barrel_rate_14",   "p_barrel_allowed_rate_30")
    df["hr_rate_edge_14_30"]        = _e("b_hr_rate_14",       "p_hr_allowed_rate_30")
    df["k_rate_edge_14_30"]         = _e("b_k_rate_14",        "p_k_rate_30")
    df["bb_rate_edge_14_30"]        = _e("b_bb_rate_14",       "p_bb_rate_30")

    if "b_k_rate_14" in df.columns and "p_k_rate_30" in df.columns:
        df["k_rate_interaction_14_30"]  = df["b_k_rate_14"]  * df["p_k_rate_30"]
        df["bb_rate_interaction_14_30"] = df["b_bb_rate_14"] * df["p_bb_rate_30"]
        df["contact_pressure_14_30"]    = (1 - df["b_k_rate_14"]) * (1 - df["p_k_rate_30"])
        df["discipline_balance_14_30"]  = (
            (df["b_bb_rate_14"] - df["b_k_rate_14"]) -
            (df["p_bb_rate_30"] - df["p_k_rate_30"])
        )

    for hand in ("L", "R"):
        b_hr   = f"b_hr_rate_14_vs{hand}"
        p_hr   = f"p_hr_allowed_rate_30_vs{hand}"
        b_hard = f"b_hardhit_rate_14_vs{hand}"
        p_hard = f"p_hardhit_allowed_rate_30_vs{hand}"
        b_bar  = f"b_barrel_rate_14_vs{hand}"
        p_bar  = f"p_barrel_allowed_rate_30_vs{hand}"
        if all(c in df.columns for c in [b_hr, p_hr, b_hard, p_hard, b_bar, p_bar]):
            df[f"hr_rate_edge_14_30_vs{hand}"]  = _e(b_hr,   p_hr)
            df[f"hardhit_edge_14_30_vs{hand}"]  = _e(b_hard, p_hard)
            df[f"barrel_edge_14_30_vs{hand}"]   = _e(b_bar,  p_bar)

    if "wind_hr_impact" in df.columns:
        b_hh = df.get("b_hardhit_rate_14", pd.Series(np.nan, index=df.index))
        b_br = df.get("b_barrel_rate_14",  pd.Series(np.nan, index=df.index))
        df["hardhit_x_wind"] = b_hh * df["wind_hr_impact"]
        df["barrel_x_wind"]  = b_br * df["wind_hr_impact"]

    def _col(c):
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    def _edge(a, b):
        return _col(a) - _col(b)

    # Opener interactions
    if "p_is_opener" in df.columns:
        df["opener_x_hardhit"] = _col("p_is_opener") * _col("b_hardhit_rate_14")
        df["opener_x_barrel"]  = _col("p_is_opener") * _col("b_barrel_rate_14")

    # ISO interactions
    if "b_iso_szn" in df.columns and "park_factor_hr" in df.columns:
        iso = _col("b_iso_szn").fillna(_col("b_iso_14")).fillna(_col("b_iso_career"))
        df["iso_x_park"] = iso * _col("park_factor_hr")
        df["iso_x_wind"] = iso * _col("wind_hr_impact")
        df["platoon_edge_x_iso"] = _col("b_platoon_hr_edge") * iso

    # Sweet spot interactions
    if "b_sweet_spot_rate_szn" in df.columns:
        ss = _col("b_sweet_spot_rate_szn").fillna(_col("b_sweet_spot_rate_14"))
        if "park_factor_hr" in df.columns:
            df["sweet_spot_x_park"] = ss * _col("park_factor_hr")
        if "wind_hr_impact" in df.columns:
            df["sweet_spot_x_wind"] = ss * _col("wind_hr_impact")
        if "p_command_30" in df.columns:
            df["sweet_spot_x_poor_command"] = ss * (1.0 - _col("p_command_30").clip(lower=0, upper=1))
        if "p_weak_stuff_rate" in df.columns:
            df["sweet_spot_x_weak_stuff"] = ss * _col("p_weak_stuff_rate")

    # 30-day edges
    df["hr_rate_edge_30_30"]  = _edge("b_hr_rate_30",      "p_hr_allowed_rate_30")
    df["hardhit_edge_30_30"]  = _edge("b_hardhit_rate_30", "p_hardhit_allowed_rate_30")
    df["barrel_edge_30_30"]   = _edge("b_barrel_rate_30",  "p_barrel_allowed_rate_30")

    # Barrel / HR interactions with poor command
    if "p_command_30" in df.columns:
        poor_cmd = 1.0 - _col("p_command_30").clip(lower=0, upper=1)
        df["barrel_x_poor_command"] = _col("b_barrel_rate_14") * poor_cmd
        df["hr30_x_poor_command"]   = _col("b_hr_rate_30") * poor_cmd

    # Weak stuff interactions
    if "p_weak_stuff_rate" in df.columns:
        df["barrel_x_weak_stuff"] = _col("b_barrel_rate_14") * _col("p_weak_stuff_rate")

    # Hot streak interactions
    if "b_ev_hot_flag" in df.columns:
        df["hot_x_park"] = _col("b_ev_hot_flag") * _col("park_factor_hr")
        df["hot_x_iso"]  = _col("b_ev_hot_flag") * _col("b_iso_szn").fillna(_col("b_iso_career"))

    # Recent HR × park
    if "b_hr_last_7d" in df.columns and "park_factor_hr" in df.columns:
        df["hr7d_x_park"] = _col("b_hr_last_7d") * _col("park_factor_hr")

    # Team HR rate interactions
    if "t_hr_rate_14" in df.columns:
        df["team_hr_x_batter_barrel"] = _col("t_hr_rate_14") * _col("b_barrel_rate_14")
        df["team_hr_x_sweet_spot"]    = _col("t_hr_rate_14") * _col("b_sweet_spot_rate_szn").fillna(
            _col("b_sweet_spot_rate_14")
        )

    # Park pull factor
    if "b_pull_air_rate_szn" in df.columns and "park_pull_factor" in df.columns:
        pull = _col("b_pull_air_rate_szn").fillna(_col("b_pull_air_rate_14"))
        df["pull_x_park_direction"] = pull * _col("park_pull_factor")

    return df


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_today_features(
    target_date: str | None = None,
    *,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Build prediction feature rows for target_date and write them to a parquet
    that predict.py will find via latest_train_table().
    Returns the path to the written file.
    """
    if target_date is None:
        target_date = _today_str()

    target_ts = pd.Timestamp(target_date)
    logger.info("Building today's features for: %s", target_date)

    # ------------------------------------------------------------------
    # 1. Fetch today's schedule
    # ------------------------------------------------------------------
    schedule_df = fetch_probable_starters_for_date(
        target_date, force_refresh=force_refresh
    )
    if schedule_df.empty:
        logger.warning("No games found for %s — MLB API may not have posted today's slate yet.", target_date)
        return _write_empty(target_date, dry_run)

    game_pks = schedule_df["game_pk"].astype(int).tolist()
    logger.info("Found %d games for %s", len(game_pks), target_date)

    # ------------------------------------------------------------------
    # 2. Fetch rosters / batting orders
    # ------------------------------------------------------------------
    starters_df, batting_df = fetch_rosters_for_games(
        game_pks, force_refresh=force_refresh
    )

    # Build game_pk → home_team lookup
    game_pk_to_home: dict[int, str] = dict(
        zip(schedule_df["game_pk"].astype(int), schedule_df["home_team"])
    )

    # ------------------------------------------------------------------
    # 3. Build matchup rows: one per (game_pk, batter, pitcher)
    # ------------------------------------------------------------------
    rows = []

    for _, game in schedule_df.iterrows():
        gp      = int(game["game_pk"])
        home    = str(game.get("home_team", ""))
        away    = str(game.get("away_team", ""))
        h_start = game.get("home_starter_id")
        a_start = game.get("away_starter_id")

        # Override with actual starters if we have them
        if not starters_df.empty:
            s = starters_df[starters_df["game_pk"] == gp]
            if not s.empty:
                h_start = s.iloc[0].get("home_starter_id", h_start)
                a_start = s.iloc[0].get("away_starter_id", a_start)

        # Capture probable pitcher names from the schedule (used as display
        # fallback when the MLBAM ID lookup fails in predict.py).
        h_start_name = str(game.get("home_starter_name", "")).strip() or None
        a_start_name = str(game.get("away_starter_name", "")).strip() or None
        # Normalise "nan" strings that pandas writes for missing values
        if h_start_name and h_start_name.lower() in ("nan", "none"):
            h_start_name = None
        if a_start_name and a_start_name.lower() in ("nan", "none"):
            a_start_name = None

        # Get batters for this game
        game_batters = batting_df[batting_df["game_pk"] == gp] if not batting_df.empty else pd.DataFrame()

        if game_batters.empty:
            logger.debug("No batting order data for game_pk=%d — skipping", gp)
            continue

        for _, batter_row in game_batters.iterrows():
            batter_id  = int(batter_row["batter"])
            team_side  = str(batter_row.get("team_side", ""))
            bat_order  = int(batter_row.get("batting_order", 0)) or 0
            is_starter = int(batter_row.get("is_starter_batter", 1))

            # Skip non-starters (substitutes) to keep predictions clean
            if not is_starter:
                continue

            # Resolve pitcher: batters face the opposing team's starter
            if team_side == "home":
                pitcher_id   = a_start       # home batters face away starter
                pitcher_name = a_start_name
                is_home      = 1
            else:
                pitcher_id   = h_start       # away batters face home starter
                pitcher_name = h_start_name
                is_home      = 0

            if pitcher_id is None or pd.isna(pitcher_id):
                pitcher_id = None  # will be filled with league avg

            rows.append({
                "game_date":              target_ts,
                "game_pk":                gp,
                "batter":                 batter_id,
                "pitcher":                int(pitcher_id) if pitcher_id is not None else None,
                "probable_pitcher_name":  pitcher_name,
                "home_team":              home,
                "batter_team":            home if is_home else away,
                "is_home_game":           is_home,
                "batting_order_pos":      bat_order,
                "is_top_of_order":        int(bat_order in range(1, 5)),
                "expected_pa_today":      _EXPECTED_PA.get(bat_order, _EXPECTED_PA_DEFAULT),
                "hr_hit":                 0,    # unknown — game not played yet
                "relief_pa_pct":          0.0,  # pre-game: no relief PAs yet
                "p_is_short_rest":        0,
                "b_days_rest":            _LEAGUE_FILL["b_days_rest"],
                "p_days_rest":            _LEAGUE_FILL["p_days_rest"],
            })

    if not rows:
        logger.warning("No batter rows built for %s — batting orders not yet posted?", target_date)
        return _write_empty(target_date, dry_run)

    today_df = pd.DataFrame(rows)
    today_df["game_date"] = pd.to_datetime(today_df["game_date"])
    logger.info("Built %d batter matchup rows for %s", len(today_df), target_date)

    # ------------------------------------------------------------------
    # 4. Carry-forward rolling features from train table
    # ------------------------------------------------------------------
    logger.info("Loading train table for feature carry-forward ...")
    train_df = _load_latest_train_table()

    # Latest batter features
    batter_latest    = _latest_batter_features(train_df)
    batter_feat_cols = [c for c in batter_latest.columns
                        if c.startswith("b_") and c not in {"batter"}]
    batter_feat_cols += ["same_hand_matchup", "pitcher_hand", "batter_hand"]
    batter_feat_cols  = [c for c in batter_feat_cols if c in batter_latest.columns]
    # KEY FIX: drop any cols already present in today_df — prevents _x/_y
    # duplicate column crash (e.g. b_days_rest already set in step 3).
    batter_feat_cols  = [c for c in batter_feat_cols if c not in today_df.columns]

    today_df = today_df.merge(
        batter_latest[["batter"] + batter_feat_cols],
        on="batter",
        how="left",
    )

    # Latest pitcher features (keyed on pitcher)
    pitcher_feat_cols = [c for c in train_df.columns
                         if c.startswith("p_") and c not in {"pitcher"}]
    pitcher_feat_cols = [c for c in pitcher_feat_cols if c in train_df.columns]

    if pitcher_feat_cols:
        pitcher_src = train_df[["pitcher", "game_date"] + pitcher_feat_cols].copy() \
            if "game_date" in train_df.columns \
            else train_df[["pitcher"] + pitcher_feat_cols].copy()
        if "game_date" in pitcher_src.columns:
            pitcher_src = pitcher_src.sort_values("game_date")
        pitcher_latest = (
            pitcher_src
            .groupby("pitcher", sort=False)
            .last()
            .reset_index()
        )
        # Drop game_date and any cols already in today_df — prevents duplicate crash
        pitcher_merge_cols = ["pitcher"] + [
            c for c in pitcher_feat_cols
            if c in pitcher_latest.columns and c not in today_df.columns
        ]
        today_df = today_df.merge(
            pitcher_latest[pitcher_merge_cols],
            on="pitcher",
            how="left",
        )
    # Latest team features (keyed on batter_team)
    team_feat_cols = [c for c in train_df.columns if c.startswith("t_")]
    team_feat_cols = [c for c in team_feat_cols if c not in today_df.columns]
    if team_feat_cols and "batter_team" in train_df.columns and "batter_team" in today_df.columns:
        team_src = train_df[["batter_team", "game_date"] + team_feat_cols].copy() \
            if "game_date" in train_df.columns \
            else train_df[["batter_team"] + team_feat_cols].copy()
        if "game_date" in team_src.columns:
            team_src = team_src.sort_values("game_date")
        team_latest = (
            team_src
            .groupby("batter_team", sort=False)
            .last()
            .reset_index()
        )
        team_merge_cols = ["batter_team"] + [
            c for c in team_feat_cols if c in team_latest.columns
        ]
        today_df = today_df.merge(
            team_latest[team_merge_cols],
            on="batter_team",
            how="left",
        )

    # Fill missing values with league averages
    for col, fill_val in _LEAGUE_FILL.items():
        if col in today_df.columns:
            today_df[col] = today_df[col].fillna(fill_val)
        else:
            today_df[col] = fill_val

    # ------------------------------------------------------------------
    # 5. Resolve same_hand_matchup if missing
    # ------------------------------------------------------------------
    if "batter_hand" not in today_df.columns or today_df["batter_hand"].isna().all():
        bhand = (
            train_df.dropna(subset=["batter", "batter_hand"])
            .groupby("batter")["batter_hand"]
            .last()
        )
        today_df["batter_hand"] = today_df["batter"].map(bhand)

    if "pitcher_hand" not in today_df.columns or today_df["pitcher_hand"].isna().all():
        phand = (
            train_df.dropna(subset=["pitcher", "pitcher_hand"])
            .groupby("pitcher")["pitcher_hand"]
            .last()
        )
        today_df["pitcher_hand"] = today_df["pitcher"].map(phand)

    if "same_hand_matchup" not in today_df.columns or today_df["same_hand_matchup"].isna().all():
        today_df["same_hand_matchup"] = np.where(
            today_df["batter_hand"].notna() & today_df["pitcher_hand"].notna(),
            (today_df["batter_hand"] == today_df["pitcher_hand"]).astype(int),
            -1,
        )

    # ------------------------------------------------------------------
    # 6. Weather
    # ------------------------------------------------------------------
    logger.info("Fetching weather for %d games ...", len(game_pks))
    game_dates_map = {gp: target_date for gp in game_pks}
    weather_df = fetch_weather_for_games_fast(
        game_pks,
        game_pk_to_home,
        game_dates=game_dates_map,
        force_refresh=force_refresh,
    )
    if not weather_df.empty:
        weather_df["game_pk"] = weather_df["game_pk"].astype(int)
        today_df["game_pk"]   = today_df["game_pk"].astype(int)
        weather_cols = ["game_pk", "temp_f", "wind_speed_mph", "wind_hr_factor",
                        "wind_hr_impact", "is_indoor", "temp_above_75", "temp_above_85",
                        "wind_out_strong", "wind_in_strong"]
        weather_cols = [c for c in weather_cols if c in weather_df.columns]
        # Drop stale weather columns before merge
        for wc in weather_cols:
            if wc != "game_pk" and wc in today_df.columns:
                today_df = today_df.drop(columns=[wc])
        today_df = today_df.merge(weather_df[weather_cols], on="game_pk", how="left")

    # ------------------------------------------------------------------
    # 7. Park factors
    # ------------------------------------------------------------------
    season_year = target_ts.year
    pf_map = get_park_factors(season=season_year)
    today_df["park_factor_hr"] = (
        today_df["home_team"].map(pf_map).fillna(DEFAULT_PARK_FACTOR / 100.0)
    )

    # ------------------------------------------------------------------
    # 8. Recompute edge features
    # ------------------------------------------------------------------
    today_df = _compute_park_direction_factor(today_df)
    today_df = _recompute_edges(today_df)

    # ------------------------------------------------------------------
    # 9. Fill remaining NaNs in stat columns with 0
    # ------------------------------------------------------------------
    non_stat = {
        "game_date", "game_pk", "batter", "pitcher", "home_team",
        "batter_team", "hr_hit", "pitcher_hand", "batter_hand",
        "pitcher_mode", "probable_pitcher_name",
    }
    stat_cols = [c for c in today_df.columns if c not in non_stat]
    today_df[stat_cols] = today_df[stat_cols].fillna(0.0)

    today_df["game_date"] = pd.to_datetime(today_df["game_date"]).dt.date

    logger.info(
        "Today's feature table: %d rows | %d columns | date=%s",
        len(today_df), len(today_df.columns), target_date,
    )

    if dry_run:
        logger.info("[DRY RUN] Would write today's features — skipping file write.")
        logger.info("Top 5 matchups by batting order:\n%s",
                    today_df[["batter", "pitcher", "batting_order_pos", "is_home_game"]]
                    .sort_values("batting_order_pos").head(5).to_string(index=False))
        return PROCESSED_DIR / f"train_table_{target_date}_today.parquet"

    # ------------------------------------------------------------------
    # 10. Write output
    # ------------------------------------------------------------------
    out_path = PROCESSED_DIR / f"train_table_{target_date}_today.parquet"
    today_df.to_parquet(out_path, index=False)
    logger.info("Written: %s", out_path.name)

    # NOTE: _append_to_combined is skipped — predict.py already prefers
    # train_table_YYYY-MM-DD_today.parquet over the combined table, so
    # appending is unnecessary and too memory-intensive for a 2GB VPS.
    # _append_to_combined(today_df, target_date)

    return out_path


def _write_empty(target_date: str, dry_run: bool) -> Path:
    """
    Called when there are no batter rows to write (batting orders not yet posted).
    Prints a clear message and exits with code 0 so the scheduler logs a clean
    skip rather than a failure, and predict.py is never invoked on empty data.
    """
    print(
        f"\n  ℹ️  No batting orders available for {target_date} yet.\n"
        "     This is normal for early-morning passes before MLB posts lineups.\n"
        "     The final pass (90 min before first pitch) will retry automatically."
    )
    import sys
    sys.exit(0)


def _append_to_combined(today_df: pd.DataFrame, target_date: str) -> None:
    """
    Load the best combined train table, remove any existing rows for
    target_date, append today's rows, and overwrite the file.

    This ensures predict.py's latest_train_table() finds today's rows
    without any changes to predict.py itself.
    """
    candidates = [
        PROCESSED_DIR / "train_table_2021_2026_full.parquet",
        PROCESSED_DIR / "train_table_2021_2025_full.parquet",
    ]
    combined_path = None
    for c in candidates:
        if c.exists() and c.stat().st_size > 10_000:
            combined_path = c
            break

    if combined_path is None:
        logger.warning(
            "No combined train table found — today's rows written only to "
            "train_table_%s_today.parquet. Update latest_train_table() in "
            "predict.py to also check this file.", target_date
        )
        return

    logger.info("Appending today's rows to combined table: %s", combined_path.name)
    existing = pd.read_parquet(combined_path)
    existing["game_date"] = pd.to_datetime(existing["game_date"])

    # Remove stale today rows if we're refreshing
    existing = existing[
        existing["game_date"] != pd.Timestamp(target_date)
    ]

    today_copy = today_df.copy()
    today_copy["game_date"] = pd.to_datetime(today_copy["game_date"])

    # Align columns — add missing cols as 0 in either direction
    for col in existing.columns:
        if col not in today_copy.columns:
            today_copy[col] = (
                0.0 if col not in
                {"pitcher_hand", "batter_hand", "pitcher_mode", "batter_team",
                 "home_team", "probable_pitcher_name"}
                else None
            )
    for col in today_copy.columns:
        if col not in existing.columns:
            existing[col] = 0.0

    combined = pd.concat(
        [existing, today_copy[existing.columns]],
        ignore_index=True,
    )
    combined = combined.sort_values("game_date").reset_index(drop=True)

    # Deduplicate — keep today's rows over any stale historical rows
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    after = len(combined)
    if before != after:
        logger.info("Deduplication removed %d rows", before - after)

    combined["game_date"] = pd.to_datetime(combined["game_date"]).dt.date
    combined.to_parquet(combined_path, index=False)
    logger.info(
        "Combined table updated: %s | total rows=%d | latest date=%s",
        combined_path.name,
        len(combined),
        target_date,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Build today's MLB HR prediction features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", default=None,
        help="Target date YYYY-MM-DD (default: today ET).",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Bypass schedule/roster caches.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be built without writing files.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    out = build_today_features(
        target_date=args.date,
        force_refresh=args.force_refresh,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"\nToday's features written to: {out.relative_to(PROJECT_ROOT)}")
        print(
            "\nNow run:\n"
            "    python -m src.model.predict\n"
            "    -- or --\n"
            f"    python -m src.model.predict --date {args.date or _today_str()}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Vectorised replacements for the three slow per-row loop functions in
build_features.py.

Import chain (no cycles):
    build_features_common  <-  build_features_fast  <-  build_features
"""

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from src.features.build_features_common import (
    MIN_PA_BATTER_SZN,
    MIN_PA_PITCHER_SZN,
    MIN_PA_WINDOW,
    FASTBALL_TYPES,
    OFFSPEED_TYPES,
    _safe_mean,
    _batter_trend_stats,
)

_PITCHER_VELO_COLS = [
    "p_fb_velo_30", "p_fb_pct_30", "p_offspeed_pct_30", "p_fb_velo_trend"
]

# ---------------------------------------------------------------------------
# Core merge helper
# ---------------------------------------------------------------------------

def _window_merge(
    pa_df: pd.DataFrame,
    need: pd.DataFrame,
    player_col: str,
    days_back: int,
    season_window: bool = False,
) -> pd.DataFrame:
    """
    Return pa_df rows inside the look-back window for each
    (player, target_date) pair in *need*.

    Window: [lower_bound, target_date)   ← strictly less-than on BOTH sides.

    Leakage notes
    -------------
    - Right bound is STRICT (<) so same-day PA never leaks into features.
    - FIX 3: Season window lower bound is also STRICT (>) so a game played
      exactly on March 1 (season start) is not included in its own features.
      Previously >= was used which could include same-day data for the very
      first day of the season.
    """
    merged = pa_df.merge(
        need[[player_col, "game_date"]].rename(columns={"game_date": "target_date"}),
        on=player_col,
        how="inner",
    )

    # Right bound: strictly before target date (no same-day leakage)
    in_window = merged["game_date"] < merged["target_date"]

    if season_window:
        # Season start = March 1 of the target year.
        # FIX 3: use strict > so March 1 games are NOT included in March 1
        # features.  The original code used >= which let same-day data leak
        # on the opening day of the season.
        szn_start = pd.to_datetime(
            merged["target_date"].dt.year.astype(str) + "-03-01"
        )
        in_window &= merged["game_date"] > szn_start
    else:
        lower = merged["target_date"] - pd.Timedelta(days=days_back)
        in_window &= merged["game_date"] >= lower

    return merged.loc[in_window].copy()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

# Columns produced by _agg_pa_stats that are only meaningful for the
# *overall* window, not platoon splits (they'd duplicate on merge).
_PLATOON_DROP_COLS = {"ev_mean", "la_mean"}


def _agg_pa_stats(
    windowed: pd.DataFrame,
    group_cols: list[str],
    min_pa: int,
    prefix: str,
    drop_means: bool = False,   # set True for platoon splits
) -> pd.DataFrame:
    """
    Aggregate PA-level stats.  Returns one row per group.
    Rates are NaN when PA < min_pa (cold-start guard).

    drop_means=True omits ev_mean and la_mean so they don't collide
    when multiple platoon DataFrames are merged onto the same base.
    """
    windowed = windowed.copy()
    ev = windowed["launch_speed"]
    la = windowed["launch_angle"]
    windowed["_ev_gte95"] = (ev >= 95).astype(float)
    windowed["_fb"]       = la.between(20, 40).astype(float)

    # Determine PA column: use "pitcher" size for pitcher-prefix aggs, else row count
    pa_src = "pitcher" if ("pitcher" in windowed.columns and prefix.startswith("p")) else "is_hr"

    agg = windowed.groupby(group_cols, sort=False).agg(
        _pa         =(pa_src,        "size"),
        _hr         =("is_hr",       "sum"),
        _ev_sum     =("launch_speed","sum"),
        _ev_count   =("launch_speed","count"),
        _la_sum     =("launch_angle","sum"),
        _la_count   =("launch_angle","count"),
        _ev_gte95   =("_ev_gte95",   "sum"),
        _fb         =("_fb",         "sum"),
        _barrel     =("is_barrel",   "sum"),
        _so         =("is_so",       "sum"),
        _bb         =("is_bb",       "sum"),
    ).reset_index()

    agg["_pa"] = agg["_pa"].astype(int)
    mask_ok = agg["_pa"] >= min_pa

    pa   = agg["_pa"]
    hr   = agg["_hr"]
    ev_c = agg["_ev_count"]
    la_c = agg["_la_count"]
    p    = prefix

    out = agg[group_cols].copy()
    out[f"{p}pa"]           = pa
    out[f"{p}hr"]           = hr
    out[f"{p}hr_rate"]      = np.where(mask_ok, hr / pa.replace(0, np.nan), np.nan)
    out[f"{p}barrel_rate"]  = np.where(mask_ok, agg["_barrel"] / pa.replace(0, np.nan), np.nan)
    out[f"{p}hardhit_rate"] = np.where(mask_ok & (ev_c > 0),
                                        agg["_ev_gte95"] / ev_c.replace(0, np.nan), np.nan)
    out[f"{p}fb_rate"]      = np.where(mask_ok & (la_c > 0),
                                        agg["_fb"] / la_c.replace(0, np.nan), np.nan)
    out[f"{p}k_rate"]       = np.where(mask_ok, agg["_so"] / pa.replace(0, np.nan), np.nan)
    out[f"{p}bb_rate"]      = np.where(mask_ok, agg["_bb"] / pa.replace(0, np.nan), np.nan)

    if not drop_means:
        out[f"{p}ev_mean"] = np.where(mask_ok & (ev_c > 0),
                                       agg["_ev_sum"] / ev_c.replace(0, np.nan), np.nan)
        out[f"{p}la_mean"] = np.where(mask_ok & (la_c > 0),
                                       agg["_la_sum"] / la_c.replace(0, np.nan), np.nan)

    return out


# ---------------------------------------------------------------------------
# Batter windows
# ---------------------------------------------------------------------------

def precompute_batter_windows_fast(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,
    batter_team_lookup: dict,
    game_pk_home_lookup: dict,
    label_game_pks: pd.DataFrame,
) -> pd.DataFrame:
    need = (
        target_dates
        .drop_duplicates(subset=["batter", "game_date"])
        .reset_index(drop=True)
    )

    # ── 14-day overall ──────────────────────────────────────────────────────
    w14 = _window_merge(pa_df, need, "batter", days_back=14)
    stats_14 = _agg_pa_stats(
        w14, ["batter", "target_date"], min_pa=MIN_PA_WINDOW, prefix="b_"
    ).rename(columns={
        "b_pa":           "b_pa_14",
        "b_hr":           "b_hr_14",
        "b_hr_rate":      "b_hr_rate_14",
        "b_barrel_rate":  "b_barrel_rate_14",
        "b_ev_mean":      "b_ev_mean_14",
        "b_la_mean":      "b_la_mean_14",
        "b_hardhit_rate": "b_hardhit_rate_14",
        "b_fb_rate":      "b_fb_rate_14",
        "b_k_rate":       "b_k_rate_14",
        "b_bb_rate":      "b_bb_rate_14",
        "target_date":    "game_date",
    })

    # ── Season overall ──────────────────────────────────────────────────────
    wszn = _window_merge(pa_df, need, "batter", days_back=0, season_window=True)
    stats_szn = _agg_pa_stats(
        wszn, ["batter", "target_date"], min_pa=MIN_PA_BATTER_SZN, prefix="b_"
    ).rename(columns={
        "b_pa":           "b_pa_szn",
        "b_hr":           "b_hr_szn",
        "b_hr_rate":      "b_hr_rate_szn",
        "b_barrel_rate":  "b_barrel_rate_szn",
        "b_ev_mean":      "b_ev_mean_szn",
        "b_la_mean":      "b_la_mean_szn",
        "b_hardhit_rate": "b_hardhit_rate_szn",
        "b_fb_rate":      "b_fb_rate_szn",
        "b_k_rate":       "b_k_rate_szn",
        "b_bb_rate":      "b_bb_rate_szn",
        "target_date":    "game_date",
    })

    # ── Platoon splits — drop_means=True prevents la_mean/ev_mean collision ─
    def _plat_14(hand):
        sub = w14[w14["p_throws"] == hand]
        if sub.empty:
            return pd.DataFrame(columns=["batter", "game_date"])
        return _agg_pa_stats(
            sub, ["batter", "target_date"], min_pa=MIN_PA_WINDOW,
            prefix="b_", drop_means=True,
        ).rename(columns={
            "b_pa":           f"b_pa_14_vs{hand}",
            "b_hr":           f"b_hr_14_vs{hand}",
            "b_hr_rate":      f"b_hr_rate_14_vs{hand}",
            "b_barrel_rate":  f"b_barrel_rate_14_vs{hand}",
            "b_hardhit_rate": f"b_hardhit_rate_14_vs{hand}",
            "b_fb_rate":      f"b_fb_rate_14_vs{hand}",
            "b_k_rate":       f"b_k_rate_14_vs{hand}",
            "b_bb_rate":      f"b_bb_rate_14_vs{hand}",
            "target_date":    "game_date",
        })

    def _plat_szn(hand):
        sub = wszn[wszn["p_throws"] == hand]
        if sub.empty:
            return pd.DataFrame(columns=["batter", "game_date"])
        return _agg_pa_stats(
            sub, ["batter", "target_date"], min_pa=MIN_PA_BATTER_SZN,
            prefix="b_", drop_means=True,
        ).rename(columns={
            "b_pa":           f"b_pa_szn_vs{hand}",
            "b_hr":           f"b_hr_szn_vs{hand}",
            "b_hr_rate":      f"b_hr_rate_szn_vs{hand}",
            "b_barrel_rate":  f"b_barrel_rate_szn_vs{hand}",
            "b_hardhit_rate": f"b_hardhit_rate_szn_vs{hand}",
            "b_fb_rate":      f"b_fb_rate_szn_vs{hand}",
            "b_k_rate":       f"b_k_rate_szn_vs{hand}",
            "b_bb_rate":      f"b_bb_rate_szn_vs{hand}",
            "target_date":    "game_date",
        })

    plat_14_L  = _plat_14("L")
    plat_14_R  = _plat_14("R")
    plat_szn_L = _plat_szn("L")
    plat_szn_R = _plat_szn("R")

    # ── Trend (7d vs 8-14d) ─────────────────────────────────────────────────
    w7    = _window_merge(pa_df, need, "batter", days_back=7)
    w8_14 = _window_merge(pa_df, need, "batter", days_back=14)
    w8_14 = w8_14[w8_14["game_date"] < (w8_14["target_date"] - pd.Timedelta(days=7))]

    w7_grp    = {k: v for k, v in w7.groupby(["batter", "target_date"])}
    w8_14_grp = {k: v for k, v in w8_14.groupby(["batter", "target_date"])}
    all_keys  = set(w7_grp) | set(w8_14_grp)
    empty_df  = pa_df.iloc[0:0]

    trend_rows = []
    for (batter_id, tdate) in all_keys:
        t = _batter_trend_stats(
            w7_grp.get((batter_id, tdate), empty_df),
            w8_14_grp.get((batter_id, tdate), empty_df),
        )
        t["batter"]    = batter_id
        t["game_date"] = tdate
        trend_rows.append(t)

    trend_df = (
        pd.DataFrame(trend_rows)
        if trend_rows
        else pd.DataFrame(columns=[
            "batter", "game_date",
            "b_ev_trend", "b_hardhit_trend", "b_barrel_trend", "b_hr_trend",
            "b_ev_mean_7", "b_hardhit_rate_7",
        ])
    )

    # ── Home/away splits (season) ───────────────────────────────────────────
    wszn_copy = wszn.copy()
    wszn_copy["batter_team"] = wszn_copy["batter"].map(batter_team_lookup)
    wszn_copy["is_home_pa"]  = (wszn_copy["home_team"] == wszn_copy["batter_team"]).astype("boolean").fillna(False).astype(int)

    def _ha_rates(sub, suffix):
        if sub.empty:
            return pd.DataFrame(columns=["batter", "game_date"])
        return _agg_pa_stats(
            sub, ["batter", "target_date"], min_pa=MIN_PA_WINDOW,
            prefix="b_", drop_means=True,
        ).rename(columns={
            "b_hr_rate":      f"b_hr_rate_{suffix}",
            "b_hardhit_rate": f"b_hardhit_rate_{suffix}",
            "b_barrel_rate":  f"b_barrel_rate_{suffix}",
            "target_date":    "game_date",
        })[[
            "batter", "game_date",
            f"b_hr_rate_{suffix}",
            f"b_hardhit_rate_{suffix}",
            f"b_barrel_rate_{suffix}",
        ]]

    home_stats = _ha_rates(wszn_copy[wszn_copy["is_home_pa"] == 1], "home")
    away_stats = _ha_rates(wszn_copy[wszn_copy["is_home_pa"] == 0], "away")

    # ── is_home_game + same_hand_matchup ────────────────────────────────────
    bgpk = label_game_pks.set_index(["batter", "game_date"])["game_pk"].to_dict()
    need_copy = need.copy()
    need_copy["game_pk"]      = [bgpk.get((r.batter, r.game_date)) for r in need_copy.itertuples()]
    need_copy["batter_team"]  = need_copy["batter"].map(batter_team_lookup)
    need_copy["home_today"]   = need_copy["game_pk"].map(game_pk_home_lookup)
    need_copy["is_home_game"] = np.where(
        need_copy["batter_team"].notna() & need_copy["home_today"].notna(),
        (need_copy["batter_team"] == need_copy["home_today"]).astype("boolean").fillna(False).astype(int),
        -1,
    )

    batter_hand_lookup = (
        pa_df.dropna(subset=["batter", "stand"])
        .groupby("batter")["stand"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )
    need_copy["batter_hand"] = need_copy["batter"].map(batter_hand_lookup)
    pitcher_hand_col = need.get("pitcher_hand") if hasattr(need, "get") else None
    if "pitcher_hand" in need.columns:
        need_copy["pitcher_hand"] = need["pitcher_hand"].values
    else:
        need_copy["pitcher_hand"] = None

    need_copy["same_hand_matchup"] = np.where(
        need_copy["batter_hand"].notna() & need_copy["pitcher_hand"].notna(),
        (need_copy["batter_hand"] == need_copy["pitcher_hand"]).astype("boolean").fillna(False).astype(int),
        -1,
    )

    # ── Final merge ─────────────────────────────────────────────────────────
    base = need_copy[["batter", "game_date", "is_home_game", "same_hand_matchup"]].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    for df_part in [stats_14, stats_szn,
                    plat_14_L, plat_14_R, plat_szn_L, plat_szn_R,
                    trend_df, home_stats, away_stats]:
        if df_part is None or df_part.empty or "game_date" not in df_part.columns:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        base = base.merge(df_part, on=["batter", "game_date"], how="left")

    if "b_hr_rate_home" in base.columns and "b_hr_rate_away" in base.columns:
        base["b_hr_rate_home_edge"] = base["b_hr_rate_home"] - base["b_hr_rate_away"]

    return base


# ---------------------------------------------------------------------------
# Pitcher windows
# ---------------------------------------------------------------------------

_PITCHER_STAT_COLS: list[str] = [
    # 30-day overall
    "p_pa_30", "p_hr_allowed_30", "p_hr_allowed_rate_30",
    "p_barrel_allowed_rate_30", "p_ev_allowed_mean_30", "p_la_allowed_mean_30",
    "p_hardhit_allowed_rate_30", "p_fb_allowed_rate_30", "p_k_rate_30", "p_bb_rate_30",
    # season overall
    "p_pa_szn", "p_hr_allowed_szn", "p_hr_allowed_rate_szn",
    "p_barrel_allowed_rate_szn", "p_ev_allowed_mean_szn", "p_la_allowed_mean_szn",
    "p_hardhit_allowed_rate_szn", "p_fb_allowed_rate_szn", "p_k_rate_szn", "p_bb_rate_szn",
    # platoon splits
    "p_pa_30_vsL", "p_hr_allowed_30_vsL", "p_hr_allowed_rate_30_vsL",
    "p_barrel_allowed_rate_30_vsL", "p_hardhit_allowed_rate_30_vsL",
    "p_fb_allowed_rate_30_vsL", "p_k_rate_30_vsL", "p_bb_rate_30_vsL",
    "p_pa_30_vsR", "p_hr_allowed_30_vsR", "p_hr_allowed_rate_30_vsR",
    "p_barrel_allowed_rate_30_vsR", "p_hardhit_allowed_rate_30_vsR",
    "p_fb_allowed_rate_30_vsR", "p_k_rate_30_vsR", "p_bb_rate_30_vsR",
    "p_pa_szn_vsL", "p_hr_allowed_szn_vsL", "p_hr_allowed_rate_szn_vsL",
    "p_barrel_allowed_rate_szn_vsL", "p_hardhit_allowed_rate_szn_vsL",
    "p_fb_allowed_rate_szn_vsL", "p_k_rate_szn_vsL", "p_bb_rate_szn_vsL",
    "p_pa_szn_vsR", "p_hr_allowed_szn_vsR", "p_hr_allowed_rate_szn_vsR",
    "p_barrel_allowed_rate_szn_vsR", "p_hardhit_allowed_rate_szn_vsR",
    "p_fb_allowed_rate_szn_vsR", "p_k_rate_szn_vsR", "p_bb_rate_szn_vsR",
]


def precompute_pitcher_windows_fast(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,
) -> pd.DataFrame:
    need = (
        target_dates
        .drop_duplicates(subset=["pitcher", "game_date"])
        .reset_index(drop=True)
    )

    # Guard: if need is empty (all starter_pitcher_id values were NaN and were
    # dropped), return a zero-row frame with the full expected column schema so
    # that the left-merge in build_features_for_range adds the columns (as NaN)
    # rather than silently omitting them — which would cause KeyErrors later in
    # _add_edge_features.
    if need.empty:
        logger.warning(
            "precompute_pitcher_windows_fast: target_dates is empty after "
            "deduplication (all starter_pitcher_id values may be NaN). "
            "Returning zero-row frame with full pitcher stat schema — all "
            "pitcher features will be NaN for this date range."
        )
        empty_cols = ["pitcher", "game_date"] + _PITCHER_STAT_COLS
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df['game_date'] = pd.to_datetime(empty_df['game_date'])
        return empty_df

    # ── 30-day overall ──────────────────────────────────────────────────────
    w30 = _window_merge(pa_df, need, "pitcher", days_back=30)
    stats_30 = _agg_pa_stats(
        w30, ["pitcher", "target_date"], min_pa=MIN_PA_WINDOW, prefix="p_"
    ).rename(columns={
        "p_pa":           "p_pa_30",
        "p_hr":           "p_hr_allowed_30",
        "p_hr_rate":      "p_hr_allowed_rate_30",
        "p_barrel_rate":  "p_barrel_allowed_rate_30",
        "p_ev_mean":      "p_ev_allowed_mean_30",
        "p_la_mean":      "p_la_allowed_mean_30",
        "p_hardhit_rate": "p_hardhit_allowed_rate_30",
        "p_fb_rate":      "p_fb_allowed_rate_30",
        "p_k_rate":       "p_k_rate_30",
        "p_bb_rate":      "p_bb_rate_30",
        "target_date":    "game_date",
    })

    # ── Season overall ──────────────────────────────────────────────────────
    wszn = _window_merge(pa_df, need, "pitcher", days_back=0, season_window=True)
    stats_szn = _agg_pa_stats(
        wszn, ["pitcher", "target_date"], min_pa=MIN_PA_PITCHER_SZN, prefix="p_"
    ).rename(columns={
        "p_pa":           "p_pa_szn",
        "p_hr":           "p_hr_allowed_szn",
        "p_hr_rate":      "p_hr_allowed_rate_szn",
        "p_barrel_rate":  "p_barrel_allowed_rate_szn",
        "p_ev_mean":      "p_ev_allowed_mean_szn",
        "p_la_mean":      "p_la_allowed_mean_szn",
        "p_hardhit_rate": "p_hardhit_allowed_rate_szn",
        "p_fb_rate":      "p_fb_allowed_rate_szn",
        "p_k_rate":       "p_k_rate_szn",
        "p_bb_rate":      "p_bb_rate_szn",
        "target_date":    "game_date",
    })

    # ── Platoon splits — drop_means=True prevents collision ─────────────────
    def _plat(windowed, hand, suffix, min_pa):
        sub = windowed[windowed["stand"] == hand]
        if sub.empty:
            # Return a DataFrame with all expected columns (filled with NaN)
            plat_cols = [
                "pitcher", "game_date",
                f"p_pa_{suffix}_vs{hand}",
                f"p_hr_allowed_{suffix}_vs{hand}",
                f"p_hr_allowed_rate_{suffix}_vs{hand}",
                f"p_barrel_allowed_rate_{suffix}_vs{hand}",
                f"p_hardhit_allowed_rate_{suffix}_vs{hand}",
                f"p_fb_allowed_rate_{suffix}_vs{hand}",
                f"p_k_rate_{suffix}_vs{hand}",
                f"p_bb_rate_{suffix}_vs{hand}",
            ]
            empty_plat = pd.DataFrame(columns=plat_cols)
            empty_plat['game_date'] = pd.to_datetime(empty_plat['game_date'])
            return empty_plat
        return _agg_pa_stats(
            sub, ["pitcher", "target_date"], min_pa=min_pa,
            prefix="p_", drop_means=True,
        ).rename(columns={
            "p_pa":           f"p_pa_{suffix}_vs{hand}",
            "p_hr":           f"p_hr_allowed_{suffix}_vs{hand}",
            "p_hr_rate":      f"p_hr_allowed_rate_{suffix}_vs{hand}",
            "p_barrel_rate":  f"p_barrel_allowed_rate_{suffix}_vs{hand}",
            "p_hardhit_rate": f"p_hardhit_allowed_rate_{suffix}_vs{hand}",
            "p_fb_rate":      f"p_fb_allowed_rate_{suffix}_vs{hand}",
            "p_k_rate":       f"p_k_rate_{suffix}_vs{hand}",
            "p_bb_rate":      f"p_bb_rate_{suffix}_vs{hand}",
            "target_date":    "game_date",
        })

    plat_30_L  = _plat(w30,  "L", "30",  MIN_PA_WINDOW)
    plat_30_R  = _plat(w30,  "R", "30",  MIN_PA_WINDOW)
    plat_szn_L = _plat(wszn, "L", "szn", MIN_PA_PITCHER_SZN)
    plat_szn_R = _plat(wszn, "R", "szn", MIN_PA_PITCHER_SZN)

    # ── Merge ───────────────────────────────────────────────────────────────
    base = need[["pitcher", "game_date"]].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    for df_part in [stats_30, stats_szn, plat_30_L, plat_30_R, plat_szn_L, plat_szn_R]:
        if df_part.empty or "game_date" not in df_part.columns:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        base = base.merge(df_part, on=["pitcher", "game_date"], how="left")

    return base


# ---------------------------------------------------------------------------
# Pitcher velo windows
# ---------------------------------------------------------------------------

def precompute_pitcher_velo_fast(
    pitches_df: pd.DataFrame,
    target_dates: pd.DataFrame,
) -> pd.DataFrame:
    need = (
        target_dates
        .drop_duplicates(subset=["pitcher", "game_date"])
        .reset_index(drop=True)
    )
    if need.empty:
        empty_cols = ["pitcher", "game_date"] + _PITCHER_VELO_COLS
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df['game_date'] = pd.to_datetime(empty_df['game_date'])
        return empty_df
    # ── 30-day FB velo / mix ────────────────────────────────────────────────
    w30 = _window_merge(pitches_df, need, "pitcher", days_back=30)

    if w30.empty:
        velo_30 = need[["pitcher", "game_date"]].copy()
        velo_30["p_fb_velo_30"]      = np.nan
        velo_30["p_fb_pct_30"]       = np.nan
        velo_30["p_offspeed_pct_30"] = np.nan
    else:
        w30 = w30.copy()
        w30["_is_fb"]       = w30["pitch_type"].isin(FASTBALL_TYPES).astype(float)
        w30["_is_offspeed"] = w30["pitch_type"].isin(OFFSPEED_TYPES).astype(float)
        w30["_fb_speed"]    = np.where(
            w30["_is_fb"] == 1,
            pd.to_numeric(w30["release_speed"], errors="coerce"),
            np.nan,
        )

        agg30 = w30.groupby(["pitcher", "target_date"], sort=False).agg(
            _total      =("_is_fb",       "size"),
            _fb_count   =("_is_fb",       "sum"),
            _os_count   =("_is_offspeed", "sum"),
            _fb_velo_sum=("_fb_speed",    "sum"),
            _fb_velo_n  =("_fb_speed",    "count"),
        ).reset_index()

        agg30["p_fb_velo_30"]      = np.where(
            agg30["_fb_velo_n"] > 0,
            agg30["_fb_velo_sum"] / agg30["_fb_velo_n"].replace(0, np.nan),
            np.nan,
        )
        agg30["p_fb_pct_30"]       = agg30["_fb_count"] / agg30["_total"].replace(0, np.nan)
        agg30["p_offspeed_pct_30"] = agg30["_os_count"] / agg30["_total"].replace(0, np.nan)

        velo_30 = agg30[["pitcher", "target_date",
                          "p_fb_velo_30", "p_fb_pct_30", "p_offspeed_pct_30"]].rename(
            columns={"target_date": "game_date"}
        )

    # ── FB velo trend (recent 3 starts vs prior 3) ──────────────────────────
    w60 = _window_merge(pitches_df, need, "pitcher", days_back=60)

    if w60.empty:
        trend_df = need[["pitcher", "game_date"]].copy()
        trend_df["p_fb_velo_trend"] = np.nan
    else:
        w60 = w60.copy()
        w60["_is_fb"]    = w60["pitch_type"].isin(FASTBALL_TYPES)
        w60["_fb_speed"] = np.where(
            w60["_is_fb"],
            pd.to_numeric(w60["release_speed"], errors="coerce"),
            np.nan,
        )

        start_velo = (
            w60[w60["_is_fb"]]
            .groupby(["pitcher", "target_date", "game_date"], sort=False)["_fb_speed"]
            .mean()
            .reset_index()
            .rename(columns={"game_date": "start_date", "_fb_speed": "start_velo"})
        )

        start_velo = start_velo.sort_values(["pitcher", "target_date", "start_date"])

        def _trend(grp):
            recent = grp.tail(3)["start_velo"].dropna().mean()
            prior  = grp.iloc[max(0, len(grp)-6): max(0, len(grp)-3)]["start_velo"].dropna().mean()
            if pd.isna(recent) or pd.isna(prior):
                return np.nan
            return recent - prior

        trend_vals = (
            start_velo
            .groupby(["pitcher", "target_date"])
            .apply(_trend)
            .reset_index()
            .rename(columns={0: "p_fb_velo_trend", "target_date": "game_date"})
        )
        trend_df = trend_vals

    # ── Merge ───────────────────────────────────────────────────────────────
    result = need[["pitcher", "game_date"]].copy()
    result["game_date"] = pd.to_datetime(result["game_date"])

    for df_part in [velo_30, trend_df]:
        if df_part.empty:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        result = result.merge(df_part, on=["pitcher", "game_date"], how="left")

    return result

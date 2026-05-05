from __future__ import annotations

"""
Vectorised replacements for the three slow per-row loop functions in
build_features.py, plus pulled-ball rate and pitch-type matchup features.

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
    PITCH_GROUP_FASTBALL,
    PITCH_GROUP_BREAKING,
    PITCH_GROUP_OFFSPEED,
    MIN_PA_PITCH_TYPE,
    MIN_PITCHES_PITCH_TYPE,
    _XWOBACON_CAP,
    _safe_mean,
    _batter_trend_stats,
    _is_pulled_airball,
)

# Min BBE counts for the rolling xwOBAcon features. Higher than MIN_PA_WINDOW
# because xwOBAcon is computed only on contact and small samples produce
# noisy means that get over-rewarded by LGBM splits.
MIN_BBE_XWOBA_14 = 10

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
    """
    merged = pa_df.merge(
        need[[player_col, "game_date"]].rename(columns={"game_date": "target_date"}),
        on=player_col,
        how="inner",
    )

    # Right bound: strictly before target date (no same-day leakage)
    in_window = merged["game_date"] < merged["target_date"]

    if season_window:
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

_PLATOON_DROP_COLS = {"ev_mean", "la_mean"}


def _agg_pa_stats(
    windowed: pd.DataFrame,
    group_cols: list[str],
    min_pa: int,
    prefix: str,
    drop_means: bool = False,
    include_xwobacon: bool = False,
    min_bbe_xwobacon: int = MIN_BBE_XWOBA_14,
) -> pd.DataFrame:
    """
    Aggregate PA-level stats.  Returns one row per group.
    Rates are NaN when PA < min_pa (cold-start guard).

    `include_xwobacon`: when True, also output `{prefix}xwobacon` =
        capped(estimated_woba_using_speedangle).mean() over BBEs in the
        window. NaN when fewer than `min_bbe_xwobacon` non-null xwOBA
        values are present. Only enable on caller-by-caller basis (e.g.,
        batter 14d window) to avoid bloating other outputs.
    """
    windowed = windowed.copy()
    ev = windowed["launch_speed"]
    la = windowed["launch_angle"]
    windowed["_ev_gte95"] = (ev >= 95).astype(float)
    windowed["_fb"]       = la.between(20, 40).astype(float)

    if include_xwobacon:
        # Cap before summing so a single freak >2.0 PA can't dominate a
        # small-sample window mean.
        xw = pd.to_numeric(
            windowed.get("estimated_woba_using_speedangle"), errors="coerce",
        )
        windowed["_xw_capped"] = xw.clip(upper=_XWOBACON_CAP)

    pa_src = "pitcher" if ("pitcher" in windowed.columns and prefix.startswith("p")) else "is_hr"

    agg_kwargs = dict(
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
    )
    if include_xwobacon:
        agg_kwargs["_xw_sum"]   = ("_xw_capped", "sum")
        agg_kwargs["_xw_count"] = ("_xw_capped", "count")

    agg = windowed.groupby(group_cols, sort=False).agg(**agg_kwargs).reset_index()

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

    if include_xwobacon:
        xw_c = agg["_xw_count"]
        out[f"{p}xwobacon"] = np.where(
            (xw_c >= min_bbe_xwobacon),
            agg["_xw_sum"] / xw_c.replace(0, np.nan),
            np.nan,
        )

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
        w14, ["batter", "target_date"],
        min_pa=MIN_PA_WINDOW, prefix="b_",
        include_xwobacon=True, min_bbe_xwobacon=MIN_BBE_XWOBA_14,
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
        "b_xwobacon":     "b_xwobacon_14",
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

    # ── Platoon splits ───────────────────────────────────────────────────────
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
    "p_pa_30", "p_hr_allowed_30", "p_hr_allowed_rate_30",
    "p_barrel_allowed_rate_30", "p_ev_allowed_mean_30", "p_la_allowed_mean_30",
    "p_hardhit_allowed_rate_30", "p_fb_allowed_rate_30", "p_k_rate_30", "p_bb_rate_30",
    "p_pa_szn", "p_hr_allowed_szn", "p_hr_allowed_rate_szn",
    "p_barrel_allowed_rate_szn", "p_ev_allowed_mean_szn", "p_la_allowed_mean_szn",
    "p_hardhit_allowed_rate_szn", "p_fb_allowed_rate_szn", "p_k_rate_szn", "p_bb_rate_szn",
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

    if need.empty:
        logger.warning(
            "precompute_pitcher_windows_fast: target_dates is empty after "
            "deduplication. Returning zero-row frame with full pitcher stat schema."
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

    # ── Platoon splits ───────────────────────────────────────────────────────
    def _plat(windowed, hand, suffix, min_pa):
        sub = windowed[windowed["stand"] == hand]
        if sub.empty:
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


# ---------------------------------------------------------------------------
# Pulled AIR BALL rate windows  (NEW)
# ---------------------------------------------------------------------------

def precompute_batter_pull_fast(
    pitches_df: pd.DataFrame,
    target_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (batter, game_date), compute rolling PULLED AIR BALL rate over
    the 14-day and season windows.

    "Pulled air ball" = fly ball or line drive hit to the pull side of the
    field.  Ground balls are excluded because they don't produce home runs
    and would dilute the signal.

    pitches_df must be the PITCH-LEVEL DataFrame (one row per pitch/event)
    so that hc_x / hc_y, bb_type, and stand are available.

    Required columns: batter, game_date, hc_x, hc_y, stand, bb_type, events

    Returns DataFrame with columns:
        batter, game_date,
        b_pull_air_rate_14,     b_pull_air_rate_szn,     ← pulled FB+LD / all FB+LD
        b_pull_air_hr_rate_14,  b_pull_air_hr_rate_szn   ← HR / pulled FB+LD
    """
    need = (
        target_dates
        .drop_duplicates(subset=["batter", "game_date"])
        .reset_index(drop=True)
    )

    empty_cols = ["batter", "game_date",
                  "b_pull_air_rate_14", "b_pull_air_rate_szn",
                  "b_pull_air_hr_rate_14", "b_pull_air_hr_rate_szn"]

    if need.empty or pitches_df.empty:
        df = pd.DataFrame(columns=empty_cols)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    # Check required spray-chart columns — absent in old cache
    missing = [c for c in ("hc_x", "hc_y", "bb_type") if c not in pitches_df.columns]
    if missing:
        logger.warning(
            "precompute_batter_pull_fast: columns %s missing from pitches_df. "
            "Delete Statcast cache files and re-download to enable pull-airball features.",
            missing,
        )
        df = pd.DataFrame(columns=empty_cols)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    p = pitches_df.copy()
    p["_events_str"] = p["events"].astype("string").str.lower().str.strip()
    p["_is_hr"]      = (p["_events_str"] == "home_run").fillna(False).astype("int8")

    # Tag pulled air balls using the updated helper (fly_ball + line_drive only)
    p["_is_pull_air"] = _is_pulled_airball(
        p["hc_x"], p["hc_y"],
        p.get("stand",   pd.Series(dtype=str)),
        p.get("bb_type", pd.Series(dtype=str)),
    ).fillna(False).astype("int8")

    # Denominator: ALL air balls (fly ball + line drive) in play, not just pulled
    bb = p.get("bb_type", pd.Series(dtype=str)).astype("string").str.lower().str.strip()
    p["_is_airball"] = bb.isin(["fly_ball", "line_drive"]).fillna(False).astype("int8")

    # Keep only rows that are air balls with valid coordinates
    # (hc_x / hc_y NaN → _is_pull_air=0 so they count in denominator but not numerator)
    air_df = p[p["_is_airball"] == 1].copy()

    if air_df.empty:
        df = pd.DataFrame(columns=empty_cols)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    def _agg_pull_air(windowed: pd.DataFrame, suffix: str, min_ab: int) -> pd.DataFrame:
        if windowed.empty:
            return pd.DataFrame(columns=["batter", "game_date",
                                          f"b_pull_air_rate_{suffix}",
                                          f"b_pull_air_hr_rate_{suffix}"])

        agg = windowed.groupby(["batter", "target_date"], sort=False).agg(
            _air_bip  =("_is_airball",   "size"),   # total air balls
            _pull_air =("_is_pull_air",  "sum"),    # pulled air balls
        ).reset_index()

        # HR on pulled air balls only
        pull_hr = (
            windowed[windowed["_is_pull_air"] == 1]
            .groupby(["batter", "target_date"], sort=False)["_is_hr"]
            .sum()
            .reset_index()
            .rename(columns={"_is_hr": "_pull_air_hr"})
        )
        agg = agg.merge(pull_hr, on=["batter", "target_date"], how="left")
        agg["_pull_air_hr"] = agg["_pull_air_hr"].fillna(0)

        mask = agg["_air_bip"] >= min_ab
        agg[f"b_pull_air_rate_{suffix}"] = np.where(
            mask,
            agg["_pull_air"] / agg["_air_bip"].replace(0, np.nan),
            np.nan,
        )
        agg[f"b_pull_air_hr_rate_{suffix}"] = np.where(
            mask & (agg["_pull_air"] > 0),
            agg["_pull_air_hr"] / agg["_pull_air"].replace(0, np.nan),
            np.nan,
        )
        return agg[["batter", "target_date",
                    f"b_pull_air_rate_{suffix}",
                    f"b_pull_air_hr_rate_{suffix}"]].rename(columns={"target_date": "game_date"})

    w14  = _window_merge(air_df, need, "batter", days_back=14)
    wszn = _window_merge(air_df, need, "batter", days_back=0, season_window=True)

    pull_14  = _agg_pull_air(w14,  "14",  min_ab=MIN_PA_WINDOW)
    pull_szn = _agg_pull_air(wszn, "szn", min_ab=MIN_PA_BATTER_SZN)

    base = need[["batter", "game_date"]].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    for df_part in [pull_14, pull_szn]:
        if df_part.empty or "game_date" not in df_part.columns:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        base = base.merge(df_part, on=["batter", "game_date"], how="left")

    return base


# ---------------------------------------------------------------------------
# Pitch-type matchup windows  (NEW)
# ---------------------------------------------------------------------------

def precompute_pitch_matchup_fast(
    pitches_df: pd.DataFrame,
    target_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pitch-type matchup features combining:
        (a) batter's HR rate vs fastballs / breaking balls / offspeed (14d + season)
        (b) pitcher's usage rate of each pitch type (30-day)
        (c) pitcher's HR-allowed rate by pitch type (30-day)
        (d) interaction: batter_hr_vs_X * pitcher_usage_X

    target_dates must have columns: batter, pitcher, game_date

    Returns DataFrame with columns:
        batter, pitcher, game_date,
        b_hr_rate_vs_fb_14,   b_hr_rate_vs_fb_szn,
        b_hr_rate_vs_brk_14,  b_hr_rate_vs_brk_szn,
        b_hr_rate_vs_os_14,   b_hr_rate_vs_os_szn,
        p_fb_usage_30,  p_brk_usage_30,  p_os_usage_30,
        p_hr_rate_vs_fb_30,  p_hr_rate_vs_brk_30,  p_hr_rate_vs_os_30,
        matchup_fb_30,  matchup_brk_30,  matchup_os_30,
        matchup_best_30
    """
    empty_cols = [
        "batter", "pitcher", "game_date",
        "b_hr_rate_vs_fb_14",  "b_hr_rate_vs_fb_szn",
        "b_hr_rate_vs_brk_14", "b_hr_rate_vs_brk_szn",
        "b_hr_rate_vs_os_14",  "b_hr_rate_vs_os_szn",
        "p_fb_usage_30",  "p_brk_usage_30",  "p_os_usage_30",
        "p_hr_rate_vs_fb_30", "p_hr_rate_vs_brk_30", "p_hr_rate_vs_os_30",
        "matchup_fb_30",  "matchup_brk_30",  "matchup_os_30",
        "matchup_best_30",
    ]

    need = (
        target_dates
        .drop_duplicates(subset=["batter", "pitcher", "game_date"])
        .reset_index(drop=True)
    )

    if need.empty or pitches_df.empty:
        df = pd.DataFrame(columns=empty_cols)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    p = pitches_df.copy()
    p["_pt_str"]     = p["pitch_type"].astype("string").str.upper().str.strip()
    p["_events_str"] = p["events"].astype("string").str.lower().str.strip()
    p["_is_hr"]      = (p["_events_str"] == "home_run").fillna(False).astype("int8")

    p["_grp_fb"]  = p["_pt_str"].isin(PITCH_GROUP_FASTBALL).fillna(False).astype("int8")
    p["_grp_brk"] = p["_pt_str"].isin(PITCH_GROUP_BREAKING).fillna(False).astype("int8")
    p["_grp_os"]  = p["_pt_str"].isin(PITCH_GROUP_OFFSPEED).fillna(False).astype("int8")

    # Only include pitches that ended PAs for rate calculations
    p_outcome = p[p["_events_str"].notna() & (p["_events_str"] != "")].copy()

    batter_need  = need[["batter", "game_date"]].drop_duplicates()
    pitcher_need = need[["pitcher", "game_date"]].drop_duplicates()

    # ── Batter HR rate vs each pitch group ──────────────────────────────────
    def _batter_hr_vs(group_col: str, grp_name: str, days: int, szn: bool,
                       min_pa: int, win_sfx: str) -> pd.DataFrame:
        col_name = f"b_hr_rate_vs_{grp_name}_{win_sfx}"
        w = _window_merge(p_outcome, batter_need, "batter",
                          days_back=days, season_window=szn)
        if w.empty:
            return pd.DataFrame(columns=["batter", "game_date", col_name])
        sub = w[w[group_col] == 1]
        if sub.empty:
            return pd.DataFrame(columns=["batter", "game_date", col_name])
        agg = sub.groupby(["batter", "target_date"], sort=False).agg(
            _pa=("_is_hr", "size"), _hr=("_is_hr", "sum"),
        ).reset_index()
        agg[col_name] = np.where(
            agg["_pa"] >= min_pa,
            agg["_hr"] / agg["_pa"].replace(0, np.nan), np.nan,
        )
        return agg[["batter", "target_date", col_name]].rename(
            columns={"target_date": "game_date"}
        )

    b_fb_14   = _batter_hr_vs("_grp_fb",  "fb",  14, False, MIN_PA_PITCH_TYPE, "14")
    b_brk_14  = _batter_hr_vs("_grp_brk", "brk", 14, False, MIN_PA_PITCH_TYPE, "14")
    b_os_14   = _batter_hr_vs("_grp_os",  "os",  14, False, MIN_PA_PITCH_TYPE, "14")
    b_fb_szn  = _batter_hr_vs("_grp_fb",  "fb",  0,  True,  MIN_PA_BATTER_SZN, "szn")
    b_brk_szn = _batter_hr_vs("_grp_brk", "brk", 0,  True,  MIN_PA_BATTER_SZN, "szn")
    b_os_szn  = _batter_hr_vs("_grp_os",  "os",  0,  True,  MIN_PA_BATTER_SZN, "szn")

    # ── Pitcher pitch-type usage (30d, ALL pitches including balls) ──────────
    w30_p = _window_merge(p, pitcher_need, "pitcher", days_back=30)

    if not w30_p.empty:
        pagg = w30_p.groupby(["pitcher", "target_date"], sort=False).agg(
            _total   =("_grp_fb",  "size"),
            _fb_cnt  =("_grp_fb",  "sum"),
            _brk_cnt =("_grp_brk", "sum"),
            _os_cnt  =("_grp_os",  "sum"),
        ).reset_index()
        mask_p = pagg["_total"] >= MIN_PITCHES_PITCH_TYPE
        pagg["p_fb_usage_30"]  = np.where(mask_p, pagg["_fb_cnt"]  / pagg["_total"].replace(0, np.nan), np.nan)
        pagg["p_brk_usage_30"] = np.where(mask_p, pagg["_brk_cnt"] / pagg["_total"].replace(0, np.nan), np.nan)
        pagg["p_os_usage_30"]  = np.where(mask_p, pagg["_os_cnt"]  / pagg["_total"].replace(0, np.nan), np.nan)
        pitcher_usage = pagg[["pitcher", "target_date",
                               "p_fb_usage_30", "p_brk_usage_30", "p_os_usage_30"]].rename(
            columns={"target_date": "game_date"}
        )
    else:
        pitcher_usage = pd.DataFrame(columns=["pitcher", "game_date",
                                               "p_fb_usage_30", "p_brk_usage_30", "p_os_usage_30"])

    # ── Pitcher HR-allowed rate by pitch group (30d) ─────────────────────────
    def _pitcher_hr_vs(group_col: str, col_name: str,
                        windowed: pd.DataFrame, p_need: pd.DataFrame) -> pd.DataFrame:
        if windowed.empty:
            return pd.DataFrame(columns=["pitcher", "game_date", col_name])
        p_out_30 = windowed[windowed["_events_str"].notna() & (windowed["_events_str"] != "")]
        sub = p_out_30[p_out_30[group_col] == 1]
        if sub.empty:
            return pd.DataFrame(columns=["pitcher", "game_date", col_name])
        agg = sub.groupby(["pitcher", "target_date"], sort=False).agg(
            _pa=("_is_hr", "size"), _hr=("_is_hr", "sum"),
        ).reset_index()
        agg[col_name] = np.where(
            agg["_pa"] >= MIN_PA_PITCH_TYPE,
            agg["_hr"] / agg["_pa"].replace(0, np.nan), np.nan,
        )
        return agg[["pitcher", "target_date", col_name]].rename(
            columns={"target_date": "game_date"}
        )

    p_hr_fb_30  = _pitcher_hr_vs("_grp_fb",  "p_hr_rate_vs_fb_30",  w30_p, pitcher_need)
    p_hr_brk_30 = _pitcher_hr_vs("_grp_brk", "p_hr_rate_vs_brk_30", w30_p, pitcher_need)
    p_hr_os_30  = _pitcher_hr_vs("_grp_os",  "p_hr_rate_vs_os_30",  w30_p, pitcher_need)

    # ── Merge everything onto (batter, pitcher, game_date) base ────────────
    base = need[["batter", "pitcher", "game_date"]].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    for df_part in [b_fb_14, b_brk_14, b_os_14, b_fb_szn, b_brk_szn, b_os_szn]:
        if df_part.empty or "game_date" not in df_part.columns:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        base = base.merge(df_part, on=["batter", "game_date"], how="left")

    for df_part in [pitcher_usage, p_hr_fb_30, p_hr_brk_30, p_hr_os_30]:
        if df_part.empty or "game_date" not in df_part.columns:
            continue
        df_part = df_part.copy()
        df_part["game_date"] = pd.to_datetime(df_part["game_date"])
        base = base.merge(df_part, on=["pitcher", "game_date"], how="left")

    # ── Matchup interaction terms ────────────────────────────────────────────
    # Season rate preferred (more stable); 14d as fallback
    def _best(szn: str, d14: str) -> pd.Series:
        s = base.get(szn, pd.Series(np.nan, index=base.index))
        d = base.get(d14, pd.Series(np.nan, index=base.index))
        return s.fillna(d)

    b_fb  = _best("b_hr_rate_vs_fb_szn",  "b_hr_rate_vs_fb_14")
    b_brk = _best("b_hr_rate_vs_brk_szn", "b_hr_rate_vs_brk_14")
    b_os  = _best("b_hr_rate_vs_os_szn",  "b_hr_rate_vs_os_14")

    fb_usage  = base.get("p_fb_usage_30",  pd.Series(np.nan, index=base.index))
    brk_usage = base.get("p_brk_usage_30", pd.Series(np.nan, index=base.index))
    os_usage  = base.get("p_os_usage_30",  pd.Series(np.nan, index=base.index))

    base["matchup_fb_30"]  = b_fb  * fb_usage
    base["matchup_brk_30"] = b_brk * brk_usage
    base["matchup_os_30"]  = b_os  * os_usage

    matchup_cols = ["matchup_fb_30", "matchup_brk_30", "matchup_os_30"]
    avail_matchup = [c for c in matchup_cols if c in base.columns]
    if avail_matchup:
        base["matchup_best_30"] = base[avail_matchup].max(axis=1)
    else:
        base["matchup_best_30"] = np.nan

    return base

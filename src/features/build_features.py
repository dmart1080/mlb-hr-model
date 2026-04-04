from __future__ import annotations

import logging
import math as _math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events, REGULAR_SEASON_GAME_TYPE
from src.data_sources.mlb_schedule import fetch_rosters_for_games, enrich_labels_with_roster
from src.features.build_labels import build_batter_game_labels
from src.features.park_factors import get_park_factors, DEFAULT_PARK_FACTOR
from src.features.build_features_common import (
    FASTBALL_TYPES,
    OFFSPEED_TYPES,
    MIN_PA_BATTER_SZN,
    MIN_PA_PITCHER_SZN,
    MIN_PA_WINDOW,
    _safe_mean,
    _is_barrel,
)
from src.features.build_features_fast import (
    precompute_batter_windows_fast,
    precompute_pitcher_windows_fast,
    precompute_pitcher_velo_fast,
    precompute_batter_pull_fast,
    precompute_pitch_matchup_fast,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_WEATHER_MAX_WORKERS = 20


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


def _load_and_clean_events(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_statcast_events(
        start_date=start_date,
        end_date=end_date,
        # game_type is fetched so regular_season_only filter runs inside
        # fetch_statcast_events, and we apply a second hard filter below
        # as a belt-and-suspenders guard against old cache files.
        columns=[
            "game_date", "game_pk", "at_bat_number",
            "batter", "pitcher", "events",
            "home_team", "launch_speed", "launch_angle",
            "p_throws", "stand",
            "release_speed", "pitch_type",
            "game_type",
            "hc_x", "hc_y", "bb_type",      # spray chart + batted ball type
            "release_spin_rate",              # pitcher stuff quality
            "release_extension",              # extension toward plate
            "pfx_x", "pfx_z",                # horizontal + vertical movement
        ],
        regular_season_only=True,
    ).df.copy()

    # Hard filter — catches old cache files pre-dating the game_type column.
    if "game_type" in raw.columns:
        before = len(raw)
        raw = raw[raw["game_type"] == REGULAR_SEASON_GAME_TYPE].copy()
        removed = before - len(raw)
        if removed:
            logger.info(
                "Hard regular-season filter: removed %d non-RS rows "
                "(spring training / postseason / All-Star).", removed,
            )
    else:
        logger.warning(
            "game_type column absent from cache — delete cache and re-run "
            "to exclude spring training and postseason games."
        )

    raw = raw.convert_dtypes(dtype_backend="numpy_nullable")
    raw["game_date"]     = pd.to_datetime(raw["game_date"])
    raw["p_throws"]      = raw["p_throws"].astype("string").str.upper().str.strip()
    raw["stand"]         = raw["stand"].astype("string").str.upper().str.strip()
    raw["pitch_type"]    = raw["pitch_type"].astype("string").str.upper().str.strip()
    raw["release_speed"] = pd.to_numeric(raw["release_speed"], errors="coerce")

    pitches_df = (
        raw.copy()
        .sort_values("game_date")
        .dropna(subset=["game_date"])
        .reset_index(drop=True)
    )
    pitches_df["game_date"] = pitches_df["game_date"].astype("datetime64[ns]")

    raw["is_hr"] = (raw["events"] == "home_run").fillna(False).astype("int8")

    # FIX 1 (part a): preserve at_bat_number in pa_df so _compute_relief_pa_pct
    # can identify which pitcher each PA was against.  We keep the raw pitch-level
    # pitcher column; the PA-level pitcher is taken as the LAST pitcher seen in
    # that at-bat (closer to the actual outcome).
    pa_df = (
        raw
        .sort_values(["game_pk", "batter", "game_date"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .agg({
            "game_date":    "first",
            "batter":       "last",
            "pitcher":      "last",
            "home_team":    "first",
            "events":       "last",
            "is_hr":        "max",
            "launch_speed": "mean",
            "launch_angle": "mean",
            "p_throws":     "last",
            "stand":        "last",
        })
    )
    # at_bat_number survives as a groupby key — confirm it's present
    assert "at_bat_number" in pa_df.columns, "at_bat_number missing from pa_df"

    ev_str = pa_df["events"].astype("string")
    pa_df["is_so"] = ev_str.str.contains("strikeout", na=False).astype("int8")
    pa_df["is_bb"] = (ev_str == "walk").fillna(False).astype("int8")

    ev = pd.to_numeric(pa_df["launch_speed"], errors="coerce")
    la = pd.to_numeric(pa_df["launch_angle"],  errors="coerce")

    pa_df["is_barrel"]    = _is_barrel(ev, la).astype("int8")
    pa_df["launch_speed"] = ev
    pa_df["launch_angle"] = la

    pa_df = (
        pa_df
        .sort_values("game_date")
        .dropna(subset=["game_date"])
        .reset_index(drop=True)
    )
    pa_df["game_date"] = pa_df["game_date"].astype("datetime64[ns]")

    logger.info(
        "Events loaded: %d pitches, %d PAs (start=%s end=%s)",
        len(pitches_df), len(pa_df), start_date, end_date,
    )
    return pitches_df, pa_df


# ---------------------------------------------------------------------------
# Days-rest helpers
# ---------------------------------------------------------------------------

def _compute_days_rest(pa_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    batter_dates  = {k: sorted(g.unique()) for k, g in pa_df.groupby("batter")["game_date"]}
    pitcher_dates = {k: sorted(g.unique()) for k, g in pa_df.groupby("pitcher")["game_date"]}

    def _gap(player_id, game_date, lookup):
        dates = lookup.get(player_id, [])
        prior = [d for d in dates if d < game_date]
        return float((game_date - max(prior)).days) if prior else np.nan

    labels = labels.copy()
    labels["b_days_rest"] = [
        _gap(int(r.batter), r.game_date, batter_dates)
        for r in labels.itertuples(index=False)
    ]
    labels["p_days_rest"] = [
        _gap(int(r.pitcher_id) if pd.notna(r.pitcher_id) else -1,
             r.game_date, pitcher_dates)
        for r in labels.itertuples(index=False)
    ]
    labels["p_is_short_rest"] = (labels["p_days_rest"] <= 3).astype("int8")
    labels["b_days_rest"] = labels["b_days_rest"].fillna(4.0)
    labels["p_days_rest"] = labels["p_days_rest"].fillna(4.0)
    return labels


# ---------------------------------------------------------------------------
# Pitcher workload + opener flag
# ---------------------------------------------------------------------------

def _compute_pitcher_workload(
    pitches_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: pitcher, game_date
) -> pd.DataFrame:
    """
    For each (pitcher, game_date), compute workload features derived from
    their most recent prior outing and rolling 30-day starts.

    Uses pitches_df (one row per pitch) for accurate pitch counts.

    Returns DataFrame with columns:
        pitcher, game_date,
        p_pitches_last_start   - total pitches thrown in last outing
        p_ip_last_start        - innings proxy (PA faced / 3.0) in last outing
        p_is_opener            - 1 if avg IP/start < 3.0 over last 30d (opener/bulk arm)
        p_workload_score       - pitches per inning in last outing (fatigue proxy)
    """
    # Count pitches per (pitcher, game_pk, game_date) from pitch-level data
    pitch_counts = (
        pitches_df.groupby(["pitcher", "game_pk", "game_date"])
        .size()
        .reset_index(name="total_pitches")
    )

    # PA count per game as innings proxy (pa_faced / 3.0)
    # Use pitches_df grouped by at_bat_number to get unique PAs
    pa_counts = (
        pitches_df.groupby(["pitcher", "game_pk", "game_date"])["at_bat_number"]
        .nunique()
        .reset_index(name="pa_faced")
    )

    game_stats = pitch_counts.merge(pa_counts, on=["pitcher", "game_pk", "game_date"])
    game_stats["ip_proxy"] = game_stats["pa_faced"] / 3.0
    game_stats["game_date"] = pd.to_datetime(game_stats["game_date"])

    need = target_dates[["pitcher", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    pitcher_games = {
        pid: grp.sort_values("game_date")
        for pid, grp in game_stats.groupby("pitcher", sort=False)
    }

    rows = []
    for _, r in need.iterrows():
        pid   = int(r["pitcher"])
        gdate = r["game_date"]
        base  = {
            "pitcher":              pid,
            "game_date":            gdate,
            "p_pitches_last_start": np.nan,
            "p_ip_last_start":      np.nan,
            "p_is_opener":          0,
            "p_workload_score":     np.nan,
        }

        grp = pitcher_games.get(pid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        last = prior.iloc[-1]
        base["p_pitches_last_start"] = float(last["total_pitches"])
        base["p_ip_last_start"]      = float(last["ip_proxy"])

        # Rolling 30d: is this pitcher an opener/bulk arm?
        w30 = prior[prior["game_date"] >= gdate - pd.Timedelta(days=30)]
        if len(w30) >= 2:
            avg_ip = w30["ip_proxy"].mean()
            base["p_is_opener"] = int(avg_ip < 3.0)
        # else: not enough data — default 0 (assume starter)

        # Workload score: pitches per inning last outing (higher = less efficient / more taxed)
        if base["p_ip_last_start"] and base["p_ip_last_start"] > 0:
            base["p_workload_score"] = base["p_pitches_last_start"] / base["p_ip_last_start"]

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ISO (Isolated Power) rolling windows
# ---------------------------------------------------------------------------

def _compute_batter_iso(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date
) -> pd.DataFrame:
    """
    Compute Isolated Power (ISO = extra bases per AB) for each (batter, game_date).
    ISO captures raw power more cleanly than HR rate in small windows because
    it counts doubles and triples too, giving more signal from fewer PA.

    ISO = (2B*1 + 3B*2 + HR*3) / AB   (extra bases above singles)

    Returns DataFrame with columns:
        batter, game_date,
        b_iso_14    - ISO over last 14 days
        b_iso_szn   - ISO season-to-date
        b_iso_career - ISO over all available history (career proxy)
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    ev_str = pa["events"].astype("string")
    pa["is_double"] = (ev_str == "double").fillna(False).astype(int)
    pa["is_triple"] = (ev_str == "triple").fillna(False).astype(int)
    pa["is_hr_"]    = (ev_str == "home_run").fillna(False).astype(int)
    # AB = PA excluding walks, HBP, sac flies (approximate: exclude BB)
    pa["is_ab"]     = (~ev_str.isin(["walk", "hit_by_pitch", "sac_fly",
                                      "sac_bunt", "intent_walk"])).fillna(True).astype(int)
    pa["extra_bases"] = pa["is_double"] * 1 + pa["is_triple"] * 2 + pa["is_hr_"] * 3

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    batter_pa = {pid: grp for pid, grp in pa.groupby("batter", sort=False)}

    MIN_AB_ISO = 10  # minimum AB before reporting ISO (cold-start guard)

    rows = []
    for _, r in need.iterrows():
        bid   = int(r["batter"])
        gdate = r["game_date"]
        base  = {
            "batter":      bid,
            "game_date":   gdate,
            "b_iso_14":    np.nan,
            "b_iso_szn":   np.nan,
            "b_iso_career": np.nan,
        }

        grp = batter_pa.get(bid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        def _iso(subset):
            ab = subset["is_ab"].sum()
            if ab < MIN_AB_ISO:
                return np.nan
            return float(subset["extra_bases"].sum()) / ab

        # 14-day window
        w14 = prior[prior["game_date"] >= gdate - pd.Timedelta(days=14)]
        base["b_iso_14"] = _iso(w14)

        # Season window
        szn_start = pd.Timestamp(gdate.year, 3, 1)
        wszn = prior[prior["game_date"] > szn_start]
        base["b_iso_szn"] = _iso(wszn)

        # Career (all history)
        base["b_iso_career"] = _iso(prior)

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweet spot rate (launch angle 8-32°, EV >= 98 mph)
# ---------------------------------------------------------------------------

def _compute_sweet_spot(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date
) -> pd.DataFrame:
    """
    Sweet spot = batted ball with launch angle 8-32° AND exit velo >= 98 mph.
    This is the tightest "home run corridor" metric — more precise than barrel
    rate (which includes slow grounders at extreme angles) and more predictive
    than EV or LA alone.

    Returns DataFrame with columns:
        batter, game_date,
        b_sweet_spot_rate_14   - % sweet spot contact last 14 days
        b_sweet_spot_rate_szn  - % sweet spot contact season-to-date
        b_sweet_spot_rate_30   - % sweet spot contact last 30 days
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    ev = pd.to_numeric(pa["launch_speed"], errors="coerce")
    la = pd.to_numeric(pa["launch_angle"], errors="coerce")
    pa["is_sweet_spot"] = (
        (la >= 8) & (la <= 32) & (ev >= 98)
    ).fillna(False).astype(int)
    pa["has_contact"] = (ev.notna() & la.notna()).astype(int)

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    batter_pa = {pid: grp for pid, grp in pa.groupby("batter", sort=False)}
    MIN_CONTACT = 8  # minimum batted balls before reporting rate

    rows = []
    for _, r in need.iterrows():
        bid   = int(r["batter"])
        gdate = r["game_date"]
        base  = {
            "batter":                bid,
            "game_date":             gdate,
            "b_sweet_spot_rate_14":  np.nan,
            "b_sweet_spot_rate_30":  np.nan,
            "b_sweet_spot_rate_szn": np.nan,
        }

        grp = batter_pa.get(bid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        def _ss_rate(subset):
            contact = subset["has_contact"].sum()
            if contact < MIN_CONTACT:
                return np.nan
            return float(subset["is_sweet_spot"].sum()) / contact

        w14  = prior[prior["game_date"] >= gdate - pd.Timedelta(days=14)]
        w30  = prior[prior["game_date"] >= gdate - pd.Timedelta(days=30)]
        szn_start = pd.Timestamp(gdate.year, 3, 1)
        wszn = prior[prior["game_date"] > szn_start]

        base["b_sweet_spot_rate_14"]  = _ss_rate(w14)
        base["b_sweet_spot_rate_30"]  = _ss_rate(w30)
        base["b_sweet_spot_rate_szn"] = _ss_rate(wszn)
        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batter 30-day rolling window
# ---------------------------------------------------------------------------

def _compute_batter_30d(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date
) -> pd.DataFrame:
    """
    Compute batter stats over a 30-day rolling window.
    Fills the gap between 14d (noisy) and season (slow to update).
    30d is the sweet spot for capturing real form while smoothing hot/cold streaks.

    Returns DataFrame with columns:
        batter, game_date,
        b_hr_rate_30, b_barrel_rate_30, b_ev_mean_30,
        b_hardhit_rate_30, b_k_rate_30, b_bb_rate_30,
        b_pa_30
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    ev_num = pd.to_numeric(pa["launch_speed"], errors="coerce")
    pa["launch_speed"] = ev_num

    ev_str = pa["events"].astype("string")
    pa["is_hr_"]  = (ev_str == "home_run").fillna(False).astype(int)
    pa["is_so_"]  = ev_str.str.contains("strikeout", na=False).astype(int)
    pa["is_bb_"]  = (ev_str == "walk").fillna(False).astype(int)

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    batter_pa = {pid: grp for pid, grp in pa.groupby("batter", sort=False)}
    MIN_PA = 10

    rows = []
    for _, r in need.iterrows():
        bid   = int(r["batter"])
        gdate = r["game_date"]
        base  = {
            "batter":           bid,
            "game_date":        gdate,
            "b_pa_30":          0,
            "b_hr_rate_30":     np.nan,
            "b_barrel_rate_30": np.nan,
            "b_ev_mean_30":     np.nan,
            "b_hardhit_rate_30":np.nan,
            "b_k_rate_30":      np.nan,
            "b_bb_rate_30":     np.nan,
        }

        grp = batter_pa.get(bid)
        if grp is None:
            rows.append(base)
            continue

        w30 = grp[
            (grp["game_date"] < gdate) &
            (grp["game_date"] >= gdate - pd.Timedelta(days=30))
        ]

        pa_count = len(w30)
        base["b_pa_30"] = pa_count
        if pa_count < MIN_PA:
            rows.append(base)
            continue

        ev = w30["launch_speed"].dropna()
        base["b_hr_rate_30"]      = float(w30["is_hr_"].sum())  / pa_count
        base["b_barrel_rate_30"]  = float(w30["is_barrel"].sum()) / pa_count if "is_barrel" in w30.columns else np.nan
        base["b_ev_mean_30"]      = float(ev.mean()) if len(ev) > 0 else np.nan
        base["b_hardhit_rate_30"] = float((ev >= 95).mean()) if len(ev) > 0 else np.nan
        base["b_k_rate_30"]       = float(w30["is_so_"].sum()) / pa_count
        base["b_bb_rate_30"]      = float(w30["is_bb_"].sum()) / pa_count
        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pitcher command metric (K% - BB%)
# ---------------------------------------------------------------------------

def _compute_pitcher_command(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: pitcher, game_date
) -> pd.DataFrame:
    """
    K% - BB% is the best single process metric for pitcher quality.
    Unlike HR allowed rate it isn't park/defence dependent, and unlike
    ERA it doesn't confound with run support or bullpen.

    Higher = better command. League average is roughly 0.14 (22% K - 8% BB).

    Returns DataFrame with columns:
        pitcher, game_date,
        p_command_30    - K% minus BB% over last 30 days
        p_command_szn   - K% minus BB% season-to-date
        p_kbb_ratio_30  - K/BB ratio last 30d (inf-capped at 10.0)
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    ev_str = pa["events"].astype("string")
    pa["is_so_"] = ev_str.str.contains("strikeout", na=False).astype(int)
    pa["is_bb_"] = (ev_str == "walk").fillna(False).astype(int)

    need = target_dates[["pitcher", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    pitcher_pa = {pid: grp for pid, grp in pa.groupby("pitcher", sort=False)}
    MIN_PA = 15

    rows = []
    for _, r in need.iterrows():
        pid   = int(r["pitcher"])
        gdate = r["game_date"]
        base  = {
            "pitcher":        pid,
            "game_date":      gdate,
            "p_command_30":   np.nan,
            "p_command_szn":  np.nan,
            "p_kbb_ratio_30": np.nan,
        }

        grp = pitcher_pa.get(pid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        def _command(subset):
            n = len(subset)
            if n < MIN_PA:
                return np.nan, np.nan
            k_pct  = float(subset["is_so_"].sum()) / n
            bb_pct = float(subset["is_bb_"].sum()) / n
            bb_cnt = subset["is_bb_"].sum()
            k_cnt  = subset["is_so_"].sum()
            ratio  = float(k_cnt / bb_cnt) if bb_cnt > 0 else 10.0
            return k_pct - bb_pct, min(ratio, 10.0)

        w30 = prior[prior["game_date"] >= gdate - pd.Timedelta(days=30)]
        szn_start = pd.Timestamp(gdate.year, 3, 1)
        wszn = prior[prior["game_date"] > szn_start]

        base["p_command_30"],  base["p_kbb_ratio_30"] = _command(w30)
        base["p_command_szn"], _                       = _command(wszn)
        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Career platoon splits
# ---------------------------------------------------------------------------

def _compute_career_platoon_splits(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date
) -> pd.DataFrame:
    """
    Compute career-level platoon splits for each (batter, game_date).
    Uses all available history prior to game_date as a career proxy.

    Career splits stabilise early-season predictions when 14d rolling
    windows have too few PA to be reliable.

    Returns DataFrame with columns:
        batter, game_date,
        b_hr_rate_career_vsL   - career HR rate vs LHP
        b_hr_rate_career_vsR   - career HR rate vs RHP
        b_iso_career_vsL       - career ISO vs LHP
        b_iso_career_vsR       - career ISO vs RHP
        b_hardhit_career_vsL   - career hard-hit rate vs LHP
        b_hardhit_career_vsR   - career hard-hit rate vs RHP
        b_platoon_hr_edge      - career HR rate advantage vs today's pitcher hand
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    ev_str = pa["events"].astype("string")
    pa["is_double"] = (ev_str == "double").fillna(False).astype(int)
    pa["is_triple"] = (ev_str == "triple").fillna(False).astype(int)
    pa["is_hr_"]    = (ev_str == "home_run").fillna(False).astype(int)
    pa["is_ab"]     = (~ev_str.isin(["walk", "hit_by_pitch", "sac_fly",
                                      "sac_bunt", "intent_walk"])).fillna(True).astype(int)
    pa["extra_bases"] = pa["is_double"] * 1 + pa["is_triple"] * 2 + pa["is_hr_"] * 3

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    # Include pitcher_hand from target_dates if available
    has_hand = "pitcher_hand" in target_dates.columns
    if has_hand:
        need = target_dates[["batter", "game_date", "pitcher_hand"]].drop_duplicates(
            subset=["batter", "game_date"]
        ).copy()

    batter_pa = {pid: grp for pid, grp in pa.groupby("batter", sort=False)}
    MIN_AB = 20

    def _rate(subset, col):
        ab = subset["is_ab"].sum()
        if ab < MIN_AB:
            return np.nan
        return float(subset[col].sum()) / ab

    def _hardhit(subset):
        ev = subset["launch_speed"].dropna()
        if len(ev) < MIN_AB:
            return np.nan
        return float((ev >= 95).mean())

    rows = []
    for _, r in need.iterrows():
        bid        = int(r["batter"])
        gdate      = r["game_date"]
        pitch_hand = r.get("pitcher_hand", None) if has_hand else None

        base = {
            "batter":               bid,
            "game_date":            gdate,
            "b_hr_rate_career_vsL": np.nan,
            "b_hr_rate_career_vsR": np.nan,
            "b_iso_career_vsL":     np.nan,
            "b_iso_career_vsR":     np.nan,
            "b_hardhit_career_vsL": np.nan,
            "b_hardhit_career_vsR": np.nan,
            "b_platoon_hr_edge":    np.nan,
        }

        grp = batter_pa.get(bid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        vs_L = prior[prior["p_throws"] == "L"]
        vs_R = prior[prior["p_throws"] == "R"]

        base["b_hr_rate_career_vsL"] = _rate(vs_L, "is_hr_")
        base["b_hr_rate_career_vsR"] = _rate(vs_R, "is_hr_")
        base["b_iso_career_vsL"]     = _rate(vs_L, "extra_bases")
        base["b_iso_career_vsR"]     = _rate(vs_R, "extra_bases")
        base["b_hardhit_career_vsL"] = _hardhit(vs_L)
        base["b_hardhit_career_vsR"] = _hardhit(vs_R)

        # Platoon edge: today's matchup hand vs opposite
        if pitch_hand in ("L", "R"):
            favoured = base[f"b_hr_rate_career_vs{pitch_hand}"]
            opposite = base[f"b_hr_rate_career_vs{'R' if pitch_hand == 'L' else 'L'}"]
            if pd.notna(favoured) and pd.notna(opposite):
                base["b_platoon_hr_edge"] = favoured - opposite

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parallelised weather fetch
# ---------------------------------------------------------------------------

def fetch_weather_for_games_fast(
    game_pks: list[int],
    home_teams: dict[int, str],
    *,
    force_refresh: bool = False,
    game_dates: dict[int, str] | None = None,
    max_workers: int = _WEATHER_MAX_WORKERS,
) -> pd.DataFrame:
    """
    Parallel replacement for weather.fetch_weather_for_games.
    Only un-cached games are fetched concurrently; cached reads stay in the
    main thread.
    """
    import json
    from src.data_sources.weather import (
        fetch_game_weather, _cache_path, _cache_is_stale,
        _neutral_weather, INDOOR_PARKS,
    )

    unique_pks = list(dict.fromkeys(game_pks))
    game_dates = game_dates or {}

    # If no games, return an empty DataFrame with the expected columns
    if not unique_pks:
        return pd.DataFrame(columns=[
            "game_pk", "temp_f", "wind_speed_mph", "wind_hr_factor",
            "wind_hr_impact", "is_indoor", "temp_above_75", "temp_above_85",
            "wind_out_strong", "wind_in_strong"
        ])

    cached_pks = [
        gp for gp in unique_pks
        if not _cache_is_stale(_cache_path(gp), game_date=game_dates.get(gp))
        and not force_refresh
    ]
    fetch_pks = [gp for gp in unique_pks if gp not in set(cached_pks)]

    logger.info(
        "Weather fetch (parallel): %d games (%d cached, %d to fetch)",
        len(unique_pks), len(cached_pks), len(fetch_pks),
    )

    results: dict[int, dict] = {}

    for gp in cached_pks:
        with open(_cache_path(gp)) as f:
            results[gp] = json.load(f)

    if fetch_pks:
        failed: list[int] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_pk = {
                pool.submit(
                    fetch_game_weather,
                    gp,
                    force_refresh=True,
                    game_date=game_dates.get(gp),
                ): gp
                for gp in fetch_pks
            }
            done = 0
            for future in as_completed(future_to_pk):
                gp = future_to_pk[future]
                try:
                    results[gp] = future.result()
                    if results[gp].get("_error"):
                        failed.append(gp)
                except Exception as e:
                    logger.error("Weather future failed game_pk=%d: %s", gp, e)
                    results[gp] = _neutral_weather(gp, error=str(e))
                    failed.append(gp)
                done += 1
                if done % 100 == 0:
                    logger.info("Weather: %d/%d fetched ...", done, len(fetch_pks))

        logger.info("Weather: %d/%d fetched", len(fetch_pks), len(fetch_pks))
        if failed:
            logger.warning(
                "Weather unavailable for %d game(s) — neutral values used: %s",
                len(failed), failed[:10],
            )

    rows = []
    for gp in unique_pks:
        r = results[gp].copy()
        home_team = home_teams.get(gp, "")
        r["is_indoor"] = int(home_team in INDOOR_PARKS)
        if r["is_indoor"]:
            r["wind_speed_mph"] = 0.0
            r["wind_hr_factor"] = 0.0
            r["wind_hr_impact"] = 0.0
            r["temp_f"]         = 72.0
        rows.append(r)

    df = pd.DataFrame(rows)
    df["temp_above_75"]   = (df["temp_f"] > 75).astype("int8")
    df["temp_above_85"]   = (df["temp_f"] > 85).astype("int8")
    df["wind_out_strong"] = ((df["wind_hr_factor"] > 0) & (df["wind_speed_mph"] >= 10)).astype("int8")
    df["wind_in_strong"]  = ((df["wind_hr_factor"] < 0) & (df["wind_speed_mph"] >= 10)).astype("int8")

    return df[[
        "game_pk", "temp_f", "wind_speed_mph", "wind_hr_factor",
        "wind_hr_impact", "is_indoor", "temp_above_75", "temp_above_85",
        "wind_out_strong", "wind_in_strong",
    ]]

# ---------------------------------------------------------------------------
# Relief PA pct
# ---------------------------------------------------------------------------

def _compute_relief_pa_pct(pa_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of each batter's PAs in a game against non-starters.

    FIX 1 (part b): This function now correctly receives the full pitch-level
    pa_df (which retains one row per PA with the actual pitcher column).
    Previously it was called with target_pa — a filtered slice — which had
    already lost at_bat_number in some code paths.

    Must be called AFTER enrich_labels_with_roster so starter_pitcher_id exists.
    """
    labels = labels.copy()
    labels["game_pk"] = labels["game_pk"].astype(int)
    labels["batter"]  = labels["batter"].astype(int)

    pa = pa_df.copy()
    pa["game_pk"] = pa["game_pk"].astype(int)
    pa["batter"]  = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)

    if "starter_pitcher_id" in labels.columns:
        batter_starter = (
            labels[["game_pk", "batter", "starter_pitcher_id"]]
            .dropna(subset=["starter_pitcher_id"])
            .copy()
        )
        batter_starter["starter_pitcher_id"] = batter_starter["starter_pitcher_id"].astype(int)
        pa = pa.merge(batter_starter, on=["game_pk", "batter"], how="left")
        pa["is_relief_pa"] = np.where(
            pa["starter_pitcher_id"].isna(),
            0,
            (pa["pitcher"] != pa["starter_pitcher_id"]).astype("boolean").fillna(False).astype(int),
        )
    else:
        logger.warning(
            "starter_pitcher_id not found in labels — using first pitcher as game-level starter"
        )
        first_pitcher = (
            pa.sort_values(["game_pk", "at_bat_number"])
            .groupby("game_pk")["pitcher"]
            .first()
            .rename("starter_pitcher")
            .reset_index()
        )
        pa = pa.merge(first_pitcher, on="game_pk", how="left")
        pa["is_relief_pa"] = (
            (pa["pitcher"] != pa["starter_pitcher"])
            .astype("boolean").fillna(False).astype(int)
        )

    relief_pct = (
        pa.groupby(["game_pk", "batter"])
        .agg(relief_pa_pct=("is_relief_pa", "mean"))
        .reset_index()
    )

    if "relief_pa_pct" in labels.columns:
        labels = labels.drop(columns=["relief_pa_pct"])
    labels = labels.merge(relief_pct, on=["game_pk", "batter"], how="left")
    labels["relief_pa_pct"] = labels["relief_pa_pct"].fillna(0.0)
    return labels


# ---------------------------------------------------------------------------
# Pitcher stuff quality (spin rate, extension, movement)
# ---------------------------------------------------------------------------

def _compute_pitcher_stuff(
    pitches_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: pitcher, game_date
) -> pd.DataFrame:
    """
    Compute pitcher stuff quality metrics from pitch-level Statcast data.

    Stuff metrics are process signals that don't depend on outcomes — a
    pitcher with elite spin/movement will suppress HRs even if recent HR
    allowed rate is high due to bad luck.

    Returns DataFrame with columns:
        pitcher, game_date,
        p_spin_rate_fb_30      - avg fastball spin rate last 30d
        p_extension_30         - avg release extension last 30d
        p_pfx_z_fb_30          - avg vertical movement (rise/drop) on FB last 30d
        p_pfx_x_fb_30          - avg horizontal break on FB last 30d
        p_stuff_score_30       - composite: spin × extension proxy
    """
    p = pitches_df.copy()
    p["game_date"] = pd.to_datetime(p["game_date"])

    for col in ("release_spin_rate", "release_extension", "pfx_x", "pfx_z"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")
        else:
            p[col] = np.nan

    is_fb = p["pitch_type"].isin(FASTBALL_TYPES)

    need = target_dates[["pitcher", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])
    pitcher_groups = {pid: grp for pid, grp in p.groupby("pitcher", sort=False)}

    rows = []
    for _, r in need.iterrows():
        pid   = int(r["pitcher"])
        gdate = r["game_date"]
        base  = {
            "pitcher":          pid,
            "game_date":        gdate,
            "p_spin_rate_fb_30": np.nan,
            "p_extension_30":    np.nan,
            "p_pfx_z_fb_30":     np.nan,
            "p_pfx_x_fb_30":     np.nan,
            "p_stuff_score_30":  np.nan,
        }

        grp = pitcher_groups.get(pid)
        if grp is None:
            rows.append(base)
            continue

        w30 = grp[
            (grp["game_date"] < gdate) &
            (grp["game_date"] >= gdate - pd.Timedelta(days=30))
        ]
        if w30.empty:
            rows.append(base)
            continue

        fb30 = w30[w30["pitch_type"].isin(FASTBALL_TYPES)]

        if not fb30.empty:
            base["p_spin_rate_fb_30"] = float(fb30["release_spin_rate"].dropna().mean()) if fb30["release_spin_rate"].notna().any() else np.nan
            base["p_pfx_z_fb_30"]     = float(fb30["pfx_z"].dropna().mean()) if fb30["pfx_z"].notna().any() else np.nan
            base["p_pfx_x_fb_30"]     = float(fb30["pfx_x"].dropna().mean()) if fb30["pfx_x"].notna().any() else np.nan

        if w30["release_extension"].notna().any():
            base["p_extension_30"] = float(w30["release_extension"].dropna().mean())

        # Composite stuff score: normalised spin × extension
        # League avg spin ~2300 rpm, extension ~6.0 ft
        spin = base["p_spin_rate_fb_30"]
        ext  = base["p_extension_30"]
        if pd.notna(spin) and pd.notna(ext):
            base["p_stuff_score_30"] = (spin / 2300.0) * (ext / 6.0)

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batter streaks & consistency (hot/cold flags, HR recency)
# ---------------------------------------------------------------------------

def _compute_batter_streaks(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date
) -> pd.DataFrame:
    """
    Compute streak and consistency features for each batter.

    Hot/cold flags are important because the model currently treats a batter
    hitting .350 in the last week the same as one hitting .150. HR recency
    captures momentum that rolling averages smooth over.

    Returns DataFrame with columns:
        batter, game_date,
        b_hr_last_7d          - HRs hit in last 7 days (raw count)
        b_games_since_last_hr - games played since last HR (recency)
        b_ev_hot_flag         - 1 if avg EV last 7d > avg EV last 30d by 2+ mph
        b_contact_hot_flag    - 1 if hard-hit rate last 7d > last 30d by 5%+
        b_hr_streak           - consecutive games with HR (current streak)
        b_avg_ev_7d           - avg EV last 7 days
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["launch_speed"] = pd.to_numeric(pa["launch_speed"], errors="coerce")
    ev_str = pa["events"].astype("string")
    pa["is_hr_"] = (ev_str == "home_run").fillna(False).astype(int)

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    need["game_date"] = pd.to_datetime(need["game_date"])
    batter_pa = {pid: grp.sort_values("game_date") for pid, grp in pa.groupby("batter", sort=False)}

    rows = []
    for _, r in need.iterrows():
        bid   = int(r["batter"])
        gdate = r["game_date"]
        base  = {
            "batter":                bid,
            "game_date":             gdate,
            "b_hr_last_7d":          0,
            "b_games_since_last_hr": np.nan,
            "b_ev_hot_flag":         0,
            "b_contact_hot_flag":    0,
            "b_hr_streak":           0,
            "b_avg_ev_7d":           np.nan,
        }

        grp = batter_pa.get(bid)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        w7  = prior[prior["game_date"] >= gdate - pd.Timedelta(days=7)]
        w30 = prior[prior["game_date"] >= gdate - pd.Timedelta(days=30)]

        # HR count last 7d
        base["b_hr_last_7d"] = int(w7["is_hr_"].sum())

        # EV stats
        ev7  = w7["launch_speed"].dropna()
        ev30 = w30["launch_speed"].dropna()
        if len(ev7) >= 3:
            base["b_avg_ev_7d"] = float(ev7.mean())
        if len(ev7) >= 3 and len(ev30) >= 8:
            ev7_mean  = float(ev7.mean())
            ev30_mean = float(ev30.mean())
            base["b_ev_hot_flag"]      = int(ev7_mean - ev30_mean >= 2.0)
            hh7  = float((ev7  >= 95).mean())
            hh30 = float((ev30 >= 95).mean())
            base["b_contact_hot_flag"] = int(hh7 - hh30 >= 0.05)

        # Games since last HR
        hr_games = prior[prior["is_hr_"] == 1]["game_date"]
        if not hr_games.empty:
            last_hr_date = hr_games.max()
            # Count distinct game dates between last HR and today
            games_after = prior[prior["game_date"] > last_hr_date]["game_date"].nunique()
            base["b_games_since_last_hr"] = float(games_after)

        # Current HR streak (consecutive games with HR, going back)
        game_dates = sorted(prior["game_date"].unique(), reverse=True)
        streak = 0
        for gd in game_dates:
            day_hrs = prior[prior["game_date"] == gd]["is_hr_"].sum()
            if day_hrs > 0:
                streak += 1
            else:
                break
        base["b_hr_streak"] = streak

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recent lineup context (team HR rate, lineup power around batter)
# ---------------------------------------------------------------------------

def _compute_lineup_context(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,  # columns: batter, game_date, game_pk
) -> pd.DataFrame:
    """
    Compute team-level and lineup-slot context features.

    A batter in a powerful lineup gets better pitches to hit. Team HR rate
    captures this lineup protection effect better than individual stats alone.

    Returns DataFrame with columns:
        batter, game_date,
        t_hr_rate_14       - team HR rate per PA last 14 days
        t_hr_rate_szn      - team HR rate per PA season-to-date
        t_hardhit_rate_14  - team hard-hit rate last 14 days
        t_ev_mean_14       - team avg exit velo last 14 days
    """
    pa = pa_df.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["launch_speed"] = pd.to_numeric(pa["launch_speed"], errors="coerce")
    ev_str = pa["events"].astype("string")
    pa["is_hr_"] = (ev_str == "home_run").fillna(False).astype(int)

    # Need batter→team mapping — use home_team + game_pk to derive it
    # We'll compute from pa_df game context: batter's team = mode of home_team
    # when they bat at home, otherwise away. Use simple approach: look up
    # from target_dates game_pk → home_team, then derive batter team from labels.
    has_game_pk = "game_pk" in target_dates.columns

    need = target_dates[["batter", "game_date"]].drop_duplicates().copy()
    if has_game_pk:
        need = target_dates[["batter", "game_date", "game_pk"]].drop_duplicates(
            subset=["batter", "game_date"]
        ).copy()
    need["game_date"] = pd.to_datetime(need["game_date"])

    # Build batter→team lookup from pa_df (mode of home_team for home batters)
    # Simpler: group by (home_team, game_date) to get all batters per team per game
    # Then for each batter, look at their team's aggregate stats
    # We'll approximate: for each batter, compute stats for all batters
    # who played on the same home_team in the same date range

    # Build batter→team from pa_df
    batter_team = (
        pa.dropna(subset=["home_team"])
        .groupby("batter")["home_team"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )

    # Group pa by team
    pa["team"] = pa["batter"].map(batter_team)
    team_pa = {t: g for t, g in pa.dropna(subset=["team"]).groupby("team", sort=False)}

    rows = []
    for _, r in need.iterrows():
        bid   = int(r["batter"])
        gdate = r["game_date"]
        team  = batter_team.get(bid)
        base  = {
            "batter":           bid,
            "game_date":        gdate,
            "t_hr_rate_14":     np.nan,
            "t_hr_rate_szn":    np.nan,
            "t_hardhit_rate_14":np.nan,
            "t_ev_mean_14":     np.nan,
        }

        if team is None:
            rows.append(base)
            continue

        grp = team_pa.get(team)
        if grp is None:
            rows.append(base)
            continue

        prior = grp[grp["game_date"] < gdate]
        if prior.empty:
            rows.append(base)
            continue

        def _team_stats(subset):
            n = len(subset)
            if n < 10:
                return np.nan, np.nan, np.nan
            hr_rate = float(subset["is_hr_"].sum()) / n
            ev = subset["launch_speed"].dropna()
            hh_rate = float((ev >= 95).mean()) if len(ev) >= 5 else np.nan
            ev_mean = float(ev.mean()) if len(ev) >= 5 else np.nan
            return hr_rate, hh_rate, ev_mean

        w14 = prior[prior["game_date"] >= gdate - pd.Timedelta(days=14)]
        szn_start = pd.Timestamp(gdate.year, 3, 1)
        wszn = prior[prior["game_date"] > szn_start]

        hr14, hh14, ev14 = _team_stats(w14)
        hrszn, _, _      = _team_stats(wszn)

        base["t_hr_rate_14"]      = hr14
        base["t_hr_rate_szn"]     = hrszn
        base["t_hardhit_rate_14"] = hh14
        base["t_ev_mean_14"]      = ev14
        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ballpark HR tendency by direction (pull vs oppo)
# ---------------------------------------------------------------------------

# Static park HR tendency by spray direction.
# Values represent the ratio of HR probability for pulled balls vs oppo balls
# in each park. Derived from Statcast hit data patterns.
# >1.0 = pull-friendly, <1.0 = more balanced/oppo-friendly
_PARK_PULL_FACTOR: dict[str, float] = {
    "LAD": 1.35, "NYY": 1.28, "COL": 1.15, "CIN": 1.20, "PHI": 1.18,
    "TOR": 1.22, "BAL": 1.19, "MIN": 1.16, "ATL": 1.12, "DET": 1.14,
    "MIL": 1.10, "ARI": 1.08, "NYM": 1.06, "HOU": 1.05, "BOS": 1.04,
    "CLE": 1.02, "CHC": 1.00, "WSN": 0.98, "STL": 0.97, "KCR": 0.96,
    "SFG": 0.95, "MIA": 0.94, "SDP": 0.93, "LAA": 0.92, "SEA": 0.91,
    "OAK": 0.90, "TEX": 0.89, "CWS": 0.88, "PIT": 0.85, "TBR": 1.02,
    "ATH": 1.10,
}


def _compute_park_direction_factor(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add park pull-factor and pull×park interaction to features_df.
    Uses home_team column + _PARK_PULL_FACTOR lookup.
    Also computes pull_air_x_park_direction: pull rate × park pull factor.
    """
    df = features_df.copy()
    if "home_team" in df.columns:
        df["park_pull_factor"] = df["home_team"].map(_PARK_PULL_FACTOR).fillna(1.0)
        if "b_pull_air_rate_szn" in df.columns:
            pull = df["b_pull_air_rate_szn"].fillna(df.get("b_pull_air_rate_14", pd.Series(0.0, index=df.index)))
            df["pull_x_park_direction"] = pull * df["park_pull_factor"]
    else:
        df["park_pull_factor"] = 1.0
        df["pull_x_park_direction"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def _add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _edge(a: str, b: str) -> pd.Series:
        # Return NaN series if either column is absent (e.g. pitcher_need was
        # empty and no pitcher stat columns were added to features_df).
        if a not in df.columns or b not in df.columns:
            missing = [c for c in (a, b) if c not in df.columns]
            logger.warning(
                "_add_edge_features: skipping edge %s - %s, missing columns: %s",
                a, b, missing,
            )
            return pd.Series(np.nan, index=df.index)
        return df[a] - df[b]

    def _col(c: str) -> pd.Series:
        """Return column or NaN series if absent."""
        if c not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[c]

    df["ev_edge_14_30"]             = _edge("b_ev_mean_14",       "p_ev_allowed_mean_30")
    df["hardhit_edge_14_30"]        = _edge("b_hardhit_rate_14",  "p_hardhit_allowed_rate_30")
    df["fb_edge_14_30"]             = _edge("b_fb_rate_14",       "p_fb_allowed_rate_30")
    df["barrel_edge_14_30"]         = _edge("b_barrel_rate_14",   "p_barrel_allowed_rate_30")
    df["hr_rate_edge_14_30"]        = _edge("b_hr_rate_14",       "p_hr_allowed_rate_30")
    df["k_rate_edge_14_30"]         = _edge("b_k_rate_14",        "p_k_rate_30")
    df["bb_rate_edge_14_30"]        = _edge("b_bb_rate_14",       "p_bb_rate_30")

    df["k_rate_interaction_14_30"]  = _col("b_k_rate_14")  * _col("p_k_rate_30")
    df["bb_rate_interaction_14_30"] = _col("b_bb_rate_14") * _col("p_bb_rate_30")
    df["contact_pressure_14_30"]    = (1 - _col("b_k_rate_14")) * (1 - _col("p_k_rate_30"))
    df["discipline_balance_14_30"]  = (
        (_col("b_bb_rate_14") - _col("b_k_rate_14")) -
        (_col("p_bb_rate_30") - _col("p_k_rate_30"))
    )

    for hand in ("L", "R"):
        b_hr   = f"b_hr_rate_14_vs{hand}"
        p_hr   = f"p_hr_allowed_rate_30_vs{hand}"
        b_hard = f"b_hardhit_rate_14_vs{hand}"
        p_hard = f"p_hardhit_allowed_rate_30_vs{hand}"
        b_bar  = f"b_barrel_rate_14_vs{hand}"
        p_bar  = f"p_barrel_allowed_rate_30_vs{hand}"
        if all(c in df.columns for c in [b_hr, p_hr, b_hard, p_hard, b_bar, p_bar]):
            df[f"hr_rate_edge_14_30_vs{hand}"]  = _edge(b_hr,   p_hr)
            df[f"hardhit_edge_14_30_vs{hand}"]  = _edge(b_hard, p_hard)
            df[f"barrel_edge_14_30_vs{hand}"]   = _edge(b_bar,  p_bar)

    if "wind_hr_impact" in df.columns:
        df["hardhit_x_wind"] = _col("b_hardhit_rate_14") * df["wind_hr_impact"]
        df["barrel_x_wind"]  = _col("b_barrel_rate_14")  * df["wind_hr_impact"]

    # Pulled air ball rate × park factor
    if "b_pull_air_rate_szn" in df.columns and "park_factor_hr" in df.columns:
        pull_col = _col("b_pull_air_rate_szn").fillna(_col("b_pull_air_rate_14"))
        df["pull_air_x_park"] = pull_col * df["park_factor_hr"]

    # Best pitch-type matchup score (max across FB / breaking / offspeed)
    matchup_cols = ["matchup_fb_30", "matchup_brk_30", "matchup_os_30"]
    avail_matchup = [c for c in matchup_cols if c in df.columns]
    if avail_matchup:
        df["matchup_best_30"] = df[avail_matchup].max(axis=1)

    # Pitcher fatigue interactions
    # High pitch count + short rest = tired pitcher → more hittable
    if "p_pitches_last_start" in df.columns and "p_days_rest" in df.columns:
        # Fatigue index: pitches thrown scaled by how little rest they got
        # Short rest (3d) with high pitch count (110+) maximises this
        df["p_fatigue_index"] = (
            _col("p_pitches_last_start") * (1.0 / _col("p_days_rest").clip(lower=1))
        )
    if "p_workload_score" in df.columns and "p_days_rest" in df.columns:
        # Inefficiency × rest: was the pitcher laboured AND came back quickly?
        df["p_workload_x_rest"] = _col("p_workload_score") * (1.0 / _col("p_days_rest").clip(lower=1))

    # Opener flag × batter hard-hit rate: opener means more reliever exposure
    # A hard-hitting batter benefits more from facing a bullpen-heavy game
    if "p_is_opener" in df.columns:
        df["opener_x_hardhit"] = _col("p_is_opener") * _col("b_hardhit_rate_14")
        df["opener_x_barrel"]  = _col("p_is_opener") * _col("b_barrel_rate_14")

    # ISO interactions
    # ISO × park factor: power hitters benefit more in hitter-friendly parks
    if "b_iso_szn" in df.columns and "park_factor_hr" in df.columns:
        iso = _col("b_iso_szn").fillna(_col("b_iso_14")).fillna(_col("b_iso_career"))
        df["iso_x_park"] = iso * _col("park_factor_hr")

    # ISO × wind: power hitters benefit more from wind blowing out
    if "b_iso_szn" in df.columns and "wind_hr_impact" in df.columns:
        iso = _col("b_iso_szn").fillna(_col("b_iso_14")).fillna(_col("b_iso_career"))
        df["iso_x_wind"] = iso * _col("wind_hr_impact")

    # Career platoon edge × season ISO: batter with strong platoon advantage + power
    if "b_platoon_hr_edge" in df.columns and "b_iso_szn" in df.columns:
        iso = _col("b_iso_szn").fillna(_col("b_iso_career"))
        df["platoon_edge_x_iso"] = _col("b_platoon_hr_edge") * iso

    # Blend career + recent platoon HR rate (weighted toward recent when sample exists)
    # If batter has enough recent PA use 14d, otherwise fall back to career
    for hand in ("L", "R"):
        recent = f"b_hr_rate_14_vs{hand}"
        career = f"b_hr_rate_career_vs{hand}"
        blend  = f"b_hr_rate_blend_vs{hand}"
        if recent in df.columns and career in df.columns:
            df[blend] = np.where(
                df[recent].notna(),
                df[recent] * 0.6 + df[career].fillna(df[recent]) * 0.4,
                df[career],
            )

    # Sweet spot interactions
    if "b_sweet_spot_rate_szn" in df.columns:
        ss = _col("b_sweet_spot_rate_szn").fillna(_col("b_sweet_spot_rate_30")).fillna(_col("b_sweet_spot_rate_14"))
        if "park_factor_hr" in df.columns:
            df["sweet_spot_x_park"] = ss * _col("park_factor_hr")
        if "wind_hr_impact" in df.columns:
            df["sweet_spot_x_wind"] = ss * _col("wind_hr_impact")
        # Sweet spot vs pitcher command — good contact vs poor command = high HR risk
        if "p_command_30" in df.columns:
            df["sweet_spot_x_poor_command"] = ss * (1.0 - _col("p_command_30").clip(lower=0, upper=1))

    # Pitcher command edges
    if "p_command_30" in df.columns:
        # Low command pitcher facing high-barrel batter = danger zone
        df["barrel_x_poor_command"] = _col("b_barrel_rate_14") * (1.0 - _col("p_command_30").clip(lower=0, upper=1))
        # 30d batter HR rate edge over pitcher command
        if "b_hr_rate_30" in df.columns:
            df["hr30_x_poor_command"] = _col("b_hr_rate_30") * (1.0 - _col("p_command_30").clip(lower=0, upper=1))

    # 30d batter window edges vs 30d pitcher
    if "b_hr_rate_30" in df.columns:
        df["hr_rate_edge_30_30"]     = _edge("b_hr_rate_30",      "p_hr_allowed_rate_30")
        df["hardhit_edge_30_30"]     = _edge("b_hardhit_rate_30", "p_hardhit_allowed_rate_30")
        df["barrel_edge_30_30"]      = _edge("b_barrel_rate_30",  "p_barrel_allowed_rate_30")

    # Pitcher stuff interactions
    # Low spin + low extension = hittable pitcher → HR risk up
    if "p_stuff_score_30" in df.columns:
        # Stuff score below league avg (1.0) × batter barrel rate = danger
        df["barrel_x_weak_stuff"] = (
            _col("b_barrel_rate_14") * (2.0 - _col("p_stuff_score_30").clip(lower=0.5, upper=2.0))
        )
        df["sweet_spot_x_weak_stuff"] = (
            _col("b_sweet_spot_rate_szn").fillna(_col("b_sweet_spot_rate_14")) *
            (2.0 - _col("p_stuff_score_30").clip(lower=0.5, upper=2.0))
        )

    # Batter streak interactions
    if "b_ev_hot_flag" in df.columns:
        df["hot_x_park"]    = _col("b_ev_hot_flag") * _col("park_factor_hr")
        df["hot_x_iso"]     = _col("b_ev_hot_flag") * _col("b_iso_szn").fillna(_col("b_iso_career"))
    if "b_hr_last_7d" in df.columns and "park_factor_hr" in df.columns:
        df["hr7d_x_park"]   = _col("b_hr_last_7d") * _col("park_factor_hr")

    # Team HR rate × batter quality (lineup protection)
    if "t_hr_rate_14" in df.columns:
        df["team_hr_x_batter_barrel"] = _col("t_hr_rate_14") * _col("b_barrel_rate_14")
        df["team_hr_x_sweet_spot"]    = _col("t_hr_rate_14") * _col("b_sweet_spot_rate_szn").fillna(
            _col("b_sweet_spot_rate_14")
        )

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features_for_range(start_date: str, end_date: str) -> FeaturesBuildResult:
    start_dt = _to_date(start_date)
    end_dt   = _to_date(end_date)
    history_start = _date_minus_days(start_dt, 60)

    logger.info(
        "build_features_for_range: %s -> %s (history from %s)",
        start_date, end_date, history_start.date(),
    )

    logger.info("Loading events %s -> %s ...", history_start.date(), end_dt.date())
    pitches_df, pa_df = _load_and_clean_events(
        start_date=history_start.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )

    game_pk_to_home = (
        pa_df.dropna(subset=["home_team"])
        .drop_duplicates(subset=["game_pk"])
        .set_index("game_pk")["home_team"]
        .to_dict()
    )

    batter_team_lookup: dict[int, str] = {}

    target_pa = pa_df[pa_df["game_date"].between(start_dt, end_dt)]
    labels = build_batter_game_labels(target_pa)
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    logger.info(
        "Labels built: %d batter-game rows, HR rate=%.4f",
        len(labels), labels["hr_hit"].mean() if len(labels) > 0 else float("nan"),
    )

    # Guard: empty date ranges (e.g. Oct 1 single-day stub with no games)
    # Write an empty parquet so the builder skips this chunk on resume.
    if labels.empty:
        logger.warning(
            "No label rows for %s -> %s — writing empty parquet and skipping.",
            start_date, end_date,
        )
        out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
        pd.DataFrame().to_parquet(out_path, index=False)
        return FeaturesBuildResult(features_df=pd.DataFrame(), output_path=out_path)

    # FIX 4: Consolidate pitcher identity to a single column called
    # `pitcher_id` used consistently throughout the pipeline.
    # starter_id (earliest at_bat_number) is preferred; pitcher_mode is the
    # fallback.  The column is renamed to `pitcher` only at the very end,
    # once all merges that key on it are complete.
    labels["pitcher_id"] = (
        pd.to_numeric(labels["starter_id"], errors="coerce")
        .fillna(pd.to_numeric(labels["pitcher_mode"], errors="coerce"))
        .astype("Int64")
    )

    pitcher_hand_lookup = (
        pa_df.dropna(subset=["pitcher", "p_throws"])
        .groupby("pitcher")["p_throws"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )
    batter_hand_lookup = (
        pa_df.dropna(subset=["batter", "stand"])
        .groupby("batter")["stand"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )

    labels["pitcher_hand"] = labels["pitcher_id"].map(
        lambda x: pitcher_hand_lookup.get(int(x)) if pd.notna(x) else None
    )
    labels["batter_hand"] = labels["batter"].map(batter_hand_lookup)

    logger.info("Computing days rest ...")
    labels = _compute_days_rest(pa_df, labels)

    # ------------------------------------------------------------------
    # Roster enrichment
    # ------------------------------------------------------------------
    target_game_pks = labels["game_pk"].dropna().astype(int).unique().tolist()
    logger.info("Fetching rosters/batting orders for %d games ...", len(target_game_pks))
    starters_df, batting_df = fetch_rosters_for_games(target_game_pks)

    if not batting_df.empty and "team_side" in batting_df.columns:
        home_games_per_batter = (
            pa_df[pa_df["game_date"].between(start_dt, end_dt)]
            .groupby("batter")["home_team"]
            .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) > 0 else None)
            .to_dict()
        )
        bd = batting_df.copy()
        bd["game_pk"] = bd["game_pk"].astype(int)
        bd["batter"]  = bd["batter"].astype(int)
        for row in bd.itertuples(index=False):
            home_abbr = game_pk_to_home.get(row.game_pk)
            if home_abbr is None:
                continue
            if row.team_side == "home":
                batter_team_lookup[row.batter] = home_abbr
            else:
                t = home_games_per_batter.get(row.batter)
                if t and t != home_abbr:
                    batter_team_lookup[row.batter] = t

    logger.debug("batter_team_lookup populated for %d batters", len(batter_team_lookup))

    labels["batter_team"] = labels["batter"].map(batter_team_lookup)
    labels = enrich_labels_with_roster(labels, starters_df, batting_df, game_pk_to_home)

    # FIX 1 (part b): pass full pa_df (not target_pa) so at_bat_number is
    # present and the starter_pitcher_id join works correctly.
    labels = _compute_relief_pa_pct(pa_df, labels)

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    game_dates_map = (
        labels.drop_duplicates("game_pk")
        .set_index("game_pk")["game_date"]
        .apply(lambda d: str(pd.Timestamp(d).date()))
        .to_dict()
    )
    weather_df = fetch_weather_for_games_fast(
        target_game_pks,
        game_pk_to_home,
        game_dates=game_dates_map,
    )
    weather_df["game_pk"] = weather_df["game_pk"].astype(int)

    # ------------------------------------------------------------------
    # Batter windows
    # ------------------------------------------------------------------
    n_b = labels[["batter", "game_date"]].drop_duplicates().shape[0]
    logger.info("Precomputing batter windows for %d (batter, date) pairs ...", n_b)
    batter_stats = precompute_batter_windows_fast(
        pa_df,
        labels[["batter", "game_date", "pitcher_hand"]],
        batter_team_lookup=batter_team_lookup,
        game_pk_home_lookup=game_pk_to_home,
        label_game_pks=labels[["batter", "game_date", "game_pk"]],
    )

    # ------------------------------------------------------------------
    # Pitcher windows
    # ------------------------------------------------------------------
    # FIX 4: use starter_pitcher_id when available (set by
    # enrich_labels_with_roster), fall back to pitcher_id.  A single
    # merge_pitcher_col is used for ALL downstream merges — no more
    # ambiguity between starter_pitcher_id and pitcher_id.
    if "starter_pitcher_id" in labels.columns:
        merge_pitcher_col = "starter_pitcher_id"
    else:
        merge_pitcher_col = "pitcher_id"

    pitcher_need = (
        labels[[merge_pitcher_col, "game_date", "batter_hand"]]
        .rename(columns={merge_pitcher_col: "pitcher"})
        .dropna(subset=["pitcher"])
        .copy()
    )
    pitcher_need["pitcher"] = pitcher_need["pitcher"].astype(int)
    n_p = pitcher_need[["pitcher", "game_date"]].drop_duplicates().shape[0]

    logger.info("Precomputing pitcher PA windows for %d (pitcher, date) pairs ...", n_p)
    pitcher_stats = precompute_pitcher_windows_fast(pa_df, pitcher_need)

    logger.info("Precomputing pitcher velo windows for %d (pitcher, date) pairs ...", n_p)
    pitcher_velo = precompute_pitcher_velo_fast(pitches_df, pitcher_need)

    # ------------------------------------------------------------------
    # Pitcher workload + opener flag
    # ------------------------------------------------------------------
    logger.info("Computing pitcher workload + opener flags ...")
    pitcher_workload = _compute_pitcher_workload(pitches_df, pitcher_need)

    # ------------------------------------------------------------------
    # Batter ISO windows
    # ------------------------------------------------------------------
    logger.info("Computing batter ISO windows ...")
    batter_iso = _compute_batter_iso(
        pa_df,
        labels[["batter", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Career platoon splits
    # ------------------------------------------------------------------
    logger.info("Computing career platoon splits ...")
    career_platoon = _compute_career_platoon_splits(
        pa_df,
        labels[["batter", "game_date", "pitcher_hand"]],
    )

    # ------------------------------------------------------------------
    # Sweet spot rate
    # ------------------------------------------------------------------
    logger.info("Computing sweet spot rates ...")
    sweet_spot_stats = _compute_sweet_spot(
        pa_df,
        labels[["batter", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Batter 30-day window
    # ------------------------------------------------------------------
    logger.info("Computing batter 30-day windows ...")
    batter_30d = _compute_batter_30d(
        pa_df,
        labels[["batter", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Pitcher command metric (K% - BB%)
    # ------------------------------------------------------------------
    logger.info("Computing pitcher command metrics ...")
    pitcher_command = _compute_pitcher_command(
        pa_df,
        pitcher_need[["pitcher", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Pitcher stuff quality (spin, extension, movement)
    # ------------------------------------------------------------------
    logger.info("Computing pitcher stuff quality ...")
    pitcher_stuff = _compute_pitcher_stuff(pitches_df, pitcher_need)

    # ------------------------------------------------------------------
    # Batter streaks & consistency
    # ------------------------------------------------------------------
    logger.info("Computing batter streaks ...")
    batter_streaks = _compute_batter_streaks(
        pa_df,
        labels[["batter", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Lineup context (team HR rate)
    # ------------------------------------------------------------------
    logger.info("Computing lineup context ...")
    lineup_context = _compute_lineup_context(
        pa_df,
        labels[["batter", "game_date", "game_pk"]],
    )

    # ------------------------------------------------------------------
    # Pulled air ball rate
    # ------------------------------------------------------------------
    logger.info("Precomputing pulled air ball rates ...")
    pull_stats = precompute_batter_pull_fast(
        pitches_df,
        labels[["batter", "game_date"]],
    )

    # ------------------------------------------------------------------
    # Pitch-type matchup features
    # ------------------------------------------------------------------
    logger.info("Precomputing pitch-type matchup features ...")
    pitch_matchup_need = (
        labels[[merge_pitcher_col, "batter", "game_date"]]
        .rename(columns={merge_pitcher_col: "pitcher"})
        .dropna(subset=["pitcher"])
        .copy()
    )
    pitch_matchup_need["pitcher"] = pitch_matchup_need["pitcher"].astype(int)
    pitch_matchup = precompute_pitch_matchup_fast(pitches_df, pitch_matchup_need)

    # ------------------------------------------------------------------
    # Merge everything
    # ------------------------------------------------------------------
    features_df = (
        labels
        .merge(batter_stats,  on=["batter", "game_date"], how="left")
        .merge(
            pitcher_stats.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(
            pitcher_velo.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(
            pitcher_workload.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(
            pitcher_command.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(
            pitcher_stuff.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(batter_iso,      on=["batter", "game_date"], how="left")
        .merge(career_platoon,  on=["batter", "game_date"], how="left")
        .merge(sweet_spot_stats, on=["batter", "game_date"], how="left")
        .merge(batter_30d,      on=["batter", "game_date"], how="left")
        .merge(batter_streaks,  on=["batter", "game_date"], how="left")
        .merge(lineup_context,  on=["batter", "game_date"], how="left")
        .merge(pull_stats,      on=["batter", "game_date"], how="left")
        .merge(
            pitch_matchup.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "batter", "game_date"], how="left",
        )
        .merge(weather_df, on="game_pk", how="left")
    )

    # Park factor
    features_df["home_team"] = features_df["game_pk"].map(game_pk_to_home)
    for _str_col in ("batter_team", "home_team", "pitcher_hand", "batter_hand"):
        if _str_col in features_df.columns:
            features_df[_str_col] = features_df[_str_col].apply(
                lambda v: None if (
                    v is None
                    or (isinstance(v, float) and _math.isnan(v))
                    or str(v) in ("nan", "None", "<NA>", "0.0", "0")
                ) else str(v)
            )

    # Handle park factor for empty DataFrame
    if not features_df.empty:
        season_year = int(pd.to_datetime(features_df["game_date"]).dt.year.mode().iloc[0])
        pf_map = get_park_factors(season=season_year)
        features_df["park_factor_hr"] = (
            features_df["home_team"].map(pf_map).fillna(DEFAULT_PARK_FACTOR / 100.0)
        )
    else:
        # Add column with NaN to maintain schema consistency
        features_df["park_factor_hr"] = np.nan

    features_df = _add_edge_features(features_df)
    features_df = _compute_park_direction_factor(features_df)

    # FIX 4: Normalise the pitcher column to a single name `pitcher`.
    # Drop starter_pitcher_id and pitcher_id; keep one clean `pitcher` column.
    if merge_pitcher_col in features_df.columns:
        features_df = features_df.rename(columns={merge_pitcher_col: "pitcher"})
    # Remove any redundant pitcher identity columns that would confuse downstream
    for _drop in ("starter_pitcher_id", "pitcher_id", "starter_id"):
        if _drop in features_df.columns and _drop != "pitcher":
            features_df = features_df.drop(columns=[_drop], errors="ignore")

    features_df["game_date"] = pd.to_datetime(features_df["game_date"]).dt.date
    features_df["hr_hit"]    = features_df["hr_hit"].astype(int)

    non_stat_cols = {
        "game_date", "game_pk", "batter", "pitcher", "home_team",
        "batter_team", "hr_hit", "pitcher_hand", "batter_hand",
        "pitcher_mode",
    }
    stat_cols = [c for c in features_df.columns if c not in non_stat_cols]
    features_df[stat_cols] = features_df[stat_cols].fillna(0.0)

    out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
    features_df.to_parquet(out_path, index=False)

    logger.info(
        "Saved: %s | rows=%d | hr_rate=%.4f | dates=%s -> %s",
        out_path.name,
        len(features_df),
        features_df["hr_hit"].mean(),
        pd.to_datetime(features_df["game_date"]).min().date(),
        pd.to_datetime(features_df["game_date"]).max().date(),
    )

    return FeaturesBuildResult(features_df=features_df, output_path=out_path)


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()

    result = build_features_for_range("2024-03-20", "2024-10-01")
    full_path = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    result.features_df.to_parquet(full_path, index=False)
    logger.info("Saved full season: %s | rows=%d", full_path.name, len(result.features_df))
    logger.info("HR rate: %.4f", result.features_df["hr_hit"].mean())

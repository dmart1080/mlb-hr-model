"""
Shared constants and pure helper functions used by both
build_features.py and build_features_fast.py.

No imports from other src.features modules — this is the base layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASTBALL_TYPES = {"FF", "SI"}
OFFSPEED_TYPES = {"SL", "CH", "CU", "KC", "FS", "ST", "SV", "CS", "EP"}

# Pitch-type groups used for matchup features
# Each batter HR rate vs pitch type × pitcher usage rate of that pitch type
PITCH_GROUP_FASTBALL  = {"FF", "SI", "FC"}          # 4-seam, sinker, cutter
PITCH_GROUP_BREAKING  = {"SL", "CU", "KC", "CS", "SV"}  # slider, curve, knuckle-curve
PITCH_GROUP_OFFSPEED  = {"CH", "FS", "EP"}           # changeup, splitter, eephus

MIN_PA_BATTER_SZN  = 10
MIN_PA_PITCHER_SZN = 10
MIN_PA_WINDOW      = 5

# Minimum pitches required to compute a reliable pitch-type usage rate
MIN_PITCHES_PITCH_TYPE = 20

# Minimum PA to trust a batter's HR rate vs a pitch group
MIN_PA_PITCH_TYPE = 10


# ---------------------------------------------------------------------------
# Pure numeric helpers
# ---------------------------------------------------------------------------

def _safe_mean(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) == 0:
        return np.nan
    m = v.mean()
    return np.nan if pd.isna(m) else float(m)


# Hard cap applied to estimated_woba_using_speedangle before averaging.
# Statcast occasionally emits values >2 on freak contact (e.g., inside-the-park
# HR with launch speed model overflow). Capping prevents one extreme PA from
# dominating a small-sample window mean. 2.0 is well above league-average
# xwOBAcon (~0.37) and individual elite-batter values (~0.50-0.55).
_XWOBACON_CAP = 2.0


def _safe_xwobacon_mean(series: pd.Series, min_n: int) -> float:
    """Mean of estimated_woba_using_speedangle on batted-ball events.

    `series` is expected to be the raw column at PA-level (NaN for non-contact
    PAs since those pitches never produced a launch_speed-driven xwOBA value).
    Outlier values are capped at _XWOBACON_CAP before averaging. Returns NaN
    if fewer than `min_n` non-null BBEs are present.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < min_n:
        return np.nan
    s = s.clip(upper=_XWOBACON_CAP)
    m = s.mean()
    return np.nan if pd.isna(m) else float(m)


def _is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    """
    Statcast barrel approximation (Baseball Savant definition).

    EV ≥ 98 mph AND LA within a widening range:
        EV 98  → LA 26–30
        EV 99  → LA 25–31
        ...
        EV 108+ → LA 16–41 (capped)
    """
    ev = pd.to_numeric(launch_speed, errors="coerce")
    la = pd.to_numeric(launch_angle,  errors="coerce")

    ev_clipped = ev.clip(upper=108)
    delta = (ev_clipped - 98).clip(lower=0)

    la_min = 26 - delta
    la_max = 30 + delta

    return (
        ev.notna() & la.notna() &
        (ev >= 98) &
        (la >= la_min) & (la <= la_max)
    ).fillna(False)


def _is_pulled_airball(
    hc_x: pd.Series,
    hc_y: pd.Series,
    stand: pd.Series,
    bb_type: pd.Series,
) -> pd.Series:
    """
    Classify a batted ball as a PULLED AIR BALL (fly ball or line drive pulled
    to the pull-side of the field).

    Ground balls and popups are excluded — only fly balls and line drives count
    because those are the trajectories that produce home runs.

    Statcast coordinate system (from catcher's perspective):
        hc_x: 0 = left foul line, ~125 = center, ~250 = right foul line
        hc_y: 0 = home plate side, increases toward outfield

    Pull direction:
        RHB pulled → left side  → hc_x < 125
        LHB pulled → right side → hc_x > 125

    We use hc_x < 100 / hc_x > 150 as tighter thresholds to stay solidly in
    the pull-side fair territory and avoid foul-ball noise near the lines.

    bb_type values (Statcast): 'fly_ball', 'line_drive', 'ground_ball', 'popup'
    Only 'fly_ball' and 'line_drive' are counted.
    """
    x   = pd.to_numeric(hc_x,  errors="coerce")
    y   = pd.to_numeric(hc_y,  errors="coerce")
    bb  = bb_type.astype("string").str.lower().str.strip()

    rhb = stand == "R"
    lhb = stand == "L"

    # Air ball = fly ball or line drive (not ground ball, not popup)
    is_air = bb.isin(["fly_ball", "line_drive"])

    # Ball in play with valid coordinates
    in_play = x.notna() & y.notna() & (y > 0) & is_air

    pulled = (
        (rhb & in_play & (x < 100)) |   # RHB pulls to left side
        (lhb & in_play & (x > 150))      # LHB pulls to right side
    )
    return pulled.fillna(False)


# ---------------------------------------------------------------------------
# Batter trend / home-away helpers (used by the fast vectorised code)
# ---------------------------------------------------------------------------

def _batter_trend_stats(w7: pd.DataFrame, w8_14: pd.DataFrame) -> dict:
    def _ev_mean(g):
        return _safe_mean(g["launch_speed"]) if len(g) > 0 else np.nan

    def _hardhit(g):
        if len(g) == 0:
            return np.nan
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else np.nan

    def _barrel(g):
        return float(g["is_barrel"].sum()) / len(g) if len(g) > 0 else np.nan

    def _hr_rate(g):
        return float(g["is_hr"].sum()) / len(g) if len(g) > 0 else np.nan

    def _diff(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return a - b

    ev7,   ev8_14   = _ev_mean(w7),  _ev_mean(w8_14)
    hh7,   hh8_14   = _hardhit(w7), _hardhit(w8_14)
    bar7,  bar8_14  = _barrel(w7),  _barrel(w8_14)
    hr7,   hr8_14   = _hr_rate(w7), _hr_rate(w8_14)

    return {
        "b_ev_trend":       _diff(ev7,  ev8_14),
        "b_hardhit_trend":  _diff(hh7,  hh8_14),
        "b_barrel_trend":   _diff(bar7, bar8_14),
        "b_hr_trend":       _diff(hr7,  hr8_14),
        "b_ev_mean_7":      ev7,
        "b_hardhit_rate_7": hh7,
    }


def _batter_home_away_stats(grp: pd.DataFrame, batter_team: str | None) -> dict:
    empty = {
        "b_hr_rate_home": np.nan, "b_hr_rate_away": np.nan,
        "b_hardhit_rate_home": np.nan, "b_hardhit_rate_away": np.nan,
        "b_barrel_rate_home": np.nan, "b_barrel_rate_away": np.nan,
        "b_hr_rate_home_edge": np.nan,
    }
    if len(grp) == 0 or batter_team is None:
        return empty

    home_mask = grp["home_team"] == batter_team
    home_grp  = grp[home_mask]
    away_grp  = grp[~home_mask]

    def _rate(g, col):
        return float(g[col].sum()) / len(g) if len(g) >= MIN_PA_WINDOW else np.nan

    def _hardhit(g):
        if len(g) < MIN_PA_WINDOW:
            return np.nan
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else np.nan

    hr_home = _rate(home_grp, "is_hr")
    hr_away = _rate(away_grp, "is_hr")
    return {
        "b_hr_rate_home":      hr_home,
        "b_hr_rate_away":      hr_away,
        "b_hardhit_rate_home": _hardhit(home_grp),
        "b_hardhit_rate_away": _hardhit(away_grp),
        "b_barrel_rate_home":  _rate(home_grp, "is_barrel"),
        "b_barrel_rate_away":  _rate(away_grp, "is_barrel"),
        "b_hr_rate_home_edge": (
            hr_home - hr_away
            if not pd.isna(hr_home) and not pd.isna(hr_away)
            else np.nan
        ),
    }

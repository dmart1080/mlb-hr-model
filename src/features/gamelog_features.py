"""
Shared rolling-outcome feature computation.

Called by BOTH the training feature builder (from Statcast PA data
aggregated per game) AND the inference builder (from MLB Stats API
gamelogs). Keeping one function ensures identical formulas across
train/predict so the model sees consistent inputs.

Input: list of dicts sorted ascending by date, each with keys:
    date (str YYYY-MM-DD), hr, ab, bb, k, h, tb, pa
Input cutoff: features computed only from games with date < cutoff.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable


_FEATURE_KEYS = (
    "b_hr_last_3g", "b_hr_last_7g", "b_hr_last_14g",
    "b_ops_last_7g", "b_ops_last_14g",
    "b_k_rate_last_7g",
    "b_ab_since_last_hr",
)


def empty_features() -> dict:
    return {k: float("nan") for k in _FEATURE_KEYS}


def _ops(ab: int, h: int, bb: int, tb: int) -> float:
    pa = ab + bb
    if pa <= 0 or ab <= 0:
        return float("nan")
    obp = (h + bb) / pa
    slg = tb / ab
    return obp + slg


def compute_rolling_gamelog_features(
    games: Iterable[dict],
    cutoff_date: str,
) -> dict:
    """
    games: iterable of per-game dicts sorted ascending by date.
    cutoff_date: 'YYYY-MM-DD' — only games strictly before this date count.

    Returns dict of the 7 rolling-outcome features. Any window with 0 games
    returns NaN for rate stats; hr counts default to 0 for empty windows.
    """
    prior = [g for g in games if g["date"] < cutoff_date]
    if not prior:
        out = empty_features()
        out["b_hr_last_3g"]  = 0
        out["b_hr_last_7g"]  = 0
        out["b_hr_last_14g"] = 0
        out["b_ab_since_last_hr"] = 200  # no games → treat as long drought
        return out

    last3  = prior[-3:]
    last7  = prior[-7:]
    last14 = prior[-14:]

    out: dict = {}
    out["b_hr_last_3g"]  = sum(g["hr"] for g in last3)
    out["b_hr_last_7g"]  = sum(g["hr"] for g in last7)
    out["b_hr_last_14g"] = sum(g["hr"] for g in last14)

    for label, window in (("7g", last7), ("14g", last14)):
        ab = sum(g["ab"] for g in window)
        h  = sum(g["h"]  for g in window)
        bb = sum(g["bb"] for g in window)
        tb = sum(g["tb"] for g in window)
        out[f"b_ops_last_{label}"] = _ops(ab, h, bb, tb)

    ab7 = sum(g["ab"] for g in last7)
    k7  = sum(g["k"]  for g in last7)
    out["b_k_rate_last_7g"] = (k7 / (ab7 + sum(g["bb"] for g in last7))) if ab7 > 0 else float("nan")

    # Count ABs since last HR. If the batter has never homered in prior games,
    # use 200 as a "long drought" sentinel — preserves monotonic semantics
    # (higher = bigger drought) and survives the downstream NaN→0 fill.
    ab_since = 0
    found_hr = False
    for g in reversed(prior):
        if g["hr"] > 0:
            found_hr = True
            break
        ab_since += g["ab"]
    out["b_ab_since_last_hr"] = ab_since if found_hr else 200

    return out

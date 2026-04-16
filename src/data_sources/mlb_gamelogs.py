"""
MLB game-log fetcher for rolling outcome-based features.

Fills the 1-2 day lag in Statcast data. MLB Stats API returns per-game
hitting stats within minutes of final out, so we can compute:
    b_hr_last_3g, b_hr_last_7g, b_hr_last_14g
    b_ops_last_7g, b_ops_last_14g
    b_k_rate_last_7g
    b_ab_since_last_hr

Cache: data/cache/gamelogs/batter_{id}_{season}.json  (TTL 12h)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "gamelogs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_TTL_HOURS = 12
_TIMEOUT         = 15


def _cache_path(batter_id: int, season: int) -> Path:
    return CACHE_DIR / f"batter_{batter_id}_{season}.json"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < _CACHE_TTL_HOURS


def fetch_batter_gamelog(
    batter_id: int,
    season: int,
    *,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return list of per-game hitting stats dicts for one batter in one season.
    Each dict contains: date, hr, ab, bb, k, h, tb (total bases), pa.
    Empty list on API miss/error — caller decides fallback.
    """
    cache = _cache_path(batter_id, season)
    if not force_refresh and _cache_is_fresh(cache):
        try:
            with open(cache) as f:
                return json.load(f)
        except Exception:
            pass

    url = f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
    params = {"stats": "gameLog", "group": "hitting", "season": season}

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("gamelog fetch failed batter=%d season=%d: %s",
                       batter_id, season, e)
        return []

    games = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            s = split.get("stat", {})
            date_str = split.get("date") or split.get("gameDate", "")
            if not date_str:
                continue
            games.append({
                "date": date_str[:10],
                "hr":   int(s.get("homeRuns",   0) or 0),
                "ab":   int(s.get("atBats",     0) or 0),
                "bb":   int(s.get("baseOnBalls", 0) or 0),
                "k":    int(s.get("strikeOuts", 0) or 0),
                "h":    int(s.get("hits",       0) or 0),
                "tb":   int(s.get("totalBases", 0) or 0),
                "pa":   int(s.get("plateAppearances", 0) or 0),
            })

    games.sort(key=lambda g: g["date"])

    try:
        with open(cache, "w") as f:
            json.dump(games, f)
    except Exception:
        pass

    return games

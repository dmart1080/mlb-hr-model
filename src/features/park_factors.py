"""
Park factor utilities for the MLB HR model.

Resolution order for any given season:
  1. Dynamic fetch from MLB Stats API (cached per season, refreshed if >30 days old)
  2. Hardcoded 2024 static table (fallback if API unavailable)
  3. DEFAULT_PARK_FACTOR = 100 (neutral) for any team not found in either source
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "park_factors"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_DAYS = 30
DEFAULT_PARK_FACTOR = 100

# Minimum number of teams required to consider an API response valid.
# The MLB has 30 teams; if fewer than this are returned the response is
# treated as partial/broken and we fall through to the static table.
_MIN_TEAMS_REQUIRED = 20

HR_PARK_FACTOR: dict[str, int] = {
    "ARI": 102, "ATL": 108, "BAL": 110, "BOS": 104, "CHC": 102,
    "CIN": 115, "CLE":  96, "COL": 120, "CWS": 101, "DET":  95,
    "HOU": 101, "KCR":  97, "LAA":  98, "LAD": 105, "MIA":  92,
    "MIL": 103, "MIN":  99, "NYM":  99, "NYY": 112, "OAK":  88,
    "PHI": 110, "PIT":  95, "SDP":  94, "SEA":  90, "SFG":  92,
    "STL":  98, "TBR":  97, "TEX": 107, "TOR": 102, "WSN":  96,
}

_TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def _cache_path(season: int) -> Path:
    return CACHE_DIR / f"park_factors_{season}.json"


def _cache_is_fresh(season: int) -> bool:
    p = _cache_path(season)
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    current_year = datetime.now().year
    if season < current_year:
        return True
    return (datetime.now() - mtime) < timedelta(days=CACHE_TTL_DAYS)


def _fetch_from_api(season: int) -> dict[str, float] | None:
    """
    Fetch HR park factors for a season from the MLB Stats API.

    FIX 5: Added explicit validation that:
      (a) the response structure contains the expected keys,
      (b) at least one split contains a non-null `parkFactor` field, and
      (c) the result covers at least _MIN_TEAMS_REQUIRED teams.

    Previously, if the sabermetrics endpoint returned splits without a
    `parkFactor` key (which happens for some season/gameType combinations),
    the function returned an empty dict that was written to cache and then
    used silently — masking the fallback to the static table.

    Returns None on any validation failure so callers fall through to the
    static table and we never cache a broken result.
    """
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=sabermetrics&group=pitching&gameType=R"
        f"&season={season}&sportId=1&limit=40"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout as e:
        logger.warning("Park factors API timeout season=%d: %s", season, e)
        return None
    except requests.HTTPError as e:
        logger.warning(
            "Park factors HTTP error %d season=%d",
            e.response.status_code, season,
        )
        return None
    except Exception as e:
        logger.error("Park factors API fetch failed season=%d: %s", season, e)
        return None

    # FIX 5a: validate top-level structure
    stats_list = data.get("stats")
    if not stats_list or not isinstance(stats_list, list):
        logger.warning(
            "Park factors: unexpected response structure for season=%d "
            "(missing 'stats' key or wrong type)",
            season,
        )
        return None

    splits = stats_list[0].get("splits", [])
    if not splits:
        logger.warning("Park factors: no splits returned for season=%d", season)
        return None

    # FIX 5b: check that at least one split actually has a parkFactor field
    sample_pf_keys = {
        k for split in splits[:5]
        for k in split.get("stat", {}).keys()
    }
    if "parkFactor" not in sample_pf_keys:
        logger.warning(
            "Park factors: 'parkFactor' key absent from stat fields for "
            "season=%d — available stat keys: %s",
            season, sorted(sample_pf_keys),
        )
        return None

    result: dict[str, float] = {}
    for split in splits:
        team_id = split.get("team", {}).get("id")
        abbr    = _TEAM_ID_TO_ABBR.get(team_id)
        pf_raw  = split.get("stat", {}).get("parkFactor")

        if abbr is None or pf_raw is None:
            continue

        try:
            pf = float(pf_raw)
        except (ValueError, TypeError):
            continue

        result[abbr] = pf

    # FIX 5c: reject partial results — fewer teams than expected means the
    # endpoint returned something but it's not reliable enough to cache.
    if len(result) < _MIN_TEAMS_REQUIRED:
        logger.warning(
            "Park factors: only %d/%d teams parsed for season=%d — "
            "result too sparse, falling back to static table",
            len(result), _MIN_TEAMS_REQUIRED, season,
        )
        return None

    logger.info("Park factors: fetched %d teams for season=%d", len(result), season)
    return result


def get_park_factors(season: int) -> dict[str, float]:
    """
    Return HR park factors for a given season as a dict mapping
    team abbreviation -> float on a 0-2 scale (1.0 = neutral).

    Resolution order:
      1. Fresh disk cache
      2. MLB Stats API -> write to cache only if response passes validation
      3. Static 2024 hardcoded table
      4. DEFAULT_PARK_FACTOR (1.0) for any missing team
    """
    cache = _cache_path(season)

    if _cache_is_fresh(season):
        with open(cache) as f:
            stored = json.load(f)
        logger.debug("Park factors cache hit for season=%d", season)
        return {k: v / 100.0 for k, v in stored.items()}

    fetched = _fetch_from_api(season)
    if fetched:
        # Only write to cache when validation passed (non-None, sufficient teams)
        with open(cache, "w") as f:
            json.dump(fetched, f)
        return {k: v / 100.0 for k, v in fetched.items()}

    # API failed or returned bad data — try stale cache before static fallback
    if cache.exists():
        logger.warning("Park factors: using stale cache for season=%d", season)
        with open(cache) as f:
            stored = json.load(f)
        return {k: v / 100.0 for k, v in stored.items()}

    logger.warning(
        "Park factors: using static 2024 fallback for season=%d", season
    )
    return {k: v / 100.0 for k, v in HR_PARK_FACTOR.items()}


def get_park_factor_for_team(team: str, season: int) -> float:
    """Convenience wrapper — returns a single team's park factor (0-2 scale)."""
    pf_map = get_park_factors(season)
    return pf_map.get(team, DEFAULT_PARK_FACTOR / 100.0)


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()

    for yr in [2023, 2024, 2025]:
        pf = get_park_factors(yr)
        logger.info("%d park factors (%d teams):", yr, len(pf))
        for abbr, val in sorted(pf.items(), key=lambda x: -x[1]):
            print(f"  {abbr:4s}  {val:.3f}  ({val*100:.0f})")

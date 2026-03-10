from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache"
WEATHER_CACHE = CACHE_DIR / "weather"
WEATHER_CACHE.mkdir(parents=True, exist_ok=True)

_PREFETCH_TTL_HOURS = 6

INDOOR_PARKS = {
    "MIA", "HOU", "SEA", "ARI", "MIL", "TOR", "TB", "TBR",
}

_WIND_DIR_TO_DEG = {
    "in from cf":  180, "out to cf":     0,
    "l to r":       90, "r to l":      270,
    "in from lf":  135, "in from rf":  225,
    "out to lf":   315, "out to rf":    45,
    "calm":          0,
}

_WIND_HR_FACTOR = {
    "out to cf":    1.0, "out to lf":    0.7, "out to rf":    0.7,
    "l to r":       0.3, "r to l":       0.3, "calm":         0.0,
    "in from lf":  -0.5, "in from rf":  -0.5, "in from cf":  -1.0,
}


def _cache_path(game_pk: int) -> Path:
    return WEATHER_CACHE / f"weather_{game_pk}.json"


def _cache_is_stale(cache: Path, game_date: Optional[str] = None) -> bool:
    if not cache.exists():
        return True

    if game_date is not None:
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            if gd.date() < now.date():
                return False
        except ValueError:
            pass

    mtime = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    return age_hours > _PREFETCH_TTL_HOURS


def _parse_wind_speed(wind_str: str) -> float:
    m = re.search(r"(\d+)\s*mph", wind_str, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def _parse_wind_direction(wind_str: str) -> str:
    wind_lower = wind_str.lower()
    for key in _WIND_HR_FACTOR:
        if key in wind_lower:
            return key
    if "calm" in wind_lower or wind_str.strip() == "0 mph":
        return "calm"
    return "unknown"


def _parse_temp(temp_str: str) -> float:
    m = re.search(r"(\d+)", str(temp_str))
    return float(m.group(1)) if m else float("nan")


def fetch_game_weather(
    game_pk: int,
    *,
    force_refresh: bool = False,
    game_date: Optional[str] = None,
) -> dict:
    """
    Fetch weather for a single game from the MLB Stats API.

    Parameters
    ----------
    game_pk       : MLB game primary key
    force_refresh : bypass all cache checks
    game_date     : YYYY-MM-DD string used to determine cache staleness.

    Returns a dict with keys:
        temp_f, wind_speed_mph, wind_direction, wind_hr_factor,
        condition, is_indoor, game_pk
    """
    cache = _cache_path(game_pk)

    if not force_refresh and not _cache_is_stale(cache, game_date=game_date):
        with open(cache) as f:
            return json.load(f)

    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout as e:
        logger.warning("Weather fetch timeout game_pk=%d: %s", game_pk, e)
        return _neutral_weather(game_pk, error=f"timeout:{e}")
    except requests.HTTPError as e:
        logger.warning(
            "Weather HTTP error %d game_pk=%d",
            e.response.status_code, game_pk,
        )
        return _neutral_weather(game_pk, error=f"http:{e.response.status_code}")
    except Exception as e:
        logger.warning("Weather fetch failed game_pk=%d: %s", game_pk, e)
        return _neutral_weather(game_pk, error=str(e))

    weather_raw = data.get("gameData", {}).get("weather", {})
    venue_raw   = data.get("gameData", {}).get("venue",   {})

    temp_str  = str(weather_raw.get("temp",      "72"))
    wind_str  = str(weather_raw.get("wind",      "0 mph, Calm"))
    condition = str(weather_raw.get("condition", "Unknown"))

    temp_f        = _parse_temp(temp_str)
    wind_speed    = _parse_wind_speed(wind_str)
    wind_dir      = _parse_wind_direction(wind_str)
    wind_hr_fac   = _WIND_HR_FACTOR.get(wind_dir, 0.0)

    result = {
        "game_pk":         game_pk,
        "temp_f":          temp_f if not pd.isna(temp_f) else 72.0,
        "wind_speed_mph":  wind_speed,
        "wind_direction":  wind_dir,
        "wind_hr_factor":  wind_hr_fac,
        "wind_hr_impact":  wind_speed * wind_hr_fac,
        "condition":       condition,
        "is_indoor":       0,
        "venue_name":      venue_raw.get("name", ""),
        "_fetched_at":     datetime.now(tz=timezone.utc).isoformat(),
    }

    with open(cache, "w") as f:
        json.dump(result, f)

    return result


def _neutral_weather(game_pk: int, error: str = "") -> dict:
    return {
        "game_pk":        game_pk,
        "temp_f":         72.0,
        "wind_speed_mph": 0.0,
        "wind_direction": "unknown",
        "wind_hr_factor": 0.0,
        "wind_hr_impact": 0.0,
        "condition":      "unknown",
        "is_indoor":      0,
        "venue_name":     "",
        "_error":         error,
    }


def fetch_weather_for_games(
    game_pks: list[int],
    home_teams: dict[int, str],
    *,
    force_refresh: bool = False,
    sleep_secs: float = 0.05,
    game_dates: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Fetch weather for a list of game_pks and return as a DataFrame.
    Marks indoor parks automatically regardless of API data.
    """
    rows = []
    unique_pks = list(dict.fromkeys(game_pks))
    game_dates = game_dates or {}

    cached_count = sum(
        1 for gp in unique_pks
        if not _cache_is_stale(_cache_path(gp), game_date=game_dates.get(gp))
    )
    logger.info(
        "Weather fetch: %d games (%d cached, %d to fetch)",
        len(unique_pks), cached_count, len(unique_pks) - cached_count,
    )

    synthetic_count = 0
    for i, gp in enumerate(unique_pks):
        gdate = game_dates.get(gp)
        was_cached = not _cache_is_stale(_cache_path(gp), game_date=gdate)

        result = fetch_game_weather(gp, force_refresh=force_refresh, game_date=gdate)

        if result.get("_error"):
            synthetic_count += 1

        home_team = home_teams.get(gp, "")
        result["is_indoor"] = int(home_team in INDOOR_PARKS)

        if result["is_indoor"]:
            result["wind_speed_mph"] = 0.0
            result["wind_hr_factor"] = 0.0
            result["wind_hr_impact"] = 0.0
            result["temp_f"]         = 72.0

        rows.append(result)

        if not was_cached:
            time.sleep(sleep_secs)

        if (i + 1) % 100 == 0:
            logger.info("Weather: %d/%d games processed ...", i + 1, len(unique_pks))

    if synthetic_count:
        logger.warning(
            "Weather unavailable for %d/%d games — neutral values used",
            synthetic_count, len(unique_pks),
        )

    df = pd.DataFrame(rows)
    df["temp_above_75"]   = (df["temp_f"] > 75).astype("int8")
    df["temp_above_85"]   = (df["temp_f"] > 85).astype("int8")
    df["wind_out_strong"] = ((df["wind_hr_factor"] > 0) & (df["wind_speed_mph"] >= 10)).astype("int8")
    df["wind_in_strong"]  = ((df["wind_hr_factor"] < 0) & (df["wind_speed_mph"] >= 10)).astype("int8")

    return df[["game_pk", "temp_f", "wind_speed_mph", "wind_hr_factor",
               "wind_hr_impact", "is_indoor", "temp_above_75", "temp_above_85",
               "wind_out_strong", "wind_in_strong"]]


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()

    test_pks   = [745456, 745457, 745458]
    test_homes = {745456: "NYY", 745457: "BOS", 745458: "HOU"}
    df = fetch_weather_for_games(test_pks, test_homes)
    print(df)

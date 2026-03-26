"""
src/data_sources/rotowire_starters.py
======================================
Scrapes probable starters from Rotowire as a fallback when the MLB Stats API
has not yet posted probablePitcher data for a game.

Rotowire posts rotation data 5-7 days in advance, making it useful for the
morning pass when MLB's API still returns None for some or all starters.

How it works
------------
1. Fetch https://www.rotowire.com/baseball/daily-lineups.php
2. Parse each lineup__box for pitcher links (class lineup__player-highlight-name)
3. Extract MLBAM ID directly from the player URL slug (e.g. paul-skenes-18838 -> 18838)
4. Team abbrs come from the lineup__top text (e.g. "PIT|NYM" -> away=PIT, home=NYM)
5. Cache results to data/cache/schedule/rotowire_{date}.json (TTL: 2h)

Usage
-----
    from src.data_sources.rotowire_starters import fetch_rotowire_starters
    starters = fetch_rotowire_starters("2026-03-27")

Integration
-----------
Called automatically by fetch_probable_starters_for_date() in mlb_schedule.py
when MLB API returns None for a game's probablePitcher.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
SCHEDULE_CACHE = PROJECT_ROOT / "data" / "cache" / "schedule"
SCHEDULE_CACHE.mkdir(parents=True, exist_ok=True)

_ROTOWIRE_URL  = "https://www.rotowire.com/baseball/daily-lineups.php"
_CACHE_TTL_SEC = 2 * 60 * 60   # 2 hours
_REQUEST_TIMEOUT = 15

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(date_str: str) -> Path:
    return SCHEDULE_CACHE / f"rotowire_{date_str}.json"


def _cache_is_fresh(date_str: str) -> bool:
    p = _cache_path(date_str)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < _CACHE_TTL_SEC


# ---------------------------------------------------------------------------
# ID extraction from URL slug
# ---------------------------------------------------------------------------

def _id_from_href(href: str) -> Optional[int]:
    """
    Extract MLBAM ID from a Rotowire player URL.
    e.g. /baseball/player/paul-skenes-18838  ->  18838
    """
    m = re.search(r"-(\d+)\s*$", href.strip().rstrip("/"))
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def _scrape_rotowire() -> list[dict]:
    """
    Fetch Rotowire daily lineups and extract probable starters.

    Returns list of dicts:
        {
            "away_team":    str,   # e.g. "PIT"
            "home_team":    str,   # e.g. "NYM"
            "away_pitcher": str | None,   # full name
            "home_pitcher": str | None,
            "away_id":      int | None,   # MLBAM ID from URL slug
            "home_id":      int | None,
        }
    """
    try:
        resp = requests.get(_ROTOWIRE_URL, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Rotowire fetch failed: %s", e)
        return []

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — run: pip install beautifulsoup4")
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    boxes = soup.find_all("div", class_="lineup__box")
    logger.info("Rotowire: found %d lineup boxes", len(boxes))

    games = []
    for box in boxes:
        # --- Team abbreviations from lineup__top (e.g. "PIT|NYM") ---
        top = box.find(class_="lineup__top")
        if not top:
            continue
        top_text = top.get_text(separator="|", strip=True)
        parts    = [p.strip() for p in top_text.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        away_abbr = parts[0]
        home_abbr = parts[1]

        # --- Pitchers: links with parent class lineup__player-highlight-name ---
        pitcher_links = box.find_all(
            "a",
            href=re.compile(r"/baseball/player/"),
        )
        # Filter to only those whose parent has class lineup__player-highlight-name
        highlight_links = [
            lnk for lnk in pitcher_links
            if "lineup__player-highlight-name" in (lnk.parent.get("class") or [])
        ]

        away_pitcher = away_id = None
        home_pitcher = home_id = None

        if len(highlight_links) >= 1:
            away_pitcher = highlight_links[0].get_text(strip=True)
            away_id      = _id_from_href(highlight_links[0].get("href", ""))

        if len(highlight_links) >= 2:
            home_pitcher = highlight_links[1].get_text(strip=True)
            home_id      = _id_from_href(highlight_links[1].get("href", ""))

        games.append({
            "away_team":    away_abbr,
            "home_team":    home_abbr,
            "away_pitcher": away_pitcher,
            "home_pitcher": home_pitcher,
            "away_id":      away_id,
            "home_id":      home_id,
            "source":       "rotowire",
        })

    return games


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_rotowire_starters(
    date_str: str,
    *,
    force_refresh: bool = False,
) -> dict[tuple[str, str], dict]:
    """
    Fetch probable starters from Rotowire for the given date.

    Returns dict keyed by (home_abbr, away_abbr):
        {
            ("NYM", "PIT"): {
                "home_team":    "NYM",
                "away_team":    "PIT",
                "home_pitcher": "Freddy Peralta",
                "away_pitcher": "Paul Skenes",
                "home_id":      int | None,
                "away_id":      int | None,
                "source":       "rotowire",
            },
            ...
        }
    """
    cache = _cache_path(date_str)

    if not force_refresh and _cache_is_fresh(date_str):
        try:
            with open(cache) as f:
                stored = json.load(f)
            result = {(e["home_team"], e["away_team"]): e for e in stored.get("games", [])}
            logger.debug("Rotowire cache hit for %s (%d games)", date_str, len(result))
            return result
        except Exception:
            pass

    logger.info("Fetching Rotowire probable starters for %s ...", date_str)
    games = _scrape_rotowire()

    if not games:
        logger.warning("Rotowire: no games found for %s", date_str)
        return {}

    result = {(g["home_team"], g["away_team"]): g for g in games}

    payload = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "date":       date_str,
        "games":      games,
    }
    try:
        with open(cache, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning("Rotowire cache write failed: %s", e)

    resolved = sum(1 for g in games if g["home_id"] or g["away_id"])
    logger.info("Rotowire: %d games, %d with pitcher IDs", len(games), resolved)
    return result


def enrich_schedule_with_rotowire(
    schedule_df: "pd.DataFrame",
    date_str: str,
    *,
    force_refresh: bool = False,
) -> "pd.DataFrame":
    """
    Fill missing starter IDs in schedule_df using Rotowire data.
    Only fills None values — confirmed MLB API data is never overwritten.
    Adds column: starter_source ("mlb_api" | "rotowire" | None)
    """
    import pandas as pd

    df = schedule_df.copy()

    df["starter_source"] = df.apply(
        lambda r: "mlb_api"
        if (r.get("home_starter_id") is not None and r.get("away_starter_id") is not None)
        else None,
        axis=1,
    )

    missing_mask = df["starter_source"].isna()
    if not missing_mask.any():
        logger.debug("All starters confirmed by MLB API — skipping Rotowire")
        return df

    logger.info(
        "Rotowire fallback: %d/%d games missing starters",
        missing_mask.sum(), len(df),
    )

    rw = fetch_rotowire_starters(date_str, force_refresh=force_refresh)
    if not rw:
        return df

    filled = 0
    for idx, row in df[missing_mask].iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        data = rw.get((home, away))
        if data is None:
            continue

        if df.at[idx, "home_starter_id"] is None and data.get("home_id"):
            df.at[idx, "home_starter_id"] = data["home_id"]
            filled += 1

        if df.at[idx, "away_starter_id"] is None and data.get("away_id"):
            df.at[idx, "away_starter_id"] = data["away_id"]
            filled += 1

        df.at[idx, "starter_source"] = "rotowire"

    logger.info("Rotowire: filled %d missing starter IDs", filled)
    return df


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.logging_config import configure_logging
    configure_logging()

    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(f"\nFetching Rotowire starters for {date_arg} ...\n")

    starters = fetch_rotowire_starters(date_arg, force_refresh=True)

    if not starters:
        print("No data returned.")
        sys.exit(1)

    print(f"{'Away':<6} {'Home':<6}  {'Away Pitcher':<25} {'Home Pitcher':<25}  {'Away ID':>9} {'Home ID':>9}")
    print("-" * 90)
    for (home, away), d in sorted(starters.items()):
        print(
            f"{away:<6} {home:<6}  "
            f"{(d['away_pitcher'] or 'TBD'):<25} "
            f"{(d['home_pitcher'] or 'TBD'):<25}  "
            f"{str(d['away_id'] or '?'):>9} "
            f"{str(d['home_id'] or '?'):>9}"
        )
    print()

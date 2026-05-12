"""
Historical HR-prop odds fetcher (paid-tier Odds API).

Wraps two endpoints:
    GET /v4/historical/sports/baseball_mlb/events?date={snapshot_iso}
    GET /v4/historical/sports/baseball_mlb/events/{event_id}/odds?date={snapshot_iso}

Credit cost (verified 2026-05-04 on the 20k/mo plan):
    /events       : 1 credit / call
    /events/.../odds : 20 credits / call (per region-set × markets)

So one slate of 15 games at one snapshot ≈ 301 credits. Aggressive caching
keeps re-runs free. Per-event responses are persisted under
    data/cache/odds/historical/odds_{date}_{snapshot}_{event_id}.json
and the events-list response under
    data/cache/odds/historical/events_{snapshot}.json

The `_parse_bookmaker_props` helper from src.data_sources.odds is reused so
returned prop dicts match the live-fetch schema exactly (event_id,
commence_time, all_books, etc.) — that lets the same enrichment / lookup
code work on historical or live data.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from src.data_sources.odds import (
    _parse_bookmaker_props,
    OddsApiQuotaExceeded,
    _MARKET,
    _SPORT,
    _QUOTA_STATUSES,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "odds" / "historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_API_BASE   = "https://api.the-odds-api.com/v4/historical"
_TIMEOUT    = 15
# us+us2 to match live behavior (HR props mostly live in us2 these days)
_REGIONS    = "us,us2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise EnvironmentError("ODDS_API_KEY not set and no api_key passed.")
    return key


def _events_cache_path(snapshot_iso: str) -> Path:
    safe = snapshot_iso.replace(":", "").replace("-", "")
    return CACHE_DIR / f"events_{safe}.json"


def _event_odds_cache_path(snapshot_iso: str, event_id: str) -> Path:
    safe = snapshot_iso.replace(":", "").replace("-", "")
    return CACHE_DIR / f"odds_{safe}_{event_id}.json"


# ---------------------------------------------------------------------------
# Snapshot-time helper
# ---------------------------------------------------------------------------

def snapshot_iso_for_game(commence_time_iso: str, lead_minutes: int = 60) -> str:
    """Pick a "closing line" snapshot for a single game.

    Returns ISO-8601 UTC at `commence_time - lead_minutes`. The historical
    API resolves to the nearest available 5-min snapshot at fetch time.
    """
    dt = datetime.fromisoformat(commence_time_iso.replace("Z", "+00:00"))
    snap = dt - timedelta(minutes=lead_minutes)
    return snap.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_iso_for_date(date_str: str, hour_utc: int = 22) -> str:
    """Slate-level snapshot: a single timestamp for the whole day (e.g.,
    18:00 ET ≈ 22:00 UTC, just before most evening games).

    Cheap-and-cheerful version of the per-game closing-line snapshot — uses
    ONE /events call per day instead of N. Good for slates where most games
    start at similar times.
    """
    return f"{date_str}T{hour_utc:02d}:00:00Z"


# ---------------------------------------------------------------------------
# Endpoint wrappers
# ---------------------------------------------------------------------------

def _fetch_events_snapshot(
    snapshot_iso: str,
    api_key: str,
    *,
    force_refresh: bool = False,
) -> list[dict]:
    """One historical /events call. Returns the inner `data` list of events."""
    cache = _events_cache_path(snapshot_iso)
    if cache.exists() and not force_refresh:
        with open(cache) as f:
            payload = json.load(f)
        events = payload.get("data") or []
        logger.debug("Events cache hit %s (%d events)", snapshot_iso, len(events))
        return events

    url = f"{_API_BASE}/sports/{_SPORT}/events"
    params = {"apiKey": api_key, "date": snapshot_iso, "dateFormat": "iso"}
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError as e:
        status = e.response.status_code
        if status in _QUOTA_STATUSES:
            raise OddsApiQuotaExceeded(
                f"HTTP {status} from historical /events (quota or rate-limit)"
            ) from e
        logger.warning("Historical /events HTTP %d for %s", status, snapshot_iso)
        return []
    except Exception as e:
        logger.error("Historical /events failed for %s: %s", snapshot_iso, e)
        return []

    payload = resp.json() if resp.content else {}
    remaining = resp.headers.get("x-requests-remaining", "?")
    last      = resp.headers.get("x-requests-last", "?")
    events    = payload.get("data") or []
    logger.info(
        "Historical /events %s -> %d events (cost=%s, remaining=%s)",
        snapshot_iso, len(events), last, remaining,
    )

    try:
        with open(cache, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning("Could not cache events for %s: %s", snapshot_iso, e)

    return events


def _fetch_event_odds(
    event_id: str,
    snapshot_iso: str,
    api_key: str,
    *,
    force_refresh: bool = False,
) -> dict:
    """One historical /events/{id}/odds call. Returns the inner `data` dict
    (which has `bookmakers`, `home_team`, etc.) or {} on failure.
    """
    cache = _event_odds_cache_path(snapshot_iso, event_id)
    if cache.exists() and not force_refresh:
        with open(cache) as f:
            payload = json.load(f)
        return payload.get("data") or {}

    url = f"{_API_BASE}/sports/{_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":     api_key,
        "date":       snapshot_iso,
        "regions":    _REGIONS,
        "markets":    _MARKET,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError as e:
        status = e.response.status_code
        if status in _QUOTA_STATUSES:
            raise OddsApiQuotaExceeded(
                f"HTTP {status} from historical /event/odds (quota or rate-limit)"
            ) from e
        logger.warning(
            "Historical /event/odds HTTP %d for event=%s @%s",
            status, event_id, snapshot_iso,
        )
        return {}
    except Exception as e:
        logger.error(
            "Historical /event/odds failed for event=%s @%s: %s",
            event_id, snapshot_iso, e,
        )
        return {}

    payload = resp.json() if resp.content else {}
    remaining = resp.headers.get("x-requests-remaining", "?")
    last      = resp.headers.get("x-requests-last", "?")
    logger.debug(
        "Historical event-odds event=%s @%s cost=%s remaining=%s",
        event_id, snapshot_iso, last, remaining,
    )

    try:
        with open(cache, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning("Could not cache event-odds for %s @%s: %s",
                       event_id, snapshot_iso, e)
    return payload.get("data") or {}


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def fetch_historical_hr_props_for_date(
    date_str: str,
    *,
    api_key: Optional[str] = None,
    snapshot_iso: Optional[str] = None,
    per_game_close: bool = True,
    close_lead_minutes: int = 60,
    force_refresh: bool = False,
) -> list[dict]:
    """Pull HR-prop lines for every event on `date_str` and return them in
    the same dict-list format as `src.data_sources.odds.fetch_hr_props`.

    Two snapshot strategies:
      per_game_close=True  (default): one /events call at 22:00 UTC of
        `date_str` to enumerate the slate, then one /event-odds call per
        event at (commence_time - close_lead_minutes). Tighter approximation
        of the closing line, but costs N+1 historical-events lookups
        (events itself is 1 credit, each event-odds call is 20 credits).
      per_game_close=False: a single slate-wide snapshot at `snapshot_iso`
        (or 22:00 UTC if not provided). Cheaper but less accurate for
        early-afternoon games.

    Returns a list of prop dicts identical in schema to the live path so
    the same lookup / enrichment code can join them.
    """
    key = _get_api_key(api_key)

    slate_snapshot = snapshot_iso or snapshot_iso_for_date(date_str)
    events = _fetch_events_snapshot(slate_snapshot, key, force_refresh=force_refresh)
    if not events:
        logger.warning("No historical events found for %s @%s", date_str, slate_snapshot)
        return []

    all_props: list[dict] = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        if per_game_close:
            commence = event.get("commence_time")
            if not commence:
                snap = slate_snapshot
            else:
                snap = snapshot_iso_for_game(commence, lead_minutes=close_lead_minutes)
        else:
            snap = slate_snapshot

        try:
            event_data = _fetch_event_odds(
                event_id, snap, key, force_refresh=force_refresh,
            )
        except OddsApiQuotaExceeded as e:
            logger.warning("Historical quota exhausted mid-slate %s: %s", date_str, e)
            break

        if not event_data:
            continue

        bookmakers = event_data.get("bookmakers", [])
        props = _parse_bookmaker_props(bookmakers, event=event_data)
        all_props.extend(props)

    logger.info(
        "Historical: %s -> %d props from %d events (per_game_close=%s)",
        date_str, len(all_props), len(events), per_game_close,
    )
    return all_props


if __name__ == "__main__":
    import argparse
    from src.logging_config import configure_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--per-game-close", action="store_true", default=True,
                        help="Use per-game close-of-line snapshots (default)")
    parser.add_argument("--slate-snapshot", default=None,
                        help="Override slate snapshot ISO (skip per-game logic)")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    configure_logging()
    props = fetch_historical_hr_props_for_date(
        args.date,
        snapshot_iso=args.slate_snapshot,
        per_game_close=args.per_game_close and args.slate_snapshot is None,
        force_refresh=args.force_refresh,
    )
    print(f"Got {len(props)} props for {args.date}")
    for p in props[:5]:
        print(f"  {p['batter_name']:<25}  over={p['over_price']:+5d}  "
              f"book={p['bookmaker']:<18}  commence={p.get('commence_time')}")

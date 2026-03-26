"""
HR prop odds fetcher for the MLB HR model.

Fetches batter HR prop lines from The Odds API, caches by date, and
exposes implied probabilities for use in predict.py edge detection.

Resolution order at predict time:
  1. Fresh disk cache for today's date (TTL = 2 hours pre-game)
  2. Live fetch from The Odds API
  3. No-match fallback (NaN) — predict.py degrades gracefully

Cache layout:
  data/cache/odds/odds_YYYY-MM-DD.json
    {
      "fetched_at": "ISO timestamp",
      "props": [
        {
          "batter_name":      "Aaron Judge",
          "market":           "batter_home_runs",
          "line":             0.5,
          "over_price":       -130,
          "under_price":      105,
          "implied_prob_over": 0.565,
          "bookmaker":        "draftkings",
          "fetched_at":       "ISO timestamp"
        },
        ...
      ]
    }

Matching strategy:
  Odds are keyed by player name (string).  At predict time we fuzzy-match
  against the resolved batter_name from playerid_reverse_lookup.  The match
  threshold is configurable (default 85 — good enough for MLB names).

API notes:
  - Endpoint: /v4/sports/baseball_mlb/events/{event_id}/odds
  - Market:   batter_home_runs  (player prop, over/under 0.5 HRs)
  - Requires: ODDS_API_KEY environment variable  OR  explicit api_key arg
  - Free tier: 500 requests/month — ~1 request per event per day is fine
    for development; a full slate (~15 games * ~9 batters each) costs ~15
    requests if fetched per-event, but we batch via the player-props endpoint
    which returns all events in one call.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_API_BASE        = "https://api.the-odds-api.com/v4"
_SPORT           = "baseball_mlb"
_MARKET          = "batter_home_runs"
_CACHE_TTL_HOURS = 2          # re-fetch if cache older than 2 hours
_FUZZY_THRESHOLD = 85         # minimum rapidfuzz score for name matching
_DEFAULT_BOOK    = "draftkings"  # preferred bookmaker; falls back to first available


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(date_str: str) -> Path:
    return CACHE_DIR / f"odds_{date_str}.json"


def _cache_is_fresh(date_str: str) -> bool:
    p = _cache_path(date_str)
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    age_h = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    return age_h < _CACHE_TTL_HOURS


# ---------------------------------------------------------------------------
# American odds → implied probability
# ---------------------------------------------------------------------------

def american_to_implied_prob(american_odds: int) -> float:
    """
    Convert American moneyline odds to implied probability (no vig removed).

    Examples:
        -130  →  0.5652  (favourite)
        +110  →  0.4762  (underdog)
        -110  →  0.5238  (standard juice)
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def remove_vig(prob_over: float, prob_under: float) -> tuple[float, float]:
    """
    Remove the bookmaker's margin (vig) from a two-sided market using
    the standard normalization method.

    Returns (fair_prob_over, fair_prob_under) that sum to 1.0.
    """
    total = prob_over + prob_under
    if total <= 0:
        return 0.5, 0.5
    return prob_over / total, prob_under / total


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "No Odds API key found.  Set the ODDS_API_KEY environment variable "
            "or pass api_key= explicitly.\n"
            "  export ODDS_API_KEY=your_key_here\n"
            "Get a free key at https://the-odds-api.com"
        )
    return key


def _fetch_events_for_date(date_str: str, api_key: str) -> list[dict]:
    """
    Fetch all MLB event IDs scheduled on date_str (ET).
    Returns a list of event dicts from The Odds API /events endpoint.

    Note: The Odds API uses UTC timestamps.  ET is UTC-4 (EDT) or UTC-5 (EST),
    so a game at 11:05pm ET = 03:05 UTC next day.  We extend commenceTimeTo to
    next day 05:00 UTC to ensure all ET games on date_str are captured.
    """
    next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{_API_BASE}/sports/{_SPORT}/events"
    params = {
        "apiKey":           api_key,
        "dateFormat":       "iso",
        "commenceTimeFrom": f"{date_str}T00:00:00Z",
        "commenceTimeTo":   f"{next_day}T05:00:00Z",  # covers through ~1am ET next day
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        logger.debug(
            "Events fetch: %s remaining requests",
            resp.headers.get("x-requests-remaining", "?"),
        )
        return resp.json()
    except requests.Timeout as e:
        logger.warning("Odds API events timeout date=%s: %s", date_str, e)
        return []
    except requests.HTTPError as e:
        logger.warning(
            "Odds API events HTTP %d date=%s",
            e.response.status_code, date_str,
        )
        return []
    except Exception as e:
        logger.error("Odds API events fetch failed date=%s: %s", date_str, e)
        return []


def _fetch_props_for_event(event_id: str, api_key: str) -> list[dict]:
    """
    Fetch HR prop lines for a single event.
    Returns a list of raw prop outcome dicts.
    """
    url = f"{_API_BASE}/sports/{_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":   api_key,
        "regions":  "us",
        "markets":  _MARKET,
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("bookmakers", [])
    except requests.Timeout as e:
        logger.warning("Odds API props timeout event=%s: %s", event_id, e)
        return []
    except requests.HTTPError as e:
        logger.warning(
            "Odds API props HTTP %d event=%s",
            e.response.status_code, event_id,
        )
        return []
    except Exception as e:
        logger.error("Odds API props fetch failed event=%s: %s", event_id, e)
        return []


def _parse_bookmaker_props(bookmakers: list[dict]) -> list[dict]:
    """
    Extract and normalise HR prop rows from the bookmakers list.
    Prefers _DEFAULT_BOOK; falls back to first available book with the market.

    Returns a list of dicts with keys:
        batter_name, market, line, over_price, under_price,
        implied_prob_over, fair_prob_over, bookmaker, fetched_at
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    rows: dict[str, dict] = {}  # batter_name -> best row

    # Sort so preferred book comes first
    sorted_books = sorted(
        bookmakers,
        key=lambda b: (0 if b.get("key") == _DEFAULT_BOOK else 1),
    )

    for book in sorted_books:
        book_key = book.get("key", "unknown")
        for market in book.get("markets", []):
            if market.get("key") != _MARKET:
                continue
            outcomes = market.get("outcomes", [])

            # Group outcomes by player name
            by_name: dict[str, dict] = {}
            for outcome in outcomes:
                name = outcome.get("description", "").strip()
                side = outcome.get("name", "").lower()  # "over" or "under"
                price = outcome.get("price")
                line  = outcome.get("point", 0.5)
                if not name or price is None:
                    continue
                if name not in by_name:
                    by_name[name] = {"line": line, "over": None, "under": None}
                if side == "over":
                    by_name[name]["over"] = int(price)
                elif side == "under":
                    by_name[name]["under"] = int(price)

            for name, sides in by_name.items():
                if sides["over"] is None or sides["under"] is None:
                    continue
                # Don't overwrite if we already have the preferred book
                if name in rows:
                    continue
                p_over  = american_to_implied_prob(sides["over"])
                p_under = american_to_implied_prob(sides["under"])
                fair_over, _ = remove_vig(p_over, p_under)
                rows[name] = {
                    "batter_name":      name,
                    "market":           _MARKET,
                    "line":             sides["line"],
                    "over_price":       sides["over"],
                    "under_price":      sides["under"],
                    "implied_prob_over": round(p_over, 4),
                    "fair_prob_over":   round(fair_over, 4),
                    "bookmaker":        book_key,
                    "fetched_at":       now_iso,
                }

    return list(rows.values())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_hr_props(
    date_str: str,
    *,
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch HR prop lines for all MLB games on date_str.

    Parameters
    ----------
    date_str      : "YYYY-MM-DD"
    api_key       : Odds API key.  Falls back to ODDS_API_KEY env var.
    force_refresh : bypass cache

    Returns
    -------
    List of prop dicts (see module docstring for schema).
    Empty list if API key is missing or API is unavailable.
    """
    cache = _cache_path(date_str)

    if not force_refresh and _cache_is_fresh(date_str):
        with open(cache) as f:
            stored = json.load(f)
        logger.debug(
            "Odds cache hit for %s (%d props)", date_str, len(stored.get("props", []))
        )
        return stored.get("props", [])

    try:
        key = _get_api_key(api_key)
    except EnvironmentError as e:
        logger.warning("%s", e)
        return []

    logger.info("Fetching HR props from Odds API for %s ...", date_str)
    events = _fetch_events_for_date(date_str, key)

    if not events:
        logger.warning("No events found for %s — odds will be unavailable", date_str)
        return []

    all_props: list[dict] = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        bookmakers = _fetch_props_for_event(event_id, key)
        props      = _parse_bookmaker_props(bookmakers)
        all_props.extend(props)
        logger.debug(
            "Event %s (%s vs %s): %d props",
            event_id,
            event.get("away_team", "?"),
            event.get("home_team", "?"),
            len(props),
        )

    logger.info(
        "Odds API: fetched %d HR props across %d events for %s",
        len(all_props), len(events), date_str,
    )

    payload = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "date":       date_str,
        "props":      all_props,
    }
    with open(cache, "w") as f:
        json.dump(payload, f, indent=2)

    return all_props


def build_odds_lookup(props: list[dict]) -> dict[str, dict]:
    """
    Build a name -> prop dict for fast lookup at predict time.
    Key is lowercased, stripped batter name.
    """
    return {p["batter_name"].lower().strip(): p for p in props}


def match_odds(
    batter_name: str,
    odds_lookup: dict[str, dict],
    threshold: int = _FUZZY_THRESHOLD,
) -> Optional[dict]:
    """
    Fuzzy-match a resolved batter name against the odds lookup.

    Uses rapidfuzz if available; falls back to exact/prefix matching.
    Returns the matched prop dict, or None if no match above threshold.
    """
    if not batter_name or not odds_lookup:
        return None

    query = batter_name.lower().strip()

    # Exact match first
    if query in odds_lookup:
        return odds_lookup[query]

    # Fuzzy match
    try:
        from rapidfuzz import process as fuzz_process
        result = fuzz_process.extractOne(query, odds_lookup.keys(), score_cutoff=threshold)
        if result:
            matched_key, score, _ = result
            logger.debug("Fuzzy match: '%s' -> '%s' (score=%d)", query, matched_key, score)
            return odds_lookup[matched_key]
    except ImportError:
        # rapidfuzz not installed — fall back to simple contains check
        for key in odds_lookup:
            last_name = query.split()[-1] if query else ""
            if last_name and last_name in key:
                return odds_lookup[key]

    return None


# ---------------------------------------------------------------------------
# Edge calculation
# ---------------------------------------------------------------------------

def compute_edge(model_prob: float, fair_prob: float) -> float:
    """
    Edge = model probability minus fair (vig-removed) market probability.

    Positive edge means the model thinks the batter is more likely to HR
    than the market implies.  Negative edge means the market is more bullish.

    Example:
        model_prob = 0.18
        fair_prob  = 0.13   (market line -130/-110 after vig removal)
        edge       = +0.05  → model is 5pp above market
    """
    return round(model_prob - fair_prob, 4)


def kelly_fraction(
    model_prob: float,
    american_odds: float,
    kelly_mult: float = 0.25,
) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.

    Uses a conservative 0.25× multiplier by default to account for model
    uncertainty.  Returns 0.0 when there is no edge.

    Parameters
    ----------
    model_prob   : model's estimated probability of the bet winning (0–1)
    american_odds: market over price in American format (e.g. -130, +110)
    kelly_mult   : fraction of full Kelly to use (default 0.25)

    Notes
    -----
    For HR props, market_over_price typically ranges from +200 to +700
    (fair prob 12–33%).  Kelly is 0.0 whenever model_prob ≤ break-even
    probability implied by the market odds, which is 1/(1 + b).

    Example
    -------
        model_prob=0.25, american_odds=+300  →  ~2.5% of bankroll
        b = 300/100 = 3.0
        break-even = 1/(1+3) = 25.0%  — model is exactly at break-even → 0.0%

        model_prob=0.30, american_odds=+300  →  ~1.25% of bankroll
        f_full = (3*0.30 - 0.70)/3 = 0.20/3 = 0.067
        f_frac = 0.067 * 0.25 = 1.67%

        model_prob=0.18, american_odds=-130  →  0.0% (model below break-even of 56.5%)
    """
    # Convert to net decimal profit per $1 risked (same as decimal_odds - 1)
    if american_odds < 0:
        b = 100.0 / abs(american_odds)
    else:
        b = american_odds / 100.0
    p = float(model_prob)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, round(f_full * kelly_mult, 4))


# ---------------------------------------------------------------------------
# DataFrame enrichment (called from predict.py)
# ---------------------------------------------------------------------------

def enrich_predictions_with_odds(
    ranked: "pd.DataFrame",  # noqa: F821  (avoid importing pd at module level)
    date_str: str,
    *,
    api_key: Optional[str] = None,
    name_col: str = "batter_name",
    force_refresh: bool = False,
) -> "pd.DataFrame":
    """
    Attach odds columns to the ranked predictions DataFrame.

    Added columns (NaN when no match):
        market_line        : 0.5 (standard; may vary if book uses 1.5)
        market_over_price  : American odds for over (e.g. -130)
        market_under_price : American odds for under
        market_implied_prob: raw implied probability (includes vig)
        market_fair_prob   : vig-removed fair probability
        edge               : model_prob - market_fair_prob
        odds_bookmaker     : which book provided the line
    """
    import pandas as pd

    props = fetch_hr_props(date_str, api_key=api_key, force_refresh=force_refresh)

    if not props:
        logger.warning("No odds available for %s — skipping odds enrichment", date_str)
        for col in ("market_line", "market_over_price", "market_under_price",
                    "market_implied_prob", "market_fair_prob", "edge",
                    "odds_bookmaker", "kelly_stake"):
            ranked[col] = float("nan")
        return ranked

    lookup = build_odds_lookup(props)

    rows_matched = 0
    result = ranked.copy()

    for col in ("market_line", "market_over_price", "market_under_price",
                "market_implied_prob", "market_fair_prob", "edge",
                "odds_bookmaker", "kelly_stake"):
        result[col] = float("nan") if col != "odds_bookmaker" else None

    for idx, row in result.iterrows():
        name = row.get(name_col)
        if not name or (isinstance(name, float)):
            continue
        prop = match_odds(str(name), lookup)
        if prop is None:
            continue
        result.at[idx, "market_line"]         = prop["line"]
        result.at[idx, "market_over_price"]   = prop["over_price"]
        result.at[idx, "market_under_price"]  = prop["under_price"]
        result.at[idx, "market_implied_prob"] = prop["implied_prob_over"]
        result.at[idx, "market_fair_prob"]    = prop["fair_prob_over"]
        result.at[idx, "edge"]                = compute_edge(
            float(row["hr_prob"]), float(prop["fair_prob_over"])
        )
        result.at[idx, "odds_bookmaker"]      = prop["bookmaker"]
        result.at[idx, "kelly_stake"]         = kelly_fraction(
            float(row["hr_prob"]), float(prop["over_price"])
        )
        rows_matched += 1

    logger.info(
        "Odds matched for %d/%d batters", rows_matched, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.logging_config import configure_logging
    load_dotenv()
    configure_logging()

    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    api_key = sys.argv[2] if len(sys.argv) > 2 else None

    props = fetch_hr_props(date, api_key=api_key)

    if not props:
        logger.warning("No props returned — check your ODDS_API_KEY and date.")
        sys.exit(1)

    logger.info("Sample HR props for %s:", date)
    for p in sorted(props, key=lambda x: x["fair_prob_over"], reverse=True)[:10]:
        print(
            f"  {p['batter_name']:<25}  "
            f"over {p['over_price']:+d}  under {p['under_price']:+d}  "
            f"implied={p['implied_prob_over']:.3f}  fair={p['fair_prob_over']:.3f}  "
            f"[{p['bookmaker']}]"
        )

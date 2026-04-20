"""
HR prop odds fetcher for the MLB HR model.

Fetches batter HR prop lines from The Odds API, caches by date, and
exposes implied probabilities for use in predict.py edge detection.

Key improvements over v1:
  - Best-line shopping: for each batter, we now scan ALL bookmakers and
    keep the highest over_price (most favourable to the bettor). This
    alone can add 2-5pp of edge vs locking to one book.
  - Vig reduced from 4.5% to 3.5%: HR props on major US books typically
    run 3-4% total margin. 4.5% was slightly pessimistic, which pushed
    fair_prob up and suppressed edge. 3.5% is more accurate for this market.
  - Both changes are non-destructive: the cache still stores per-book lines
    so you can see which book sourced the best price.

Resolution order at predict time:
  1. Fresh disk cache for today's date (TTL 2h)
  2. Live fetch from The Odds API
  3. No-match fallback (NaN) — predict.py degrades gracefully

Cache layout:
  data/cache/odds/odds_YYYY-MM-DD.json
    {
      "fetched_at": "ISO timestamp",
      "props": [
        {
          "batter_name":       "Aaron Judge",
          "market":            "batter_home_runs",
          "line":              0.5,
          "over_price":        -110,        ← best price across all books
          "under_price":       -110,
          "implied_prob_over": 0.524,
          "fair_prob_over":    0.496,
          "bookmaker":         "draftkings", ← which book had the best over price
          "fetched_at":        "ISO timestamp",
          "all_books":         {             ← full per-book breakdown for reference
            "draftkings": {"over": -110, "under": -110},
            "fanduel":    {"over": -115, "under": -105},
            ...
          }
        },
        ...
      ]
    }
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
_CACHE_TTL_HOURS = 12  # extended from 2h → 12h to conserve free-tier API credits

# Reduced from 4.5% → 3.5%: HR props on DK/FD/BetMGM typically run
# 3–4% total margin. The previous 4.5% was pessimistic and suppressed edge.
_VIG_PCT         = 0.035

_FUZZY_THRESHOLD = 85
_DEFAULT_BOOK    = "draftkings"

# Odds API returns 401 for quota-exhausted and 429 for rate-limit.
_QUOTA_STATUSES = {401, 429}


class OddsApiQuotaExceeded(Exception):
    """Raised when the Odds API responds with a quota/rate-limit status."""


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


def _load_stale_fallback(date_str: str) -> list[dict]:
    """
    Return the most recent cached odds when live fetch is unavailable.
    Preference order:
      1. Today's cache (ignoring TTL) — same-day lines even if hours old.
      2. Most recent cache of any prior date — yesterday's lines as last resort.
      3. [] — no cache anywhere; caller runs without edge/EV.
    """
    today = _cache_path(date_str)
    if today.exists():
        try:
            with open(today) as f:
                stored = json.load(f)
            props = stored.get("props", [])
            if props:
                logger.warning(
                    "Odds API unavailable — using stale today-cache for %s "
                    "(fetched %s, %d props).",
                    date_str, stored.get("fetched_at", "?"), len(props),
                )
                return props
        except Exception as e:
            logger.warning("Could not read stale today-cache: %s", e)

    try:
        candidates = sorted(
            CACHE_DIR.glob("odds_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        candidates = []
    for cand in candidates:
        if cand.name == f"odds_{date_str}.json":
            continue
        try:
            with open(cand) as f:
                stored = json.load(f)
            props = stored.get("props", [])
            if not props:
                continue  # skip empty caches — keep searching for one with data
            logger.warning(
                "Odds API unavailable and no cache for %s — falling back to "
                "%s (fetched %s, %d props). Lines may be significantly stale.",
                date_str, cand.name, stored.get("fetched_at", "?"), len(props),
            )
            return props
        except Exception as e:
            logger.warning("Could not read fallback cache %s: %s", cand.name, e)

    logger.warning(
        "No odds cache available anywhere — predict will run without edge/EV."
    )
    return []


# ---------------------------------------------------------------------------
# American odds ↔ implied probability
# ---------------------------------------------------------------------------

def american_to_implied_prob(american_odds: int) -> float:
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def remove_vig(prob_over: float, prob_under: float) -> tuple[float, float]:
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
            "No Odds API key found. Set ODDS_API_KEY in your .env file.\n"
            "Get a free key at https://the-odds-api.com"
        )
    return key


def _fetch_events_for_date(date_str: str, api_key: str) -> list[dict]:
    next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{_API_BASE}/sports/{_SPORT}/events"
    params = {
        "apiKey":           api_key,
        "dateFormat":       "iso",
        "commenceTimeFrom": f"{date_str}T00:00:00Z",
        "commenceTimeTo":   f"{next_day}T05:00:00Z",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        logger.debug("Events fetch: %s remaining requests",
                     resp.headers.get("x-requests-remaining", "?"))
        return resp.json()
    except requests.Timeout as e:
        logger.warning("Odds API events timeout date=%s: %s", date_str, e)
        return []
    except requests.HTTPError as e:
        status = e.response.status_code
        if status in _QUOTA_STATUSES:
            raise OddsApiQuotaExceeded(
                f"HTTP {status} from Odds API /events (quota or rate-limit)"
            ) from e
        logger.warning("Odds API events HTTP %d date=%s", status, date_str)
        return []
    except Exception as e:
        logger.error("Odds API events fetch failed date=%s: %s", date_str, e)
        return []


def _fetch_props_for_event(event_id: str, api_key: str) -> list[dict]:
    url = f"{_API_BASE}/sports/{_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    _MARKET,
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
        status = e.response.status_code
        if status in _QUOTA_STATUSES:
            raise OddsApiQuotaExceeded(
                f"HTTP {status} from Odds API /odds (quota or rate-limit)"
            ) from e
        logger.warning("Odds API props HTTP %d event=%s", status, event_id)
        return []
    except Exception as e:
        logger.error("Odds API props fetch failed event=%s: %s", event_id, e)
        return []


def _parse_bookmaker_props(bookmakers: list[dict]) -> list[dict]:
    """
    Extract HR prop rows from the bookmakers list.

    BEST-LINE SHOPPING: Instead of stopping at the first (preferred) book,
    we now scan ALL bookmakers and keep the highest over_price for each batter.
    The winning book is recorded in the 'bookmaker' field so you know where
    to place the bet. All per-book prices are stored in 'all_books' for
    reference and debugging.

    Uses the updated _VIG_PCT (3.5%) for fair probability calculation.
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # Collect all prices per batter across all books
    # Structure: {batter_name: {book_key: {"over": int, "under": int, "line": float}}}
    all_prices: dict[str, dict[str, dict]] = {}

    for book in bookmakers:
        book_key = book.get("key", "unknown")
        for market in book.get("markets", []):
            if market.get("key") != _MARKET:
                continue
            outcomes = market.get("outcomes", [])

            # Keep only the 0.5 (To Hit A HR) line. Many books offer alt
            # lines (1.5, 2.5) as separate "Over" outcomes under the same
            # player name, which corrupt by_name if we key only by name.
            by_name: dict[str, dict] = {}
            for outcome in outcomes:
                name  = outcome.get("description", "").strip()
                side  = outcome.get("name", "").lower()
                price = outcome.get("price")
                line  = outcome.get("point", 0.5)
                if not name or price is None:
                    continue
                if float(line) != 0.5:
                    continue
                if name not in by_name:
                    by_name[name] = {"line": 0.5, "over": None, "under": None}
                if side == "over":
                    by_name[name]["over"] = int(price)
                elif side == "under":
                    by_name[name]["under"] = int(price)

            for name, sides in by_name.items():
                # Over price is required; under is optional (some books
                # like BetOnline only post Over on HR props).
                if sides["over"] is None:
                    continue
                if name not in all_prices:
                    all_prices[name] = {}
                all_prices[name][book_key] = {
                    "over":  sides["over"],
                    "under": sides["under"],
                    "line":  sides["line"],
                }

    # For each batter, pick the book with the highest over_price (best for bettor)
    rows: list[dict] = []
    for name, book_prices in all_prices.items():
        best_book = max(book_prices, key=lambda b: book_prices[b]["over"])
        best      = book_prices[best_book]

        p_over  = american_to_implied_prob(best["over"])
        if best["under"] is not None:
            p_under = american_to_implied_prob(best["under"])
            fair_over, _ = remove_vig(p_over, p_under)
        else:
            # Over-only market (e.g. BetOnline): no under price to de-vig
            # against. Apply a flat assumed vig (_VIG_PCT) to the over
            # side so fair_prob stays comparable to two-sided quotes.
            fair_over = p_over / (1 + _VIG_PCT)

        rows.append({
            "batter_name":       name,
            "market":            _MARKET,
            "line":              best["line"],
            "over_price":        best["over"],
            "under_price":       best["under"],
            "implied_prob_over": round(p_over, 4),
            "fair_prob_over":    round(fair_over, 4),
            "bookmaker":         best_book,
            "fetched_at":        now_iso,
            "all_books":         book_prices,  # full breakdown stored for reference
        })

        if len(book_prices) > 1:
            prices_str = "  ".join(
                f"{b}:{book_prices[b]['over']:+d}" for b in sorted(book_prices)
            )
            logger.debug("Best line for %s: %s@%s (+%d) vs alternatives: %s",
                         name, best_book, best["over"],
                         best["over"] - min(bp["over"] for bp in book_prices.values()),
                         prices_str)

    return rows


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
    Returns list of prop dicts with best-line-shopped prices.
    Empty list if API key is missing or API is unavailable.
    """
    cache = _cache_path(date_str)

    if not force_refresh and _cache_is_fresh(date_str):
        with open(cache) as f:
            stored = json.load(f)
        logger.debug("Odds cache hit for %s (%d props)",
                     date_str, len(stored.get("props", [])))
        return stored.get("props", [])

    try:
        key = _get_api_key(api_key)
    except EnvironmentError as e:
        logger.warning("%s", e)
        return _load_stale_fallback(date_str)

    logger.info("Fetching HR props from Odds API for %s ...", date_str)
    try:
        events = _fetch_events_for_date(date_str, key)
    except OddsApiQuotaExceeded as e:
        logger.warning("Odds API quota exhausted on events fetch: %s", e)
        return _load_stale_fallback(date_str)

    if not events:
        logger.warning("No events found for %s — odds will be unavailable", date_str)
        return _load_stale_fallback(date_str)

    all_props: list[dict] = []
    quota_hit = False
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        try:
            bookmakers = _fetch_props_for_event(event_id, key)
        except OddsApiQuotaExceeded as e:
            logger.warning(
                "Odds API quota exhausted during props loop (%d/%d events done): %s",
                len(all_props), len(events), e,
            )
            quota_hit = True
            break
        props = _parse_bookmaker_props(bookmakers)
        all_props.extend(props)
        logger.debug("Event %s (%s vs %s): %d props",
                     event_id,
                     event.get("away_team", "?"),
                     event.get("home_team", "?"),
                     len(props))

    if quota_hit and not all_props:
        return _load_stale_fallback(date_str)

    logger.info("Odds API: fetched %d HR props across %d events for %s",
                len(all_props), len(events), date_str)

    # Don't overwrite a valid prior cache with an empty result — keeps the
    # stale-fallback path usable on later retries within the same day.
    if all_props:
        payload = {
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "date":       date_str,
            "props":      all_props,
        }
        with open(cache, "w") as f:
            json.dump(payload, f, indent=2)

    return all_props


def build_odds_lookup(props: list[dict]) -> dict[str, dict]:
    return {p["batter_name"].lower().strip(): p for p in props}


def match_odds(
    batter_name: str,
    odds_lookup: dict[str, dict],
    threshold: int = _FUZZY_THRESHOLD,
) -> Optional[dict]:
    if not batter_name or not odds_lookup:
        return None

    query = batter_name.lower().strip()

    if query in odds_lookup:
        return odds_lookup[query]

    try:
        from rapidfuzz import process as fuzz_process
        result = fuzz_process.extractOne(query, odds_lookup.keys(), score_cutoff=threshold)
        if result:
            matched_key, score, _ = result
            logger.debug("Fuzzy match: '%s' -> '%s' (score=%d)", query, matched_key, score)
            return odds_lookup[matched_key]
    except ImportError:
        for key in odds_lookup:
            last_name = query.split()[-1] if query else ""
            if last_name and last_name in key:
                return odds_lookup[key]

    return None


# ---------------------------------------------------------------------------
# Edge calculation
# ---------------------------------------------------------------------------

def compute_edge(model_prob: float, fair_prob: float) -> float:
    return round(model_prob - fair_prob, 4)


def kelly_fraction(
    model_prob: float,
    american_odds: float,
    kelly_mult: float = 0.25,
) -> float:
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
    ranked: "pd.DataFrame",
    date_str: str,
    *,
    api_key: Optional[str] = None,
    name_col: str = "batter_name",
    force_refresh: bool = False,
) -> "pd.DataFrame":
    """
    Attach best-line-shopped odds columns to the ranked predictions DataFrame.

    Added columns (NaN when no match):
        market_line         : 0.5
        market_over_price   : best American odds across all books
        market_under_price  : corresponding under price from same book
        market_implied_prob : raw implied probability (includes vig)
        market_fair_prob    : vig-removed fair probability (using 3.5% vig)
        edge                : model_prob - market_fair_prob
        odds_bookmaker      : which book had the best over line
    """
    import pandas as pd

    props = fetch_hr_props(date_str, api_key=api_key, force_refresh=force_refresh)

    if not props:
        logger.warning("No odds available for %s — skipping odds enrichment", date_str)
        for col in ("market_line", "market_over_price", "market_under_price",
                    "market_implied_prob", "market_fair_prob", "edge",
                    "odds_bookmaker", "kelly_stake"):
            ranked[col] = float("nan") if col != "odds_bookmaker" else None
        return ranked

    lookup = build_odds_lookup(props)

    rows_matched = 0
    result = ranked.copy()

    for col in ("market_line", "market_over_price", "market_under_price",
                "market_implied_prob", "market_fair_prob", "edge", "kelly_stake"):
        result[col] = float("nan")
    result["odds_bookmaker"] = None

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

    logger.info("Odds matched for %d/%d batters", rows_matched, len(result))
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

    props = fetch_hr_props(date, api_key=api_key, force_refresh=True)

    if not props:
        logger.warning("No props returned — check your ODDS_API_KEY and date.")
        sys.exit(1)

    logger.info("Sample HR props for %s (best line per batter):", date)
    for p in sorted(props, key=lambda x: x["fair_prob_over"], reverse=True)[:15]:
        n_books = len(p.get("all_books", {}))
        print(
            f"  {p['batter_name']:<25}  "
            f"best: {p['bookmaker']:<12} over {p['over_price']:+d}  "
            f"fair={p['fair_prob_over']:.3f}  "
            f"({n_books} book{'s' if n_books != 1 else ''})"
        )

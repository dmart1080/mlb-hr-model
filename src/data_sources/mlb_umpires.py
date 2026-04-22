"""
Home-plate umpire fetcher for the MLB HR model.

Pulls the HP umpire from `/game/{game_pk}/boxscore` (reliable pre-game,
usually populated a few hours before first pitch) and caches per game_pk.

The HP ump has a measurable effect on HR rates: a bigger strike zone means
more strikeouts and fewer PAs per game; a smaller zone means more walks and
HR-eligible contact opportunities. Treating `hp_ump_id` as a LightGBM
categorical lets the model learn per-ump tendencies without us hard-coding
factors.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "umpires"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_API_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT  = 10
_MAX_WORKERS = 8


def _cache_path(game_pk: int) -> Path:
    return CACHE_DIR / f"ump_{game_pk}.json"


def fetch_hp_umpire(game_pk: int, *, force_refresh: bool = False) -> Optional[dict]:
    """
    Return {'game_pk': int, 'hp_ump_id': int|None, 'hp_ump_name': str|None}.
    Empty fields (name=None, id=None) when the assignment isn't posted yet.
    Returns None only on total fetch failure.
    """
    cache = _cache_path(game_pk)
    if cache.exists() and not force_refresh:
        try:
            with open(cache) as f:
                cached = json.load(f)
            # Only trust cache if it has a real ump id — empty caches get
            # retried so the once-posted assignment lands eventually.
            if cached.get("hp_ump_id"):
                return cached
        except Exception as e:
            logger.warning("Bad ump cache for %d: %s", game_pk, e)

    url = f"{_API_BASE}/game/{game_pk}/boxscore"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        officials = resp.json().get("officials", []) or []
    except Exception as e:
        logger.warning("Ump fetch failed for game_pk=%d: %s", game_pk, e)
        return None

    hp_id: Optional[int]   = None
    hp_name: Optional[str] = None
    for o in officials:
        if o.get("officialType") == "Home Plate":
            hp_id   = o.get("official", {}).get("id")
            hp_name = o.get("official", {}).get("fullName")
            break

    result = {
        "game_pk":      int(game_pk),
        "hp_ump_id":    int(hp_id) if hp_id is not None else None,
        "hp_ump_name":  hp_name,
    }
    try:
        with open(cache, "w") as f:
            json.dump(result, f)
    except Exception as e:
        logger.warning("Could not write ump cache for %d: %s", game_pk, e)
    return result


def fetch_hp_umpires_for_games(
    game_pks: list[int],
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Parallel fetch HP ump for a list of game_pks.
    Returns DataFrame with columns: game_pk, hp_ump_id, hp_ump_name.
    """
    unique = list(dict.fromkeys(int(g) for g in game_pks))
    if not unique:
        return pd.DataFrame(columns=["game_pk", "hp_ump_id", "hp_ump_name"])

    # Count how many would be cache hits vs live fetches — logged for
    # visibility during backfills.
    cached_count = sum(
        1 for g in unique
        if _cache_path(g).exists() and not force_refresh
    )
    logger.info(
        "Umpire fetch: %d games total (%d cached, %d to fetch)",
        len(unique), cached_count, len(unique) - cached_count,
    )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        fut_to_pk = {
            ex.submit(fetch_hp_umpire, pk, force_refresh=force_refresh): pk
            for pk in unique
        }
        for fut in as_completed(fut_to_pk):
            pk  = fut_to_pk[fut]
            try:
                r = fut.result()
            except Exception as e:
                logger.warning("Ump future failed game_pk=%d: %s", pk, e)
                r = None
            if r is None:
                rows.append({
                    "game_pk":     pk,
                    "hp_ump_id":   None,
                    "hp_ump_name": None,
                })
            else:
                rows.append(r)

    df = pd.DataFrame(rows)
    # Ensure id column is nullable int so downstream merges don't coerce NaN-only
    # columns to float64 silently. Int64 accepts pd.NA and survives parquet.
    df["hp_ump_id"] = df["hp_ump_id"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.logging_config import configure_logging
    configure_logging()

    if len(sys.argv) < 2:
        print("usage: python -m src.data_sources.mlb_umpires <game_pk> [<game_pk>...]")
        sys.exit(1)

    pks = [int(x) for x in sys.argv[1:]]
    df  = fetch_hp_umpires_for_games(pks, force_refresh=True)
    print(df.to_string(index=False))
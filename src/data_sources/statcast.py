from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball.statcast import statcast


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StatcastFetchResult:
    df: pd.DataFrame
    cache_path: Path
    from_cache: bool


def _normalize_date(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%Y-%m-%d")


def _cache_filename(start_date: str, end_date: str) -> str:
    return f"statcast_{start_date}_to_{end_date}.parquet"


# Columns we always want cached — superset of everything build_features needs.
# game_type is required so we can filter to regular season (R) only.
# hc_x / hc_y are required for pulled-ball rate features.
REQUIRED_CACHE_COLS = {
    "p_throws", "stand", "release_speed", "pitch_type", "game_type",
    "hc_x", "hc_y",    # spray chart coordinates for pull-airball rate
    "bb_type",          # batted ball type (fly_ball/line_drive/ground_ball/popup)
}

# Regular season game type code in the Statcast / MLB Stats API.
REGULAR_SEASON_GAME_TYPE = "R"


# Default freshness window — Statcast revises pitch classifications and
# batted-ball outcomes for ~1-2 days after a game, and late West Coast games
# may not be fully ingested until ~6am ET the next day. Within this window
# we always re-fetch and overwrite the cache; older dates are trusted.
DEFAULT_FRESH_WINDOW_DAYS = 3


def fetch_statcast_events(
    start_date: str,
    end_date: str,
    *,
    force_refresh: bool = False,
    columns: Optional[list[str]] = None,
    regular_season_only: bool = True,
    fresh_window_days: int = DEFAULT_FRESH_WINDOW_DAYS,
) -> StatcastFetchResult:
    """
    Download Statcast events between start_date and end_date (inclusive),
    cache the results, and return a DataFrame.

    Freshness handling
    ------------------
    Recent dates (within `fresh_window_days` of today) are always re-fetched
    and the cache for those dates is overwritten on every call. Older dates
    are trusted from cache. When the requested range straddles the cutoff,
    the call is split: an "old" sub-range (cached) plus a "fresh" sub-range
    (re-downloaded), each written to its own range-keyed cache file. The
    existing cache layout is preserved — we just may produce two files
    instead of one for straddling requests.

    Set `fresh_window_days=0` to fully trust cache (useful for backfills /
    reproducible runs). `force_refresh=True` overrides everything and
    re-downloads the entire requested range as a single file.

    Parameters
    ----------
    start_date : str  YYYY-MM-DD
    end_date   : str  YYYY-MM-DD
    force_refresh : bool
        If True, re-download the full requested range as a single file,
        bypassing the freshness split.
    columns : Optional[list[str]]
        If provided, returns only these columns (if they exist).
    regular_season_only : bool  (default True)
        If True, filter to game_type == 'R' before returning.
        Full unfiltered dataset is still cached.
    fresh_window_days : int  (default 3)
        Days back from today that are always re-fetched. 0 = trust cache.
    """
    start_date = _normalize_date(start_date)
    end_date   = _normalize_date(end_date)

    if force_refresh or fresh_window_days <= 0:
        # No split — single-range path with whatever force_refresh the
        # caller asked for. This is also the recursive base case.
        return _fetch_single_range(
            start_date, end_date,
            force_refresh=force_refresh,
            columns=columns,
            regular_season_only=regular_season_only,
        )

    today  = pd.Timestamp.today().normalize()
    cutoff = (today - pd.Timedelta(days=fresh_window_days)).strftime("%Y-%m-%d")
    # `cutoff` is the first day we no longer trust the cache for.

    if end_date < cutoff:
        logger.info(
            "Statcast cache: trusting [%s, %s] (all older than %s)",
            start_date, end_date, cutoff,
        )
        return _fetch_single_range(
            start_date, end_date,
            force_refresh=False,
            columns=columns,
            regular_season_only=regular_season_only,
        )

    if start_date >= cutoff:
        logger.info(
            "Statcast cache: refreshing [%s, %s] (within %d-day fresh window)",
            start_date, end_date, fresh_window_days,
        )
        return _fetch_single_range(
            start_date, end_date,
            force_refresh=True,
            columns=columns,
            regular_season_only=regular_season_only,
        )

    # Straddle: split at the cutoff.
    older_end = (pd.to_datetime(cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(
        "Statcast cache: trusting [%s, %s], refreshing [%s, %s]",
        start_date, older_end, cutoff, end_date,
    )
    older = _fetch_single_range(
        start_date, older_end,
        force_refresh=False,
        columns=columns,
        regular_season_only=regular_season_only,
    )
    fresh = _fetch_single_range(
        cutoff, end_date,
        force_refresh=True,
        columns=columns,
        regular_season_only=regular_season_only,
    )
    df = pd.concat([older.df, fresh.df], ignore_index=True)
    # Report the fresh-side cache_path (more useful for debugging recent issues)
    # and from_cache=False because at least part of the result was re-fetched.
    return StatcastFetchResult(df=df, cache_path=fresh.cache_path, from_cache=False)


def _fetch_single_range(
    start_date: str,
    end_date: str,
    *,
    force_refresh: bool,
    columns: Optional[list[str]],
    regular_season_only: bool,
) -> StatcastFetchResult:
    """Single (start, end) range fetch with the original cache logic."""
    cache_path = CACHE_DIR / _cache_filename(start_date, end_date)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)

        # If cache is missing any required columns, force a refresh
        missing_cols = REQUIRED_CACHE_COLS - set(df.columns)
        if missing_cols:
            logger.warning(
                "Cache missing columns %s — re-downloading %s -> %s",
                missing_cols, start_date, end_date,
            )
            cache_path.unlink()
        else:
            logger.debug("Cache hit: %s (%d rows)", cache_path.name, len(df))
            df = _apply_filters(df, regular_season_only=regular_season_only)
            if columns:
                keep = [c for c in columns if c in df.columns]
                df = df[keep]
            return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=True)

    logger.info("Downloading from Statcast: %s -> %s", start_date, end_date)
    df = statcast(start_dt=start_date, end_dt=end_date)
    logger.info("Downloaded %d rows from Statcast (all game types)", len(df))

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Cache the full unfiltered dataset so we never need to re-download
    # just because the filter changes.
    df.to_parquet(cache_path, index=False)
    logger.debug("Cached to %s", cache_path.name)

    df = _apply_filters(df, regular_season_only=regular_season_only)

    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]

    return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=False)


def _apply_filters(df: pd.DataFrame, *, regular_season_only: bool) -> pd.DataFrame:
    """
    Apply post-cache filters.  Currently only regular-season filtering,
    but structured so future filters (e.g. min pitch speed) are easy to add.
    """
    if not regular_season_only:
        return df

    if "game_type" not in df.columns:
        logger.warning(
            "game_type column not found in Statcast data — "
            "cannot filter to regular season. All game types will be used. "
            "Delete the cache file and re-download to pick up game_type."
        )
        return df

    before = len(df)
    df = df[df["game_type"] == REGULAR_SEASON_GAME_TYPE].copy()
    removed = before - len(df)

    if removed > 0:
        logger.info(
            "Regular-season filter: removed %d rows (%.1f%%) of non-regular-season "
            "game types. %d rows remain.",
            removed, 100 * removed / max(before, 1), len(df),
        )
    else:
        logger.debug("Regular-season filter: all %d rows are game_type=R", len(df))

    return df


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()

    result = fetch_statcast_events(
        start_date="2024-06-01",
        end_date="2024-06-03",
        columns=[
            "game_date", "game_pk", "batter", "pitcher", "events",
            "p_throws", "stand", "release_speed", "pitch_type", "game_type",
            "hc_x", "hc_y",
        ],
    )
    logger.info("Loaded rows: %d", len(result.df))
    logger.info("Cache file:  %s", result.cache_path)
    logger.info("From cache:  %s", result.from_cache)
    if "game_type" in result.df.columns:
        logger.info("Game types:  %s", result.df["game_type"].value_counts().to_dict())
    if "hc_x" in result.df.columns:
        logger.info("hc_x nulls:  %d / %d", result.df["hc_x"].isna().sum(), len(result.df))
    print(result.df.head(10))

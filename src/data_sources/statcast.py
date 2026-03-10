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


# Columns we always want cached — superset of everything build_features needs
REQUIRED_CACHE_COLS = {"p_throws", "stand", "release_speed", "pitch_type"}


def fetch_statcast_events(
    start_date: str,
    end_date: str,
    *,
    force_refresh: bool = False,
    columns: Optional[list[str]] = None,
) -> StatcastFetchResult:
    """
    Download Statcast events between start_date and end_date (inclusive),
    cache the results, and return a DataFrame.

    Parameters
    ----------
    start_date : str  YYYY-MM-DD
    end_date   : str  YYYY-MM-DD
    force_refresh : bool
        If True, re-download even if cached file exists.
    columns : Optional[list[str]]
        If provided, returns only these columns (if they exist).
    """
    start_date = _normalize_date(start_date)
    end_date   = _normalize_date(end_date)

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
            if columns:
                keep = [c for c in columns if c in df.columns]
                df = df[keep]
            return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=True)

    logger.info("Downloading from Statcast: %s -> %s", start_date, end_date)
    df = statcast(start_dt=start_date, end_dt=end_date)
    logger.info("Downloaded %d rows from Statcast", len(df))

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    df.to_parquet(cache_path, index=False)
    logger.debug("Cached to %s", cache_path.name)

    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]

    return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=False)


if __name__ == "__main__":
    from src.logging_config import configure_logging
    configure_logging()

    result = fetch_statcast_events(
        start_date="2024-06-01",
        end_date="2024-06-03",
        columns=[
            "game_date", "game_pk", "batter", "pitcher", "events",
            "p_throws", "stand", "release_speed", "pitch_type",
        ],
    )
    logger.info("Loaded rows: %d", len(result.df))
    logger.info("Cache file:  %s", result.cache_path)
    logger.info("From cache:  %s", result.from_cache)
    print(result.df.head(10))

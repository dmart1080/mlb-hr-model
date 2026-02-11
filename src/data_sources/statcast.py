from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball.statcast import statcast



# Project root is the folder that contains /src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StatcastFetchResult:
    df: pd.DataFrame
    cache_path: Path
    from_cache: bool


def _normalize_date(date_str: str) -> str:
    """
    Ensures a date string is in YYYY-MM-DD format.
    Raises ValueError if it isn't.
    """
    return pd.to_datetime(date_str).strftime("%Y-%m-%d")


def _cache_filename(start_date: str, end_date: str) -> str:
    return f"statcast_{start_date}_to_{end_date}.parquet"


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
    start_date : str
        YYYY-MM-DD
    end_date : str
        YYYY-MM-DD
    force_refresh : bool
        If True, re-download even if cached file exists.
    columns : Optional[list[str]]
        If provided, returns only these columns (if they exist).

    Returns
    -------
    StatcastFetchResult
        Contains df, cache_path, from_cache
    """
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)

    cache_path = CACHE_DIR / _cache_filename(start_date, end_date)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=True)

    # Download via pybaseball
    df = statcast(start_dt=start_date, end_dt=end_date)

    # Basic cleanup: ensure game_date is a date (not string)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Save cache
    df.to_parquet(cache_path, index=False)

    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]

    return StatcastFetchResult(df=df, cache_path=cache_path, from_cache=False)


if __name__ == "__main__":
    # Quick manual test:
    # python src/data_sources/statcast.py
    result = fetch_statcast_events(
        start_date="2024-06-01",
        end_date="2024-06-03",
        columns=["game_date", "game_pk", "batter", "pitcher", "events"],
    )
    print(f"Loaded rows: {len(result.df):,}")
    print(f"Cache file: {result.cache_path}")
    print(f"From cache: {result.from_cache}")
    print(result.df.head(10))

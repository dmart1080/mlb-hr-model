"""
Collect Statcast data for yesterday (or a specified date) and cache it locally.

Usage
-----
    # Fetch yesterday (default):
    python -m src.data_sources.collect_daily_statcast

    # Fetch a specific date:
    python -m src.data_sources.collect_daily_statcast --date 2026-06-01

    # Force re-download even if cache already exists:
    python -m src.data_sources.collect_daily_statcast --force

This script is designed to be the first step in the GitHub Actions morning
workflow. It populates data/cache/ so that build_today_features.py (called by
predict.py) finds a warm cache and doesn't need to download during the
prediction run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from src.data_sources.statcast import fetch_statcast_events
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def collect(target_date: str, *, force: bool = False) -> None:
    logger.info("Collecting Statcast data for %s (force=%s)", target_date, force)

    result = fetch_statcast_events(
        start_date=target_date,
        end_date=target_date,
        force_refresh=force,
        # fresh_window_days=0 would trust cache; keep default (3) so
        # yesterday is always refreshed (Statcast revises data overnight).
        fresh_window_days=3,
    )

    logger.info(
        "Statcast %s: %d rows | cache=%s | from_cache=%s",
        target_date,
        len(result.df),
        result.cache_path.name,
        result.from_cache,
    )

    if result.df.empty:
        logger.warning("No data returned for %s — may be an off-day.", target_date)
        sys.exit(0)

    logger.info("Done. Cache path: %s", result.cache_path)


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Download yesterday's Statcast data into the local cache.",
    )
    parser.add_argument(
        "--date",
        default=_yesterday(),
        help="Date to collect (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if a cache file already exists.",
    )
    args = parser.parse_args()

    collect(args.date, force=args.force)

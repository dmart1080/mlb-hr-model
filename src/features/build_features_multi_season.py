from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from src.logging_config import configure_logging
from src.features.build_features import build_features_for_range

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    ("2021-04-01", "2021-10-03"),
    ("2022-04-07", "2022-10-05"),
    ("2023-03-30", "2023-10-01"),
    ("2024-03-20", "2024-10-01"),
    ("2025-03-27", "2025-10-01"),
]

# ---------------------------------------------------------------------------
# TEST MODE
# Set to True to build only one month of 2025 data and verify the full
# pipeline before committing to a full multi-season rebuild.
# ---------------------------------------------------------------------------
TEST_MODE = True
TEST_RANGE = ("2025-03-27", "2025-04-30")


def _build_month_ranges(seasons: list[tuple[str, str]]) -> list[tuple[str, str]]:
    ranges: list[tuple[str, str]] = []
    for season_start, season_end in seasons:
        s = pd.Timestamp(season_start)
        e = pd.Timestamp(season_end)
        cursor = s
        while cursor <= e:
            month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
            chunk_end = min(month_end, e)
            ranges.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            cursor = chunk_end + pd.Timedelta(days=1)
    return ranges


def build_month(start: str, end: str) -> Path:
    out_path = PROCESSED_DIR / f"train_table_{start}_to_{end}.parquet"
    if out_path.exists():
        logger.info("Skipping (already exists): %s", out_path.name)
        return out_path

    result = build_features_for_range(start, end)
    logger.info(
        "Saved: %s | rows=%d | hr_rate=%.4f",
        result.output_path.name,
        len(result.features_df),
        result.features_df["hr_hit"].mean(),
    )
    return result.output_path


def build_multi_season(output_name: str = "train_table_2021_2025_full.parquet") -> Path:
    """
    Build (or resume) feature tables for all seasons in SEASONS,
    then concatenate into a single parquet file.

    If TEST_MODE is True, only builds TEST_RANGE and saves a separate
    test output so it doesn't overwrite any existing full table.
    """
    if TEST_MODE:
        start, end = TEST_RANGE
        logger.info("=" * 50)
        logger.info("TEST MODE — building %s -> %s only", start, end)
        logger.info("=" * 50)
        month_files = [build_month(start, end)]
        out_name = f"train_table_TEST_{start}_to_{end}.parquet"
    else:
        month_ranges = _build_month_ranges(SEASONS)
        month_files = []
        for i, (start, end) in enumerate(month_ranges, 1):
            logger.info("[%d/%d] Building %s -> %s", i, len(month_ranges), start, end)
            month_files.append(build_month(start, end))
        out_name = output_name

    logger.info("Concatenating %d month files ...", len(month_files))
    dfs = [pd.read_parquet(p) for p in month_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    after = len(combined)
    if before != after:
        logger.info("Dropped %d duplicate (game_pk, batter) rows", before - after)

    out_path = PROCESSED_DIR / out_name
    combined.to_parquet(out_path, index=False)

    logger.info(
        "%sfile saved: %s | rows=%d | hr_rate=%.4f | dates=%s -> %s",
        "TEST " if TEST_MODE else "",
        out_path.name,
        len(combined),
        combined["hr_hit"].mean(),
        combined["game_date"].min().date(),
        combined["game_date"].max().date(),
    )

    new_cols = ["batting_order_pos", "is_top_of_order", "expected_pa_today", "relief_pa_pct"]
    present = [c for c in new_cols if c in combined.columns]
    if present:
        logger.debug("New feature columns present: %s", present)

    if not TEST_MODE:
        _log_season_breakdown(combined)

    if TEST_MODE:
        logger.info("Test run complete. Set TEST_MODE = False to build all seasons.")

    return out_path


def _log_season_breakdown(df: pd.DataFrame) -> None:
    df = df.copy()
    df["season"] = df["game_date"].dt.year
    summary = (
        df.groupby("season")
        .agg(rows=("hr_hit", "size"), hr_rate=("hr_hit", "mean"))
        .reset_index()
    )
    logger.info("Season breakdown:")
    for _, row in summary.iterrows():
        logger.info("  %d: %8d rows  hr_rate=%.4f", int(row.season), int(row.rows), row.hr_rate)


if __name__ == "__main__":
    configure_logging()
    build_multi_season()

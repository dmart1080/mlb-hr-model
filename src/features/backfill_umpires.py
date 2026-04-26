"""
One-off backfill: add hp_ump_id / hp_ump_name to historical monthly parquets
so the model can use umpire features on prior seasons.

Monthly parquets were built before `fetch_hp_umpires_for_games` was wired into
build_features.py — this script fills in the missing columns without a full
rebuild. Run once, then the next monthly retrain will produce umpire-aware
parquets natively.

Usage:
    python -m src.features.backfill_umpires              # backfill all monthlies + regenerate full
    python -m src.features.backfill_umpires --skip-full  # monthlies only (faster iteration)
    python -m src.features.backfill_umpires --force      # rewrite even if hp_ump_id already present
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.data_sources.mlb_umpires import fetch_hp_umpires_for_games
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MONTHLY_RE = re.compile(r"^train_table_\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}\.parquet$")
FULL_NAME  = "train_table_2021_2026_full.parquet"


def _monthly_parquets() -> list[Path]:
    return sorted(p for p in PROCESSED_DIR.glob("train_table_*.parquet") if MONTHLY_RE.match(p.name))


def backfill_parquet(path: Path, *, force: bool = False) -> tuple[int, int]:
    """Add hp_ump_id + hp_ump_name columns to a monthly parquet in place.

    Returns (n_games, n_with_ump). Skips (and returns 0,0) when already populated
    unless force=True.
    """
    df = pd.read_parquet(path)
    if df.empty:
        logger.info("  %s: empty, skipping", path.name)
        return 0, 0

    if "hp_ump_id" in df.columns and not force:
        n_with = int(df["hp_ump_id"].notna().sum())
        logger.info("  %s: already has hp_ump_id (%d rows populated), skipping", path.name, n_with)
        return 0, 0

    game_pks = sorted(df["game_pk"].dropna().astype(int).unique().tolist())
    logger.info("  %s: fetching umpires for %d games", path.name, len(game_pks))
    ump_df = fetch_hp_umpires_for_games(game_pks)

    if "hp_ump_id" in df.columns:
        df = df.drop(columns=["hp_ump_id", "hp_ump_name"], errors="ignore")

    df["game_pk"] = df["game_pk"].astype(int)
    ump_df["game_pk"] = ump_df["game_pk"].astype(int)
    merged = df.merge(ump_df, on="game_pk", how="left")

    n_games    = len(game_pks)
    n_with_ump = int(merged.loc[merged["hp_ump_id"].notna(), "game_pk"].nunique())
    logger.info(
        "  %s: %d/%d games have hp_ump_id (%.1f%%)",
        path.name, n_with_ump, n_games, 100 * n_with_ump / max(n_games, 1),
    )

    merged.to_parquet(path, index=False)
    return n_games, n_with_ump


def regenerate_full(output_name: str = FULL_NAME) -> Path:
    monthlies = _monthly_parquets()
    logger.info("Concatenating %d monthly parquets into %s ...", len(monthlies), output_name)
    dfs = []
    for p in monthlies:
        try:
            d = pd.read_parquet(p)
            if not d.empty:
                dfs.append(d)
        except Exception as e:
            logger.warning("  Skipping unreadable %s: %s", p.name, e)
    if not dfs:
        raise ValueError("No monthly parquets to concatenate.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    after = len(combined)
    if before != after:
        logger.info("  Dropped %d duplicate (game_pk, batter) rows", before - after)

    out_path = PROCESSED_DIR / output_name
    combined.to_parquet(out_path, index=False)

    n_ump_rows = int(combined["hp_ump_id"].notna().sum()) if "hp_ump_id" in combined.columns else 0
    logger.info(
        "Wrote %s | rows=%d | dates=%s -> %s | hp_ump_id populated on %d/%d rows (%.1f%%)",
        out_path.name, len(combined),
        combined["game_date"].min().date(), combined["game_date"].max().date(),
        n_ump_rows, len(combined), 100 * n_ump_rows / max(len(combined), 1),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-full", action="store_true", help="Don't regenerate the combined full parquet")
    parser.add_argument("--force",     action="store_true", help="Re-fetch even if hp_ump_id already present")
    args = parser.parse_args()

    configure_logging()
    monthlies = _monthly_parquets()
    logger.info("Found %d monthly parquets", len(monthlies))

    total_games = 0
    total_with  = 0
    for p in monthlies:
        n_g, n_w = backfill_parquet(p, force=args.force)
        total_games += n_g
        total_with  += n_w

    logger.info("Backfill done: %d/%d games matched (%.1f%%)",
                total_with, total_games, 100 * total_with / max(total_games, 1))

    if not args.skip_full:
        regenerate_full()


if __name__ == "__main__":
    main()
"""
Export a slim (last 90 days) slice of the full train table.

Run this locally after every monthly retrain:

    python scripts/export_slim_table.py

Writes data/processed/train_table_slim_90d.parquet (~5-20 MB), which is
committed to git so that GitHub Actions workflows can run build_today_features
carry-forward without needing the full multi-GB history file.

The 90-day window is enough for all rolling features (max window = 30 days
plus carry-forward lookback). It comfortably fits in a GitHub repo without LFS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "train_table_slim_90d.parquet"

WINDOW_DAYS = 90

FULL_TABLE_CANDIDATES = [
    PROCESSED_DIR / "train_table_2021_2026_full.parquet",
    PROCESSED_DIR / "train_table_2021_2025_full.parquet",
    PROCESSED_DIR / "train_table_2024_full_season.parquet",
]


def main() -> None:
    src_path = next((p for p in FULL_TABLE_CANDIDATES if p.exists()), None)
    if src_path is None:
        print(
            "ERROR: No full train table found in data/processed/.\n"
            "Run build_features.py first to generate the training data.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {src_path.name} ...")
    df = pd.read_parquet(src_path)

    if "game_date" not in df.columns:
        print("ERROR: 'game_date' column not found in train table.", file=sys.stderr)
        sys.exit(1)

    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = df["game_date"].max() - pd.Timedelta(days=WINDOW_DAYS)
    slim = df[df["game_date"] >= cutoff].copy()

    print(
        f"Full table: {len(df):,} rows  |  "
        f"Slim ({WINDOW_DAYS}d): {len(slim):,} rows  |  "
        f"Date range: {slim['game_date'].min().date()} → {slim['game_date'].max().date()}"
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Wrote {OUT_PATH.name}  ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  git add data/processed/train_table_slim_90d.parquet")
    print('  git commit -m "chore: update slim train table (90d)"')
    print("  git push")


if __name__ == "__main__":
    main()

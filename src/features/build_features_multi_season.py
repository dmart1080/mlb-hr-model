from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import build_features_for_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Season definitions
# Each tuple is (season_start, season_end).
# build_features_for_range pulls 60 days of history before season_start
# automatically for rolling-window features.
# ---------------------------------------------------------------------------
SEASONS = [
    ("2021-04-01", "2021-10-03"),
    ("2022-04-07", "2022-10-05"),
    ("2023-03-30", "2023-10-01"),
    ("2024-03-20", "2024-10-01"),
    ("2025-03-27", "2025-10-01"),  # update end date as season progresses
]

# ---------------------------------------------------------------------------
# TEST MODE
# Set to True to build only one month of 2025 data and verify the full
# pipeline (MLB schedule API, batting orders, starter assignment) works
# before committing to a full multi-season rebuild.
#
# Usage:
#   TEST_MODE = True   → builds 2025-03-27 → 2025-04-30 only
#   TEST_MODE = False  → builds all seasons defined in SEASONS above
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
        print(f"  Skipping (already exists): {out_path.name}")
        return out_path

    result = build_features_for_range(start, end)
    print(
        f"  Saved: {result.output_path.name} "
        f"| rows={len(result.features_df):,} "
        f"| hr_rate={result.features_df['hr_hit'].mean():.4f}"
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
        print(f"\n{'='*50}")
        print(f"TEST MODE — building {start} → {end} only")
        print(f"{'='*50}\n")
        month_files = [build_month(start, end)]
        out_name = f"train_table_TEST_{start}_to_{end}.parquet"
    else:
        month_ranges = _build_month_ranges(SEASONS)
        month_files = []
        for i, (start, end) in enumerate(month_ranges, 1):
            print(f"\n[{i}/{len(month_ranges)}] Building {start} → {end}")
            month_files.append(build_month(start, end))
        out_name = output_name

    print("\nConcatenating months …")
    dfs = [pd.read_parquet(p) for p in month_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    # Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    after = len(combined)
    if before != after:
        print(f"  Dropped {before - after:,} duplicate (game_pk, batter) rows.")

    out_path = PROCESSED_DIR / out_name
    combined.to_parquet(out_path, index=False)

    print(f"\n✅  {'TEST ' if TEST_MODE else ''}file saved: {out_path.name}")
    print(f"   Total rows : {len(combined):,}")
    print(f"   Date range : {combined['game_date'].min().date()} → {combined['game_date'].max().date()}")
    print(f"   HR rate    : {combined['hr_hit'].mean():.4f}")

    # Show new columns if present
    new_cols = ["batting_order_pos", "is_top_of_order", "expected_pa_today", "relief_pa_pct"]
    present = [c for c in new_cols if c in combined.columns]
    if present:
        print(f"\n  New feature columns present: {present}")
        print(combined[present].describe().to_string())

    if not TEST_MODE:
        _print_season_breakdown(combined)

    if TEST_MODE:
        print(
            "\n  ✔  Test run complete. Set TEST_MODE = False to build all seasons."
        )

    return out_path


def _print_season_breakdown(df: pd.DataFrame) -> None:
    df = df.copy()
    df["season"] = df["game_date"].dt.year
    summary = (
        df.groupby("season")
        .agg(rows=("hr_hit", "size"), hr_rate=("hr_hit", "mean"))
        .reset_index()
    )
    print("\n  Season breakdown:")
    for _, row in summary.iterrows():
        print(f"    {int(row.season)}: {int(row.rows):>8,} rows  hr_rate={row.hr_rate:.4f}")


if __name__ == "__main__":
    build_multi_season()

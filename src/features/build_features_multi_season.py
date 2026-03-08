from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import build_features_for_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Each tuple is (season_start, season_end).
# build_features_for_range will automatically pull 60 days of history
# before season_start for rolling-window features.
SEASONS = [
    ("2021-04-01", "2021-10-03"),
    ("2022-04-07", "2022-10-05"),
    ("2023-03-30", "2023-10-01"),
    ("2024-03-20", "2024-10-01"),
    ("2025-03-27", "2025-10-01"),  # update end date as season progresses
]

# Build month-by-month within each season to keep Statcast fetches small
# and allow resume-on-failure (each month file is cached individually).
MONTH_RANGES: list[tuple[str, str]] = []
for season_start, season_end in SEASONS:
    s = pd.Timestamp(season_start)
    e = pd.Timestamp(season_end)
    cursor = s
    while cursor <= e:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        chunk_end = min(month_end, e)
        MONTH_RANGES.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)


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
    """
    month_files: list[Path] = []

    for i, (start, end) in enumerate(MONTH_RANGES, 1):
        print(f"\n[{i}/{len(MONTH_RANGES)}] Building {start} → {end}")
        month_files.append(build_month(start, end))

    print("\nConcatenating all months …")
    dfs: list[pd.DataFrame] = []
    for p in month_files:
        df = pd.read_parquet(p)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    # Deduplicate: keep last occurrence of any (game_pk, batter) duplicate
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk", "batter"], keep="last")
    after = len(combined)
    if before != after:
        print(f"  Dropped {before - after:,} duplicate (game_pk, batter) rows.")

    out_path = PROCESSED_DIR / output_name
    combined.to_parquet(out_path, index=False)

    print(f"\n✅  Multi-season file saved: {out_path.name}")
    print(f"   Total rows : {len(combined):,}")
    print(f"   Date range : {combined['game_date'].min().date()} → {combined['game_date'].max().date()}")
    print(f"   HR rate    : {combined['hr_hit'].mean():.4f}")
    _print_season_breakdown(combined)
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

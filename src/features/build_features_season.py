from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import build_features_for_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Season definitions — extend here to add more years
# ---------------------------------------------------------------------------

SEASON_MONTHS: dict[int, list[tuple[str, str]]] = {
    2022: [
        ("2022-04-01", "2022-04-30"),
        ("2022-05-01", "2022-05-31"),
        ("2022-06-01", "2022-06-30"),
        ("2022-07-01", "2022-07-31"),
        ("2022-08-01", "2022-08-31"),
        ("2022-09-01", "2022-09-30"),
        ("2022-10-01", "2022-10-05"),
    ],
    2023: [
        ("2023-03-30", "2023-03-31"),
        ("2023-04-01", "2023-04-30"),
        ("2023-05-01", "2023-05-31"),
        ("2023-06-01", "2023-06-30"),
        ("2023-07-01", "2023-07-31"),
        ("2023-08-01", "2023-08-31"),
        ("2023-09-01", "2023-09-30"),
        ("2023-10-01", "2023-10-01"),
    ],
    2024: [
        ("2024-03-20", "2024-03-31"),
        ("2024-04-01", "2024-04-30"),
        ("2024-05-01", "2024-05-31"),
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-07-31"),
        ("2024-08-01", "2024-08-31"),
        ("2024-09-01", "2024-09-30"),
        ("2024-10-01", "2024-10-01"),
    ],
}


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

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


def build_season(year: int) -> Path:
    """Build and concatenate all monthly feature tables for a given season year."""
    months = SEASON_MONTHS.get(year)
    if months is None:
        raise ValueError(f"No season definition for year {year}. Add it to SEASON_MONTHS.")

    print(f"\n{'='*50}")
    print(f"Building season {year}")
    print(f"{'='*50}")

    month_files = []
    for start, end in months:
        print(f"\n--- {start} to {end} ---")
        month_files.append(build_month(start, end))

    dfs = []
    for p in month_files:
        print(f"Loading {p.name}")
        dfs.append(pd.read_parquet(p))

    season_df = pd.concat(dfs, ignore_index=True)
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])
    season_df = season_df.sort_values("game_date").reset_index(drop=True)

    out_path = PROCESSED_DIR / f"train_table_{year}_full_season.parquet"
    season_df.to_parquet(out_path, index=False)

    print(f"\n✅ Season {year} saved: {out_path.name}")
    print(f"   Rows: {len(season_df):,}  |  HR rate: {season_df['hr_hit'].mean():.4f}")
    return out_path


def build_multi_season(years: list[int]) -> Path:
    """
    Build each season independently (resume-friendly), then concatenate
    into a single combined training table.
    """
    season_files = []
    for year in years:
        season_files.append(build_season(year))

    print(f"\n{'='*50}")
    print(f"Combining seasons: {years}")
    print(f"{'='*50}")

    dfs = []
    for p in season_files:
        df = pd.read_parquet(p)
        print(f"  {p.name}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    label = "_".join(str(y) for y in years)
    out_path = PROCESSED_DIR / f"train_table_{label}_combined.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\n✅ Combined table saved: {out_path.name}")
    print(f"   Total rows: {len(combined):,}  |  HR rate: {combined['hr_hit'].mean():.4f}")
    return out_path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MLB HR feature tables by season.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Season years to build (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        default=True,
        help="After building each season, concatenate into a combined table (default: True)",
    )
    args = parser.parse_args()

    if args.combine:
        build_multi_season(args.years)
    else:
        for year in args.years:
            build_season(year)

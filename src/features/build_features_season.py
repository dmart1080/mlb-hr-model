from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.features.build_features import build_features_for_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Season start/end months — adjust if a year had unusual start/end dates
SEASON_MONTHS = [
    ("03-01", "03-31"),
    ("04-01", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
    ("10-01", "10-31"),
]

# Known season overrides for unusual years
SEASON_OVERRIDES: dict[int, list[tuple[str, str]]] = {
    2020: [  # 60-game COVID season
        ("07-01", "07-31"),
        ("08-01", "08-31"),
        ("09-01", "09-27"),
    ],
}


def get_months_for_year(year: int) -> list[tuple[str, str]]:
    overrides = SEASON_OVERRIDES.get(year)
    if overrides:
        return [(f"{year}-{s}", f"{year}-{e}") for s, e in overrides]
    return [(f"{year}-{s}", f"{year}-{e}") for s, e in SEASON_MONTHS]


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
    months = get_months_for_year(year)
    out_path = PROCESSED_DIR / f"train_table_{year}_full_season.parquet"

    if out_path.exists():
        print(f"Season table already exists: {out_path.name} — skipping full rebuild.")
        print("Pass --force to overwrite.")
        return out_path

    print(f"\n{'='*50}")
    print(f"Building {year} season features")
    print(f"{'='*50}")

    month_files = []
    for start, end in months:
        print(f"\n--- {start} to {end} ---")
        month_files.append(build_month(start, end))

    dfs = [pd.read_parquet(p) for p in month_files]
    season_df = pd.concat(dfs, ignore_index=True)
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])
    season_df = season_df.sort_values("game_date").reset_index(drop=True)

    season_df.to_parquet(out_path, index=False)

    print(f"\n✅ {year} season saved: {out_path.name}")
    print(f"   Rows:    {len(season_df):,}")
    print(f"   HR rate: {season_df['hr_hit'].mean():.4f}")
    print(f"   Dates:   {season_df['game_date'].min().date()} → {season_df['game_date'].max().date()}")

    return out_path


def combine_seasons(years: list[int]) -> Path:
    """Merge multiple season parquets into one combined training table."""
    dfs = []
    for year in years:
        p = PROCESSED_DIR / f"train_table_{year}_full_season.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Season table for {year} not found: {p}\n"
                f"Run: python -m src.features.build_features_season --year {year}"
            )
        print(f"Loading {p.name} ...")
        dfs.append(pd.read_parquet(p))

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    label = "_".join(str(y) for y in sorted(years))
    out_path = PROCESSED_DIR / f"train_table_{label}_combined.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\n✅ Combined table saved: {out_path.name}")
    print(f"   Rows:    {len(combined):,}")
    print(f"   HR rate: {combined['hr_hit'].mean():.4f}")
    print(f"   Dates:   {combined['game_date'].min().date()} → {combined['game_date'].max().date()}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build season feature tables for the HR model.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build a single season
    build_p = subparsers.add_parser("build", help="Build features for a single season year.")
    build_p.add_argument("--year", type=int, required=True, help="Season year, e.g. 2021")
    build_p.add_argument("--force", action="store_true", help="Overwrite existing output.")

    # combine multiple seasons
    combine_p = subparsers.add_parser("combine", help="Combine multiple season tables into one.")
    combine_p.add_argument("--years", type=int, nargs="+", required=True, help="Years to combine, e.g. 2021 2022 2023 2024")

    args = parser.parse_args()

    if args.command == "build":
        if args.force:
            out = PROCESSED_DIR / f"train_table_{args.year}_full_season.parquet"
            if out.exists():
                out.unlink()
                print(f"Deleted existing: {out.name}")
        build_season(args.year)

    elif args.command == "combine":
        combine_seasons(args.years)


if __name__ == "__main__":
    main()

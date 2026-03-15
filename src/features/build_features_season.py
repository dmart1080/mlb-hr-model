from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.logging_config import configure_logging
from src.features.build_features import build_features_for_range
from src.features.build_features_multi_season import build_month

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEASON_MONTHS = [
    ("03-01", "03-31"), ("04-01", "04-30"), ("05-01", "05-31"),
    ("06-01", "06-30"), ("07-01", "07-31"), ("08-01", "08-31"),
    ("09-01", "09-30"), ("10-01", "10-31"),
]

SEASON_OVERRIDES: dict[int, list[tuple[str, str]]] = {
    2020: [
        ("07-01", "07-31"), ("08-01", "08-31"), ("09-01", "09-27"),
    ],
}


def get_months_for_year(year: int) -> list[tuple[str, str]]:
    overrides = SEASON_OVERRIDES.get(year)
    if overrides:
        return [(f"{year}-{s}", f"{year}-{e}") for s, e in overrides]
    return [(f"{year}-{s}", f"{year}-{e}") for s, e in SEASON_MONTHS]


def build_season(year: int) -> Path:
    months = get_months_for_year(year)
    out_path = PROCESSED_DIR / f"train_table_{year}_full_season.parquet"

    if out_path.exists():
        logger.info(
            "Season table already exists: %s — skipping. Pass --force to overwrite.",
            out_path.name,
        )
        return out_path

    logger.info("=" * 50)
    logger.info("Building %d season features", year)
    logger.info("=" * 50)

    month_files = []
    for start, end in months:
        logger.info("--- %s to %s ---", start, end)
        month_files.append(build_month(start, end))

    dfs = [pd.read_parquet(p) for p in month_files]
    season_df = pd.concat(dfs, ignore_index=True)
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])
    season_df = season_df.sort_values("game_date").reset_index(drop=True)

    season_df.to_parquet(out_path, index=False)

    logger.info(
        "%d season saved: %s | rows=%d | hr_rate=%.4f | dates=%s -> %s",
        year,
        out_path.name,
        len(season_df),
        season_df["hr_hit"].mean(),
        season_df["game_date"].min().date(),
        season_df["game_date"].max().date(),
    )
    return out_path


def combine_seasons(years: list[int]) -> Path:
    """Merge multiple season parquets into one combined training table."""
    dfs = []
    for year in years:
        p = PROCESSED_DIR / f"train_table_{year}_full_season.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Season table for {year} not found: {p}\n"
                f"Run: python -m src.features.build_features_season build --year {year}"
            )
        logger.info("Loading %s ...", p.name)
        dfs.append(pd.read_parquet(p))

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.sort_values("game_date").reset_index(drop=True)

    label = "_".join(str(y) for y in sorted(years))
    out_path = PROCESSED_DIR / f"train_table_{label}_combined.parquet"
    combined.to_parquet(out_path, index=False)

    logger.info(
        "Combined table saved: %s | rows=%d | hr_rate=%.4f | dates=%s -> %s",
        out_path.name,
        len(combined),
        combined["hr_hit"].mean(),
        combined["game_date"].min().date(),
        combined["game_date"].max().date(),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build season feature tables for the HR model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.features.build_features_season build --year 2024\n"
            "  python -m src.features.build_features_season --debug build --year 2024 --force\n"
            "  python -m src.features.build_features_season --log-file build.log build --year 2024\n"
            "  python -m src.features.build_features_season combine --years 2021 2022 2023 2024\n"
        ),
    )

    # Global flags come BEFORE the subcommand, e.g.:
    #   python -m src.features.build_features_season --debug build --year 2024
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional path to write full DEBUG log (e.g. build.log).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build features for a single season year.")
    build_p.add_argument("--year", type=int, required=True)
    build_p.add_argument("--force", action="store_true", help="Overwrite existing output.")

    combine_p = subparsers.add_parser("combine", help="Combine multiple season tables into one.")
    combine_p.add_argument("--years", type=int, nargs="+", required=True)

    args = parser.parse_args()

    configure_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=args.log_file,
    )

    if args.command == "build":
        if args.force:
            out = PROCESSED_DIR / f"train_table_{args.year}_full_season.parquet"
            if out.exists():
                out.unlink()
                logger.info("Deleted existing: %s", out.name)
        build_season(args.year)

    elif args.command == "combine":
        combine_seasons(args.years)


if __name__ == "__main__":
    main()

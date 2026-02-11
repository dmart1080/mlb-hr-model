from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_features import build_features_for_range

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MONTHS_2024 = [
    ("2024-03-01", "2024-03-31"),
    ("2024-04-01", "2024-04-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-08-01", "2024-08-31"),
    ("2024-09-01", "2024-09-30"),
    ("2024-10-01", "2024-10-31"),
]


def build_month(start: str, end: str) -> Path:
    out_path = PROCESSED_DIR / f"train_table_{start}_to_{end}.parquet"

    # resume-friendly
    if out_path.exists():
        print(f"Skipping (already exists): {out_path.name}")
        return out_path

    result = build_features_for_range(start, end)
    print(
        f"Saved: {result.output_path.name} | rows={len(result.features_df):,} | hr_rate={result.features_df['hr_hit'].mean():.4f}"
    )
    return result.output_path


def build_full_season() -> Path:
    month_files = []
    for start, end in MONTHS_2024:
        print(f"\n=== Building {start} to {end} ===")
        month_files.append(build_month(start, end))

    dfs = []
    for p in month_files:
        print(f"Loading {p.name}")
        dfs.append(pd.read_parquet(p))

    season_df = pd.concat(dfs, ignore_index=True)
    season_df["game_date"] = pd.to_datetime(season_df["game_date"])
    season_df = season_df.sort_values("game_date").reset_index(drop=True)

    out_path = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    season_df.to_parquet(out_path, index=False)

    print(f"\n✅ Full season saved: {out_path.name}")
    print(f"Rows: {len(season_df):,}")
    print(f"Label HR rate: {season_df['hr_hit'].mean():.4f}")
    return out_path


if __name__ == "__main__":
    build_full_season()

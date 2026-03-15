"""
MLB HR Model — Daily Predictions
==================================
Loads the trained model, scores today's batters, prints ranked HR candidates,
enriches with odds/edge data, and saves a CSV for backtest analysis.

Usage
-----
    # Manual one-off run (auto-detects latest date in train table):
    python -m src.model.predict

    # Morning pass — probable starters, early lines:
    python -m src.model.predict --run-type morning

    # Final pass — confirmed lineups, final lines (bypasses caches):
    python -m src.model.predict --run-type final --force-refresh

    # Score a specific date:
    python -m src.model.predict --date 2026-04-01

    # Force-refresh all caches (schedule + odds) without a run-type label:
    python -m src.model.predict --force-refresh

Output
------
    data/predictions/predictions_YYYY-MM-DD.csv              ← manual / canonical
    data/predictions/predictions_YYYY-MM-DD_morning.csv      ← morning pass
    data/predictions/predictions_YYYY-MM-DD_final.csv        ← final pass
    (final pass also overwrites the canonical un-stamped file)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import glob
import joblib
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

_LOOKUP_CHUNK_SIZE = 500
TOP_N = 20


# ---------------------------------------------------------------------------
# Model + table loading
# ---------------------------------------------------------------------------

def latest_train_table() -> Path:
    """
    Preference order:
      1. 2021-2025 multi-season file
      2. 2024 full-season file (legacy)
      3. Most-recently-modified train_table_*.parquet
    """
    multi = PROCESSED_DIR / "train_table_2021_2025_full.parquet"
    if multi.exists():
        return multi

    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season

    files = sorted(
        glob.glob(str(PROCESSED_DIR / "train_table_*.parquet")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed.")
    return Path(files[0])


def load_model():
    """
    Load the best available model.  Preference order:
      1. 2021-2025 LightGBM calibrated
      2. 2021-2025 LogReg calibrated
      3. Legacy 2024 models
    """
    candidates = [
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_lightgbm_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_edges_calibrated_2024.joblib",
    ]
    for path in candidates:
        if path.exists():
            print(f"Loading model: {path.name}")
            bundle = joblib.load(path)
            return bundle["model"], bundle["feature_cols"], bundle.get("apply_shrinkage", False)

    raise FileNotFoundError(
        "No trained model found in models/. Run src/model/train.py first."
    )


# ---------------------------------------------------------------------------
# Player name resolution
# ---------------------------------------------------------------------------

def _build_id_name_map(ids: list[int]) -> dict[int, str]:
    """
    Resolve a list of MLBAM IDs to full names using playerid_reverse_lookup.
    Chunked to avoid API limits.
    """
    unique_ids = list(set(ids))
    mapping: dict[int, str] = {}
    for i in range(0, len(unique_ids), _LOOKUP_CHUNK_SIZE):
        chunk = unique_ids[i : i + _LOOKUP_CHUNK_SIZE]
        try:
            look = playerid_reverse_lookup(chunk, key_type="mlbam")
            look["full_name"] = (
                look["name_first"].fillna("") + " " + look["name_last"].fillna("")
            ).str.strip()
            for _, row in look.iterrows():
                mapping[int(row["key_mlbam"])] = row["full_name"]
        except Exception as e:
            print(f"⚠️  Name lookup failed for chunk {i}: {e}")
    return mapping


def add_player_names(
    df: pd.DataFrame,
    id_name_map: dict[int, str],
    id_col: str,
    out_col: str,
) -> pd.DataFrame:
    df[out_col] = (
        df[id_col]
        .astype("Int64")
        .map(lambda x: id_name_map.get(int(x)) if pd.notna(x) else None)
    )
    return df


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _format_pct(val) -> str:
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return "  —  "


def _format_odds(val) -> str:
    try:
        v = int(val)
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "  —  "


def _format_edge(val) -> str:
    try:
        v = float(val)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v*100:.1f}pp"
    except (TypeError, ValueError):
        return "  —  "


def print_ranked_table(ranked: pd.DataFrame, *, has_odds: bool) -> None:
    if has_odds:
        print(f"\n{'='*85}")
        print(f"  TOP HR CANDIDATES  (sorted by edge, then model prob)")
        print(f"{'='*85}")
        print(
            f"  {'Batter':<26} {'Pitcher':<22} "
            f"{'Model':>7}  {'Market':>7}  {'Edge':>8}  {'Odds':>6}  {'Kelly':>6}  {'Slot':>4}"
        )
        print(f"  {'-'*26} {'-'*22} {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*4}")

        bettable = ranked[ranked["edge"].notna() & (ranked["edge"] > 0)].sort_values(
            ["edge", "hr_prob"], ascending=False
        )
        rest = ranked[~ranked.index.isin(bettable.index)].sort_values("hr_prob", ascending=False)
        display = pd.concat([bettable, rest]).head(TOP_N)

        for _, row in display.iterrows():
            batter  = str(row.get("batter_name", ""))[:26]
            pitcher = str(row.get("pitcher_name", ""))[:22]
            model   = _format_pct(row.get("hr_prob"))
            market  = _format_pct(row.get("market_fair_prob"))
            edge    = _format_edge(row.get("edge"))
            odds    = _format_odds(row.get("market_over_price"))
            kelly   = _format_pct(row.get("kelly_stake"))
            pos     = int(row.get("batting_order_pos", 0))
            flag    = "  ← BET" if (pd.notna(row.get("edge")) and float(row.get("edge", 0)) > 0) else ""
            print(
                f"  {batter:<26} {pitcher:<22} "
                f"{model:>7}  {market:>7}  {edge:>8}  {odds:>6}  {kelly:>6}  {pos:>4}{flag}"
            )

        n_edge = len(bettable)
        if n_edge:
            avg_edge  = bettable["edge"].mean() * 100
            avg_kelly = bettable["kelly_stake"].mean() * 100 if "kelly_stake" in bettable else 0
            print(f"\n  {n_edge} bets with positive edge | avg edge {avg_edge:.1f}pp | avg Kelly {avg_kelly:.1f}%")
        else:
            print(f"\n  No positive-edge bets today.")
        print(f"{'='*85}")

    else:
        print(f"\n{'='*70}")
        print(f"  TOP HR CANDIDATES  (no odds data)")
        print(f"{'='*70}")
        print(
            f"  {'Batter':<26} {'Pitcher':<24} "
            f"{'Model':>7}  {'Slot':>4}  {'xPA':>5}"
        )
        print(f"  {'-'*26} {'-'*24} {'-'*7}  {'-'*4}  {'-'*5}")

        top = ranked.sort_values("hr_prob", ascending=False).head(TOP_N)
        for _, row in top.iterrows():
            batter  = str(row.get("batter_name", ""))[:26]
            pitcher = str(row.get("pitcher_name", ""))[:24]
            model   = _format_pct(row.get("hr_prob"))
            pos     = int(row.get("batting_order_pos", 0))
            xpa     = f"{float(row.get('expected_pa_today', 0)):.1f}"
            print(f"  {batter:<26} {pitcher:<24} {model:>7}  {pos:>4}  {xpa:>5}")

        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score today's batters and print ranked HR candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-type", choices=["morning", "final"], default=None,
        help=(
            "Pass label stamped into the output CSV filename. "
            "'morning' = probable starters, early lines (no cache bypass). "
            "'final'   = confirmed lineups, final lines (use with --force-refresh). "
            "Omit for a one-off manual run."
        ),
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help=(
            "Bypass schedule and odds disk caches. "
            "Use on the final pass to pick up confirmed lineups and latest lines."
        ),
    )
    parser.add_argument(
        "--date", default=None,
        help="Score a specific date (YYYY-MM-DD) instead of the latest in the train table.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model + train table
    # ------------------------------------------------------------------
    model, feature_cols, apply_shrinkage_flag = load_model()
    train_path = latest_train_table()

    df = pd.read_parquet(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    if apply_shrinkage_flag:
        from src.model.train import apply_shrinkage as _apply_shrinkage
        print("Applying empirical Bayes shrinkage ...")
        df = _apply_shrinkage(df)

    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = df["game_date"].max()

    today_df = df[df["game_date"] == target_date].copy()

    if today_df.empty:
        print(f"No rows found for {target_date.date()} in the train table.")
        print("Run build_features_multi_season (or build_features_season) to add today.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    missing = [c for c in feature_cols if c not in today_df.columns]
    if missing:
        print(f"⚠️  {len(missing)} feature(s) missing at inference — filling with 0:")
        for c in missing:
            print(f"     - {c}")
            today_df[c] = 0.0

    X = today_df[feature_cols].fillna(0.0)
    today_df["hr_prob"] = model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Resolve player names
    # ------------------------------------------------------------------
    all_ids: list[int] = []
    for col in ("batter", "pitcher"):
        if col in today_df.columns:
            all_ids.extend(today_df[col].dropna().astype(int).unique().tolist())
    id_name_map = _build_id_name_map(all_ids)

    today_df = add_player_names(today_df, id_name_map, "batter",  "batter_name")
    today_df = add_player_names(today_df, id_name_map, "pitcher", "pitcher_name")

    ranked = today_df.sort_values("hr_prob", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Odds enrichment
    # ------------------------------------------------------------------
    date_str = str(target_date.date())
    has_odds = False

    odds_api_key = os.environ.get("ODDS_API_KEY")
    if odds_api_key:
        try:
            from src.data_sources.odds import enrich_predictions_with_odds
            ranked = enrich_predictions_with_odds(
                ranked,
                date_str,
                api_key=odds_api_key,
                force_refresh=args.force_refresh,
            )
            has_odds = ranked["edge"].notna().any()
        except Exception as e:
            print(f"⚠️  Odds enrichment failed: {e}")
    else:
        print(
            "ℹ️  No ODDS_API_KEY found — running without odds data.\n"
            "   Set the env var to enable edge detection:\n"
            "     export ODDS_API_KEY=your_key_here"
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    run_label = f" [{args.run_type.upper()} PASS]" if args.run_type else ""
    print(f"\nPredictions for {date_str}{run_label}:")

    if args.run_type == "morning":
        print("  ⚠️  Lineups may not be confirmed yet — batting order positions are estimates.")
    elif args.run_type == "final":
        print("  ✓  Final pass — use this output for betting decisions.")

    print_ranked_table(ranked, has_odds=has_odds)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    save_cols = [
        "batter_name", "pitcher_name", "hr_prob", "batter", "pitcher",
        "batting_order_pos", "expected_pa_today",
    ]
    if has_odds:
        save_cols += [
            "market_over_price", "market_fair_prob", "edge",
            "odds_bookmaker", "kelly_stake",
        ]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_type:
        out_path = PREDICTIONS_DIR / f"predictions_{date_str}_{args.run_type}.csv"
    else:
        out_path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"

    out_df = ranked[[c for c in save_cols if c in ranked.columns]].copy()
    out_df["run_type"] = args.run_type or "manual"
    out_df["run_at"]   = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Final pass also overwrites the canonical un-stamped file so any tool
    # expecting predictions_YYYY-MM-DD.csv always gets the best snapshot.
    if args.run_type == "final":
        canonical = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
        out_df.to_csv(canonical, index=False)
        print(f"  Also saved: {canonical.relative_to(PROJECT_ROOT)}  (canonical)")

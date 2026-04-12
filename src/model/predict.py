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

    # Final pass — confirmed lineups, final lines (--force-refresh auto-enabled):
    python -m src.model.predict --run-type final

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
from datetime import datetime, timezone, timedelta
from pathlib import Path
import glob
import joblib
import pandas as pd
from src.model.train import _IsotonicWrappedModel # noqa: F401
from dotenv import load_dotenv
load_dotenv()

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

_LOOKUP_CHUNK_SIZE = 500
TOP_N = 20

# Minimum model probability to flag a bet. Positive edge on low-probability
# predictions is usually calibration noise — the model is least reliable in
# the tails. This filters out "edge" that isn't actionable.
MIN_BET_PROB = 0.08


# ---------------------------------------------------------------------------
# Model + table loading
# ---------------------------------------------------------------------------

def latest_train_table() -> Path:
    """
    Preference order:
      1. Today's _today.parquet written by build_today_features (most current)
      2. 2021-2026 combined multi-season file (updated by build_today_features)
      3. 2021-2025 combined multi-season file (legacy)
      4. 2024 full-season file (legacy)
      5. Most-recently-modified train_table_*.parquet

    The _today.parquet is preferred when it exists because it's guaranteed
    to contain exactly today's slate with fresh features.  The combined file
    is second because build_today_features appends to it, so it always has
    today's rows after build_today_features runs.
    """
    from datetime import date as _date
    today_str = _date.today().strftime("%Y-%m-%d")

    # 1. Today's dedicated file — written fresh by build_today_features
    today_file = PROCESSED_DIR / f"train_table_{today_str}_today.parquet"
    if today_file.exists() and today_file.stat().st_size > 1_000:
        return today_file

    # 2. Combined 2021-2026 (build_today_features appends here)
    multi_2026 = PROCESSED_DIR / "train_table_2021_2026_full.parquet"
    if multi_2026.exists():
        return multi_2026

    # 3. Combined 2021-2025 (legacy)
    multi_2025 = PROCESSED_DIR / "train_table_2021_2025_full.parquet"
    if multi_2025.exists():
        return multi_2025

    # 4. 2024 single-season (legacy)
    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season

    # 5. Glob fallback — most recently modified
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
      1. 2021-2026 LightGBM calibrated
      2. 2021-2026 LogReg calibrated
      3. 2021-2025 LightGBM calibrated
      4. 2021-2025 LogReg calibrated
      5. Legacy 2024 models
    """
    candidates = [
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2026.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2026.joblib",
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
# Game status filter
# ---------------------------------------------------------------------------

def fetch_upcoming_game_pks(date_str: str) -> set[int] | None:
    """
    Query the MLB Stats API for today's games and return the set of game_pks
    whose status is NOT yet Live or Final.

    Returns None if the API call fails (caller should skip the filter).
    A 30-minute buffer is NOT needed here — the API status field is the
    ground truth: 'Preview' / 'Warmup' = upcoming, 'Live' / 'Final' = skip.
    """
    import requests as _req

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=game(status),team"
    try:
        resp = _req.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"⚠️  Game status fetch failed (filter skipped): {e}")
        return None

    upcoming: set[int] = set()
    skipped:  list[str] = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            pk     = int(game["gamePk"])
            state  = game.get("status", {}).get("abstractGameState", "Preview")
            detail = game.get("status", {}).get("detailedState", "")
            away   = game.get("teams", {}).get("away",  {}).get("team", {}).get("abbreviation", "?")
            home   = game.get("teams", {}).get("home",  {}).get("team", {}).get("abbreviation", "?")
            matchup = f"{away}@{home}"

            if state in ("Live", "Final"):
                skipped.append(f"{matchup} ({detail})")
            else:
                upcoming.add(pk)

    if skipped:
        print(f"ℹ️  Skipping {len(skipped)} game(s) already in progress or final:")
        for g in skipped:
            print(f"     • {g}")

    return upcoming


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
# Display helpers
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
        v    = float(val)
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

        bettable = ranked[
            ranked["edge"].notna()
            & (ranked["edge"] > 0)
            & (ranked["hr_prob"] >= MIN_BET_PROB)
        ].sort_values(["edge", "hr_prob"], ascending=False)
        rest    = ranked[~ranked.index.isin(bettable.index)].sort_values("hr_prob", ascending=False)
        display = pd.concat([bettable, rest]).head(TOP_N)

        for _, row in display.iterrows():
            batter  = str(row.get("batter_name",  ""))[:26]
            pitcher = str(row.get("pitcher_name", ""))[:22]
            model   = _format_pct(row.get("hr_prob"))
            market  = _format_pct(row.get("market_fair_prob"))
            edge    = _format_edge(row.get("edge"))
            odds    = _format_odds(row.get("market_over_price"))
            kelly   = _format_pct(row.get("kelly_stake"))
            pos     = int(row.get("batting_order_pos", 0))
            _has_edge = pd.notna(row.get("edge")) and float(row.get("edge", 0)) > 0
            _above_floor = float(row.get("hr_prob", 0)) >= MIN_BET_PROB
            flag = "  ← BET" if (_has_edge and _above_floor) else ""
            print(
                f"  {batter:<26} {pitcher:<22} "
                f"{model:>7}  {market:>7}  {edge:>8}  {odds:>6}  {kelly:>6}  {pos:>4}{flag}"
            )

        n_edge = len(bettable)
        if n_edge:
            avg_edge  = bettable["edge"].mean() * 100
            avg_kelly = bettable["kelly_stake"].mean() * 100 if "kelly_stake" in bettable else 0
            print(
                f"\n  {n_edge} bets with positive edge | "
                f"avg edge {avg_edge:.1f}pp | avg Kelly {avg_kelly:.1f}%"
            )
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
            batter  = str(row.get("batter_name",  ""))[:26]
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
            "'final'   = confirmed lineups, final lines (--force-refresh auto-enabled). "
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
    parser.add_argument(
        "--min-prob", type=float, default=None,
        help=(
            f"Minimum model probability to flag a bet (default {MIN_BET_PROB:.0%}). "
            "Filters out low-confidence 'edge' that is likely calibration noise."
        ),
    )
    args = parser.parse_args()

    if args.min_prob is not None:
        MIN_BET_PROB = args.min_prob

    # Final passes should always bypass caches — confirmed lineups and the
    # latest odds lines must be fetched fresh, not served from a stale cache.
    if args.run_type == "final" and not args.force_refresh:
        args.force_refresh = True
        print("ℹ️  Final pass: --force-refresh enabled automatically.")

    # ------------------------------------------------------------------
    # Load model + train table
    # ------------------------------------------------------------------
    model, feature_cols, apply_shrinkage_flag = load_model()
    train_path = latest_train_table()
    print(f"Using train table: {train_path.name}")

    df = pd.read_parquet(train_path)

    # Guard: empty parquet (no batting orders posted yet) — exit cleanly
    # rather than crashing with KeyError: 'game_date'
    if df.empty or "game_date" not in df.columns:
        print(
            f"\n  ℹ️  No batter rows in {train_path.name} — "
            "batting orders not yet posted. Skipping predictions.\n"
            "     The final pass will retry automatically before first pitch."
        )
        sys.exit(0)

    df["game_date"] = pd.to_datetime(df["game_date"])

    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = df["game_date"].max()

    today_df = df[df["game_date"] == target_date].copy()

    if apply_shrinkage_flag:
        from src.model.train import apply_shrinkage as _apply_shrinkage
        print("Applying empirical Bayes shrinkage ...")
        today_df = _apply_shrinkage(today_df)

    if today_df.empty:
        print(f"\nNo rows found for {target_date.date()} in {train_path.name}.")
        print(
            "Run build_today_features first to inject today's slate:\n"
            "    python -m src.features.build_today_features --force-refresh"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Filter out games that are already Live or Final
    # This prevents the no-edge fallback (and any other pass) from surfacing
    # picks for games that have already started or finished.
    # ------------------------------------------------------------------
    date_str = str(target_date.date())

    if "game_pk" in today_df.columns:
        upcoming_pks = fetch_upcoming_game_pks(date_str)
        if upcoming_pks is not None:
            before = len(today_df)
            today_df = today_df[today_df["game_pk"].astype(int).isin(upcoming_pks)].copy()
            dropped = before - len(today_df)
            if dropped:
                print(f"  Removed {dropped} batter row(s) for games already started or final.")
            if today_df.empty:
                print(
                    f"\n  ℹ️  All games for {date_str} are already Live or Final. "
                    "No predictions to score.\n"
                )
                sys.exit(0)
    else:
        print("⚠️  game_pk column not found — skipping game status filter.")

    print(f"Scoring {len(today_df)} batter rows for {target_date.date()} ...")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    # Warn about leaked features that are always 0 at inference time.
    # These are computed from completed-game data and unavailable pre-game.
    _LEAKED_FEATURES = {"relief_pa_pct"}
    leaked_present = _LEAKED_FEATURES & set(feature_cols)
    if leaked_present:
        print(
            f"⚠️  Model uses {len(leaked_present)} leaked feature(s) that are "
            f"unavailable at inference (always 0): {sorted(leaked_present)}\n"
            "   Retrain the model to remove these — predictions are degraded."
        )

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
    has_odds = False

    # Gate: don't burn Odds API tokens until rolling windows are filling.
    # 14-day features are reliable after ~14 days; 30-day features benefit
    # from empirical Bayes shrinkage so partial windows are usable.
    _ROLLING_WINDOW_DAYS = 14
    _SEASON_STARTS = {2025: (3, 27), 2026: (3, 26)}
    _pred_date = datetime.strptime(date_str, "%Y-%m-%d")
    _month, _day = _SEASON_STARTS.get(_pred_date.year, (3, 27))
    _season_start = datetime(_pred_date.year, _month, _day)
    _days_into_season = (_pred_date - _season_start).days
    _windows_ready = _days_into_season >= _ROLLING_WINDOW_DAYS

    odds_api_key = os.environ.get("ODDS_API_KEY")
    if odds_api_key and _windows_ready:
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
    elif odds_api_key and not _windows_ready:
        print(
            f"ℹ️  Skipping odds enrichment — only {_days_into_season}d into season "
            f"(need {_ROLLING_WINDOW_DAYS}d for rolling windows to fill).\n"
            "   Odds API tokens will be conserved until features are reliable."
        )
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

    # ------------------------------------------------------------------
    # Discord notification
    # ------------------------------------------------------------------
    try:
        from src.notifications.discord import send_to_discord
        pass_label = (args.run_type or "manual").upper()
        sent = send_to_discord(
            ranked,
            pass_label=pass_label,
            date_str=date_str,
        )
        if sent:
            print(f"  Discord: picks posted ✓")
    except Exception as _disc_err:
        print(f"  Discord: notification failed — {_disc_err}")

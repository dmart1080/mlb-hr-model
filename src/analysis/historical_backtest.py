"""
Historical-odds backtest.

The existing src.analysis.backtest only works on dates where the model wrote
a prediction CSV with live odds attached (last ~30-60 days). This module
removes that constraint: it scores the saved model on any past date range,
joins to historical Odds API lines (closing-price snapshots), and produces
the same edge sweep + calibration + ROI report.

Workflow
--------
1. Load the train_table for the requested date range.
2. Apply the saved model (with apply_shrinkage) to get hr_prob per row.
3. Resolve MLBAM batter ids to names (reuse predict.py helpers).
4. Pull historical odds for every date (cached on first fetch).
5. Match (batter_name, game_date) -> prop, compute edge / Kelly / P&L.
6. Reuse the existing backtest.py reporting helpers.

The intended scope is the 30-day OOS holdout (2026-04-04 -> 2026-05-03),
which is the only window the model has not been trained on. Other windows
will produce optimistic numbers and should be interpreted as in-sample.

Usage
-----
    # Default: 30-day holdout window
    python -m src.analysis.historical_backtest

    # Specific range:
    python -m src.analysis.historical_backtest --start 2026-04-15 --end 2026-04-17

    # Free run on cached dates only (no new API calls):
    python -m src.analysis.historical_backtest --no-fetch

    # Dump merged results to CSV:
    python -m src.analysis.historical_backtest --save-results
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.logging_config import configure_logging
from src.data_sources.statcast import fetch_statcast_events
from src.data_sources.odds_historical import fetch_historical_hr_props_for_date
from src.data_sources.odds import (
    build_odds_lookup,
    match_odds,
    compute_edge,
    kelly_fraction,
    OddsApiQuotaExceeded,
)
from src.model.train import apply_shrinkage, HOLDOUT_DAYS, _IsotonicWrappedModel  # noqa: F401 (needed for pickle.load)
from src.model.predict import _build_id_name_map, add_player_names
from src.analysis.backtest import (
    _bet_pnl,
    _american_to_decimal,
    edge_threshold_sweep,
    calibration_table,
    find_recommended_threshold,
    print_report,
    KELLY_FRACTION,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
RESULTS_DIR     = PROJECT_ROOT / "data" / "backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2026.joblib"


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def default_holdout_window(train_path: Path = None) -> tuple[str, str]:
    """The 30-day OOS holdout — same slice train.py uses for evaluation."""
    train_path = train_path or PROCESSED_DIR / "train_table_2021_2026_full.parquet"
    df = pd.read_parquet(train_path, columns=["game_date"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    max_d = df["game_date"].max()
    start = (max_d - pd.Timedelta(days=HOLDOUT_DAYS - 1)).strftime("%Y-%m-%d")
    end   = max_d.strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# Model loading + scoring
# ---------------------------------------------------------------------------

def load_model_bundle() -> tuple:
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"], bundle.get("apply_shrinkage", True)


def score_window(start: str, end: str) -> pd.DataFrame:
    """Load train_table rows in [start, end] and score them with the model.

    Returns a DataFrame with: game_date, game_pk, batter, batter_name,
    hr_prob, hr_hit (= actual_hr from the train_table labels).
    """
    parquet = PROCESSED_DIR / "train_table_2021_2026_full.parquet"
    df = pd.read_parquet(parquet)
    df["game_date"] = pd.to_datetime(df["game_date"])

    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    window = df[(df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)].copy()
    if window.empty:
        raise SystemExit(f"No train_table rows in {start} -> {end}.")
    logger.info("Loaded %d batter-game rows for %s -> %s", len(window), start, end)

    model, feature_cols, do_shrink = load_model_bundle()
    if do_shrink:
        window = apply_shrinkage(window)

    missing = [c for c in feature_cols if c not in window.columns]
    if missing:
        logger.warning(
            "%d feature(s) missing from train_table — filling with 0: %s",
            len(missing), missing,
        )
        for c in missing:
            window[c] = 0.0

    X = window[feature_cols].fillna(0.0)
    probs = model.predict_proba(X)[:, 1]
    window["hr_prob"] = probs

    # Resolve batter names (single MLB lookup batched per backtest run)
    batter_ids = window["batter"].dropna().astype(int).unique().tolist()
    logger.info("Resolving %d unique batter names ...", len(batter_ids))
    id_name_map = _build_id_name_map(batter_ids)
    window = add_player_names(window, id_name_map, "batter", "batter_name")

    return window[["game_date", "game_pk", "batter", "batter_name",
                   "hr_prob", "hr_hit"]].rename(columns={"hr_hit": "actual_hr"})


# ---------------------------------------------------------------------------
# Odds collection (caches every per-event response — re-runs cost zero)
# ---------------------------------------------------------------------------

def collect_historical_odds(
    dates: list[str],
    *,
    per_game_close: bool = True,
    close_lead_minutes: int = 60,
    force_refresh: bool = False,
    no_fetch: bool = False,
) -> dict[str, list[dict]]:
    """Return {date_str: [prop_dict, ...]}.

    When `no_fetch=True`, only reads from cache and skips dates without one.
    Useful for free re-runs after a prior fetch session.
    """
    out: dict[str, list[dict]] = {}
    for d in dates:
        try:
            if no_fetch:
                # Read-only path: rely on cache. fetch_historical_hr_props_for_date
                # falls through to the API on cache miss; we skip the call by
                # checking cache existence manually for the slate snapshot.
                from src.data_sources.odds_historical import (
                    _events_cache_path, snapshot_iso_for_date,
                )
                slate_snap = snapshot_iso_for_date(d)
                if not _events_cache_path(slate_snap).exists():
                    logger.info("No cache for %s — skipping (--no-fetch).", d)
                    out[d] = []
                    continue

            props = fetch_historical_hr_props_for_date(
                d,
                per_game_close=per_game_close,
                close_lead_minutes=close_lead_minutes,
                force_refresh=force_refresh,
            )
        except OddsApiQuotaExceeded as e:
            logger.error("Quota exhausted on %s: %s", d, e)
            out[d] = []
            break
        out[d] = props
    return out


# ---------------------------------------------------------------------------
# Join predictions to odds
# ---------------------------------------------------------------------------

def enrich_with_historical_odds(
    scored: pd.DataFrame,
    odds_by_date: dict[str, list[dict]],
    *,
    kelly_mult: float = KELLY_FRACTION,
) -> pd.DataFrame:
    """Attach odds + edge + P&L to scored predictions. Mirrors the schema
    src.analysis.backtest.build_results produces so reporting can reuse it.
    """
    result = scored.copy()
    for col in ("market_line", "market_over_price", "market_under_price",
                "market_implied_prob", "market_fair_prob", "edge", "kelly_stake",
                "pnl_1u", "decimal_odds", "kelly_pnl"):
        result[col] = float("nan")
    result["odds_bookmaker"] = None

    matched = 0
    for date_str, props in odds_by_date.items():
        if not props:
            continue
        lookup = build_odds_lookup(props)
        date_ts = pd.Timestamp(date_str)
        mask = result["game_date"] == date_ts
        if not mask.any():
            continue
        for idx in result[mask].index:
            name = result.at[idx, "batter_name"]
            if not isinstance(name, str) or not name:
                continue
            prop = match_odds(name, lookup)
            if prop is None:
                continue
            ov = prop["over_price"]
            fp = prop["fair_prob_over"]
            mp = float(result.at[idx, "hr_prob"])
            actual = int(result.at[idx, "actual_hr"]) if not pd.isna(result.at[idx, "actual_hr"]) else None
            result.at[idx, "market_line"]         = prop["line"]
            result.at[idx, "market_over_price"]   = ov
            result.at[idx, "market_under_price"]  = prop["under_price"]
            result.at[idx, "market_implied_prob"] = prop["implied_prob_over"]
            result.at[idx, "market_fair_prob"]    = fp
            result.at[idx, "edge"]                = compute_edge(mp, fp)
            result.at[idx, "odds_bookmaker"]      = prop["bookmaker"]
            result.at[idx, "decimal_odds"]        = _american_to_decimal(ov)
            kstake                                = kelly_fraction(mp, ov, kelly_mult)
            result.at[idx, "kelly_stake"]         = kstake
            if actual is not None:
                pnl = _bet_pnl(ov, bool(actual))
                result.at[idx, "pnl_1u"]    = pnl
                result.at[idx, "kelly_pnl"] = pnl * kstake
            matched += 1

    logger.info("Joined odds to %d/%d rows (%.1f%%)",
                matched, len(result), 100 * matched / max(len(result), 1))
    # Drop rows without odds — that's the bettable universe
    bettable = result[result["edge"].notna()].copy()
    logger.info("Bettable rows (edge not NaN): %d", len(bettable))
    return bettable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # backtest.py's print_report uses unicode arrows; force UTF-8 stdout
    # so Windows cp1252 doesn't crash the report.
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: holdout start)")
    parser.add_argument("--end",   default=None, help="YYYY-MM-DD (default: holdout end)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API calls — use cache only")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch even if cached (burns credits)")
    parser.add_argument("--per-game-close", action="store_true", default=True,
                        help="Per-game close-of-line snapshot (default)")
    parser.add_argument("--close-lead-min", type=int, default=60,
                        help="Snapshot N minutes before first pitch (default 60)")
    parser.add_argument("--save-results", action="store_true",
                        help="Write merged results to data/backtest/historical_results.csv")
    parser.add_argument("--kelly-mult", type=float, default=KELLY_FRACTION,
                        help=f"Fractional Kelly multiplier (default {KELLY_FRACTION})")
    args = parser.parse_args()

    configure_logging()

    if args.start is None or args.end is None:
        h_start, h_end = default_holdout_window()
        start = args.start or h_start
        end   = args.end   or h_end
    else:
        start, end = args.start, args.end
    logger.info("Backtest range: %s -> %s", start, end)

    # Build list of dates in window
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    dates = [(sd + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range((ed - sd).days + 1)]

    # 1. Score model on the window
    scored = score_window(start, end)

    # 2. Pull historical odds
    odds_by_date = collect_historical_odds(
        dates,
        per_game_close=args.per_game_close,
        close_lead_minutes=args.close_lead_min,
        force_refresh=args.force_refresh,
        no_fetch=args.no_fetch,
    )
    total_props = sum(len(v) for v in odds_by_date.values())
    days_with_odds = sum(1 for v in odds_by_date.values() if v)
    logger.info("Odds collected: %d props across %d/%d days",
                total_props, days_with_odds, len(dates))

    # 3. Join + compute bets
    results = enrich_with_historical_odds(scored, odds_by_date, kelly_mult=args.kelly_mult)
    if results.empty:
        logger.error("No bettable rows after join — nothing to report.")
        return

    # 4. Analyze using backtest.py helpers (drop-in)
    sweep = edge_threshold_sweep(results, kelly_mult=args.kelly_mult)
    cal   = calibration_table(results)
    rec   = find_recommended_threshold(sweep)
    print_report(results, sweep, cal, rec, show_calibration=True)

    if args.save_results:
        out = RESULTS_DIR / f"historical_results_{start}_to_{end}.csv"
        results.to_csv(out, index=False)
        logger.info("Saved merged results: %s", out)


if __name__ == "__main__":
    main()

"""
Backtesting module for the MLB HR model.

Joins saved daily prediction CSVs against actual Statcast outcomes to measure:
  - Model calibration: does hr_prob=0.15 actually hit ~15% of the time?
  - Edge quality: do positive-edge bets have positive ROI?
  - Kelly sizing: recommended bet size per play given edge and odds
  - Optimal edge threshold: the cutoff that maximises historical ROI

Usage
-----
    # Run the full backtest report:
    python -m src.analysis.backtest

    # Specify a date range:
    python -m src.analysis.backtest --start 2025-04-01 --end 2025-09-30

    # Show calibration curve only:
    python -m src.analysis.backtest --calibration-only

    # Save merged results to CSV for external analysis:
    python -m src.analysis.backtest --save-results

Output structure
----------------
  1. Summary header — date range, total bets, overall win rate
  2. Edge-threshold sweep — ROI / win-rate / units at every +1pp threshold
  3. Recommended threshold — the edge cutoff with best risk-adjusted ROI
  4. Kelly sizing — fractional Kelly stake per active bet
  5. Calibration check — model prob vs actual HR rate across prob buckets

Bet accounting
--------------
  All bets are ON THE OVER (HR yes) at the market_over_price.
  A $1 unit bet at American odds X returns:
    Win:  +$( 100/abs(X) ) if X < 0   or   +$(X/100) if X > 0
    Loss: -$1

Kelly fraction
--------------
  f* = (b*p - q) / b      where b = decimal odds - 1, p = model prob, q = 1-p
  We use fractional Kelly (default 0.25) to account for model uncertainty.
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import configure_logging
from src.data_sources.statcast import fetch_statcast_events

logger = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
RESULTS_DIR    = PROJECT_ROOT / "data" / "backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fractional Kelly multiplier — 0.25 is conservative but standard for sports
KELLY_FRACTION = 0.25
# Minimum edge to include a bet in the recommended strategy
DEFAULT_MIN_EDGE = 0.03
# Minimum bets required at a threshold before we trust its ROI estimate
MIN_BETS_FOR_RELIABLE_ROI = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _american_to_decimal(american: float) -> float:
    """Convert American moneyline to decimal odds (profit + stake)."""
    if american < 0:
        return 1 + (100 / abs(american))
    return 1 + (american / 100)


def _bet_pnl(american: float, won: bool) -> float:
    """
    Return profit/loss of a $1 unit bet.
    Win returns profit only (not stake); loss returns -1.
    """
    dec = _american_to_decimal(american)
    return (dec - 1) if won else -1.0


def kelly_fraction(
    model_prob: float,
    american_odds: float,
    kelly_mult: float = KELLY_FRACTION,
) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.

    Returns 0.0 when there is no edge (would recommend betting against yourself).

    Parameters
    ----------
    model_prob   : model's estimated probability of HR
    american_odds: market over price (e.g. -130, +110)
    kelly_mult   : fraction of full Kelly to use (default 0.25)
    """
    b = _american_to_decimal(american_odds) - 1  # net decimal profit per $1
    p = model_prob
    q = 1 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, round(f_full * kelly_mult, 4))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load all predictions_YYYY-MM-DD.csv files in data/predictions/.
    Filters to rows that have odds data (edge column present and not NaN).
    """
    files = sorted(glob.glob(str(PREDICTIONS_DIR / "predictions_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {PREDICTIONS_DIR}.\n"
            "Run src/model/predict.py with ODDS_API_KEY set to generate them."
        )

    # Priority for deduplication per (game_date, batter): lower = preferred
    _PASS_PRIORITY = {"final": 0, "manual": 1, "morning": 2, "unknown": 3}

    dfs = []
    for f in files:
        stem     = Path(f).stem.replace("predictions_", "")
        parts    = stem.split("_")
        date_str = parts[0]                                   # YYYY-MM-DD
        pass_sfx = parts[1] if len(parts) > 1 else None      # morning/final/None

        # When a canonical un-stamped file exists for the same date, skip the
        # stamped morning/final files — the final pass overwrites the canonical.
        if pass_sfx in ("morning", "final"):
            canonical = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
            if canonical.exists():
                continue

        try:
            df = pd.read_csv(f)
            df["game_date"] = pd.to_datetime(date_str)
            if "run_type" not in df.columns:
                df["run_type"] = pass_sfx or "manual"
            dfs.append(df)
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)

    if not dfs:
        raise ValueError("All prediction files failed to load.")

    preds = pd.concat(dfs, ignore_index=True)
    preds["game_date"] = pd.to_datetime(preds["game_date"])

    # Deduplicate: keep highest-priority pass per (game_date, batter)
    if "run_type" in preds.columns and "batter" in preds.columns:
        preds["_priority"] = preds["run_type"].map(
            lambda x: _PASS_PRIORITY.get(str(x), 3)
        )
        preds = (
            preds
            .sort_values("_priority")
            .drop_duplicates(subset=["game_date", "batter"], keep="first")
            .drop(columns=["_priority"])
            .reset_index(drop=True)
        )

    if start_date:
        preds = preds[preds["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        preds = preds[preds["game_date"] <= pd.to_datetime(end_date)]

    logger.info(
        "Loaded %d predictions across %d dates (%s → %s)",
        len(preds),
        preds["game_date"].nunique(),
        preds["game_date"].min().date(),
        preds["game_date"].max().date(),
    )
    return preds


def load_outcomes(dates: list[str]) -> pd.DataFrame:
    """
    Fetch actual HR outcomes for the given dates from Statcast.
    Uses the existing cache — no new downloads if already cached.

    Returns a DataFrame with columns: game_date, batter, hr_hit (0/1).
    """
    if not dates:
        return pd.DataFrame(columns=["game_date", "batter", "hr_hit"])

    dates_sorted = sorted(dates)
    start = dates_sorted[0]
    end   = dates_sorted[-1]

    logger.info("Loading Statcast outcomes %s → %s ...", start, end)

    # Do NOT filter to regular season here — predictions may have been made
    # for playoff games (Wild Card, etc.) and we want the actual outcomes
    # for whatever dates predictions exist, regardless of game_type.
    events = fetch_statcast_events(
        start_date=start,
        end_date=end,
        columns=["game_date", "game_pk", "batter", "events", "game_type"],
        regular_season_only=False,
    ).df

    events["game_date"] = pd.to_datetime(events["game_date"])
    events["is_hr"] = (events["events"] == "home_run").fillna(False).astype(int)

    # Collapse to one row per (batter, game_date): did they HR at all?
    outcomes = (
        events
        .groupby(["game_date", "batter"], as_index=False)
        .agg(hr_hit=("is_hr", "max"))
    )
    # Filter to only the dates we actually have predictions for
    date_set = {pd.to_datetime(d) for d in dates}
    outcomes = outcomes[outcomes["game_date"].isin(date_set)]

    logger.info(
        "Outcomes loaded: %d batter-games, HR rate=%.4f",
        len(outcomes),
        outcomes["hr_hit"].mean(),
    )
    return outcomes


# ---------------------------------------------------------------------------
# Merge + annotate
# ---------------------------------------------------------------------------

def build_results(preds: pd.DataFrame, outcomes: pd.DataFrame, kelly_mult: float = KELLY_FRACTION) -> pd.DataFrame:
    """
    Join predictions to outcomes and compute per-row bet P&L.

    Only rows with valid odds (edge not NaN, market_over_price not NaN)
    are included — these are the bets we could have actually placed.
    """
    # Need batter as int for the join
    preds = preds.copy()
    preds["batter"] = pd.to_numeric(preds["batter"], errors="coerce").astype("Int64")
    outcomes["batter"] = pd.to_numeric(outcomes["batter"], errors="coerce").astype("Int64")

    merged = preds.merge(
        outcomes.rename(columns={"hr_hit": "actual_hr"}),
        on=["game_date", "batter"],
        how="left",
    )

    # Rows without an outcome match = game not yet played or data missing
    missing_outcome = merged["actual_hr"].isna().sum()
    if missing_outcome:
        logger.warning(
            "%d predictions have no Statcast outcome — "
            "likely future dates or missing cache. Dropping.",
            missing_outcome,
        )
    merged = merged.dropna(subset=["actual_hr"]).copy()
    merged["actual_hr"] = merged["actual_hr"].astype(int)

    # Only keep rows where we had an odds line
    if "edge" not in merged.columns or "market_over_price" not in merged.columns:
        logger.warning(
            "Predictions are missing odds columns (edge, market_over_price). "
            "These files were generated without ODDS_API_KEY set. "
            "Run predict.py with ODDS_API_KEY to generate bettable predictions."
        )
        return pd.DataFrame()

    has_odds = merged["edge"].notna() & merged["market_over_price"].notna()
    bettable = merged[has_odds].copy()

    if bettable.empty:
        logger.warning(
            "No rows with odds data found. Did predictions include ODDS_API_KEY?"
        )
        return bettable

    bettable["decimal_odds"] = bettable["market_over_price"].apply(_american_to_decimal)
    bettable["pnl_1u"]       = bettable.apply(
        lambda r: _bet_pnl(r["market_over_price"], bool(r["actual_hr"])), axis=1
    )
    bettable["kelly_stake"]  = bettable.apply(
        lambda r: kelly_fraction(r["hr_prob"], r["market_over_price"], kelly_mult), axis=1
    )
    bettable["kelly_pnl"]    = bettable["pnl_1u"] * bettable["kelly_stake"]

    logger.info(
        "Results built: %d bettable rows, %d unique dates, overall win rate=%.3f",
        len(bettable),
        bettable["game_date"].nunique(),
        bettable["actual_hr"].mean(),
    )
    return bettable


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def edge_threshold_sweep(results: pd.DataFrame, kelly_mult: float = KELLY_FRACTION) -> pd.DataFrame:
    """
    For every edge threshold from 0 to 0.10 (step 0.01), compute:
      - n_bets    : bets placed
      - win_rate  : fraction won
      - roi_1u    : return on investment (flat $1/bet)
      - units_1u  : total units profit
      - roi_kelly : ROI using fractional Kelly stakes
    """
    rows = []
    thresholds = np.arange(0.00, 0.11, 0.01)

    for t in thresholds:
        subset = results[results["edge"] >= t]
        n = len(subset)
        if n == 0:
            continue
        win_rate  = subset["actual_hr"].mean()
        units_1u  = subset["pnl_1u"].sum()
        roi_1u    = units_1u / n
        kelly_pnl = subset.apply(
            lambda r: _bet_pnl(r["market_over_price"], bool(r["actual_hr"]))
                      * kelly_fraction(r["hr_prob"], r["market_over_price"], kelly_mult),
            axis=1,
        ).sum()
        roi_kelly = kelly_pnl / n if n > 0 else np.nan
        rows.append({
            "min_edge":    round(t, 2),
            "n_bets":      n,
            "win_rate":    round(win_rate, 4),
            "units_1u":    round(units_1u, 3),
            "roi_1u":      round(roi_1u, 4),
            "roi_kelly":   round(roi_kelly, 4),
            "reliable":    n >= MIN_BETS_FOR_RELIABLE_ROI,
        })

    return pd.DataFrame(rows)


def calibration_table(results: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """
    Bucket bets by model hr_prob and compare to actual HR rate.
    A well-calibrated model should have actual ≈ predicted in each bucket.
    """
    results = results.copy()
    results["prob_bucket"] = pd.cut(results["hr_prob"], bins=n_bins)

    cal = (
        results.groupby("prob_bucket", observed=True)
        .agg(
            n=("actual_hr", "size"),
            predicted=("hr_prob", "mean"),
            actual=("actual_hr", "mean"),
        )
        .reset_index()
    )
    cal["calibration_error"] = (cal["predicted"] - cal["actual"]).abs().round(4)
    return cal


def find_recommended_threshold(sweep: pd.DataFrame) -> float:
    """
    Return the minimum edge threshold that maximises flat-bet ROI
    subject to having at least MIN_BETS_FOR_RELIABLE_ROI bets.
    Falls back to DEFAULT_MIN_EDGE if no threshold clears the bar.
    """
    reliable = sweep[sweep["reliable"]]
    if reliable.empty:
        logger.warning(
            "No edge threshold has >= %d bets. Using default %.2f.",
            MIN_BETS_FOR_RELIABLE_ROI, DEFAULT_MIN_EDGE,
        )
        return DEFAULT_MIN_EDGE

    best_idx = reliable["roi_1u"].idxmax()
    return float(reliable.loc[best_idx, "min_edge"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x*100:+.1f}%"

def _pp(x: float) -> str:
    return f"{x*100:+.1f}pp"


def print_report(
    results: pd.DataFrame,
    sweep: pd.DataFrame,
    cal: pd.DataFrame,
    rec_threshold: float,
    *,
    show_calibration: bool = True,
) -> None:
    date_min = results["game_date"].min().date()
    date_max = results["game_date"].max().date()
    n_dates  = results["game_date"].nunique()

    print(f"\n{'='*65}")
    print(f"  MLB HR MODEL — BACKTEST REPORT")
    print(f"{'='*65}")
    print(f"  Period  : {date_min}  →  {date_max}  ({n_dates} days)")
    print(f"  Bets    : {len(results):,}  (all rows with odds lines)")
    print(f"  Win rate: {results['actual_hr'].mean()*100:.1f}%")
    print(f"  ROI (flat $1): {_pct(results['pnl_1u'].sum() / len(results))}")

    # --- Edge sweep ---
    print(f"\n  {'Min Edge':>8}  {'Bets':>6}  {'Win%':>6}  {'Units':>7}  {'ROI':>7}  {'Note'}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*4}")
    for _, row in sweep.iterrows():
        flag = "  ◀ recommended" if row["min_edge"] == rec_threshold else ""
        flag = flag or ("" if row["reliable"] else "  (low n)")
        print(
            f"  {_pp(row['min_edge']):>8}  {int(row['n_bets']):>6}  "
            f"{row['win_rate']*100:>5.1f}%  {row['units_1u']:>+7.2f}u  "
            f"{_pct(row['roi_1u']):>7}{flag}"
        )

    # --- Recommended strategy summary ---
    rec_bets = results[results["edge"] >= rec_threshold]
    print(f"\n{'='*65}")
    print(f"  RECOMMENDED STRATEGY  (edge ≥ {_pp(rec_threshold)})")
    print(f"{'='*65}")
    if rec_bets.empty:
        print("  No bets meet this threshold in the backtest period.")
    else:
        avg_kelly = rec_bets["kelly_stake"].mean()
        kelly_roi = rec_bets["kelly_pnl"].sum() / len(rec_bets)
        print(f"  Bets placed   : {len(rec_bets):,}")
        print(f"  Win rate      : {rec_bets['actual_hr'].mean()*100:.1f}%")
        print(f"  Flat ROI      : {_pct(rec_bets['pnl_1u'].sum() / len(rec_bets))}")
        print(f"  Avg Kelly stake  : {avg_kelly*100:.2f}% of bankroll per bet")
        print(f"  Kelly ROI     : {_pct(kelly_roi)}")
        print(f"\n  Sample bets (top edge, this period):")
        print(f"  {'Date':<12} {'Batter':<24} {'Prob':>6} {'Fair':>6} {'Edge':>7} {'Odds':>6} {'Won':>4}")
        print(f"  {'-'*12} {'-'*24} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*4}")
        top = rec_bets.sort_values("edge", ascending=False).head(10)
        for _, r in top.iterrows():
            won = "✓" if r["actual_hr"] else "✗"
            name = str(r.get("batter_name", r["batter"]))[:24]
            print(
                f"  {str(r['game_date'].date()):<12} {name:<24} "
                f"{r['hr_prob']*100:>5.1f}% {r['market_fair_prob']*100:>5.1f}% "
                f"{_pp(r['edge']):>7} {int(r['market_over_price']):>+6} {won:>4}"
            )

    # --- Calibration ---
    if show_calibration:
        print(f"\n{'='*65}")
        print(f"  MODEL CALIBRATION  (predicted vs actual HR rate)")
        print(f"{'='*65}")
        print(f"  {'Bucket':<18} {'N':>5}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
        print(f"  {'-'*18} {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")
        for _, row in cal.iterrows():
            print(
                f"  {str(row['prob_bucket']):<18} {int(row['n']):>5}  "
                f"{row['predicted']*100:>9.1f}%  {row['actual']*100:>7.1f}%  "
                f"{row['calibration_error']*100:>7.1f}pp"
            )
        mean_err = cal["calibration_error"].mean()
        print(f"\n  Mean absolute calibration error: {mean_err*100:.2f}pp")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the MLB HR model against historical predictions and outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.analysis.backtest\n"
            "  python -m src.analysis.backtest --start 2025-04-01 --end 2025-09-30\n"
            "  python -m src.analysis.backtest --save-results\n"
            "  python -m src.analysis.backtest --calibration-only\n"
        ),
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--min-edge", type=float, default=None,
                        help="Override recommended edge threshold (e.g. 0.04)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save merged results CSV to data/backtest/")
    parser.add_argument("--calibration-only", action="store_true",
                        help="Show calibration table only, skip betting analysis")
    parser.add_argument("--kelly-fraction", type=float, default=KELLY_FRACTION,
                        help=f"Fractional Kelly multiplier (default {KELLY_FRACTION})")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(level=logging.DEBUG if args.debug else logging.INFO)

    kelly_mult = args.kelly_fraction

    # --- Load predictions ---
    preds = load_predictions(start_date=args.start, end_date=args.end)

    # --- Load outcomes ---
    dates = preds["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()
    outcomes = load_outcomes(dates)

    if outcomes.empty:
        print("No Statcast outcomes found for the prediction dates.")
        print("The Statcast cache may not cover this period yet.")
        return

    # --- Calibration-only mode: works even without odds columns ---
    if args.calibration_only:
        preds_with_outcomes = preds.copy()
        preds_with_outcomes["batter"] = pd.to_numeric(preds_with_outcomes["batter"], errors="coerce").astype("Int64")
        outcomes["batter"] = pd.to_numeric(outcomes["batter"], errors="coerce").astype("Int64")
        merged_cal = preds_with_outcomes.merge(
            outcomes.rename(columns={"hr_hit": "actual_hr"}),
            on=["game_date", "batter"], how="inner",
        ).dropna(subset=["actual_hr", "hr_prob"])
        if merged_cal.empty:
            print("No matched predictions+outcomes for calibration.")
            return
        cal = calibration_table(merged_cal)
        print(f"\n{'='*65}")
        print(f"  MODEL CALIBRATION  (predicted vs actual HR rate)")
        print(f"  {len(merged_cal):,} batter-games  |  actual HR rate: {merged_cal['actual_hr'].mean()*100:.1f}%")
        print(f"{'='*65}")
        print(f"  {'Bucket':<18} {'N':>5}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
        print(f"  {'-'*18} {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")
        for _, row in cal.iterrows():
            print(
                f"  {str(row['prob_bucket']):<18} {int(row['n']):>5}  "
                f"{row['predicted']*100:>9.1f}%  {row['actual']*100:>7.1f}%  "
                f"{row['calibration_error']*100:>7.1f}pp"
            )
        print(f"\n  Mean absolute calibration error: {cal['calibration_error'].mean()*100:.2f}pp")
        print(f"{'='*65}\n")
        return

    # --- Build results (requires odds columns) ---
    results = build_results(preds, outcomes, kelly_mult=kelly_mult)

    if results.empty:
        print("\nNo bettable results found — predictions are missing odds columns.")
        print("This means predict.py was run without ODDS_API_KEY set.")
        print("\nTo generate predictions with odds:")
        print("  1. Set ODDS_API_KEY in your .env or environment")
        print("  2. Run: python -m src.model.predict")
        print("  3. Re-run backtest once you have a few days of odds-enriched predictions")
        print(f"\n  Calibration-only mode works on existing predictions:")
        print(f"     python -m src.analysis.backtest --calibration-only")
        return

    # --- Analysis ---
    sweep = edge_threshold_sweep(results, kelly_mult=kelly_mult)
    cal   = calibration_table(results)
    rec_threshold = args.min_edge if args.min_edge is not None else find_recommended_threshold(sweep)

    # --- Report ---
    print_report(
        results, sweep, cal, rec_threshold,
        show_calibration=not args.calibration_only,
    )

    # --- Optional CSV export ---
    if args.save_results:
        out = RESULTS_DIR / f"backtest_results_{results['game_date'].min().date()}_{results['game_date'].max().date()}.csv"
        results.to_csv(out, index=False)
        print(f"  Results saved to: {out.relative_to(PROJECT_ROOT)}")

        sweep_out = RESULTS_DIR / "edge_sweep.csv"
        sweep.to_csv(sweep_out, index=False)
        print(f"  Edge sweep saved to: {sweep_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

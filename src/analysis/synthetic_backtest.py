"""
MLB HR Model — Historical Synthetic Odds Backtest
===================================================
Since real historical prop lines aren't available on the free Odds API tier,
this script synthesizes realistic market lines from the model's own hr_prob
and runs a full backtest over the historical train table (2021–2025).

How synthetic lines work
------------------------
Real sportsbooks price HR props using their own model + vig. We approximate
this using a two-component market model:

  1. Market fair probability  = beta_0 + beta_1 * hr_prob
     Books don't perfectly agree with any one model. We shrink the model
     probability toward the league-average base rate (~3.3%) to simulate
     a more conservative market estimate.

  2. Vig (overround)          = ~4.5% total (typical for player props)
     Applied symmetrically to get over/under American odds.

The shrinkage parameter controls how much the synthetic market disagrees
with the model. At shrinkage=0.0, market = model (no edge by construction).
At shrinkage=0.3, market is 30% regressed toward league avg — this is
the most realistic setting based on observed book behavior for HR props.

Shrinkage range sweep:
  --shrinkage 0.0   → market perfectly agrees with model (null hypothesis)
  --shrinkage 0.15  → market slightly conservative (optimistic for model)
  --shrinkage 0.30  → realistic (default)
  --shrinkage 0.50  → very conservative market (stress test)

Usage
-----
    # Full backtest with default settings:
    python -m src.analysis.synthetic_backtest

    # Specify date range:
    python -m src.analysis.synthetic_backtest --start 2024-04-01 --end 2024-09-30

    # Sweep shrinkage values to see sensitivity:
    python -m src.analysis.synthetic_backtest --sweep-shrinkage

    # Save results to CSV:
    python -m src.analysis.synthetic_backtest --save-results

    # Calibration-only (no betting sim):
    python -m src.analysis.synthetic_backtest --calibration-only

Output
------
  Console: full backtest report (same format as backtest.py)
  CSV:     data/backtest/synthetic_backtest_results_<date_range>.csv
           data/backtest/synthetic_edge_sweep_<date_range>.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
BACKTEST_DIR  = PROJECT_ROOT / "data" / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_AVG_HR_RATE  = 0.033   # ~3.3% per PA, long-run MLB average
VIG_PCT             = 0.045   # 4.5% total book margin on HR props (typical)
DEFAULT_SHRINKAGE   = 0.30    # market regresses 30% toward league avg
KELLY_FRACTION      = 0.25    # fractional Kelly multiplier
MIN_EDGE            = 0.03    # minimum edge to count as a bet
MIN_BETS_RELIABLE   = 50      # minimum bets for reliable ROI estimate


# ---------------------------------------------------------------------------
# Market model
# ---------------------------------------------------------------------------

def synthesize_market(
    hr_prob: pd.Series,
    shrinkage: float = DEFAULT_SHRINKAGE,
    vig: float = VIG_PCT,
    league_avg: float = LEAGUE_AVG_HR_RATE,
) -> pd.DataFrame:
    """
    Given model HR probabilities, synthesize realistic market lines.

    The market fair probability is a shrinkage-blend of model prob and
    league average, simulating a market that uses a more conservative model:

        market_fair = (1 - shrinkage) * hr_prob + shrinkage * league_avg

    Vig is applied symmetrically to convert fair probability to American odds:
        implied_over  = market_fair * (1 + vig/2)
        implied_under = (1 - market_fair) * (1 + vig/2)
        over_price    = implied_to_american(implied_over)

    Returns a DataFrame with columns:
        market_fair_prob, market_over_price, market_under_price,
        market_implied_prob
    """
    market_fair = (1 - shrinkage) * hr_prob + shrinkage * league_avg

    # Apply vig: inflate both sides proportionally
    implied_over  = market_fair * (1 + vig / 2)
    implied_under = (1 - market_fair) * (1 + vig / 2)

    def implied_to_american(p: pd.Series) -> pd.Series:
        p = p.clip(0.001, 0.999)
        return np.where(
            p >= 0.5,
            -(p / (1 - p) * 100).round(),
            ((1 - p) / p * 100).round(),
        ).astype(float)

    return pd.DataFrame({
        "market_fair_prob":    market_fair.round(4),
        "market_implied_prob": implied_over.round(4),
        "market_over_price":   implied_to_american(implied_over),
        "market_under_price":  implied_to_american(implied_under),
        "odds_bookmaker":      "synthetic",
    })


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

def _american_to_decimal(american_odds: float) -> float:
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def kelly_stake(
    model_prob: float,
    american_odds: float,
    kelly_mult: float = KELLY_FRACTION,
) -> float:
    b = _american_to_decimal(american_odds) - 1
    p = model_prob
    q = 1 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, round(f_full * kelly_mult, 4))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_table(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load the multi-season train table and filter to date range."""
    # Explicit candidates in preference order — add new filenames here when
    # train.py saves under a new name (e.g. 2021_2027 next year).
    candidates = [
        PROCESSED_DIR / "train_table_2021_2026_full.parquet",
        PROCESSED_DIR / "train_table_2021_2025_full.parquet",
        PROCESSED_DIR / "train_table_2024_full_season.parquet",
    ]
    path = None
    for c in candidates:
        if c.exists() and c.stat().st_size > 10_000:
            path = c
            break
    if path is None:
        # Glob fallback: prefer largest non-empty combined file over per-month stubs.
        files = [
            p for p in PROCESSED_DIR.glob("train_table_*.parquet")
            if p.stat().st_size > 10_000 and "full" in p.name
        ]
        if not files:
            files = [
                p for p in PROCESSED_DIR.glob("train_table_*.parquet")
                if p.stat().st_size > 10_000
            ]
        if not files:
            raise FileNotFoundError(
                "No non-empty train_table_*.parquet found in data/processed/.\n"
                "Run src/features/build_features_multi_season.py first."
            )
        files.sort(key=lambda p: p.stat().st_size, reverse=True)
        path = files[0]

    logger.info("Loading train table: %s", path.name)
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    if start_date:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    logger.info(
        "Train table: %d rows | %s → %s",
        len(df),
        df["game_date"].min().date(),
        df["game_date"].max().date(),
    )
    return df


def score_with_model(df: pd.DataFrame) -> pd.DataFrame:
    """Load the trained model and score all rows in df."""
    import joblib
    import glob as _glob

    # Preference order: newest combined model first.
    # Add the next year's filename here each offseason after retraining.
    candidates = [
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2026.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2026.joblib",
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_lightgbm_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2024.joblib",
    ]
    bundle = None
    for c in candidates:
        if c.exists():
            bundle = joblib.load(c)
            logger.info("Loaded model: %s", c.name)
            break
    if bundle is None:
        raise FileNotFoundError(
            "No trained model found in models/. Run src/model/train.py first."
        )

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    apply_shrink = bundle.get("apply_shrinkage", False)

    if apply_shrink:
        from src.model.train import apply_shrinkage as _apply_shrinkage
        logger.info("Applying empirical Bayes shrinkage ...")
        df = _apply_shrinkage(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("%d feature(s) missing — filling with 0: %s", len(missing), missing[:5])
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols].fillna(0.0)
    df = df.copy()
    df["hr_prob"] = model.predict_proba(X)[:, 1]
    return df


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def build_synthetic_results(
    df: pd.DataFrame,
    shrinkage: float = DEFAULT_SHRINKAGE,
    kelly_mult: float = KELLY_FRACTION,
) -> pd.DataFrame:
    """
    Given a scored DataFrame (with hr_prob and hr_hit), synthesize market
    lines and compute per-row P&L.

    Required columns: hr_prob, hr_hit
    Returns: same DataFrame enriched with market + betting columns.
    """
    if "hr_prob" not in df.columns:
        raise ValueError("df must have hr_prob column. Call score_with_model() first.")
    if "hr_hit" not in df.columns:
        raise ValueError("df must have hr_hit column.")

    result = df.copy()

    # Synthesize market
    market = synthesize_market(result["hr_prob"], shrinkage=shrinkage)
    result = pd.concat([result.reset_index(drop=True), market.reset_index(drop=True)], axis=1)

    # Edge
    result["edge"] = (result["hr_prob"] - result["market_fair_prob"]).round(4)

    # Kelly stake
    result["kelly_stake"] = [
        kelly_stake(float(p), float(o), kelly_mult)
        for p, o in zip(result["hr_prob"], result["market_over_price"])
    ]

    # P&L per unit flat bet
    dec_odds = result["market_over_price"].apply(_american_to_decimal)
    result["pnl_1u"] = np.where(
        result["hr_hit"] == 1,
        dec_odds - 1,
        -1.0,
    )

    # P&L Kelly
    result["kelly_pnl"] = result["pnl_1u"] * result["kelly_stake"]

    return result


def edge_threshold_sweep(
    results: pd.DataFrame,
    kelly_mult: float = KELLY_FRACTION,
) -> pd.DataFrame:
    """Sweep edge thresholds 0–10pp and report ROI at each."""
    rows = []
    for threshold in np.arange(0.00, 0.101, 0.01):
        bets = results[results["edge"] >= threshold]
        if len(bets) == 0:
            continue
        win_rate  = bets["hr_hit"].mean()
        units_1u  = bets["pnl_1u"].sum()
        roi_1u    = bets["pnl_1u"].mean()
        roi_kelly = bets["kelly_pnl"].mean() if "kelly_pnl" in bets else float("nan")
        rows.append({
            "min_edge":  round(threshold, 2),
            "n_bets":    len(bets),
            "win_rate":  round(win_rate, 4),
            "units_1u":  round(units_1u, 2),
            "roi_1u":    round(roi_1u, 4),
            "roi_kelly": round(roi_kelly, 4),
            "reliable":  len(bets) >= MIN_BETS_RELIABLE,
        })
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bucket hr_prob into deciles and compare predicted vs actual HR rate."""
    df = df.copy().dropna(subset=["hr_prob", "hr_hit"])
    df["prob_bucket"] = pd.qcut(df["hr_prob"], q=n_bins, duplicates="drop")
    cal = (
        df.groupby("prob_bucket", observed=True)
        .agg(
            n=("hr_hit", "count"),
            predicted=("hr_prob", "mean"),
            actual=("hr_hit", "mean"),
        )
        .reset_index()
    )
    cal["calibration_error"] = (cal["predicted"] - cal["actual"]).abs()
    return cal


def find_recommended_threshold(sweep: pd.DataFrame) -> float:
    """Return edge threshold with best flat-bet ROI among reliable buckets."""
    reliable = sweep[sweep["reliable"]]
    if reliable.empty:
        return MIN_EDGE
    best = reliable.loc[reliable["roi_1u"].idxmax()]
    return float(best["min_edge"])


# ---------------------------------------------------------------------------
# Shrinkage sweep
# ---------------------------------------------------------------------------

def shrinkage_sweep(
    df: pd.DataFrame,
    shrinkage_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Test multiple shrinkage values to understand sensitivity.
    Returns a DataFrame summarizing ROI at each shrinkage level.
    """
    if shrinkage_values is None:
        shrinkage_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    rows = []
    for s in shrinkage_values:
        results = build_synthetic_results(df, shrinkage=s)
        bets    = results[results["edge"] >= MIN_EDGE]
        if len(bets) == 0:
            continue
        rows.append({
            "shrinkage":    s,
            "n_bets":       len(bets),
            "avg_edge_pp":  round(bets["edge"].mean() * 100, 2),
            "win_rate":     round(bets["hr_hit"].mean(), 4),
            "roi_1u":       round(bets["pnl_1u"].mean(), 4),
            "total_units":  round(bets["pnl_1u"].sum(), 2),
            "roi_kelly":    round(bets["kelly_pnl"].mean(), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    results: pd.DataFrame,
    sweep: pd.DataFrame,
    cal: pd.DataFrame,
    rec_threshold: float,
    shrinkage: float,
    date_range: tuple[str, str],
    min_prob: float | None = None,
) -> None:
    start, end = date_range
    n_total  = len(results)
    n_bets   = len(results[results["edge"] >= rec_threshold])
    hr_rate  = results["hr_hit"].mean()
    prob_filter = f"  |  min-prob: {min_prob:.3f}" if min_prob is not None else ""

    print(f"\n{'='*72}")
    print(f"  SYNTHETIC ODDS BACKTEST")
    print(f"  {start} → {end}  |  market shrinkage: {shrinkage:.0%}  |  Kelly: {KELLY_FRACTION:.0%}x{prob_filter}")
    print(f"{'='*72}")
    print(f"  Total batter-games scored : {n_total:,}")
    print(f"  Actual HR rate            : {hr_rate*100:.2f}%")
    print(f"  League avg HR rate        : {LEAGUE_AVG_HR_RATE*100:.2f}%")
    print(f"  Recommended edge threshold: +{rec_threshold*100:.0f}pp")
    print(f"  Bets at recommended edge  : {n_bets:,}")

    print(f"\n  ── Edge Threshold Sweep ──────────────────────────────────────────")
    print(f"  {'Edge':>6}  {'Bets':>6}  {'Win%':>6}  {'Units':>7}  {'ROI/bet':>8}  {'Reliable':>8}")
    print(f"  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}  {'--------':>8}  {'--------':>8}")
    for _, row in sweep.iterrows():
        reliable = "✓" if row["reliable"] else "  "
        marker   = " ← rec" if abs(row["min_edge"] - rec_threshold) < 0.005 else ""
        print(
            f"  {row['min_edge']*100:>5.0f}pp  "
            f"{int(row['n_bets']):>6,}  "
            f"{row['win_rate']*100:>5.1f}%  "
            f"{row['units_1u']:>+7.1f}u  "
            f"{row['roi_1u']*100:>+7.2f}%  "
            f"  {reliable}{marker}"
        )

    # Recommended threshold detail
    rec_bets = results[results["edge"] >= rec_threshold]
    if len(rec_bets) > 0:
        avg_edge  = rec_bets["edge"].mean() * 100
        avg_kelly = rec_bets["kelly_stake"].mean() * 100
        total_u   = rec_bets["pnl_1u"].sum()
        roi_kelly = rec_bets["kelly_pnl"].mean() * 100
        print(f"\n  ── Recommended Strategy (edge ≥ +{rec_threshold*100:.0f}pp) ──────────────────")
        print(f"  Bets         : {len(rec_bets):,}")
        print(f"  Win rate     : {rec_bets['hr_hit'].mean()*100:.1f}%")
        print(f"  Avg edge     : +{avg_edge:.1f}pp")
        print(f"  Avg Kelly    : {avg_kelly:.1f}% of bankroll")
        print(f"  Total P&L    : {total_u:+.1f} units (flat $1/bet)")
        print(f"  ROI per bet  : {rec_bets['pnl_1u'].mean()*100:+.2f}%")
        print(f"  ROI (Kelly)  : {roi_kelly:+.2f}% per bet")

        # Sample top bets
        sample = rec_bets.nlargest(5, "edge")
        print(f"\n  Sample highest-edge bets:")
        for _, row in sample.iterrows():
            result_str = "✓ HR" if row["hr_hit"] == 1 else "✗"
            print(
                f"    {str(row.get('game_date', ''))[:10]}  "
                f"batter={int(row.get('batter', 0))}  "
                f"model={row['hr_prob']*100:.1f}%  "
                f"fair={row['market_fair_prob']*100:.1f}%  "
                f"edge={row['edge']*100:+.1f}pp  "
                f"odds={int(row['market_over_price']):+d}  "
                f"{result_str}"
            )

    # Calibration
    print(f"\n  ── Model Calibration ─────────────────────────────────────────────")
    print(f"  {'Bucket':<22}  {'N':>6}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
    print(f"  {'-'*22}  {'------':>6}  {'----------':>10}  {'--------':>8}  {'--------':>8}")
    for _, row in cal.iterrows():
        print(
            f"  {str(row['prob_bucket']):<22}  "
            f"{int(row['n']):>6,}  "
            f"{row['predicted']*100:>9.2f}%  "
            f"{row['actual']*100:>7.2f}%  "
            f"{row['calibration_error']*100:>7.2f}pp"
        )
    mae = cal["calibration_error"].mean() * 100
    print(f"\n  Mean absolute calibration error: {mae:.2f}pp")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Synthetic odds backtest using historical train table data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start",   default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--shrinkage", type=float, default=DEFAULT_SHRINKAGE,
        help=f"Market shrinkage toward league avg (default {DEFAULT_SHRINKAGE}). "
             "0=market agrees with model, 0.5=very conservative market.",
    )
    parser.add_argument(
        "--sweep-shrinkage", action="store_true",
        help="Sweep multiple shrinkage values to test sensitivity.",
    )
    parser.add_argument(
        "--min-edge", type=float, default=None,
        help=f"Override recommended edge threshold (default: auto-detected).",
    )
    parser.add_argument(
        "--min-prob", type=float, default=None,
        help=(
            "Filter out batter-games where model hr_prob is below this threshold "
            "before computing edge/ROI. E.g. --min-prob 0.06 drops the weakest "
            "candidates where the model is most overconfident. "
            "Affects all modes: full backtest, shrinkage sweep, calibration."
        ),
    )
    parser.add_argument(
        "--save-results", action="store_true",
        help="Save results CSVs to data/backtest/.",
    )
    parser.add_argument(
        "--calibration-only", action="store_true",
        help="Only show calibration table, skip betting sim.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Load + score ---
    df = load_train_table(start_date=args.start, end_date=args.end)
    df = score_with_model(df)

    if "hr_hit" not in df.columns:
        raise ValueError("Train table missing hr_hit column.")

    # --- Apply min-prob filter ---
    if args.min_prob is not None:
        n_before = len(df)
        df = df[df["hr_prob"] >= args.min_prob].copy()
        n_dropped = n_before - len(df)
        logger.info(
            "min-prob filter (%.3f): dropped %d rows (%.1f%%), %d remain",
            args.min_prob, n_dropped, n_dropped / n_before * 100, len(df),
        )

    date_range = (
        str(df["game_date"].min().date()),
        str(df["game_date"].max().date()),
    )

    # --- Calibration only ---
    if args.calibration_only:
        cal = calibration_table(df)
        prob_filter = f"  |  min-prob: {args.min_prob:.3f}" if args.min_prob else ""
        print(f"\n{'='*65}")
        print(f"  MODEL CALIBRATION  ({date_range[0]} → {date_range[1]}){prob_filter}")
        print(f"  {len(df):,} batter-games  |  actual HR rate: {df['hr_hit'].mean()*100:.2f}%")
        print(f"{'='*65}")
        print(f"  {'Bucket':<22}  {'N':>6}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
        print(f"  {'-'*22}  {'------':>6}  {'----------':>10}  {'--------':>8}  {'--------':>8}")
        for _, row in cal.iterrows():
            print(
                f"  {str(row['prob_bucket']):<22}  "
                f"{int(row['n']):>6,}  "
                f"{row['predicted']*100:>9.2f}%  "
                f"{row['actual']*100:>7.2f}%  "
                f"{row['calibration_error']*100:>7.2f}pp"
            )
        print(f"\n  Mean absolute calibration error: {cal['calibration_error'].mean()*100:.2f}pp")
        print(f"{'='*65}\n")
        return

    # --- Shrinkage sweep ---
    if args.sweep_shrinkage:
        prob_filter = f"  |  min-prob: {args.min_prob:.3f}" if args.min_prob else ""
        print(f"\n{'='*72}")
        print(f"  SHRINKAGE SENSITIVITY SWEEP  ({date_range[0]} → {date_range[1]}){prob_filter}")
        print(f"  Edge threshold: +{MIN_EDGE*100:.0f}pp  |  Kelly: {KELLY_FRACTION:.0%}x")
        print(f"{'='*72}")
        print(f"  {'Shrink':>7}  {'Bets':>6}  {'AvgEdge':>8}  {'Win%':>6}  {'ROI/bet':>8}  {'TotalU':>8}  {'KellyROI':>9}")
        print(f"  {'-------':>7}  {'------':>6}  {'--------':>8}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'---------':>9}")
        sweep_df = shrinkage_sweep(df)
        for _, row in sweep_df.iterrows():
            print(
                f"  {row['shrinkage']*100:>6.0f}%  "
                f"{int(row['n_bets']):>6,}  "
                f"{row['avg_edge_pp']:>+7.2f}pp  "
                f"{row['win_rate']*100:>5.1f}%  "
                f"{row['roi_1u']*100:>+7.2f}%  "
                f"{row['total_units']:>+8.1f}u  "
                f"{row['roi_kelly']*100:>+8.2f}%"
            )
        print(f"{'='*72}\n")
        print("Note: shrinkage=0% means market perfectly mirrors model → no edge by construction.")
        print(f"      shrinkage={DEFAULT_SHRINKAGE*100:.0f}% is the default (realistic market disagreement).\n")
        return

    # --- Full backtest ---
    results = build_synthetic_results(df, shrinkage=args.shrinkage)
    sweep   = edge_threshold_sweep(results)
    cal     = calibration_table(results)
    rec     = args.min_edge if args.min_edge is not None else find_recommended_threshold(sweep)

    print_report(results, sweep, cal, rec, args.shrinkage, date_range, min_prob=args.min_prob)

    if args.save_results:
        prob_tag  = f"_minprob{int(args.min_prob*1000)}" if args.min_prob else ""
        stem = f"synthetic_{date_range[0]}_{date_range[1]}_shrink{int(args.shrinkage*100)}{prob_tag}"
        results_path = BACKTEST_DIR / f"{stem}_results.csv"
        sweep_path   = BACKTEST_DIR / f"{stem}_sweep.csv"
        results.to_csv(results_path, index=False)
        sweep.to_csv(sweep_path, index=False)
        print(f"  Results saved: {results_path.relative_to(PROJECT_ROOT)}")
        print(f"  Sweep saved  : {sweep_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()

"""
MLB HR Model — Weekly Recap
============================
Generates a weekly performance summary comparing predictions against actual
Statcast outcomes. Shows P&L, calibration, win rate, and ROI by edge bucket.

Usage
-----
    # Last 7 days:
    python -m src.analysis.weekly_recap

    # Specific week (Monday–Sunday):
    python -m src.analysis.weekly_recap --week 2026-03-10

    # All-time summary:
    python -m src.analysis.weekly_recap --all-time

    # Save to CSV:
    python -m src.analysis.weekly_recap --save

    # Post to Discord:
    python -m src.analysis.weekly_recap --discord

Kelly Sizing Note
-----------------
This module uses ONE canonical Kelly implementation (below) that matches
the formula used at training time. The two redundant copies in odds.py and
backtest.py should be replaced with imports from here.

Full Kelly formula:
    f* = (b·p - q) / b    where b = decimal_odds - 1, p = win_prob, q = 1 - p

We apply a fractional multiplier (default 0.25×) to account for model
uncertainty. This is the Kelly stake as a FRACTION OF BANKROLL, not a
dollar amount.

Common mistake: confusing the Kelly stake with a unit count. If your
bankroll is $1,000 and Kelly says 4%, you bet $40 — not 4 units.

Profitability check: ROI per bet > 0 is necessary but not sufficient.
You also need:
  - Enough bets (n ≥ 30) for the estimate to be reliable
  - Positive ROI at the edge threshold you're actually using
  - The synthetic backtest ROI to be consistent with real backtest ROI
    (large gaps indicate overfitting or market disagreement model is wrong)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.logging_config import configure_logging
from src.data_sources.statcast import fetch_statcast_events

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
BACKTEST_DIR    = PROJECT_ROOT / "data" / "backtest"
RECAP_DIR       = PROJECT_ROOT / "data" / "recaps"
RECAP_DIR.mkdir(parents=True, exist_ok=True)

KELLY_FRACTION   = 0.25   # fractional Kelly multiplier (matches training)
DEFAULT_MIN_EDGE = 0.03   # minimum edge to count as a bet


# ---------------------------------------------------------------------------
# CANONICAL Kelly implementation — single source of truth
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """Convert American moneyline to decimal odds (includes stake return)."""
    if american < 0:
        return 1.0 + (100.0 / abs(american))
    return 1.0 + (american / 100.0)


def kelly_stake(
    model_prob: float,
    american_odds: float,
    kelly_mult: float = KELLY_FRACTION,
) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    model_prob   : model's probability the bet wins (0–1)
    american_odds: market over price in American format (e.g. -130, +110)
    kelly_mult   : fraction of full Kelly to use (default 0.25)

    Returns
    -------
    Fraction of bankroll to bet (0.0 if no edge).

    Example
    -------
        kelly_stake(0.18, -130, 0.25)
        b = 100/130 = 0.769
        f_full = (0.769 * 0.18 - 0.82) / 0.769 = -0.887  → 0.0 (no edge)

        kelly_stake(0.18, +110, 0.25)
        b = 110/100 = 1.10
        f_full = (1.10 * 0.18 - 0.82) / 1.10 = 0.034
        f_frac = 0.034 * 0.25 = 0.0085  → bet 0.85% of bankroll
    """
    b = american_to_decimal(american_odds) - 1.0  # net profit per $1 risked
    p = float(model_prob)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, round(f_full * kelly_mult, 6))


def bet_pnl(american_odds: float, won: bool) -> float:
    """P&L of a $1 flat bet. Win returns net profit; loss returns -1."""
    return (american_to_decimal(american_odds) - 1.0) if won else -1.0


def kelly_pnl(model_prob: float, american_odds: float, won: bool,
              kelly_mult: float = KELLY_FRACTION) -> float:
    """P&L of a Kelly-sized bet (as fraction of bankroll)."""
    stake = kelly_stake(model_prob, american_odds, kelly_mult)
    return bet_pnl(american_odds, won) * stake


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _week_bounds(reference_date: Optional[str] = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (Monday, Sunday) of the week containing reference_date."""
    if reference_date:
        ref = pd.Timestamp(reference_date)
    else:
        ref = pd.Timestamp.now()
    monday = ref - pd.Timedelta(days=ref.dayofweek)
    sunday = monday + pd.Timedelta(days=6)
    return monday.normalize(), sunday.normalize()


def load_predictions_for_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load all prediction CSVs in the date range [start, end]."""
    files = sorted(PREDICTIONS_DIR.glob("predictions_????-??-??.csv"))
    dfs = []
    for f in files:
        date_str = f.stem.replace("predictions_", "")
        try:
            dt = pd.Timestamp(date_str)
        except Exception:
            continue
        if start <= dt <= end:
            try:
                df = pd.read_csv(f)
                df["game_date"] = dt
                dfs.append(df)
            except Exception as e:
                logger.warning("Could not load %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()
    preds = pd.concat(dfs, ignore_index=True)
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    return preds


def load_outcomes_for_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch actual HR outcomes from Statcast for the date range."""
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    logger.info("Loading Statcast outcomes %s → %s ...", start_str, end_str)
    try:
        events = fetch_statcast_events(
            start_date=start_str,
            end_date=end_str,
            columns=["game_date", "batter", "events"],
            regular_season_only=False,
        ).df
    except Exception as e:
        logger.error("Failed to load Statcast outcomes: %s", e)
        return pd.DataFrame(columns=["game_date", "batter", "hr_hit"])

    events["game_date"] = pd.to_datetime(events["game_date"])
    events["is_hr"] = (events["events"] == "home_run").fillna(False).astype(int)
    outcomes = (
        events
        .groupby(["game_date", "batter"], as_index=False)
        .agg(hr_hit=("is_hr", "max"))
    )
    return outcomes


# ---------------------------------------------------------------------------
# Results assembly
# ---------------------------------------------------------------------------

def build_weekly_results(
    preds: pd.DataFrame,
    outcomes: pd.DataFrame,
    kelly_mult: float = KELLY_FRACTION,
) -> pd.DataFrame:
    """
    Join predictions to outcomes. Computes flat and Kelly P&L per row.
    Only rows with odds data (edge + market_over_price not NaN) are included
    in betting analysis; all rows are included for calibration.
    """
    if preds.empty or outcomes.empty:
        return pd.DataFrame()

    preds = preds.copy()
    outcomes = outcomes.copy()
    preds["batter"]    = pd.to_numeric(preds["batter"],    errors="coerce").astype("Int64")
    outcomes["batter"] = pd.to_numeric(outcomes["batter"], errors="coerce").astype("Int64")

    merged = preds.merge(
        outcomes.rename(columns={"hr_hit": "actual_hr"}),
        on=["game_date", "batter"],
        how="left",
    ).dropna(subset=["actual_hr"]).copy()
    merged["actual_hr"] = merged["actual_hr"].astype(int)

    # Ensure odds columns exist — predictions run without ODDS_API_KEY won't have them.
    for col in ("edge", "market_over_price", "market_fair_prob", "kelly_stake"):
        if col not in merged.columns:
            merged[col] = np.nan

    # has_odds: True only when both edge and market_over_price are present and finite.
    merged["has_odds"] = (
        merged["edge"].notna() & merged["market_over_price"].notna()
    )

    # Compute P&L vectorised — avoids row-wise apply() crashing on missing columns.
    dec_odds = merged["market_over_price"].apply(
        lambda x: american_to_decimal(float(x)) if pd.notna(x) else np.nan
    )
    merged["pnl_1u"] = np.where(
        merged["has_odds"],
        np.where(merged["actual_hr"] == 1, dec_odds - 1, -1.0),
        np.nan,
    )

    merged["kelly_stake"] = merged.apply(
        lambda r: kelly_stake(float(r["hr_prob"]), float(r["market_over_price"]), kelly_mult)
        if r["has_odds"] else np.nan,
        axis=1,
    )
    merged["kelly_pnl"] = merged["pnl_1u"] * merged["kelly_stake"]

    return merged


# ---------------------------------------------------------------------------
# Edge bucket analysis
# ---------------------------------------------------------------------------

def edge_bucket_table(results: pd.DataFrame) -> pd.DataFrame:
    """P&L breakdown by edge bucket (rows with odds data only)."""
    bettable = results[results["has_odds"] & results["edge"].notna()].copy()
    if bettable.empty:
        return pd.DataFrame()

    thresholds = [0.00, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    rows = []
    for t in thresholds:
        sub = bettable[bettable["edge"] >= t]
        if len(sub) == 0:
            continue
        rows.append({
            "min_edge":  round(t, 2),
            "n_bets":    len(sub),
            "win_rate":  round(sub["actual_hr"].mean(), 4),
            "units_1u":  round(sub["pnl_1u"].sum(), 3),
            "roi_1u":    round(sub["pnl_1u"].mean(), 4),
            "roi_kelly": round(sub["kelly_pnl"].mean(), 4) if sub["kelly_pnl"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(results: pd.DataFrame, n_bins: int = 6) -> pd.DataFrame:
    """
    Model calibration: predicted vs actual HR rate across probability buckets.

    Uses quantile-based bins (pd.qcut) so each bucket has roughly equal sample
    size — unlike equal-width bins which pile everything into the low-prob bucket
    and leave tiny unreliable samples at the top.

    Adds a 95% CI on the actual HR rate and a 'reliable' flag so the MAE
    headline only counts buckets with enough data to be meaningful.
    """
    sub = results.dropna(subset=["hr_prob", "actual_hr"])
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    sub["prob_bucket"] = pd.qcut(sub["hr_prob"], q=n_bins, duplicates="drop")
    cal = (
        sub.groupby("prob_bucket", observed=True)
        .agg(
            n=("actual_hr", "size"),
            predicted=("hr_prob", "mean"),
            actual=("actual_hr", "mean"),
        )
        .reset_index()
    )
    cal["error_pp"] = ((cal["predicted"] - cal["actual"]) * 100).round(1)
    cal["ci_95_pp"] = (
        1.96 * np.sqrt(cal["actual"] * (1 - cal["actual"]) / cal["n"].clip(lower=1)) * 100
    ).round(1)
    cal["reliable"] = cal["ci_95_pp"] < 5.0
    return cal


# ---------------------------------------------------------------------------
# Profitability validator
# ---------------------------------------------------------------------------

def check_kelly_correctness(results: pd.DataFrame) -> list[str]:
    """
    Audit Kelly stakes for correctness. Returns a list of warning strings
    (empty list = all checks passed).
    """
    warnings = []
    bettable = results[results["has_odds"] & results["edge"].notna()].copy()
    if bettable.empty:
        return ["No bettable rows found — cannot validate Kelly."]

    # Check 1: Kelly stake should be 0 when edge <= 0
    neg_edge = bettable[bettable["edge"] <= 0]
    bad_stakes = neg_edge[neg_edge["kelly_stake"] > 0]
    if len(bad_stakes) > 0:
        warnings.append(
            f"⚠️  KELLY BUG: {len(bad_stakes)} rows have edge ≤ 0 but Kelly stake > 0. "
            "Kelly should never recommend a bet with no edge."
        )

    # Check 2: Kelly stake should be a fraction (0–1), not a unit count
    huge_stakes = bettable[bettable["kelly_stake"] > 0.5]
    if len(huge_stakes) > 0:
        warnings.append(
            f"⚠️  KELLY WARNING: {len(huge_stakes)} rows have Kelly stake > 50% of bankroll. "
            "This suggests the kelly_mult multiplier may not be applied, or odds are extreme."
        )

    # Check 3: Average Kelly stake should be sane for HR props (~1-8%)
    pos_edge = bettable[bettable["edge"] > DEFAULT_MIN_EDGE]
    if len(pos_edge) > 0:
        avg_stake = pos_edge["kelly_stake"].mean()
        if avg_stake > 0.15:
            warnings.append(
                f"⚠️  KELLY WARNING: Average Kelly stake at edge ≥ {DEFAULT_MIN_EDGE:.0%} is "
                f"{avg_stake:.1%}. Expected 1–8% for HR props. Check kelly_mult ({KELLY_FRACTION}×)."
            )
        elif avg_stake < 0.001:
            warnings.append(
                f"⚠️  KELLY WARNING: Average Kelly stake at edge ≥ {DEFAULT_MIN_EDGE:.0%} is "
                f"{avg_stake:.2%}. Very small — odds may be extremely unfavorable."
            )

    # Check 4: Verify a specific row manually
    sample = bettable[bettable["edge"] > 0.03].head(1)
    if not sample.empty:
        row = sample.iloc[0]
        expected = kelly_stake(float(row["hr_prob"]), float(row["market_over_price"]))
        actual   = float(row["kelly_stake"])
        if abs(expected - actual) > 1e-4:
            warnings.append(
                f"⚠️  KELLY MISMATCH: Sample row has kelly_stake={actual:.4f} "
                f"but canonical formula gives {expected:.4f}. "
                f"(model_prob={row['hr_prob']:.3f}, odds={row['market_over_price']:+.0f})"
            )

    return warnings


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct(x: float, decimals: int = 1) -> str:
    try:
        return f"{float(x)*100:.{decimals}f}%"
    except Exception:
        return "—"

def _pp(x: float) -> str:
    try:
        v = float(x) * 100
        return f"{v:+.1f}pp"
    except Exception:
        return "—"

def _u(x: float) -> str:
    try:
        return f"{float(x):+.2f}u"
    except Exception:
        return "—"


def print_weekly_report(
    results: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    min_edge: float = DEFAULT_MIN_EDGE,
) -> None:
    """Print a formatted weekly recap to stdout."""
    n_total    = len(results)
    n_bettable = results["has_odds"].sum() if "has_odds" in results.columns else 0
    hr_rate    = results["actual_hr"].mean() if n_total > 0 else 0.0

    print(f"\n{'='*72}")
    print(f"  ⚾  MLB HR MODEL — WEEKLY RECAP")
    print(f"  Week of {week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}")
    print(f"{'='*72}")
    print(f"  Batter-games scored : {n_total:,}")
    print(f"  With odds data      : {n_bettable:,}")
    print(f"  Actual HR rate      : {_pct(hr_rate)}")

    # --- Kelly audit (only meaningful when odds data exists) ---
    has_any_odds = "has_odds" in results.columns and bool(results["has_odds"].any())
    if has_any_odds:
        kelly_warnings = check_kelly_correctness(results)
        if kelly_warnings:
            print(f"\n  ── Kelly Sizing Audit ────────────────────────────────────────────")
            for w in kelly_warnings:
                print(f"  {w}")
        else:
            print(f"\n  ✓ Kelly sizing checks passed (formula correct, stakes in range)")
    else:
        print(f"\n  ℹ️  No odds data — set ODDS_API_KEY and rerun predict.py for betting analysis.")

    # --- Recommended threshold performance ---
    if has_any_odds:
        bettable = results[
            results["has_odds"] &
            results["edge"].notna() &
            (results["edge"] >= min_edge)
        ].copy()
    else:
        bettable = pd.DataFrame()

    print(f"\n  ── Bets at Edge ≥ {_pp(min_edge)} ─────────────────────────────────────")
    if len(bettable) == 0:
        print(f"  No bets at edge ≥ {_pp(min_edge)} — either no ODDS_API_KEY or no positive-edge picks found.")
    else:
        n_bets    = len(bettable)
        win_rate  = bettable["actual_hr"].mean()
        flat_pnl  = bettable["pnl_1u"].sum()
        flat_roi  = bettable["pnl_1u"].mean()
        kelly_roi = bettable["kelly_pnl"].mean() if bettable["kelly_pnl"].notna().any() else np.nan
        avg_edge  = bettable["edge"].mean()
        avg_kelly = bettable["kelly_stake"].mean() if bettable["kelly_stake"].notna().any() else np.nan

        print(f"  Bets placed     : {n_bets}")
        print(f"  Win rate        : {_pct(win_rate)}")
        print(f"  Avg edge        : {_pp(avg_edge)}")
        print(f"  Avg Kelly stake : {_pct(avg_kelly, 2)} of bankroll per bet")
        print(f"  Flat P&L        : {_u(flat_pnl)}   (ROI: {_pct(flat_roi)})")
        print(f"  Kelly ROI/bet   : {_pct(kelly_roi)}")

        # Day-by-day breakdown
        print(f"\n  ── Day-by-Day ────────────────────────────────────────────────────")
        print(f"  {'Date':<12}  {'Bets':>5}  {'Won':>4}  {'Flat P&L':>9}  {'ROI':>7}")
        print(f"  {'-'*12}  {'-'*5}  {'-'*4}  {'-'*9}  {'-'*7}")
        daily = bettable.groupby(bettable["game_date"].dt.date).agg(
            n=("actual_hr", "count"),
            won=("actual_hr", "sum"),
            flat_pnl=("pnl_1u", "sum"),
            flat_roi=("pnl_1u", "mean"),
        )
        for date, row in daily.iterrows():
            print(
                f"  {str(date):<12}  {int(row['n']):>5}  {int(row['won']):>4}  "
                f"{_u(row['flat_pnl']):>9}  {_pct(row['flat_roi']):>7}"
            )

        # Top picks of the week
        hits = bettable[bettable["actual_hr"] == 1].sort_values("edge", ascending=False)
        misses = bettable[bettable["actual_hr"] == 0].sort_values("edge", ascending=False)
        if len(hits) > 0:
            print(f"\n  ── Best Hits This Week ───────────────────────────────────────────")
            print(f"  {'Date':<12}  {'Batter':<24}  {'Prob':>6}  {'Edge':>7}  {'Odds':>6}")
            for _, r in hits.head(5).iterrows():
                name = str(r.get("batter_name", r["batter"]))[:24]
                odds = int(r["market_over_price"])
                odds_str = f"+{odds}" if odds > 0 else str(odds)
                print(
                    f"  {str(r['game_date'].date()):<12}  {name:<24}  "
                    f"{_pct(r['hr_prob']):>6}  {_pp(r['edge']):>7}  {odds_str:>6}"
                )

        if len(misses) > 0 and len(misses) <= n_bets:
            print(f"\n  ── Notable Misses ────────────────────────────────────────────────")
            for _, r in misses.head(3).iterrows():
                name = str(r.get("batter_name", r["batter"]))[:24]
                print(
                    f"  {str(r['game_date'].date())}  {name}  "
                    f"prob={_pct(r['hr_prob'])}  edge={_pp(r['edge'])}  ✗"
                )

    # --- Edge bucket sweep ---
    sweep = edge_bucket_table(results)
    if not sweep.empty:
        print(f"\n  ── Edge Threshold Sweep ──────────────────────────────────────────")
        print(f"  {'Edge':>6}  {'Bets':>5}  {'Win%':>6}  {'Units':>7}  {'ROI/bet':>8}")
        print(f"  {'------':>6}  {'-----':>5}  {'------':>6}  {'-------':>7}  {'--------':>8}")
        for _, row in sweep.iterrows():
            marker = " ← recommended" if abs(row["min_edge"] - min_edge) < 0.005 else ""
            print(
                f"  {_pp(row['min_edge']):>6}  {int(row['n_bets']):>5}  "
                f"{_pct(row['win_rate']):>6}  {_u(row['units_1u']):>7}  "
                f"{_pct(row['roi_1u']):>8}{marker}"
            )

    # --- Calibration ---
    cal = calibration_table(results)
    if not cal.empty:
        n_total_cal = cal["n"].sum()
        print(f"\n  ── Model Calibration  ({n_total_cal:,} batter-games, equal-frequency bins) ──────")
        print(f"  {'Bucket':<22}  {'N':>5}  {'Predicted':>10}  {'Actual':>8}  {'±95%CI':>7}  {'Error':>8}")
        print(f"  {'-'*22}  {'-----':>5}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}")
        for _, row in cal.iterrows():
            flag = "  ⚠ low-n" if not row["reliable"] else ""
            print(
                f"  {str(row['prob_bucket']):<22}  {int(row['n']):>5}  "
                f"{_pct(row['predicted']):>10}  {_pct(row['actual']):>8}  "
                f"{row['ci_95_pp']:>+6.1f}pp  {row['error_pp']:>+7.1f}pp{flag}"
            )
        reliable = cal[cal["reliable"]]
        if len(reliable) > 0:
            mae_r = reliable["error_pp"].abs().mean()
            mae_a = cal["error_pp"].abs().mean()
            print(f"\n  MAE (reliable buckets, n≥~50): {mae_r:.1f}pp")
            if len(reliable) < len(cal):
                print(f"  MAE (all buckets incl. low-n): {mae_a:.1f}pp  ← inflated by small samples")
                print(f"  ℹ️  Run --all-time after a full season for stable calibration (need ~2,000+ rows).")
        else:
            print(f"\n  MAE: {cal['error_pp'].abs().mean():.1f}pp  ⚠ All buckets low-n — more data needed.")

    print(f"\n{'='*72}\n")


def format_discord_recap(
    results: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    min_edge: float = DEFAULT_MIN_EDGE,
) -> list[dict]:
    """Build Discord webhook payloads for the weekly recap."""
    bettable = pd.DataFrame()
    if "has_odds" in results.columns and "edge" in results.columns:
        bettable = results[
            results["has_odds"] &
            results["edge"].notna() &
            (results["edge"] >= min_edge)
        ].copy()

    n_bets   = len(bettable)
    hr_rate  = results["actual_hr"].mean() if len(results) > 0 else 0.0
    win_rate = bettable["actual_hr"].mean() if n_bets > 0 else 0.0
    flat_pnl = bettable["pnl_1u"].sum() if n_bets > 0 else 0.0
    flat_roi = bettable["pnl_1u"].mean() if n_bets > 0 else 0.0
    avg_edge = bettable["edge"].mean() if n_bets > 0 else 0.0

    colour_pos = 0x00B16A  # green
    colour_neg = 0xE74C3C  # red
    colour_neu = 0x95A5A6  # grey
    colour = colour_pos if flat_pnl > 0 else (colour_neg if flat_pnl < 0 else colour_neu)

    pnl_emoji = "🟢" if flat_pnl > 0 else ("🔴" if flat_pnl < 0 else "⚪")

    header_embed = {
        "title": (
            f"⚾ MLB HR Model — Weekly Recap\n"
            f"{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"
        ),
        "color": colour,
        "fields": [
            {"name": "Bets Placed",    "value": str(n_bets),              "inline": True},
            {"name": "Win Rate",       "value": _pct(win_rate),           "inline": True},
            {"name": "Actual HR Rate", "value": _pct(hr_rate),            "inline": True},
            {"name": f"{pnl_emoji} Flat P&L",
             "value": f"{_u(flat_pnl)} (ROI: {_pct(flat_roi)})",         "inline": True},
            {"name": "Avg Edge",       "value": _pp(avg_edge),            "inline": True},
            {"name": "Min Edge Used",  "value": _pp(min_edge),            "inline": True},
        ],
        "footer": {"text": f"MLB HR Model • Weekly Recap • {week_start.strftime('%Y-W%W')}"},
    }

    payloads = [{"embeds": [header_embed]}]

    # Day-by-day breakdown
    if n_bets > 0:
        daily = bettable.groupby(bettable["game_date"].dt.date).agg(
            n=("actual_hr", "count"),
            won=("actual_hr", "sum"),
            flat_pnl=("pnl_1u", "sum"),
        )
        day_lines = []
        for date, row in daily.iterrows():
            emoji = "🟢" if row["flat_pnl"] > 0 else "🔴"
            day_lines.append(
                f"{emoji} **{date}**: {int(row['won'])}/{int(row['n'])} wins, "
                f"{_u(row['flat_pnl'])}"
            )

        hits = bettable[bettable["actual_hr"] == 1].sort_values("edge", ascending=False)
        hit_lines = []
        for _, r in hits.head(5).iterrows():
            name = str(r.get("batter_name", r["batter"]))[:20]
            hit_lines.append(f"✅ {name} ({_pct(r['hr_prob'])} / {_pp(r['edge'])} edge)")

        detail_embed = {
            "title": "Day-by-Day Results",
            "color": colour,
            "fields": [
                {
                    "name": "📅 Daily Breakdown",
                    "value": "\n".join(day_lines) or "No data",
                    "inline": False,
                },
            ],
        }
        if hit_lines:
            detail_embed["fields"].append({
                "name": "🏆 Top HR Hits",
                "value": "\n".join(hit_lines),
                "inline": False,
            })
        payloads.append({"embeds": [detail_embed]})

    return payloads


def post_recap_to_discord(
    results: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    min_edge: float = DEFAULT_MIN_EDGE,
    webhook_url: Optional[str] = None,
) -> bool:
    """POST the weekly recap to Discord. Returns True on success."""
    import requests as req
    from dotenv import load_dotenv
    load_dotenv()

    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        logger.warning("No DISCORD_WEBHOOK_URL set — cannot post recap.")
        return False

    payloads = format_discord_recap(results, week_start, week_end, min_edge)
    import time
    for i, payload in enumerate(payloads):
        try:
            resp = req.post(url, json=payload, timeout=10)
            if resp.status_code == 204:
                logger.debug("Discord recap message %d/%d sent", i + 1, len(payloads))
            else:
                logger.warning("Discord returned %d for message %d", resp.status_code, i + 1)
            time.sleep(0.5)
        except Exception as e:
            logger.error("Discord post failed: %s", e)
            return False
    logger.info("Weekly recap posted to Discord (%d messages)", len(payloads))
    return True


def save_recap_csv(results: pd.DataFrame, week_start: pd.Timestamp) -> Path:
    """Save detailed results to CSV in data/recaps/."""
    out = RECAP_DIR / f"recap_{week_start.strftime('%Y-W%W')}_{week_start.strftime('%Y-%m-%d')}.csv"
    cols = [c for c in [
        "game_date", "batter_name", "pitcher_name", "hr_prob",
        "edge", "market_over_price", "market_fair_prob", "kelly_stake",
        "actual_hr", "pnl_1u", "kelly_pnl",
    ] if c in results.columns]
    results[cols].to_csv(out, index=False)
    logger.info("Recap saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Weekly recap: model performance vs actual outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--week",      default=None,
                        help="Reference date (any day in the target week) YYYY-MM-DD. "
                             "Default: current week.")
    parser.add_argument("--all-time",  action="store_true",
                        help="Summarise all available prediction dates.")
    parser.add_argument("--min-edge",  type=float, default=DEFAULT_MIN_EDGE,
                        help=f"Minimum edge threshold (default {DEFAULT_MIN_EDGE}).")
    parser.add_argument("--discord",   action="store_true",
                        help="Post recap to Discord webhook.")
    parser.add_argument("--save",      action="store_true",
                        help="Save detailed results CSV to data/recaps/.")
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.all_time:
        # Find date range from all prediction files
        files = sorted(PREDICTIONS_DIR.glob("predictions_????-??-??.csv"))
        if not files:
            print("No prediction files found.")
            return
        dates = [pd.Timestamp(f.stem.replace("predictions_", "")) for f in files]
        week_start, week_end = min(dates), max(dates)
        label = "All-Time"
    else:
        week_start, week_end = _week_bounds(args.week)
        label = f"Week of {week_start.strftime('%b %d')} – {week_end.strftime('%b %d')}"

    print(f"\nLoading predictions for: {label} ...")
    preds = load_predictions_for_dates(week_start, week_end)
    if preds.empty:
        print(f"No predictions found for {label}.")
        print(f"Check that data/predictions/ contains files in this date range.")
        return

    print(f"Loading Statcast outcomes ...")
    outcomes = load_outcomes_for_dates(week_start, week_end)
    if outcomes.empty:
        print("No Statcast outcomes found — games may not have been played yet.")
        return

    results = build_weekly_results(preds, outcomes, kelly_mult=KELLY_FRACTION)
    if results.empty:
        print("Could not join predictions to outcomes.")
        return

    print_weekly_report(results, week_start, week_end, min_edge=args.min_edge)

    if args.save:
        path = save_recap_csv(results, week_start)
        print(f"  Results saved: {path.relative_to(PROJECT_ROOT)}")

    if args.discord:
        sent = post_recap_to_discord(results, week_start, week_end, min_edge=args.min_edge)
        print(f"  Discord: {'posted ✓' if sent else 'failed ✗'}")


if __name__ == "__main__":
    main()

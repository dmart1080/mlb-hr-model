"""
Closing Line Value (CLV) tracker.

CLV = implied_prob(closing_line) - implied_prob(bet_line)

Positive CLV means the market moved *towards* your pick after you bet —
the strongest leading indicator of long-term profit, and measurable with
10x less sample than raw ROI. Over ~200 bets, median CLV predicts ROI
reliably where win rate is still noise.

Run pattern:
  - Final pass writes predictions_{date}_final.csv with market_over_price
    (bet-time line, best across books).
  - ~10 min before first pitch of each wave, run this tracker to re-fetch
    odds and append a snapshot. The closing line is the last snapshot
    captured before game status flips to Live.

Usage:
    python -m src.analysis.clv_tracker                    # today, force refresh
    python -m src.analysis.clv_tracker --date 2026-04-14
    python -m src.analysis.clv_tracker --summarize        # season CLV report

Output:
    data/predictions/clv_YYYY-MM-DD.csv   — append-only snapshot log
      batter_name, bet_price, bet_fair_prob, snap_price, snap_fair_prob,
      clv_pp, snap_time, bookmaker
"""
from __future__ import annotations

import argparse
import glob
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data_sources.odds import (
    american_to_implied_prob,
    build_odds_lookup,
    fetch_hr_props,
    match_odds,
    remove_vig,
)
from src.logging_config import configure_logging

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"


def _load_final_picks(date_str: str) -> pd.DataFrame:
    """Load today's final-pass picks — only rows we actually bet on."""
    final_path = PREDICTIONS_DIR / f"predictions_{date_str}_final.csv"
    canon_path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    path = final_path if final_path.exists() else canon_path

    if not path.exists():
        logger.warning("No prediction CSV found for %s", date_str)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "edge" not in df.columns:
        logger.warning("No odds columns in %s — CLV not applicable", path.name)
        return pd.DataFrame()

    # Focus on picks with positive edge (the ones we'd actually bet)
    bet_mask = df["edge"].notna() & (df["edge"] > 0)
    return df[bet_mask].copy()


def snapshot_clv(date_str: str) -> pd.DataFrame:
    """
    Re-fetch current market and compute CLV vs bet-time price for each pick.
    Appends to data/predictions/clv_{date}.csv.
    """
    picks = _load_final_picks(date_str)
    if picks.empty:
        logger.info("No positive-edge picks to track for %s", date_str)
        return pd.DataFrame()

    logger.info("Fetching current odds for %s ...", date_str)
    props = fetch_hr_props(date_str, force_refresh=True)
    if not props:
        logger.warning("No live odds available — CLV snapshot skipped")
        return pd.DataFrame()

    lookup = build_odds_lookup(props)
    now_iso = datetime.now().isoformat(timespec="seconds")
    rows = []

    for _, pick in picks.iterrows():
        name = pick.get("batter_name")
        if not isinstance(name, str):
            continue

        bet_price = pick.get("market_over_price")
        if pd.isna(bet_price):
            continue

        prop = match_odds(str(name), lookup)
        if prop is None:
            logger.debug("No current line for %s — skipping", name)
            continue

        snap_price = prop["over_price"]
        snap_p_over  = american_to_implied_prob(snap_price)
        snap_p_under = american_to_implied_prob(prop["under_price"])
        snap_fair, _ = remove_vig(snap_p_over, snap_p_under)

        bet_fair = float(pick.get("market_fair_prob", float("nan")))
        clv_pp = (snap_fair - bet_fair) * 100  # percentage points

        rows.append({
            "batter_name":    name,
            "bet_price":      int(bet_price),
            "bet_fair_prob":  round(bet_fair, 4),
            "snap_price":     int(snap_price),
            "snap_fair_prob": round(snap_fair, 4),
            "clv_pp":         round(clv_pp, 3),
            "snap_time":      now_iso,
            "bookmaker":      prop["bookmaker"],
        })

    if not rows:
        logger.info("No matched picks to snapshot")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = PREDICTIONS_DIR / f"clv_{date_str}.csv"
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)

    avg_clv = df["clv_pp"].mean()
    pos     = (df["clv_pp"] > 0).mean() * 100
    logger.info(
        "CLV snapshot: %d picks | avg CLV %+.2fpp | %.0f%% positive",
        len(df), avg_clv, pos,
    )
    logger.info("Appended: %s", out_path.relative_to(PROJECT_ROOT))
    return df


def summarize_season(year: int = None) -> None:
    """Aggregate all CLV snapshots — returns season-to-date CLV metrics."""
    year = year or datetime.now().year
    files = sorted(glob.glob(str(PREDICTIONS_DIR / f"clv_{year}-*.csv")))
    if not files:
        logger.info("No CLV snapshot files found for %d", year)
        return

    dfs = [pd.read_csv(f) for f in files]
    all_snaps = pd.concat(dfs, ignore_index=True)

    # For each batter_name per date, keep only the latest snapshot
    # (that's the closing line)
    all_snaps["snap_time"] = pd.to_datetime(all_snaps["snap_time"])
    all_snaps["date"]      = all_snaps["snap_time"].dt.date
    closing = (
        all_snaps.sort_values("snap_time")
        .groupby(["date", "batter_name"], as_index=False)
        .tail(1)
    )

    n = len(closing)
    avg_clv = closing["clv_pp"].mean()
    med_clv = closing["clv_pp"].median()
    pos_pct = (closing["clv_pp"] > 0).mean() * 100

    logger.info("=" * 50)
    logger.info("SEASON %d CLV SUMMARY (closing lines only)", year)
    logger.info("=" * 50)
    logger.info("Picks tracked:   %d", n)
    logger.info("Average CLV:     %+.2fpp", avg_clv)
    logger.info("Median CLV:      %+.2fpp", med_clv)
    logger.info("Positive CLV:    %.1f%%", pos_pct)
    logger.info("")
    logger.info("Interpretation:")
    logger.info("  > +1pp avg CLV → strong positive-EV (long-term profitable)")
    logger.info("  ~  0pp avg CLV → break-even, likely negative after vig")
    logger.info("  < -1pp avg CLV → reduce volume / tighten thresholds")
    logger.info("=" * 50)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Track Closing Line Value for bets")
    parser.add_argument("--date", default=None,
                        help="Snapshot a specific date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--summarize", action="store_true",
                        help="Print season-to-date CLV summary instead of snapshotting.")
    parser.add_argument("--year", type=int, default=None,
                        help="Year for --summarize (default: current year)")
    args = parser.parse_args()

    if args.summarize:
        summarize_season(args.year)
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        snapshot_clv(date_str)

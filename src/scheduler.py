"""
MLB HR Model — Daily Scheduler
===============================
Runs predict.py at two key points each day:

  Pass 1 — MORNING  (~10:00 AM ET): Probable starters are posted, early
            prop lines may be available. Good for spotting early value
            before the market tightens.

  Pass 2 — FINAL    (~75 min before earliest first pitch): Official lineups
            are confirmed, lines are final. This is the actionable output.

The scheduler detects which pass to run based on current time, refreshes
the appropriate caches, and logs timing metadata to data/predictions/run_log.csv.

Usage
-----
    # Run the correct pass for the current time automatically:
    python -m src.scheduler

    # Force a specific pass:
    python -m src.scheduler --pass morning
    python -m src.scheduler --pass final

    # Dry-run: show what would run without executing:
    python -m src.scheduler --dry-run

    # List today's run history:
    python -m src.scheduler --status

Windows Task Scheduler setup
-----------------------------
Create two tasks in Task Scheduler pointing at this script:

  Task 1 — Morning
    Program:  C:\\path\\to\\.venv\\Scripts\\python.exe
    Arguments: -m src.scheduler --pass morning
    Start in:  C:\\path\\to\\mlb-hr-model
    Trigger:   Daily at 10:00 AM

  Task 2 — Final
    Program:  C:\\path\\to\\.venv\\Scripts\\python.exe
    Arguments: -m src.scheduler --pass final
    Start in:  C:\\path\\to\\mlb-hr-model
    Trigger:   Daily at 5:00 PM

  (Adjust the final trigger based on your local timezone relative to ET.
   Most day games start at 1pm ET; most night games at 7pm ET.
   5pm ET gives ~2hr buffer before the typical first pitch.)

Data availability by pass
--------------------------
  Morning pass:
    - Probable starters:  Usually posted by 9–10am ET (not guaranteed)
    - Batting orders:     NOT confirmed (lineups ~90 min before first pitch)
    - Prop lines:         Sometimes open, sometimes not yet listed
    - Best for:           Early-line value, overnight line moves

  Final pass:
    - Probable starters:  Confirmed (or scratched — schedule refresh catches this)
    - Batting orders:     Confirmed for most games
    - Prop lines:         Final pre-game lines, tightest spreads
    - Best for:           Actual betting decisions

Output
------
  data/predictions/predictions_YYYY-MM-DD_morning.csv   ← morning snapshot
  data/predictions/predictions_YYYY-MM-DD_final.csv     ← final snapshot (use this one)
  data/predictions/predictions_YYYY-MM-DD.csv           ← symlink/copy of final pass
  data/predictions/run_log.csv                          ← timing + data-quality log
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG = PREDICTIONS_DIR / "run_log.csv"

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Pass definitions
# ---------------------------------------------------------------------------

# Morning pass: run at 10am ET, cache TTL = none (always fresh at this time)
# Final pass:   run at 5pm ET  (or manually), force-refresh everything
PASS_DEFINITIONS = {
    "morning": {
        "label":          "MORNING",
        "run_hour_et":    10,
        "force_refresh":  False,   # schedule cache may already be warm from yesterday
        "description":    "Probable starters + early lines. Lineups not yet confirmed.",
    },
    "final": {
        "label":          "FINAL",
        "run_hour_et":    17,
        "force_refresh":  True,    # always bypass caches for final pass
        "description":    "Confirmed lineups + final pre-game lines. Use this for bets.",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_et() -> datetime:
    return datetime.now(tz=ET)


def detect_pass() -> str:
    """Detect which pass to run based on current ET time."""
    hour = now_et().hour
    # Before 1pm → morning pass; 1pm+ → final pass
    return "morning" if hour < 13 else "final"


def today_str() -> str:
    return now_et().strftime("%Y-%m-%d")


def log_run(
    pass_name: str,
    date_str: str,
    status: str,
    starters_confirmed: bool | None,
    lineups_confirmed: bool | None,
    odds_available: bool | None,
    n_predictions: int | None,
    n_with_edge: int | None,
    elapsed_secs: float | None,
    notes: str = "",
) -> None:
    """Append a row to the run log CSV."""
    write_header = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_time_utc", "game_date", "pass", "status",
                "starters_confirmed", "lineups_confirmed", "odds_available",
                "n_predictions", "n_with_edge", "elapsed_secs", "notes",
            ])
        writer.writerow([
            datetime.now(tz=timezone.utc).isoformat(),
            date_str,
            pass_name,
            status,
            starters_confirmed,
            lineups_confirmed,
            odds_available,
            n_predictions,
            n_with_edge,
            round(elapsed_secs, 1) if elapsed_secs is not None else None,
            notes,
        ])


def check_data_availability(date_str: str) -> dict:
    """
    Inspect the cached data for today to determine what's confirmed.
    Returns a dict of availability flags without making any network calls.
    """
    import json

    schedule_cache = PROJECT_ROOT / "data" / "cache" / "schedule"
    odds_cache     = PROJECT_ROOT / "data" / "cache" / "odds"

    # --- Probable starters ---
    date_cache = schedule_cache / f"schedule_{date_str}.json"
    starters_confirmed = False
    if date_cache.exists():
        with open(date_cache) as f:
            games = json.load(f)
        if isinstance(games, list) and games:
            has_home = any(g.get("home_starter_id") is not None for g in games)
            has_away = any(g.get("away_starter_id") is not None for g in games)
            starters_confirmed = has_home and has_away

    # --- Lineups (batting orders from live feed) ---
    lineups_confirmed = False
    game_caches = list(schedule_cache.glob("game_*.json"))
    if game_caches:
        confirmed_count = 0
        for gc in game_caches[:5]:  # sample first 5 games
            try:
                with open(gc) as f:
                    roster = json.load(f)
                if roster.get("batting_orders"):
                    confirmed_count += 1
            except Exception:
                pass
        lineups_confirmed = confirmed_count >= 3

    # --- Odds ---
    odds_file = odds_cache / f"odds_{date_str}.json"
    odds_available = False
    if odds_file.exists():
        try:
            with open(odds_file) as f:
                odds_data = json.load(f)
            odds_available = len(odds_data.get("props", [])) > 0
        except Exception:
            pass

    return {
        "starters_confirmed": starters_confirmed,
        "lineups_confirmed":  lineups_confirmed,
        "odds_available":     odds_available,
    }


def inspect_predictions_csv(path: Path) -> dict:
    """Read a predictions CSV and return summary stats."""
    import pandas as pd
    if not path.exists():
        return {"n_predictions": 0, "n_with_edge": 0}
    df = pd.read_csv(path)
    n_edge = int(df["edge"].notna().sum()) if "edge" in df.columns else 0
    return {"n_predictions": len(df), "n_with_edge": n_edge}


def print_status() -> None:
    """Print today's run history from the log."""
    date_str = today_str()
    if not RUN_LOG.exists():
        print(f"No run log found. Nothing has run yet.")
        return

    import csv as _csv
    rows = []
    with open(RUN_LOG) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("game_date") == date_str:
                rows.append(row)

    if not rows:
        print(f"No runs logged for {date_str}.")
        return

    print(f"\n{'='*72}")
    print(f"  Run history for {date_str}")
    print(f"{'='*72}")
    for row in rows:
        t = row.get("run_time_utc", "")[:19].replace("T", " ")
        p = row.get("pass", "").upper()
        status = row.get("status", "")
        n   = row.get("n_predictions", "?")
        ne  = row.get("n_with_edge", "?")
        sec = row.get("elapsed_secs", "?")
        starters = "✓" if row.get("starters_confirmed") == "True" else "✗"
        lineups  = "✓" if row.get("lineups_confirmed")  == "True" else "✗"
        odds     = "✓" if row.get("odds_available")     == "True" else "✗"
        notes = row.get("notes", "")
        print(f"\n  [{p}]  {t} UTC  —  {status}")
        print(f"    Starters: {starters}   Lineups: {lineups}   Odds: {odds}")
        print(f"    Predictions: {n}   With edge: {ne}   Elapsed: {sec}s")
        if notes:
            print(f"    Notes: {notes}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pass(pass_name: str, *, dry_run: bool = False) -> int:
    """
    Execute one prediction pass. Returns exit code.
    """
    defn        = PASS_DEFINITIONS[pass_name]
    date_str    = today_str()
    start_time  = datetime.now(tz=timezone.utc)

    avail = check_data_availability(date_str)

    print(f"\n{'='*72}")
    print(f"  MLB HR MODEL — {defn['label']} PASS  ({date_str})")
    print(f"{'='*72}")
    print(f"  {defn['description']}")
    print(f"\n  Data availability (pre-run):")
    print(f"    Probable starters : {'confirmed' if avail['starters_confirmed'] else 'not yet posted'}")
    print(f"    Batting lineups   : {'confirmed' if avail['lineups_confirmed'] else 'not yet confirmed'}")
    print(f"    Prop lines        : {'available' if avail['odds_available'] else 'not yet available / no API key'}")
    print(f"\n  Force-refresh caches: {defn['force_refresh']}")

    if dry_run:
        print(f"\n  [DRY RUN] Would run: python -m src.model.predict --run-type {pass_name}"
              + (" --force-refresh" if defn["force_refresh"] else ""))
        print(f"{'='*72}\n")
        return 0

    # Build the predict.py command
    cmd = [sys.executable, "-m", "src.model.predict", "--run-type", pass_name]
    if defn["force_refresh"]:
        cmd.append("--force-refresh")

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        exit_code = result.returncode
        status = "success" if exit_code == 0 else f"failed (exit {exit_code})"
    except Exception as e:
        logger.error("predict.py crashed: %s", e)
        exit_code = 1
        status = f"error: {e}"

    elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()

    # Inspect the output CSV for summary stats
    out_csv = PREDICTIONS_DIR / f"predictions_{date_str}_{pass_name}.csv"
    if not out_csv.exists():
        out_csv = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    stats = inspect_predictions_csv(out_csv)

    # Log the run
    log_run(
        pass_name=pass_name,
        date_str=date_str,
        status=status,
        elapsed_secs=elapsed,
        notes=f"force_refresh={defn['force_refresh']}",
        **avail,
        **stats,
    )

    print(f"\n  {'='*70}")
    print(f"  {defn['label']} pass complete — {status}")
    print(f"  Predictions: {stats['n_predictions']}   With edge: {stats['n_with_edge']}   Elapsed: {elapsed:.0f}s")
    print(f"  {'='*70}\n")

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="MLB HR model daily scheduler — runs predict.py at the right time",
    )
    parser.add_argument(
        "--pass", dest="pass_name", choices=["morning", "final"],
        help="Which pass to run. Auto-detected from current time if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print today's run history and exit.",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    pass_name = args.pass_name or detect_pass()
    logger.info("Running %s pass (date: %s)", pass_name.upper(), today_str())

    sys.exit(run_pass(pass_name, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

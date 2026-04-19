"""
MLB HR Model — Daily Scheduler
===============================
Runs predict.py at the right time for EVERY game wave each day.

Problem with a fixed 5pm final pass:
  The original scheduler hardcoded 5pm ET as the final pass time.
  That works for night games (7pm first pitch) but completely misses
  afternoon games (1pm, 4pm starts) — either the pass fires after the
  game has started, or lineups aren't confirmed yet when it fires early.

Solution — game-aware pass timing:
  This scheduler fetches actual first-pitch times from the MLB Stats API
  and groups games into waves by start time. Each wave gets its own final
  pass fired ~90 minutes before that wave's earliest first pitch.

  Typical daily schedule:
    Noon games      → first pitch ~12:05 ET  → final pass ~10:30 ET
    Afternoon games → first pitch ~4:05 ET   → final pass ~2:30 ET
    Evening games   → first pitch ~7:05 ET   → final pass ~5:30 ET
    Late games      → first pitch ~10:05 ET  → final pass ~8:30 ET

  The morning pass still fires once at 10am ET and covers all games
  (probable starters, early lines, no confirmed lineups yet).

Pass structure
--------------
  MORNING  — 10:00 AM ET daily
    Probable starters posted; lineups NOT yet confirmed.
    Good for spotting early line value.

  FINAL_WAVE_{N} — 90 min before each game wave's earliest first pitch
    Confirmed lineups, final pre-game lines. USE THIS FOR BETS.
    One Discord post per wave so you get picks as each group locks.

Windows Task Scheduler setup
-----------------------------
  Task 1 — Morning (fixed time, runs daily)
    Program:   C:\\path\\to\\.venv\\Scripts\\python.exe
    Arguments: -m src.scheduler --pass morning
    Start in:  C:\\path\\to\\mlb-hr-model
    Trigger:   Daily at 10:00 AM

  Task 2 — Game-aware final passes (runs every 30 min, self-manages)
    Program:   C:\\path\\to\\.venv\\Scripts\\python.exe
    Arguments: -m src.scheduler --auto-final
    Start in:  C:\\path\\to\\mlb-hr-model
    Trigger:   Daily, repeat every 30 minutes from 10:00 AM to 11:00 PM

  The --auto-final flag checks the current time against today's game
  schedule and fires a final pass for any wave whose window just opened.
  Running it every 30 min means you never miss a wave by more than 30 min.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_CACHE  = PROJECT_ROOT / "data" / "cache" / "schedule"
SCHEDULE_CACHE.mkdir(parents=True, exist_ok=True)

RUN_LOG = PREDICTIONS_DIR / "run_log.csv"
ET      = ZoneInfo("America/New_York")

# How far before first pitch to fire the final pass for a wave
FINAL_PASS_LEAD_MINUTES = 90

# CLV snapshot: fire this many minutes before each wave's first pitch.
# The closing line = the last snapshot captured before games go Live.
# Runs on its own Task Scheduler cadence (every 15 min is fine).
CLV_SNAPSHOT_LEAD_MINUTES = 15

# Games within this many minutes of each other are grouped into the same wave
WAVE_GAP_MINUTES = 60

# Morning pass fires at this ET hour regardless of game schedule
MORNING_PASS_HOUR_ET = 10


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_et() -> datetime:
    return datetime.now(tz=ET)


def today_str() -> str:
    return now_et().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Fetch today's game start times from MLB Stats API
# ---------------------------------------------------------------------------

def fetch_game_times(date_str: str, *, force_refresh: bool = False) -> list[datetime]:
    """
    Return a sorted list of first-pitch datetimes (ET) for all games on date_str.
    Caches to data/cache/schedule/game_times_{date_str}.json.
    Falls back to [] on any error so the scheduler degrades gracefully.
    """
    cache_file = SCHEDULE_CACHE / f"game_times_{date_str}.json"

    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file) as f:
                raw = json.load(f)
            return [datetime.fromisoformat(t) for t in raw]
        except Exception:
            pass

    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        times: list[datetime] = []
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                game_time_str = game.get("gameDate")
                if game_time_str:
                    try:
                        utc_dt = datetime.fromisoformat(
                            game_time_str.replace("Z", "+00:00")
                        )
                        et_dt = utc_dt.astimezone(ET)
                        times.append(et_dt)
                    except Exception:
                        continue

        times.sort()

        with open(cache_file, "w") as f:
            json.dump([t.isoformat() for t in times], f)

        return times

    except Exception as e:
        logger.warning("Could not fetch game times for %s: %s", date_str, e)
        return []


# ---------------------------------------------------------------------------
# Group games into waves
# ---------------------------------------------------------------------------

def build_waves(game_times: list[datetime]) -> list[dict]:
    """
    Group game_times into waves (clusters of games within WAVE_GAP_MINUTES
    of each other). Returns a list of wave dicts, each containing:
        wave_id         : int (1-based)
        earliest        : datetime  — earliest first pitch in the wave
        latest          : datetime  — latest first pitch in the wave
        game_count      : int
        final_pass_time : datetime  — when to fire the final pass (earliest - lead)
        label           : str       — human-readable label e.g. "NOON", "AFTERNOON"
    """
    if not game_times:
        return []

    waves: list[list[datetime]] = []
    current_wave = [game_times[0]]

    for t in game_times[1:]:
        gap = (t - current_wave[-1]).total_seconds() / 60
        if gap <= WAVE_GAP_MINUTES:
            current_wave.append(t)
        else:
            waves.append(current_wave)
            current_wave = [t]
    waves.append(current_wave)

    result = []
    for i, wave in enumerate(waves, start=1):
        earliest = wave[0]
        latest   = wave[-1]
        fp_time  = earliest - timedelta(minutes=FINAL_PASS_LEAD_MINUTES)
        hour     = earliest.hour

        if hour < 13:
            label = "NOON"
        elif hour < 16:
            label = "AFTERNOON"
        elif hour < 19:
            label = "EARLY_EVENING"
        elif hour < 22:
            label = "EVENING"
        else:
            label = "LATE"

        result.append({
            "wave_id":         i,
            "earliest":        earliest,
            "latest":          latest,
            "game_count":      len(wave),
            "final_pass_time": fp_time,
            "label":           label,
            "pass_name":       f"final_{label.lower()}_{i}",
        })

    return result


# ---------------------------------------------------------------------------
# Determine which wave passes are due right now
# ---------------------------------------------------------------------------

def get_due_passes(
    waves: list[dict],
    *,
    already_run: set[str],
    lookback_minutes: int = 240,
) -> list[dict]:
    """
    Return waves whose final_pass_time fell within the last `lookback_minutes`
    AND have not already been run today.

    lookback_minutes should be slightly longer than the Task Scheduler repeat
    interval (default 30 min) to handle slight timing drift.
    """
    now  = now_et()
    due  = []
    for wave in waves:
        fp          = wave["final_pass_time"]
        age_minutes = (now - fp).total_seconds() / 60
        if 0 <= age_minutes <= lookback_minutes:
            if wave["pass_name"] not in already_run:
                due.append(wave)
    return due


# ---------------------------------------------------------------------------
# Run log helpers
# ---------------------------------------------------------------------------

def load_already_run_today(date_str: str) -> set[str]:
    """Return set of pass names that have already completed successfully today."""
    if not RUN_LOG.exists():
        return set()
    already = set()
    try:
        with open(RUN_LOG) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("game_date") == date_str and row.get("status") == "success":
                    already.add(row.get("pass", ""))
    except Exception:
        pass
    return already


def log_run(
    *,
    pass_name: str,
    date_str: str,
    status: str,
    elapsed_secs: float,
    starters_confirmed: bool,
    lineups_confirmed: bool,
    odds_available: bool,
    n_predictions: int,
    n_with_edge: int,
    notes: str = "",
) -> None:
    write_header = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_time_utc", "game_date", "pass", "status",
                "elapsed_secs", "starters_confirmed", "lineups_confirmed",
                "odds_available", "n_predictions", "n_with_edge", "notes",
            ])
        writer.writerow([
            datetime.now(tz=timezone.utc).isoformat(),
            date_str,
            pass_name,
            status,
            f"{elapsed_secs:.1f}",
            starters_confirmed,
            lineups_confirmed,
            odds_available,
            n_predictions,
            n_with_edge,
            notes,
        ])


# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

def check_data_availability(date_str: str) -> dict:
    """
    Heuristically check what data is available for today's slate.
    Returns a dict with boolean flags:
        starters_confirmed : True if the roster cache has probable starters
        lineups_confirmed  : True if at least one game cache has batting orders
        odds_available     : True if the odds cache has HR props
    """
    starters_confirmed = False
    lineups_confirmed  = False
    odds_available     = False

    roster_cache_dir = PROJECT_ROOT / "data" / "cache" / "schedule"
    if roster_cache_dir.exists():
        # Check the schedule_{date}.json for actual starter IDs (not just any file)
        schedule_cache = roster_cache_dir / f"schedule_{date_str}.json"
        if schedule_cache.exists():
            try:
                with open(schedule_cache) as f:
                    games = json.load(f)
                if isinstance(games, list) and games:
                    has_home = any(g.get("home_starter_id") is not None for g in games)
                    has_away = any(g.get("away_starter_id") is not None for g in games)
                    starters_confirmed = has_home and has_away
            except Exception:
                pass

        # Check game_*.json files for confirmed batting orders
        game_cache_files = list(roster_cache_dir.glob(f"game_*{date_str}*.json"))
        if not game_cache_files:
            game_cache_files = list(roster_cache_dir.glob("game_*.json"))
        for path in game_cache_files:
            if _game_cache_has_lineups(path):
                lineups_confirmed = True
                break

    odds_cache_dir = PROJECT_ROOT / "data" / "cache" / "odds"
    if odds_cache_dir.exists():
        # Check all known naming conventions, then fall back to any file for today
        candidate_names = [
            f"odds_{date_str}.json",
            f"hr_props_{date_str}.json",
        ]
        odds_file = None
        for name in candidate_names:
            p = odds_cache_dir / name
            if p.exists():
                odds_file = p
                break
        if odds_file is None:
            today_files = list(odds_cache_dir.glob(f"*{date_str}*"))
            if today_files:
                odds_file = today_files[0]
        if odds_file is not None:
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


def _game_cache_has_lineups(path: Path) -> bool:
    try:
        with open(path) as f:
            roster = json.load(f)
        return bool(roster.get("batting_orders"))
    except Exception:
        return False


def inspect_predictions_csv(path: Path) -> dict:
    try:
        import pandas as pd
        if not path.exists():
            return {"n_predictions": 0, "n_with_edge": 0}
        df = pd.read_csv(path)
        n_edge = int(df["edge"].notna().sum()) if "edge" in df.columns else 0
        return {"n_predictions": len(df), "n_with_edge": n_edge}
    except Exception:
        return {"n_predictions": 0, "n_with_edge": 0}


# ---------------------------------------------------------------------------
# Core pass runner
# ---------------------------------------------------------------------------

def run_pass(
    pass_name: str,
    *,
    dry_run: bool = False,
    force_refresh: bool = True,
    description: str = "",
) -> int:
    """
    Execute one prediction pass. Returns exit code.
    pass_name is used as the --run-type label in predict.py AND as the
    key logged to run_log.csv.
    """
    date_str   = today_str()
    start_time = datetime.now(tz=timezone.utc)
    avail      = check_data_availability(date_str)

    print(f"\n{'='*72}")
    print(f"  MLB HR MODEL — {pass_name.upper()} PASS  ({date_str})")
    print(f"{'='*72}")
    if description:
        print(f"  {description}")
    print(f"\n  Data availability:")
    print(f"    Probable starters : {'confirmed' if avail['starters_confirmed'] else 'not yet posted'}")
    print(f"    Batting lineups   : {'confirmed' if avail['lineups_confirmed'] else 'not yet confirmed'}")
    print(f"    Prop lines        : {'available' if avail['odds_available'] else 'not yet available / no API key'}")
    print(f"  Force-refresh caches: {force_refresh}")

    # Warn when a final pass fires before books have posted HR props.
    is_final = not pass_name.startswith("morning")
    if is_final and not avail["odds_available"]:
        print(
            "\n  ⚠️  WARNING: Prop lines are not yet available for this final pass.\n"
            "     Output will be model-only — no edge, Kelly, or odds columns.\n"
            "     Once lines post, re-run to get edge-filtered picks:\n"
            "       python -m src.scheduler --pass final"
        )

    # Map wave pass names (e.g. "final_evening") back to "final" for predict.py
    # so it saves to the right canonical filename.
    predict_run_type = "morning" if pass_name == "morning" else "final"

    if dry_run:
        cmd_str = (
            f"python -m src.features.build_today_features"
            + (" --force-refresh" if force_refresh else "")
            + f"\npython -m src.model.predict --date {date_str} --run-type {predict_run_type}"
            + (" --force-refresh" if force_refresh
               else (" --no-force-refresh" if predict_run_type == "final" else ""))
        )
        print(f"\n  [DRY RUN] Would run:\n    {cmd_str}")
        print(f"{'='*72}\n")
        return 0

    # ------------------------------------------------------------------
    # Step 1: Build today's features
    # This injects ALL of today's matchup rows (full game slate) into the
    # combined train table so predict.py sees every batter, not just stale rows.
    # ------------------------------------------------------------------
    build_cmd = [sys.executable, "-m", "src.features.build_today_features"]
    if force_refresh:
        build_cmd.append("--force-refresh")

    logger.info("Building today's features: %s", " ".join(build_cmd))
    print(f"\n  Step 1/2 — Building today's features ...")

    try:
        build_result = subprocess.run(build_cmd, cwd=PROJECT_ROOT)
        if build_result.returncode != 0:
            logger.warning(
                "build_today_features exited with code %d — predict will use "
                "whatever rows are already in the train table.",
                build_result.returncode,
            )
            print(
                f"  ⚠️  build_today_features exited with code {build_result.returncode}.\n"
                "     Proceeding with predict using existing train table rows."
            )
    except Exception as e:
        logger.error("build_today_features crashed: %s", e)
        print(f"  ⚠️  build_today_features crashed: {e}\n     Proceeding anyway.")

    # ------------------------------------------------------------------
    # Step 2: Score
    # ------------------------------------------------------------------
    cmd = [sys.executable, "-m", "src.model.predict", "--date", date_str, "--run-type", predict_run_type]
    if force_refresh:
        cmd.append("--force-refresh")
    elif predict_run_type == "final":
        # Suppress predict.py's auto-force so subsequent waves hit the 12h
        # odds cache instead of spending ~16 credits on a fresh pull.
        cmd.append("--no-force-refresh")

    logger.info("Running predict: %s", " ".join(cmd))
    print(f"\n  Step 2/2 — Scoring predictions ...")

    try:
        result    = subprocess.run(cmd, cwd=PROJECT_ROOT)
        exit_code = result.returncode
        status    = "success" if exit_code == 0 else f"failed (exit {exit_code})"
    except Exception as e:
        logger.error("predict.py crashed: %s", e)
        exit_code = 1
        status    = f"error: {e}"

    elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()

    out_csv = PREDICTIONS_DIR / f"predictions_{date_str}_{predict_run_type}.csv"
    if not out_csv.exists():
        out_csv = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
    stats = inspect_predictions_csv(out_csv)

    log_run(
        pass_name=pass_name,
        date_str=date_str,
        status=status,
        elapsed_secs=elapsed,
        starters_confirmed=avail["starters_confirmed"],
        lineups_confirmed=avail["lineups_confirmed"],
        odds_available=avail["odds_available"],
        n_predictions=stats["n_predictions"],
        n_with_edge=stats["n_with_edge"],
        notes=f"force_refresh={force_refresh}",
    )

    print(f"\n  {'='*70}")
    print(f"  {pass_name.upper()} complete — {status}")
    print(f"  Predictions: {stats['n_predictions']}   With edge: {stats['n_with_edge']}   Elapsed: {elapsed:.0f}s")
    print(f"  {'='*70}\n")

    return exit_code


# ---------------------------------------------------------------------------
# Show waves (diagnostic)
# ---------------------------------------------------------------------------

def show_waves(date_str: str) -> None:
    game_times = fetch_game_times(date_str, force_refresh=True)
    waves      = build_waves(game_times)
    already    = load_already_run_today(date_str)

    print(f"\n{'='*72}")
    print(f"  MLB game waves for {date_str}  ({len(game_times)} games, {len(waves)} wave(s))")
    print(f"{'='*72}")

    if not waves:
        print("  No games found (or MLB API unavailable).")
        print(f"{'='*72}\n")
        return

    for w in waves:
        fp_str = w["final_pass_time"].strftime("%I:%M %p ET")
        e_str  = w["earliest"].strftime("%I:%M %p ET")
        l_str  = w["latest"].strftime("%I:%M %p ET")
        status = "✓ already run" if w["pass_name"] in already else "pending"
        print(
            f"\n  Wave {w['wave_id']}  [{w['label']}]  —  {w['game_count']} game(s)"
            f"\n    First pitches : {e_str} – {l_str}"
            f"\n    Final pass at : {fp_str}  ({w['pass_name']})  {status}"
        )

    print(f"\n{'='*72}\n")


# ---------------------------------------------------------------------------
# Status (run log)
# ---------------------------------------------------------------------------

def print_status() -> None:
    date_str = today_str()
    if not RUN_LOG.exists():
        print("No run log found. Nothing has run yet.")
        return

    rows = []
    with open(RUN_LOG) as f:
        reader = csv.DictReader(f)
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
        t        = row.get("run_time_utc", "")[:19].replace("T", " ")
        p        = row.get("pass", "").upper()
        status   = row.get("status", "")
        sec      = row.get("elapsed_secs", "?")
        starters = "✓" if row.get("starters_confirmed") == "True" else "✗"
        lineups  = "✓" if row.get("lineups_confirmed")  == "True" else "✗"
        odds     = "✓" if row.get("odds_available")     == "True" else "✗"
        notes    = row.get("notes", "")

        # Safely parse prediction counts — guard against old schema where
        # booleans were accidentally stored in these columns.
        raw_n  = row.get("n_predictions", "?")
        raw_ne = row.get("n_with_edge", "?")
        try:
            n  = int(raw_n)  if str(raw_n).lstrip("-").isdigit() else "?"
            ne = int(raw_ne) if str(raw_ne).lstrip("-").isdigit() else "?"
        except (ValueError, TypeError):
            n, ne = "?", "?"

        print(f"\n  [{p}]  {t} UTC  —  {status}")
        print(f"    Starters: {starters}   Lineups: {lineups}   Odds: {odds}")
        print(f"    Predictions: {n}   With edge: {ne}   Elapsed: {sec}s")
        if notes:
            print(f"    Notes: {notes}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="MLB HR model scheduler — game-aware pass timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pass", dest="pass_name", choices=["morning", "final"],
        help="Run a specific pass immediately (morning or final).",
    )
    mode.add_argument(
        "--auto-final", action="store_true",
        help=(
            "Check today's game schedule and fire final passes for any wave "
            "whose window just opened. Schedule this every 30 min in Task Scheduler."
        ),
    )
    mode.add_argument(
        "--auto-final-once", action="store_true",
        help=(
            "Fire final pass ONCE per day when 90 min before earliest first "
            "pitch. Cheap to run hourly — uses a sentinel file to dedupe."
        ),
    )
    mode.add_argument(
        "--auto-clv", action="store_true",
        help=(
            "Snapshot current odds vs bet-time lines for today's picks if a "
            "wave's first pitch is within the CLV window. Schedule every 15 min."
        ),
    )
    mode.add_argument(
        "--show-waves", action="store_true",
        help="Print today's game waves and their scheduled pass times, then exit.",
    )
    mode.add_argument(
        "--status", action="store_true",
        help="Print today's run history and exit.",
    )

    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing.")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    date_str = today_str()

    # ---- --status ----
    if args.status:
        print_status()
        return

    # ---- --show-waves ----
    if args.show_waves:
        show_waves(date_str)
        return

    # ---- --pass morning | final ----
    if args.pass_name:
        is_morning = args.pass_name == "morning"
        sys.exit(run_pass(
            args.pass_name,
            dry_run=args.dry_run,
            force_refresh=not is_morning,
            description=(
                "Probable starters + early lines. Lineups not yet confirmed."
                if is_morning else
                "Confirmed lineups + final pre-game lines. Use this for bets."
            ),
        ))

    # ---- --auto-final ----
    if args.auto_final:
        game_times = fetch_game_times(date_str, force_refresh=False)
        waves      = build_waves(game_times)

        if not waves:
            print(f"No games found for {date_str} — nothing to do.")
            return

        already = load_already_run_today(date_str)
        due     = get_due_passes(waves, already_run=already)

        if not due:
            now_str = now_et().strftime("%I:%M %p ET")
            print(f"[{now_str}] No final passes due right now.")
            if args.dry_run:
                show_waves(date_str)
            return

        # Only the first final of the day force-refreshes the odds cache.
        # Waves 2+ reuse the 12h cache (saves ~16 credits each, keeps us
        # under the 500/mo free tier). Lines stay reasonably fresh because
        # the cache TTL covers a typical single-day slate.
        had_prior_final = any(p.startswith("final") for p in already)

        for wave in due:
            print(
                f"\n⏰  Wave {wave['wave_id']} [{wave['label']}] final pass triggered "
                f"({wave['game_count']} games, first pitch {wave['earliest'].strftime('%I:%M %p ET')})"
            )
            force_refresh = not had_prior_final
            if not force_refresh:
                print("  (Reusing cached odds — not the first final of the day.)")
            rc = run_pass(
                wave["pass_name"],
                dry_run=args.dry_run,
                force_refresh=force_refresh,
                description=(
                    f"Final pass for {wave['label']} wave — "
                    f"{wave['game_count']} game(s), first pitch {wave['earliest'].strftime('%I:%M %p ET')}"
                ),
            )
            if rc != 0:
                logger.warning("Pass %s exited with code %d", wave["pass_name"], rc)
            # After the first successful wave, subsequent waves in this same
            # cron tick should also use cache.
            had_prior_final = True

        return

    # ---- --auto-final-once ----
    # Cheap hourly check: fire ONE final pass per day, 90 min before the
    # **biggest wave's** first pitch. Uses a sentinel file so multiple hourly
    # ticks inside the window don't re-fire. No Odds API cost for the check
    # itself (MLB schedule is free); Odds API is only hit when the pass fires.
    #
    # Why the biggest wave, not the earliest? Mixed schedules (e.g. 1 afternoon
    # game + 14 night games) used to burn the daily fire on the lone afternoon
    # game — producing a Discord post with 1 pick and missing the full slate.
    # Targeting the largest wave ensures the fire covers the most matchups.
    if args.auto_final_once:
        sentinel = PROJECT_ROOT / "data" / f".final_fired_{date_str}"
        if sentinel.exists():
            print(f"Final pass already fired today ({date_str}) — skipping.")
            return

        game_times = fetch_game_times(date_str, force_refresh=False)
        waves      = build_waves(game_times)
        if not waves:
            print(f"No games found for {date_str} — nothing to do.")
            return

        # Pick the wave with the most games. On ties (same game count), prefer
        # the later wave — books typically have tighter lines closer to the
        # main slate.
        target_wave = max(waves, key=lambda w: (w["game_count"], w["earliest"]))
        target_first_pitch = target_wave["earliest"]
        fire_at = target_first_pitch - timedelta(minutes=FINAL_PASS_LEAD_MINUTES)
        now = now_et()

        if now < fire_at:
            mins_until = int((fire_at - now).total_seconds() / 60)
            print(
                f"[{now.strftime('%I:%M %p ET')}] Too early — biggest wave "
                f"({target_wave['game_count']} games) first pitch "
                f"{target_first_pitch.strftime('%I:%M %p ET')}, fire at "
                f"{fire_at.strftime('%I:%M %p ET')} ({mins_until} min from now)."
            )
            return

        print(
            f"\n⏰  Final pass fired — biggest wave ({target_wave['game_count']} games) "
            f"first pitch {target_first_pitch.strftime('%I:%M %p ET')}"
        )
        rc = run_pass(
            "final",
            dry_run=args.dry_run,
            force_refresh=True,
            description=(
                f"Daily final pass — {target_wave['game_count']} games, "
                f"first pitch {target_first_pitch.strftime('%I:%M %p ET')}"
            ),
        )
        if rc == 0 and not args.dry_run:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.touch()
        elif rc != 0:
            logger.warning("Final pass exited with code %d", rc)
        return

    # ---- --auto-clv ----
    if args.auto_clv:
        # CLV snapshots re-fetch odds and cost ~16 API credits each. On the
        # free tier (500/mo) that's unaffordable. Gate behind CLV_ENABLED.
        if os.environ.get("CLV_ENABLED", "false").lower() not in ("1", "true", "yes"):
            print("CLV disabled (set CLV_ENABLED=true to enable — costs API credits).")
            return

        game_times = fetch_game_times(date_str, force_refresh=False)
        waves      = build_waves(game_times)

        if not waves:
            print(f"No games found for {date_str} — nothing to do.")
            return

        # Fire one CLV snapshot per day — right before the LATEST wave's first
        # pitch. clv_tracker snapshots every positive-edge pick for the date in
        # a single pass, so we only need one run per day. Firing per-wave would
        # cost ~16 credits each = 4×/day = blows past the 500/mo free tier.
        now = now_et()
        latest_wave = max(waves, key=lambda w: w["earliest"])
        in_window = (
            now >= latest_wave["earliest"] - timedelta(minutes=CLV_SNAPSHOT_LEAD_MINUTES)
            and now <= latest_wave["earliest"] + timedelta(minutes=5)
        )

        if not in_window:
            now_str = now.strftime("%I:%M %p ET")
            fp_str  = latest_wave["earliest"].strftime("%I:%M %p ET")
            print(
                f"[{now_str}] No CLV snapshot due — latest wave first pitch {fp_str} "
                f"(snapshot window opens {CLV_SNAPSHOT_LEAD_MINUTES} min prior)."
            )
            return

        # Dedupe: clv_{date}.csv gets written once per snapshot run. If it
        # already exists for today, we've already snapshotted — skip.
        clv_file = PREDICTIONS_DIR / f"clv_{date_str}.csv"
        if clv_file.exists():
            print(f"CLV snapshot already taken for {date_str} ({clv_file.name}) — skipping.")
            return

        print(f"⏰  CLV snapshot — latest wave imminent: {latest_wave['label']}")

        cmd = [sys.executable, "-m", "src.analysis.clv_tracker", "--date", date_str]
        if args.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return

        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT)
        except Exception as e:
            logger.error("clv_tracker crashed: %s", e)
        return

    # No mode specified — print help
    parser.print_help()


if __name__ == "__main__":
    main()

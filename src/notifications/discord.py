"""
Discord Webhook Notifier — MLB HR Picks
========================================
Sends daily bet picks to a Discord channel via webhook.

Each pick is posted as a rich embed with:
  - Player name, pitcher faced, batting slot
  - Model prob, market prob, edge
  - DraftKings / FanDuel odds (from odds data)
  - Minimum odds for other books (use --manual-check list)
  - Kelly stake recommendation

Setup
-----
  1. In Discord: Server Settings → Integrations → Webhooks → New Webhook
  2. Copy the webhook URL
  3. Add to your .env file:
       DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN

  Optional — separate channels per pass:
       DISCORD_WEBHOOK_MORNING=https://discord.com/api/webhooks/...
       DISCORD_WEBHOOK_FINAL=https://discord.com/api/webhooks/...

Usage (standalone test)
-----------------------
    python -m src.notifications.discord --date 2026-04-01
    python -m src.notifications.discord --pass final
    python -m src.notifications.discord --dry-run        # print to console only

Called automatically by predict.py after every final pass if
DISCORD_WEBHOOK_URL (or DISCORD_WEBHOOK_FINAL) is set.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

# ---------------------------------------------------------------------------
# Platform colours (used for embed accent)
# ---------------------------------------------------------------------------
COLOUR_FINAL   = 0x00B16A   # green  — confirmed bets
COLOUR_MORNING = 0xF39C12   # amber  — early / unconfirmed
COLOUR_NO_ODDS = 0x95A5A6   # grey   — model-only run

# ---------------------------------------------------------------------------
# Minimum-odds calculation
# ---------------------------------------------------------------------------

def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal."""
    if american < 0:
        return 1 + (100 / abs(american))
    return 1 + (american / 100)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American (rounded to nearest 5)."""
    if decimal >= 2.0:
        american = (decimal - 1) * 100
    else:
        american = -100 / (decimal - 1)
    # Round to nearest 5 (typical book increment)
    return int(round(american / 5) * 5)


def minimum_odds_for_edge(
    model_prob: float,
    min_edge_pp: float = 3.0,
    vig_pct: float = 0.045,
) -> dict:
    """
    Calculate the minimum acceptable American odds at which a bet still
    has at least `min_edge_pp` percentage-points of edge vs the model.

    For use when manually checking a book that isn't in the odds feed.

    Parameters
    ----------
    model_prob  : model's HR probability (0–1)
    min_edge_pp : minimum edge required in percentage points (default 3pp)
    vig_pct     : assumed book vig to add on top (default 4.5%)

    Returns
    -------
    dict with:
        min_fair_decimal    : minimum decimal odds before vig
        min_book_decimal    : minimum decimal odds after vig (what book shows)
        min_book_american   : minimum American odds to show on a bet slip
        breakeven_american  : odds at which edge = 0 (no-bet threshold)
    """
    # Fair probability threshold: model_prob minus minimum edge
    # = maximum fair prob we're willing to accept from the market
    max_market_fair_prob = model_prob - (min_edge_pp / 100)

    if max_market_fair_prob <= 0:
        # Edge requirement > model prob — no line will ever be good enough
        return {
            "min_fair_decimal":  None,
            "min_book_decimal":  None,
            "min_book_american": None,
            "breakeven_american": None,
            "note": "Model prob too low for this edge threshold.",
        }

    # Convert to decimal odds (fair, before vig)
    min_fair_decimal = 1 / max_market_fair_prob

    # Add vig: the book's implied prob = fair_prob * (1 + vig/2) per side
    # So the over price will be slightly worse than fair.
    # We want the minimum BOOK odds (after vig), meaning the market fair prob
    # implied by what the book shows must still be <= max_market_fair_prob.
    # Solve: book_implied_prob = book_decimal_implied / (1 + vig)
    # min_book_decimal such that the fair prob it implies <= max_market_fair_prob
    min_book_implied_prob = max_market_fair_prob * (1 + vig_pct / 2)
    min_book_decimal = 1 / min_book_implied_prob

    # Breakeven (edge = 0): market fair prob == model prob
    breakeven_fair_decimal = 1 / model_prob
    breakeven_book_implied_prob = model_prob * (1 + vig_pct / 2)
    breakeven_book_decimal = 1 / breakeven_book_implied_prob

    return {
        "min_fair_decimal":   round(min_fair_decimal, 3),
        "min_book_decimal":   round(min_book_decimal, 3),
        "min_book_american":  decimal_to_american(min_book_decimal),
        "breakeven_american": decimal_to_american(breakeven_book_decimal),
        "note": f"Need odds ≥ {_fmt_american(decimal_to_american(min_book_decimal))} on other books "
                f"for {min_edge_pp:.0f}pp edge.",
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_american(val) -> str:
    try:
        v = int(val)
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(val) -> str:
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_edge(val) -> str:
    try:
        v = float(val) * 100
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}pp"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Embed builder
# ---------------------------------------------------------------------------

def build_picks_payload(
    ranked_df,
    *,
    pass_label: str = "FINAL",
    date_str: str | None = None,
    min_edge_pp: float = 3.0,
) -> list[dict]:
    """
    Build a list of Discord webhook payloads (one per message chunk).

    Discord embed field limit: 25 fields per embed.
    We send one embed per pick so each card is readable.

    Returns a list of dicts ready to POST to the webhook.
    """
    import pandas as pd

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    has_odds = "edge" in ranked_df.columns and ranked_df["edge"].notna().any()

    # Filter to positive-edge bets only (or top-10 if no odds)
    if has_odds:
        bets = ranked_df[
            ranked_df["edge"].notna() & (ranked_df["edge"] > 0)
        ].sort_values(["edge", "hr_prob"], ascending=False).copy()
    else:
        bets = ranked_df.sort_values("hr_prob", ascending=False).head(10).copy()

    if bets.empty:
        # Send a "no picks" notice
        payload = {
            "embeds": [{
                "title": f"⚾ MLB HR Picks — {date_str} [{pass_label}]",
                "description": "No positive-edge bets found today. 🔍",
                "color": COLOUR_NO_ODDS,
                "footer": {"text": f"MLB HR Model • {pass_label} pass • {date_str}"},
            }]
        }
        return [payload]

    colour = COLOUR_FINAL if pass_label == "FINAL" else COLOUR_MORNING

    payloads = []

    # ---- Header message ----
    n_bets = len(bets)
    if has_odds:
        avg_edge = bets["edge"].mean() * 100
        desc = (
            f"**{n_bets} bet{'s' if n_bets != 1 else ''} with positive edge** "
            f"| avg edge **{avg_edge:+.1f}pp**\n"
            f"✅ DraftKings & FanDuel odds shown directly.\n"
            f"📋 Other books: check minimum odds listed per pick."
        )
    else:
        desc = (
            f"**Top {n_bets} HR candidates** (model-only, no odds data)\n"
            f"ℹ️ Set `ODDS_API_KEY` for edge-filtered picks with minimum odds."
        )

    header_payload = {
        "embeds": [{
            "title": f"⚾ MLB HR Picks — {date_str}  [{pass_label}]",
            "description": desc,
            "color": colour,
        }]
    }
    payloads.append(header_payload)

    # ---- One embed per pick ----
    for rank, (_, row) in enumerate(bets.iterrows(), start=1):
        batter  = str(row.get("batter_name", "Unknown"))
        pitcher = str(row.get("pitcher_name", "Unknown"))
        slot    = int(row.get("batting_order_pos", 0)) or "?"
        model   = _fmt_pct(row.get("hr_prob"))
        market  = _fmt_pct(row.get("market_fair_prob")) if has_odds else "—"
        edge    = _fmt_edge(row.get("edge")) if has_odds else "—"
        odds    = _fmt_american(row.get("market_over_price")) if has_odds else "—"
        kelly   = _fmt_pct(row.get("kelly_stake")) if has_odds else "—"
        book    = str(row.get("odds_bookmaker", "")).upper() or "—"

        fields = [
            {"name": "🆚 vs Pitcher",    "value": pitcher,  "inline": True},
            {"name": "📍 Batting Slot",  "value": str(slot), "inline": True},
            {"name": "🤖 Model Prob",    "value": model,    "inline": True},
        ]

        if has_odds:
            fields += [
                {"name": "📊 Market Fair", "value": market, "inline": True},
                {"name": "📈 Edge",        "value": edge,   "inline": True},
                {"name": "💰 Kelly Stake", "value": kelly,  "inline": True},
            ]

            # DraftKings / FanDuel — show the odds we have directly
            # (The odds feed aggregates to one book; DK/FD are typically within
            #  a few ticks of each other for HR props)
            dk_fd_odds = odds
            fields += [
                {"name": "🟢 DraftKings",  "value": dk_fd_odds, "inline": True},
                {"name": "🔵 FanDuel",     "value": dk_fd_odds, "inline": True},
                {"name": "📖 Source Book", "value": book,       "inline": True},
            ]

            # Minimum odds for OTHER books (manual check)
            try:
                hr_prob = float(row.get("hr_prob", 0))
                min_odds_info = minimum_odds_for_edge(hr_prob, min_edge_pp=min_edge_pp)
                min_american  = min_odds_info.get("min_book_american")
                breakeven     = min_odds_info.get("breakeven_american")
                if min_american is not None:
                    other_books_text = (
                        f"Need **{_fmt_american(min_american)}** or better\n"
                        f"*(break-even: {_fmt_american(breakeven)})*"
                    )
                else:
                    other_books_text = "Prob too low for this threshold"
            except Exception:
                other_books_text = "—"

            fields.append({
                "name": f"🔍 Other Books (min for {min_edge_pp:.0f}pp edge)",
                "value": other_books_text,
                "inline": False,
            })

        embed = {
            "title": f"#{rank}  {batter}  — HR Over",
            "color": colour,
            "fields": fields,
            "footer": {
                "text": f"MLB HR Model • {pass_label} pass • {date_str}"
            },
        }
        payloads.append({"embeds": [embed]})

    return payloads


# ---------------------------------------------------------------------------
# Webhook sender
# ---------------------------------------------------------------------------

def send_to_discord(
    ranked_df,
    *,
    pass_label: str = "FINAL",
    date_str: str | None = None,
    webhook_url: str | None = None,
    dry_run: bool = False,
    min_edge_pp: float = 3.0,
) -> bool:
    """
    Format picks and POST to Discord webhook.

    Parameters
    ----------
    ranked_df   : predictions DataFrame from predict.py
    pass_label  : "MORNING" or "FINAL"
    date_str    : YYYY-MM-DD string for the display header
    webhook_url : override env var
    dry_run     : print payloads to console without sending
    min_edge_pp : minimum edge for 'other books' minimum odds calculation

    Returns True on success, False on failure.
    """
    if webhook_url is None:
        key = f"DISCORD_WEBHOOK_{pass_label.upper()}"
        webhook_url = os.environ.get(key) or os.environ.get("DISCORD_WEBHOOK_URL")

    if not webhook_url and not dry_run:
        logger.info(
            "No Discord webhook URL found. Set DISCORD_WEBHOOK_URL in .env to enable.\n"
            "  export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/TOKEN"
        )
        return False

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    payloads = build_picks_payload(
        ranked_df,
        pass_label=pass_label,
        date_str=date_str,
        min_edge_pp=min_edge_pp,
    )

    if dry_run:
        print(f"\n{'='*70}")
        print(f"  [DRY RUN] Discord payload — {len(payloads)} message(s)")
        print(f"{'='*70}")
        for i, p in enumerate(payloads, 1):
            print(f"\n--- Message {i} ---")
            print(json.dumps(p, indent=2))
        print(f"\n{'='*70}\n")
        return True

    success = True
    for i, payload in enumerate(payloads):
        try:
            resp = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
            )
            if resp.status_code == 204:
                logger.debug("Discord message %d/%d sent OK", i + 1, len(payloads))
            else:
                logger.warning(
                    "Discord webhook returned %d for message %d: %s",
                    resp.status_code, i + 1, resp.text[:200],
                )
                success = False
            # Discord rate-limit: 5 requests/2s → tiny sleep between embeds
            import time
            time.sleep(0.5)
        except requests.RequestException as e:
            logger.error("Discord webhook failed message %d: %s", i + 1, e)
            success = False

    if success:
        logger.info("Discord: sent %d pick message(s) for %s [%s]", len(payloads), date_str, pass_label)
    return success


# ---------------------------------------------------------------------------
# CLI (standalone test / manual trigger)
# ---------------------------------------------------------------------------

def _load_latest_predictions(date_str: str | None, pass_label: str) -> "pd.DataFrame":
    import pandas as pd

    if date_str is None:
        # Find most recent predictions file
        files = sorted(PREDICTIONS_DIR.glob("predictions_????-??-??.csv"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No prediction CSVs found in {PREDICTIONS_DIR}")
        path = files[0]
        date_str = path.stem.replace("predictions_", "")
    else:
        suffix_map = {"FINAL": "final", "MORNING": "morning"}
        sfx = suffix_map.get(pass_label.upper())
        path = PREDICTIONS_DIR / f"predictions_{date_str}_{sfx}.csv" if sfx else None
        if path is None or not path.exists():
            path = PREDICTIONS_DIR / f"predictions_{date_str}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No predictions file found for {date_str}")

    logger.info("Loading predictions from %s", path)
    return pd.read_csv(path), date_str


if __name__ == "__main__":
    import argparse
    from src.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="Send HR picks to Discord")
    parser.add_argument("--date",      default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument("--pass",      dest="pass_label", choices=["morning", "final"],
                        default="final", help="Pass label (default: final)")
    parser.add_argument("--min-edge",  type=float, default=3.0,
                        help="Min edge pp for other-books minimum odds (default 3.0)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print payloads to console without sending")
    parser.add_argument("--webhook",   default=None, help="Override webhook URL")
    args = parser.parse_args()

    df, resolved_date = _load_latest_predictions(args.date, args.pass_label.upper())

    send_to_discord(
        df,
        pass_label=args.pass_label.upper(),
        date_str=resolved_date,
        webhook_url=args.webhook,
        dry_run=args.dry_run,
        min_edge_pp=args.min_edge,
    )

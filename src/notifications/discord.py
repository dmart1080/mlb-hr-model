"""
Discord Webhook Notifier — MLB HR Picks
========================================
Sends daily bet picks to a Discord channel via webhook.

Pick tiers
----------
  🟢 BET    edge > 0pp         — positive edge, post as full pick embed
  🟡 WATCH  -2pp ≤ edge ≤ 0pp — near-edge, worth monitoring or shopping
  (below -2pp is not shown)
  📊 TOP 10 MODEL PLAYS        — only when no edge bets: top 10 hr_prob,
                                 no odds checked, purely model-based, posted after watchlist

The watch-list is posted as a compact single embed (not individual cards)
so it doesn't flood the channel on days with no strong edges.

Setup
-----
  1. In Discord: Server Settings → Integrations → Webhooks → New Webhook
  2. Copy the webhook URL
  3. Add to your .env file:
       DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN

  Optional — separate channels per pass:
       DISCORD_WEBHOOK_MORNING=https://discord.com/api/webhooks/...
       DISCORD_WEBHOOK_FINAL=https://discord.com/api/webhooks/...
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

COLOUR_FINAL    = 0x00B16A   # green  — confirmed bets
COLOUR_MORNING  = 0xF39C12   # amber  — early / unconfirmed
COLOUR_WATCH    = 0xF39C12   # amber  — near-edge watch list
COLOUR_NO_ODDS  = 0x95A5A6   # grey   — model-only run
COLOUR_MODEL    = 0x3498DB   # blue   — model favourite (no odds check)

# Near-edge window: picks between WATCH_FLOOR and 0pp are shown as watch list
WATCH_FLOOR_PP = -2.0


# ---------------------------------------------------------------------------
# Platform odds helpers
# ---------------------------------------------------------------------------

def american_to_decimal(american: int | float) -> float:
    if american < 0:
        return 1 + (100 / abs(american))
    return 1 + (american / 100)


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        american = (decimal - 1) * 100
    else:
        american = -100 / (decimal - 1)
    return int(round(american / 5) * 5)


def minimum_odds_for_edge(
    model_prob: float,
    min_edge_pp: float = 3.0,
    vig_pct: float = 0.035,   # updated to match odds.py
) -> dict:
    """
    Calculate the minimum acceptable American odds for at least min_edge_pp
    of edge. Uses 3.5% vig to match the updated odds.py assumption.
    """
    max_market_fair_prob = model_prob - (min_edge_pp / 100)

    if max_market_fair_prob <= 0:
        return {
            "min_fair_decimal":  None,
            "min_book_decimal":  None,
            "min_book_american": None,
            "breakeven_american": None,
            "note": "Model prob too low for this edge threshold.",
        }

    min_fair_decimal = 1 / max_market_fair_prob

    min_book_implied_prob = max_market_fair_prob * (1 + vig_pct / 2)
    min_book_decimal = 1 / min_book_implied_prob

    breakeven_book_implied_prob = model_prob * (1 + vig_pct / 2)
    breakeven_book_decimal = 1 / breakeven_book_implied_prob

    return {
        "min_fair_decimal":   round(min_fair_decimal, 3),
        "min_book_decimal":   round(min_book_decimal, 3),
        "min_book_american":  decimal_to_american(min_book_decimal),
        "breakeven_american": decimal_to_american(breakeven_book_decimal),
        "note": f"Need odds ≥ {_fmt_american(decimal_to_american(min_book_decimal))} "
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


def _safe_name(val, fallback: str = "TBD") -> str:
    """Return a clean player name string, replacing NaN/None/empty with fallback."""
    if val is None:
        return fallback
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return fallback
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return fallback
    return s


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------

def _build_bet_embed(
    rank: int,
    row,
    *,
    colour: int,
    pass_label: str,
    date_str: str,
    min_edge_pp: float,
    has_odds: bool,
) -> dict:
    """Build a full pick embed for a positive-edge bet."""
    batter  = _safe_name(row.get("batter_name"), fallback="Unknown")
    pitcher = _safe_name(row.get("pitcher_name"), fallback="TBD")
    slot    = int(row.get("batting_order_pos", 0)) or "?"
    model   = _fmt_pct(row.get("hr_prob"))
    market  = _fmt_pct(row.get("market_fair_prob")) if has_odds else "—"
    edge    = _fmt_edge(row.get("edge")) if has_odds else "—"
    odds    = _fmt_american(row.get("market_over_price")) if has_odds else "—"
    kelly   = _fmt_pct(row.get("kelly_stake")) if has_odds else "—"
    book    = str(row.get("odds_bookmaker", "")).upper() or "—"

    fields = [
        {"name": "🆚 vs Pitcher",   "value": pitcher,   "inline": True},
        {"name": "📍 Batting Slot", "value": str(slot), "inline": True},
        {"name": "🤖 Model Prob",   "value": model,     "inline": True},
    ]

    if has_odds:
        fields += [
            {"name": "📊 Market Fair", "value": market, "inline": True},
            {"name": "📈 Edge",        "value": edge,   "inline": True},
            {"name": "💰 Kelly Stake", "value": kelly,  "inline": True},
            {"name": "🟢 DraftKings",  "value": odds,   "inline": True},
            {"name": "🔵 FanDuel",     "value": odds,   "inline": True},
            {"name": "📖 Source Book", "value": book,   "inline": True},
        ]

        try:
            hr_prob = float(row.get("hr_prob", 0))
            min_odds_info = minimum_odds_for_edge(hr_prob, min_edge_pp=min_edge_pp)
            min_american  = min_odds_info.get("min_book_american")
            breakeven     = min_odds_info.get("breakeven_american")
            other_books_text = (
                f"Need **{_fmt_american(min_american)}** or better\n"
                f"*(break-even: {_fmt_american(breakeven)})*"
                if min_american is not None
                else "Prob too low for this threshold"
            )
        except Exception:
            other_books_text = "—"

        fields.append({
            "name":   f"🔍 Other Books (min for {min_edge_pp:.0f}pp edge)",
            "value":  other_books_text,
            "inline": False,
        })

    return {
        "title":  f"#{rank}  {batter}  — HR Over",
        "color":  colour,
        "fields": fields,
        "footer": {"text": f"MLB HR Model • {pass_label} pass • {date_str}"},
    }


def _build_watchlist_embed(
    watch_rows,
    *,
    date_str: str,
    pass_label: str,
) -> dict:
    """
    Build a compact single embed for near-edge watch-list picks.
    These are picks where -2pp ≤ edge < 0pp — worth monitoring but not bets.
    """
    lines = []
    for _, row in watch_rows.iterrows():
        name    = _safe_name(row.get("batter_name"),  fallback="Unknown")
        pitcher = _safe_name(row.get("pitcher_name"), fallback="TBD")
        model   = _fmt_pct(row.get("hr_prob"))
        edge    = _fmt_edge(row.get("edge"))
        odds    = _fmt_american(row.get("market_over_price"))
        book    = str(row.get("odds_bookmaker", "")).upper() or "—"
        slot    = int(row.get("batting_order_pos", 0))
        lines.append(
            f"**{name}** vs {pitcher} (slot {slot})\n"
            f"  Model {model} | Edge {edge} | {odds} [{book}]"
        )

    return {
        "title":       f"🟡 Watch List — Near Edge ({date_str})",
        "description": (
            f"These {len(lines)} pick(s) are within 2pp of positive edge. "
            f"Worth shopping other books or monitoring for line movement.\n\n"
            + "\n\n".join(lines)
        ),
        "color":  COLOUR_WATCH,
        "footer": {"text": f"MLB HR Model • {pass_label} pass • {date_str} • not a bet recommendation"},
    }


def _build_top10_model_embed(top_rows, *, date_str: str, pass_label: str) -> dict:
    """
    Compact embed showing the top 10 players by model HR probability.
    Only posted when there are no positive-edge bets today.
    No odds API is consulted; this is purely model output.
    """
    lines = []
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        batter  = _safe_name(row.get("batter_name"), fallback="Unknown")
        pitcher = _safe_name(row.get("pitcher_name"), fallback="TBD")
        prob    = _fmt_pct(row.get("hr_prob"))
        slot    = int(row.get("batting_order_pos", 0)) or "?"
        lines.append(f"**{rank}.** **{batter}** vs {pitcher} (slot {slot}) — {prob}")

    return {
        "title":       f"📊 Top 10 Model Plays — {date_str}",
        "description": (
            "No edge plays today. Here are the top 10 players by model HR probability.\n"
            "No odds data used — not a bet recommendation.\n\n"
            + "\n".join(lines)
        ),
        "color": COLOUR_MODEL,
        "footer": {
            "text": f"MLB HR Model • {pass_label} pass • {date_str} • model only — not a bet recommendation"
        },
    }


# ---------------------------------------------------------------------------
# Main payload builder
# ---------------------------------------------------------------------------

def build_picks_payload(
    ranked_df,
    *,
    pass_label: str = "FINAL",
    date_str: str | None = None,
    min_edge_pp: float = 3.0,
) -> list[dict]:
    """
    Build Discord webhook payloads.

    Tier logic:
      edge > 0pp             → full pick embed (green)  — bet
      WATCH_FLOOR ≤ edge ≤ 0 → compact watch-list embed (amber) — monitor
      edge < WATCH_FLOOR     → not shown
      no odds at all         → top-10 model-only fallback (grey)
      no edge bets today     → top-10 model plays embed (blue) posted after watch list
    """
    import pandas as pd

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    has_odds = "edge" in ranked_df.columns and ranked_df["edge"].notna().any()

    payloads = []

    # ---- No odds fallback ----
    if not has_odds:
        top = ranked_df.sort_values("hr_prob", ascending=False).head(10)
        lines = []
        for _, row in top.iterrows():
            lines.append(
                f"**{_safe_name(row.get('batter_name'))}** vs {_safe_name(row.get('pitcher_name'), fallback='TBD')} "
                f"— {_fmt_pct(row.get('hr_prob'))} (slot {int(row.get('batting_order_pos',0))})"
            )
        payloads.append({"embeds": [{
            "title":       f"⚾ MLB HR Picks — {date_str}  [{pass_label}]",
            "description": "**Model-only picks** (no odds data)\n\n" + "\n".join(lines),
            "color":       COLOUR_NO_ODDS,
        }]})
        # Model favourite is already the first line above; no separate embed needed
        return payloads

    # ---- Split into bet / watch tiers ----
    watch_floor = WATCH_FLOOR_PP / 100.0

    bets = ranked_df[
        ranked_df["edge"].notna() & (ranked_df["edge"] > 0)
    ].sort_values(["edge", "hr_prob"], ascending=False).copy()

    watch = ranked_df[
        ranked_df["edge"].notna() &
        (ranked_df["edge"] >= watch_floor) &
        (ranked_df["edge"] <= 0)
    ].sort_values(["edge", "hr_prob"], ascending=False).copy()

    colour = COLOUR_FINAL if pass_label == "FINAL" else COLOUR_MORNING

    # ---- Header ----
    if len(bets) > 0:
        avg_edge = bets["edge"].mean() * 100
        desc = (
            f"**{len(bets)} bet{'s' if len(bets) != 1 else ''} with positive edge** "
            f"| avg edge **{avg_edge:+.1f}pp**\n"
            f"✅ DraftKings & FanDuel odds shown directly.\n"
            f"📋 Other books: check minimum odds listed per pick."
        )
    else:
        desc = (
            f"**No positive-edge bets today.**\n"
            f"Check the watch list below for near-edge candidates."
        )

    if len(watch) > 0:
        desc += f"\n🟡 **{len(watch)} watch-list pick{'s' if len(watch) != 1 else ''}** (within 2pp of edge)"

    payloads.append({"embeds": [{
        "title":       f"⚾ MLB HR Picks — {date_str}  [{pass_label}]",
        "description": desc,
        "color":       colour,
    }]})

    # ---- One embed per bet ----
    for rank, (_, row) in enumerate(bets.iterrows(), start=1):
        embed = _build_bet_embed(
            rank, row,
            colour=colour,
            pass_label=pass_label,
            date_str=date_str,
            min_edge_pp=min_edge_pp,
            has_odds=True,
        )
        payloads.append({"embeds": [embed]})

    # ---- Compact watch-list embed ----
    if not watch.empty:
        payloads.append({"embeds": [
            _build_watchlist_embed(watch, date_str=date_str, pass_label=pass_label)
        ]})

    # ---- Top 10 model plays — only when no positive-edge bets today ----
    if bets.empty:
        top10 = ranked_df.sort_values("hr_prob", ascending=False).head(10)
        payloads.append({"embeds": [
            _build_top10_model_embed(top10, date_str=date_str, pass_label=pass_label)
        ]})

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
    if webhook_url is None:
        key = f"DISCORD_WEBHOOK_{pass_label.upper()}"
        webhook_url = os.environ.get(key) or os.environ.get("DISCORD_WEBHOOK_URL")

    if not webhook_url and not dry_run:
        logger.info(
            "No Discord webhook URL found. Set DISCORD_WEBHOOK_URL in .env to enable."
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
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code == 204:
                logger.debug("Discord message %d/%d sent OK", i + 1, len(payloads))
            else:
                logger.warning("Discord webhook returned %d for message %d: %s",
                               resp.status_code, i + 1, resp.text[:200])
                success = False
            import time
            time.sleep(0.5)
        except requests.RequestException as e:
            logger.error("Discord webhook failed message %d: %s", i + 1, e)
            success = False

    if success:
        logger.info("Discord: sent %d pick message(s) for %s [%s]",
                    len(payloads), date_str, pass_label)
    return success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_latest_predictions(date_str: str | None, pass_label: str):
    import pandas as pd

    if date_str is None:
        files = sorted(PREDICTIONS_DIR.glob("predictions_????-??-??.csv"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No prediction CSVs found in {PREDICTIONS_DIR}")
        path = files[0]
        date_str = path.stem.replace("predictions_", "")
    else:
        suffix_map = {"FINAL": "final", "MORNING": "morning"}
        sfx  = suffix_map.get(pass_label.upper())
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
                        default="final")
    parser.add_argument("--min-edge",  type=float, default=3.0)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--webhook",   default=None)
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

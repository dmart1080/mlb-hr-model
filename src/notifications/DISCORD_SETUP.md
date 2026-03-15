## ============================================================
## Discord + Platform Odds Setup
## ============================================================

## --- 1. Add to your .env file ---

# Required — one webhook for all passes:
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN

# Optional — separate channels per pass:
DISCORD_WEBHOOK_MORNING=https://discord.com/api/webhooks/YOUR_MORNING_CHANNEL_TOKEN
DISCORD_WEBHOOK_FINAL=https://discord.com/api/webhooks/YOUR_FINAL_CHANNEL_TOKEN


## --- 2. How to create a Discord webhook ---
##
##   a. Open Discord → go to the channel you want picks in
##   b. Channel Settings (⚙️) → Integrations → Webhooks → New Webhook
##   c. Name it (e.g. "HR Picks Bot"), copy the Webhook URL
##   d. Paste into .env above


## --- 3. How picks are posted ---
##
##   Each run (morning + final) sends:
##     • One header card: date, pass label, number of bets, avg edge
##     • One embed per bet pick showing:
##
##       #1  Aaron Judge — HR Over
##       ┌─────────────────────────────────────────────────────┐
##       │ vs Pitcher     Gerrit Cole        Batting Slot  3   │
##       │ Model Prob     14.2%              Market Fair   9.8%│
##       │ Edge           +4.4pp             Kelly Stake   6.1%│
##       │ DraftKings     +115               FanDuel       +115│
##       │ Source Book    DRAFTKINGS                           │
##       │ Other Books (min for 3pp edge)                      │
##       │   Need +105 or better  (break-even: -108)           │
##       └─────────────────────────────────────────────────────┘


## --- 4. DraftKings & FanDuel odds ---
##
##   The odds feed (Odds API) pulls live lines. DraftKings and FanDuel
##   are both shown as the same line in each pick embed because their
##   HR prop prices are typically within 5 ticks of each other.
##
##   If you want truly separate DK vs FD lines, upgrade your Odds API
##   plan to include multiple regions/bookmakers and set:
##     _DEFAULT_BOOK = "draftkings"   (in src/data_sources/odds.py)
##   then run a second enrichment pass for FanDuel.


## --- 5. Other books — minimum odds (manual check) ---
##
##   Every pick shows a "minimum odds" line for books NOT in the feed.
##   This is the worst American odds line you should accept and still
##   maintain at least 3pp of edge vs the model.
##
##   Example:
##     Model prob = 14.2%,  min edge = 3pp
##     → You need the book to offer at least +105
##     → If their line is +100 or worse (e.g. -110), skip it.
##
##   To change the minimum edge threshold:
##     python -m src.notifications.discord --min-edge 5.0
##   Or set it as an env var:
##     DISCORD_MIN_EDGE_PP=5.0    (read in discord.py if you add that)


## --- 6. Manual test ---
##
##   # Print picks to console (no actual Discord POST):
##   python -m src.notifications.discord --dry-run
##
##   # Send today's final picks right now:
##   python -m src.notifications.discord --pass final
##
##   # Send a specific date:
##   python -m src.notifications.discord --date 2026-04-15 --pass final


## --- 7. Automatic posting ---
##
##   After applying predict_py_discord_patch.txt to src/model/predict.py,
##   Discord posts happen automatically at the end of every scheduler run:
##
##     python -m src.scheduler --pass morning   → posts to DISCORD_WEBHOOK_MORNING
##     python -m src.scheduler --pass final     → posts to DISCORD_WEBHOOK_FINAL
##     (falls back to DISCORD_WEBHOOK_URL if the pass-specific key is absent)

# MLB HR Model

A machine-learning pipeline that predicts daily MLB home run probabilities, identifies edges against sportsbook lines, and posts picks to Discord automatically — with no manual intervention required.

---

## How it works

1. **Morning pass (10 AM ET)** — scores every batter on today's slate using probable starters and early lines. Good for spotting value before the market moves.
2. **Final passes (90 min before each game wave)** — re-scores with confirmed lineups and live prop lines. These are the picks to bet.
3. **Discord** gets one embed per pick, per wave, automatically.

The scheduler is game-aware: it groups games into waves by start time and fires a separate final pass for each wave (noon games, afternoon games, evening games, etc.) so you never miss a slate.

---

## Project structure

```
mlb-hr-model/
├── data/
│   ├── cache/
│   │   ├── statcast/           # Raw Statcast parquets (auto-created, gitignored)
│   │   ├── schedule/           # MLB API game roster + batting order cache
│   │   ├── weather/            # MLB API weather cache
│   │   ├── odds/               # Odds API HR prop cache (TTL 2h)
│   │   └── park_factors/       # Park factor cache (TTL 30d)
│   ├── processed/              # Feature tables and train tables (gitignored)
│   ├── predictions/            # Daily prediction CSVs + run_log.csv
│   ├── backtest/               # Backtest results
│   └── recaps/                 # Weekly recap CSVs
├── models/                     # Saved .joblib model bundles + train_runs.csv (gitignored)
├── src/
│   ├── logging_config.py
│   ├── scheduler.py                        # Game-aware daily scheduler (main entry point)
│   ├── data_sources/
│   │   ├── statcast.py                     # Statcast fetch + disk cache
│   │   ├── weather.py                      # MLB Stats API weather
│   │   ├── mlb_schedule.py                 # Probable starters + confirmed lineups
│   │   └── odds.py                         # The Odds API: HR props + edge calc
│   ├── features/
│   │   ├── build_labels.py                 # Collapse events → batter-game labels
│   │   ├── build_features_common.py        # Shared constants + pure helpers
│   │   ├── build_features_fast.py          # Vectorised rolling window computation
│   │   ├── build_features.py               # Main feature pipeline
│   │   ├── build_features_season.py        # Single-season build CLI
│   │   ├── build_features_multi_season.py  # Multi-season build (2021–2026)
│   │   ├── build_today_features.py         # Today's matchup features (called by scheduler)
│   │   └── park_factors.py                 # HR park factor fetch + static fallback
│   ├── model/
│   │   ├── train.py                        # Train + calibrate + save model
│   │   └── predict.py                      # Score today's slate, save CSV, post to Discord
│   ├── analysis/
│   │   └── weekly_recap.py                 # Weekly P&L + calibration report
│   └── notifications/
│       ├── discord.py                      # Discord webhook: pick embeds + weekly recap
│       └── DISCORD_SETUP.md               # Webhook setup guide
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/YOUR_USERNAME/mlb-hr-model.git
cd mlb-hr-model

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
# Required for odds/edge detection (free tier: 500 req/month)
# Get a key at https://the-odds-api.com
ODDS_API_KEY=your_key_here

# Required for Discord pick notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN

# Optional — separate channels per pass type
DISCORD_WEBHOOK_MORNING=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_FINAL=https://discord.com/api/webhooks/...
```

---

## One-time: build training data + train model

### 1. Build features (multi-season, 2021–2026)

```bash
python -m src.features.build_features_multi_season
```

Builds month-by-month, skipping already-built months. First run takes 20–40 min due to MLB API fetches (cached after first run). Saves `data/processed/train_table_2021_2026_full.parquet`.

> Set `TEST_MODE = True` at the top of `build_features_multi_season.py` to build just one month first and verify the pipeline.

### 2. Train the model

```bash
python -m src.model.train
```

Saves a calibrated LightGBM bundle to `models/`.

---

## Daily usage (automated)

### Scheduler commands

```bash
# Show today's game waves and scheduled pass times
python -m src.scheduler --show-waves

# Run the morning pass manually
python -m src.scheduler --pass morning

# Check if any final passes are due now and fire them
python -m src.scheduler --auto-final

# See today's run history
python -m src.scheduler --status

# Dry-run without executing or posting to Discord
python -m src.scheduler --auto-final --dry-run
```

### Manual predict (any date)

```bash
python -m src.model.predict --date 2026-04-01
```

---

## Automated deployment (Hetzner VPS)

The project runs fully automated on a $5/mo Linux VPS — no laptop required.

### Server setup

```bash
# On the server
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git

git clone https://github.com/YOUR_USERNAME/mlb-hr-model.git
cd mlb-hr-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy your `.env` from your laptop:
```bash
# On your laptop
scp .env root@YOUR_SERVER_IP:/root/mlb-hr-model/.env
```

### Cron jobs (Linux)

```bash
crontab -e
```

Add the following. Note the `ODDS_API_KEY=` line at the top — cron doesn't source `.env`, so the key must be injected inline (or use `env -i … source .env …` in each job).

```
# Odds API key — cron doesn't source .env, inject here
ODDS_API_KEY=your_key_here

# Nightly retrain — 4:00 AM ET (refreshes isotonic recalibration with live outcomes)
0 8 * * * cd /root/mlb-hr-model && /root/mlb-hr-model/.venv/bin/python -m src.model.train >> /root/mlb-hr-model/cron.log 2>&1

# Morning pass — 10:00 AM ET (UTC-4 in summer)
0 14 * * * cd /root/mlb-hr-model && /root/mlb-hr-model/.venv/bin/python -m src.scheduler --pass morning >> /root/mlb-hr-model/cron.log 2>&1

# Auto final — every 30 min, all day. Fires each wave 90 min before first pitch.
# All 24 hours needed because late games (9+ PM ET) fire their pass after midnight UTC.
*/30 * * * * cd /root/mlb-hr-model && /root/mlb-hr-model/.venv/bin/python -m src.scheduler --auto-final >> /root/mlb-hr-model/cron.log 2>&1
```

> **Note:** Server runs UTC. 10 AM ET = 14:00 UTC in summer (EDT). Change `14` → `15` when clocks fall back in November.

### Verify it's working

```bash
tail -f /root/mlb-hr-model/cron.log
python -m src.scheduler --status
```

---

## Weekly recap

Review model performance vs actual outcomes:

```bash
# Last 7 days
python -m src.analysis.weekly_recap

# Specific week
python -m src.analysis.weekly_recap --week 2026-04-07

# All-time summary
python -m src.analysis.weekly_recap --all-time

# Post to Discord
python -m src.analysis.weekly_recap --discord

# Save to CSV
python -m src.analysis.weekly_recap --save
```

---

## Typical daily schedule

| Time (ET)   | Event                                      |
|-------------|--------------------------------------------|
| 10:00 AM    | Morning pass — probable starters, early lines |
| ~10:30 AM   | Final pass for noon game wave              |
| ~2:30 PM    | Final pass for afternoon game wave         |
| ~5:30 PM    | Final pass for evening game wave           |
| ~8:30 PM    | Final pass for late game wave              |

Each final pass posts picks to Discord ~90 minutes before first pitch.

---

## Discord pick format

Each pick is posted as a rich embed:

```
#1  Aaron Judge — HR Over
┌─────────────────────────────────────────────────────┐
│ vs Pitcher     Gerrit Cole        Batting Slot  3   │
│ Model Prob     14.2%              Market Fair  9.8% │
│ Edge           +4.4pp             Suggested    1.0u │
│ Best Price     +115               Book         DK   │
│ Other Books    Need +105 or better (break-even -108)│
└─────────────────────────────────────────────────────┘
```

Unit sizing is derived from fractional Kelly:

| Kelly fraction | Units |
|----------------|-------|
| < 0.5%         | skip  |
| 0.5–1.5%       | 0.5u  |
| 1.5–3.0%       | 1.0u  |
| 3.0–5.0%       | 1.5u  |
| 5.0–7.0%       | 2.0u  |
| ≥ 7.0%         | 3.0u  |

---

## Bet quality gates

Calibration analysis on live 2026 data showed the model overshoots in the top probability bucket and that huge "edges" are usually calibration noise rather than real mispricings. Four filters gate every bet:

| Gate | Value | Rationale |
|------|-------|-----------|
| `MAX_PRED_PROB` | 0.20 | Empirical ceiling — top calibration bucket converts at ~20% actual |
| `MIN_BET_PROB`  | 0.10 | Below the MLB base rate, positive edge is usually noise |
| `MIN_REL_EDGE`  | 0.30 | Edge must be ≥30% of fair prob (keeps quality constant across buckets) |
| `MAX_BET_EDGE`  | 0.10 | Edges > +10pp are typically model errors, not market mispricings |

All four live in [src/model/predict.py](src/model/predict.py) as module constants — adjust there and both the terminal output and Discord embeds will follow.

---

## Operating costs

See [docs/costs.md](docs/costs.md) for VPS + Odds API credit math, free-tier
headroom, and the upgrade break-even rule.

---

## Key files

| File | Purpose |
|------|---------|
| `src/scheduler.py` | Main entry point — game-aware pass timing |
| `src/model/predict.py` | Scores batters, applies odds, posts to Discord |
| `src/model/train.py` | Trains and calibrates the LightGBM model |
| `src/features/build_features.py` | Main feature pipeline — add new features here |
| `src/features/build_features_multi_season.py` | Multi-season build orchestration + TEST_MODE |
| `src/data_sources/odds.py` | Fetches HR prop lines from The Odds API |
| `src/notifications/discord.py` | Formats and sends Discord pick embeds |
| `src/analysis/weekly_recap.py` | Weekly P&L and calibration report |
| `monthly_retrain.ps1` | Monthly retrain automation script (run locally) |
| `mlb_hr_model_feature_reference.docx` | Full feature inventory and architecture guide |
| `data/predictions/run_log.csv` | History of every pass run |

---

## Model maintenance

The model runs on three layers of automation:

| Layer | Frequency | How | What it does |
|-------|-----------|-----|--------------|
| Isotonic recalibration | Nightly (4 AM ET) | VPS cron | Retrains calibration layer using accumulated 2026 prediction outcomes. Fast (~30s), no Statcast download |
| Current season refresh | Monthly | `monthly_retrain.ps1` (local) | Pulls fresh prediction CSVs from VPS, rebuilds current season Statcast months, full retrain, deploys new model |
| Full rebuild | Mid-season + offseason | `monthly_retrain.ps1 -FullRebuild` (local) | Rebuilds all seasons (2021–present), full retrain |

### Monthly retrain (run from your laptop)

```powershell
# Standard monthly refresh — rebuilds current season months only (~20-40 min)
.\monthly_retrain.ps1

# Preview what would happen without making changes
.\monthly_retrain.ps1 -DryRun

# Full 5-season rebuild — use at All-Star break and offseason (~3-4 hours)
.\monthly_retrain.ps1 -FullRebuild
```

The script automatically:
1. Pulls latest prediction CSVs from VPS (for isotonic recalibration)
2. Deletes stale feature parquets for the current season
3. Rebuilds feature tables with fresh Statcast data
4. Retrains the model
5. Commits the new `.joblib`, pushes to GitHub, deploys to VPS

> **First-time setup:** Update `$VPS_HOST` at the top of `monthly_retrain.ps1` with your server IP.

### Recommended cadence

- **Every ~4 weeks** — run `.\monthly_retrain.ps1` to keep rolling windows fresh
- **All-Star break (July)** — run `.\monthly_retrain.ps1 -FullRebuild` for a full half-season of in-season data
- **Offseason (October)** — add new season year to `SEASONS` in `build_features_multi_season.py`, then full rebuild

### Adding new features

See `mlb_hr_model_feature_reference.docx` for the full feature inventory and step-by-step instructions for adding new features while keeping all files consistent.

Current model: **184 features · ROC-AUC 0.669 · 3.03x top-1% lift**

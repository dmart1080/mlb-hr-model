# mlb-hr-model

MLB home run prediction model using Statcast data. Given a batter–pitcher matchup and game context, the model outputs the probability that the batter hits at least one home run in that game.

## Results (2024 season, single-season baseline)

| Metric | Value |
|---|---|
| ROC-AUC | 0.623 |
| Log loss | 0.323 |
| Baseline HR rate | 10.1% |
| Top 10% bucket HR rate | 15.1% (1.49× baseline) |
| Top 1% bucket HR rate | 22.2% (2.19× baseline) |

Model: LightGBM + sigmoid calibration. Trained on 2021–2024, tested on 2025 (out-of-time) once the multi-season data build completes. Results above are from the 2024 single-season baseline; multi-season results will be updated once the full build runs.

---

## How it works

Each row in the training table represents one batter–game opportunity. The label is `hr_hit = 1` if the batter hit at least one home run in that game.

**Only regular season games are used** (`game_type = "R"`). Spring training, postseason, All-Star, and exhibition games are excluded at the Statcast fetch level and double-filtered in the feature builder.

Features are computed from Statcast pitch/event data using rolling windows calculated strictly before the game date (no leakage):

### Batter features

| Group | Features |
|---|---|
| **14-day rolling** | PA count, HR rate, barrel rate, EV mean, LA mean, hard-hit rate, FB rate, K rate, BB rate |
| **14-day platoon** | HR rate, barrel rate, hard-hit rate vs LHP and vs RHP separately |
| **Season-to-date** | Same contact-quality set as 14-day |
| **Season platoon** | HR rate, barrel rate, hard-hit rate vs LHP and vs RHP |
| **7-day trend** | EV trend, hard-hit trend, barrel trend, HR trend (7d minus prior 7d) |
| **Home/away splits** | HR rate, hard-hit rate, barrel rate at home vs away (season) |
| **Context** | `is_home_game`, `same_hand_matchup` (batter hand vs pitcher hand) |
| **Lineup** | `batting_order_pos` (1–9), `is_top_of_order` (slots 1–4), `expected_pa_today` |
| **Rest** | `b_days_rest` (days since last game) |

### Pitcher features

All pitcher features are computed against the **actual starting pitcher** resolved via the MLB Stats API live feed, not the mode-pitcher proxy.

| Group | Features |
|---|---|
| **30-day rolling** | PA faced, HR allowed rate, EV allowed mean, hard-hit allowed rate, FB allowed rate, barrel allowed rate, K rate, BB rate |
| **30-day platoon** | HR allowed rate, hard-hit allowed rate, barrel allowed rate vs LHB and vs RHB |
| **Season-to-date** | Same set as 30-day |
| **Season platoon** | HR allowed rate, hard-hit allowed rate, barrel allowed rate vs LHB and vs RHB |
| **Velo (30-day)** | Fastball velocity mean, fastball usage %, offspeed usage % |
| **Velo trend** | FB velo change (recent 3 starts vs prior 3 starts) |
| **Rest** | `p_days_rest`, `p_is_short_rest` (≤3 days) |

### Edge / interaction features

Batter 14-day minus pitcher 30-day for EV, hard-hit rate, FB rate, barrel rate, HR rate, K rate, BB rate. Also: platoon-split edges (vsL, vsR), K-rate interaction, BB-rate interaction, contact pressure composite, discipline balance.

### Context features

| Feature | Description |
|---|---|
| `park_factor_hr` | HR park factor fetched from MLB Stats API per season (cached 30 days; falls back to static 2024 table if API returns fewer than 20 teams or missing `parkFactor` field) |
| `temp_f`, `wind_hr_impact` | Game-day temperature and wind impact on HR (direction × speed) |
| `is_indoor` | 1 for domed stadiums (MIA, HOU, SEA, ARI, MIL, TOR, TB); wind and temp forced to neutral |
| `temp_above_75`, `temp_above_85`, `wind_out_strong`, `wind_in_strong` | Binary weather flags |
| `relief_pa_pct` | Fraction of in-game PAs taken against non-starters; discounts starter-based matchup features |

### Cold-start handling

All rate features use **empirical Bayes shrinkage** toward league-average priors so low-PA players regress to the mean rather than showing misleading zeros or extreme rates:

- Batter prior: 50 PA
- Pitcher prior: 75 PA
- League rates: HR 3.3%, barrel 7.3%, hard-hit 38.0%, FB 35.0%, K 22.7%, BB 8.3%

Shrinkage is applied at both **training time** (in `train.py`) and **inference time** (in `predict.py` when `apply_shrinkage=True` is stored in the model bundle).

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
│   └── predictions/            # Daily prediction CSVs: predictions_YYYY-MM-DD.csv
├── models/                     # Saved .joblib bundles + train_runs.csv log (gitignored)
├── src/
│   ├── logging_config.py           # Centralised logging setup
│   ├── data_sources/
│   │   ├── statcast.py             # Statcast fetch + disk cache + RS filter
│   │   ├── weather.py              # MLB Stats API weather fetch + cache
│   │   ├── mlb_schedule.py         # MLB Stats API: probable/actual starters + batting orders
│   │   └── odds.py                 # The Odds API: HR prop lines + edge calculation
│   ├── features/
│   │   ├── build_labels.py             # Collapse events → batter-game labels
│   │   ├── build_features_common.py    # Shared constants + pure helpers (base layer)
│   │   ├── build_features_fast.py      # Vectorised rolling window computation
│   │   ├── build_features.py           # Main feature pipeline (calls fast + enrichments)
│   │   ├── build_features_season.py    # Single-season build CLI
│   │   ├── build_features_multi_season.py  # Multi-season build (2021–2025)
│   │   └── park_factors.py             # HR park factor fetch + static fallback
│   └── model/
│       ├── train.py        # Train + calibrate + save model
│       └── predict.py      # Load model, score latest date, print ranked list + save CSV
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install pybaseball lightgbm scikit-learn pandas numpy joblib pyarrow \
            requests python-dotenv rapidfuzz
```

Optional — for odds/edge detection:
```bash
export ODDS_API_KEY=your_key_here   # get a free key at https://the-odds-api.com
# or add to a .env file in the project root:
echo "ODDS_API_KEY=your_key_here" >> .env
```

---

## Usage

### 1. Build features

**Multi-season (recommended, resume-safe):**
```bash
python -m src.features.build_features_multi_season
```
Builds month-by-month for 2021–2025. Already-built months are skipped automatically. First run takes 20–40 min due to MLB API roster fetches (cached after first run). Saves `data/processed/train_table_2021_2025_full.parquet`.

> **Test mode:** `build_features_multi_season.py` has a `TEST_MODE = True` flag at the top. In test mode it builds only one month of 2025 data so you can verify the full pipeline quickly before committing to a full rebuild. Set `TEST_MODE = False` to build all seasons.

**Single season (faster, for development):**
```bash
python -m src.features.build_features_season build --year 2024
python -m src.features.build_features_season build --year 2024 --debug          # verbose
python -m src.features.build_features_season build --year 2024 --log-file b.log # log to file

# Combine multiple seasons into one table:
python -m src.features.build_features_season combine --years 2022 2023 2024
```

### 2. Train

```bash
python -m src.model.train
```

Uses a **hard calendar split**: train = everything before 2025-03-27, test = 2025 season (true out-of-time validation). Falls back to an 80/20 percentage split if no 2025 data is present yet.

- Trains both LightGBM and Logistic Regression, keeps whichever scores higher on raw ROC-AUC
- Calibrates with Platt scaling (sigmoid) on a held-out chronological slice
- Prints a full summary and appends a row to `models/train_runs.csv`
- Saves model bundle to `models/hr_model_<name>_2021_2025.joblib`

### 3. Predict

```bash
python -m src.model.predict
```

Scores all batter–pitcher matchups on the latest date in the feature table. Prints the top 20 HR candidates.

If `ODDS_API_KEY` is set, also fetches today's HR prop lines from The Odds API and shows:
- Market over price (American odds)
- Fair probability (vig-removed)
- Edge = model probability − fair market probability

Saves predictions to `data/predictions/predictions_YYYY-MM-DD.csv`.

---

## Cleaning data cache (for testing / forcing fresh downloads)

Use these commands when testing new code that changes how features are computed or when you need to force a full re-download.

Each section shows both **PowerShell** (Windows) and **bash** (macOS/Linux) versions. The `python` commands are the same on both platforms.

---

### Clear everything (nuclear)

Removes all cached data and processed feature tables.

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache, data\processed
```
```bash
# bash
rm -rf data/cache/ data/processed/
```

---

### Clear only Statcast cache

Forces a full re-download from Baseball Savant on the next run.

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\statcast

# or a specific date range only:
Remove-Item -ErrorAction SilentlyContinue data\cache\statcast\statcast_2024-03-20_to_2024-10-01.parquet
```
```bash
# bash
rm -rf data/cache/statcast/

# or a specific date range only:
rm data/cache/statcast/statcast_2024-03-20_to_2024-10-01.parquet
```

> **Note:** The Statcast cache now stores `game_type` so the regular-season filter works correctly. Old cache files missing this column are detected automatically on the next run and re-downloaded. You can also force it manually with the commands above.

---

### Clear roster / batting order cache

Forces a re-fetch of actual starters and batting orders from the MLB Stats API.

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\schedule

# or a specific game:
Remove-Item -ErrorAction SilentlyContinue data\cache\schedule\game_745456.json
```
```bash
# bash
rm -rf data/cache/schedule/
rm data/cache/schedule/game_745456.json
```

---

### Clear weather cache

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\weather
```
```bash
# bash
rm -rf data/cache/weather/
```

---

### Clear park factors cache

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\park_factors
```
```bash
# bash
rm -rf data/cache/park_factors/
```

---

### Clear odds cache

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\odds
```
```bash
# bash
rm -rf data/cache/odds/
```

---

### Clear only processed feature tables

Keeps the raw Statcast cache (no re-download needed) but forces feature recomputation.

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\processed

# or a specific month:
Remove-Item -ErrorAction SilentlyContinue data\processed\train_table_2024-04-01_to_2024-04-30.parquet
```
```bash
# bash
rm -rf data/processed/

# or a specific month:
rm data/processed/train_table_2024-04-01_to_2024-04-30.parquet
```

---

### Clear models

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue models
```
```bash
# bash
rm -rf models/
```

---

### Recommended workflow: testing new feature code

When you change feature logic (rolling windows, edge features, new columns) — clear processed tables but keep the Statcast cache to avoid re-downloading ~GB of data:

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\processed
python -m src.features.build_features_multi_season
python -m src.model.train
```
```bash
# bash
rm -rf data/processed/
python -m src.features.build_features_multi_season
python -m src.model.train
```

---

### Recommended workflow: testing new data pipeline code

When you change how Statcast data is fetched, cleaned, or filtered (e.g. the regular-season `game_type` filter):

```powershell
# PowerShell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue data\cache\statcast, data\processed
python -m src.features.build_features_multi_season
python -m src.model.train
```
```bash
# bash
rm -rf data/cache/statcast/ data/processed/
python -m src.features.build_features_multi_season
python -m src.model.train
```

---

## Data

All pitch/event data is sourced from [Baseball Savant](https://baseballsavant.mlb.com/) via the [pybaseball](https://github.com/jldbc/pybaseball) library. Starter and batting order data is fetched from the [MLB Stats API](https://statsapi.mlb.com). HR prop odds are fetched from [The Odds API](https://the-odds-api.com). All raw data is cached locally — re-runs do not re-download unless caches are cleared.

`data/` and `models/` are gitignored and should never be committed.

---

## Model details

Two models are trained and compared on raw ROC-AUC; the better one is kept:

- **Logistic Regression** — StandardScaler → L2-regularized LR with `class_weight="balanced"` to handle the ~10% HR base rate
- **LightGBM** — `num_leaves=31`, `min_child_samples=300`, `reg_lambda=10.0`, `reg_alpha=0.1`, `subsample=0.8`, `colsample_bytree=0.6`, early stopping on AUC with a calibration holdout

Both are calibrated with **Platt scaling** (sigmoid) on a held-out chronological calibration slice (last 20% of the training period). The final model bundle is saved to `models/hr_model_<name>_2021_2025.joblib` and includes:
- `model`: calibrated classifier
- `feature_cols`: ordered list of feature column names
- `apply_shrinkage`: bool flag — when True, `predict.py` applies empirical Bayes shrinkage before scoring

### Barrel approximation

The Statcast barrel definition is approximated directly from launch speed and angle since the raw `barrel` column is not reliably populated in all pybaseball versions:

- EV ≥ 98 mph AND launch angle within a widening range:
  - At 98 mph: LA 26°–30°
  - Each additional mph widens the range by ±1° (e.g. 103 mph → 21°–35°)
  - Capped at 108 mph: LA 16°–41°

This matches the [Baseball Savant barrel definition](https://www.mlb.com/glossary/statcast/barrels).

---

## Known limitations

- Pitcher features are only as good as starter resolution. If the MLB Stats API returns no starter for a game, the model falls back to `pitcher_mode` (most-common pitcher faced), which may be a reliever in bulk-usage games.
- Weather data is pulled from the MLB Stats API game feed and may not be available for games that haven't started yet. Pre-game weather uses neutral values (72°F, 0 mph wind).
- The odds integration requires a paid Odds API key; the free tier (500 requests/month) covers roughly one full day's slate per day.
- `relief_pa_pct` is computed post-game and is 0.0 at prediction time (before the game starts). It is included as a training signal but not meaningful for live predictions.

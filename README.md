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

Model: LightGBM + sigmoid calibration. Trained on 2021–2024, tested on 2025 (out-of-time) once the multi-season data build completes.

---

## How it works

Each row in the training table represents one batter–game opportunity. The label is `hr_hit = 1` if the batter hit at least one home run in that game.

Features are computed from Statcast pitch/event data using rolling windows calculated strictly before the game date (no leakage):

- **Batter (14-day):** PA count, HR rate, barrel rate, EV mean, launch angle mean, hard-hit rate, FB rate, K rate, BB rate
- **Batter (season-to-date):** same set of contact-quality metrics
- **Pitcher allowed (30-day):** PA count, HR allowed rate, EV allowed mean, hard-hit allowed rate, FB allowed rate, barrel allowed rate, K rate, BB rate
- **Pitcher allowed (season-to-date):** same set
- **Edge features:** batter 14d minus pitcher 30d for EV, hard-hit rate, FB rate, barrel rate, HR rate, K rate, BB rate — plus interaction terms and a contact pressure composite
- **Park factor:** HR park factor on a 100 = neutral scale (per ballpark)

---

## Project structure

```
mlb-hr-model/
├── data/
│   ├── cache/              # Raw Statcast parquets (auto-created, gitignored)
│   └── processed/          # Feature tables and train tables
├── models/                 # Saved .joblib model files + train_runs.csv log
├── src/
│   ├── data_sources/
│   │   └── statcast.py     # Statcast fetch + disk cache
│   ├── features/
│   │   ├── build_labels.py              # Collapse events → batter-game labels
│   │   ├── build_features.py            # Rolling feature computation
│   │   ├── build_features_season_2024.py  # Single-season build (legacy)
│   │   ├── build_features_multi_season.py # Multi-season build (2021–2025)
│   │   └── park_factors.py              # HR park factor lookup table
│   └── model/
│       ├── train.py        # Train + calibrate + save model
│       └── predict.py      # Load model, score latest date, print ranked list
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install pybaseball lightgbm scikit-learn pandas numpy joblib pyarrow
```

---

## Usage

### 1. Build features

**First run (multi-season, recommended):**
```bash
python -m src.features.build_features_multi_season
```
Builds month-by-month for 2021–2025. Resume-safe — already-built months are skipped. Expect 20–40 min on first run. Saves `data/processed/train_table_2021_2025_full.parquet`.

**Single season (faster, for development):**
```bash
python -m src.features.build_features_season_2024
```

### 2. Train

```bash
python -m src.model.train
```

Uses a **hard calendar split**: train = everything before 2025-03-27, test = 2025 season (true out-of-time validation). Falls back to a 80/20 percentage split if no 2025 data is present yet.

Prints a full summary and appends a row to `models/train_runs.csv`.

### 3. Predict

```bash
python -m src.model.predict
```

Scores all batter–pitcher matchups on the latest date in the feature table and prints the top 15 HR candidates ranked by predicted probability.

---

## Data

All data is sourced from [Baseball Savant](https://baseballsavant.mlb.com/) via the [pybaseball](https://github.com/jldbc/pybaseball) library. Raw Statcast event data is cached locally in `data/cache/` as parquet files keyed by date range — re-runs do not re-download.

`data/` and `models/` should be added to `.gitignore`.

---

## Model details

Two models are trained and compared on raw ROC-AUC; the better one is kept:

- **Logistic Regression** — StandardScaler → L2-regularized LR with `class_weight="balanced"` to handle the ~10% HR base rate
- **LightGBM** — shallow trees (`num_leaves=15`), heavy regularization (`reg_lambda=5`, `min_child_samples=200`), subsampling to reduce overfitting on a noisy label

Both are then calibrated with **Platt scaling** (sigmoid) on a held-out chronological calibration slice (last 20% of the train period). The final model is saved to `models/hr_model_<name>_2021_2025.joblib`.

The barrel approximation used in features is `EV ≥ 95 mph AND launch angle 20°–35°`, derived directly from Statcast pitch data since the raw `barrel` column is not always populated.

---

## Known limitations

- **Pitcher assignment:** uses the mode pitcher faced per batter-game as a matchup proxy. Doesn't model bullpen usage or handedness splits.
- **No lineup context:** doesn't model batting order position or lineup protection.
- **Park factors:** static 2024 values. Not updated intra-season.
- **Cold-start:** batters/pitchers with fewer than ~10 PA in the rolling window get `0.0` for rate features. Model compensates via season-to-date fallbacks but early-season predictions are noisier.

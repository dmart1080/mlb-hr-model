from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
from xml.parsers.expat import model
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.frozen import FrozenEstimator
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hard calendar cut-point: train on everything before this date, test on/after
# ---------------------------------------------------------------------------
TEST_START_DATE = "2025-03-27"


def pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def f3(x: float) -> str:
    return f"{x:.3f}"

def f2(x: float) -> str:
    return f"{x:.2f}"

def fmt_int(x: int) -> str:
    return f"{x:,}"

from datetime import datetime
import csv

def print_summary(train_path: Path, model_path: Path, feature_cols: list[str], metrics: dict, extra: dict):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = model_path.name

    print("\n" + "=" * 30)
    print("MLB HR MODEL — TRAIN SUMMARY")
    print("=" * 30)
    print(f"Run time: {run_time}")
    print(f"Training table: {train_path.name}\n")

    print(f"Date range: train={metrics['train_start']}→{metrics['train_end']}  "
          f"test={metrics['test_start']}→{metrics['test_end']}\n")

    print(f"Rows: train={fmt_int(metrics['train_rows'])}  "
          f"test={fmt_int(metrics['test_rows'])}")
    print(f"Test HR rate (baseline): {pct(metrics['test_hr_rate'])}\n")

    print("Performance (test):")
    print(f"  ROC-AUC:   {f3(metrics['roc_auc'])}")
    print(f"  Log loss:  {f3(metrics['log_loss'])}")
    print(f"  Avg pred:  {pct(metrics['avg_pred_prob'])}")
    print(f"  Max pred:  {f3(extra['max_pred_prob'])}\n")

    print("Lift checks (test):")
    print(f"  Top 10% HR rate: {pct(extra['top10_hr_rate'])}  "
          f"({extra['top10_lift']:.2f}x baseline)")
    print(f"  Top 1%  HR rate: {pct(extra['top1_hr_rate'])}  "
          f"({extra['top1_lift']:.2f}x baseline)  "
          f"(n={fmt_int(extra['top1_count'])})")
    print(f"  Top 1%  avg b_pa_14: {f2(extra['top1_avg_b_pa_14'])}")
    print(f"  Top 1%  avg p_pa_30: {f2(extra['top1_avg_p_pa_30'])}\n")

    print(f"Calibration delta: {extra['avg_minus_base_pp']:+.2f} pp")

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Model saved: {model_name}")
    print("=" * 30 + "\n")

    # -----------------------
    # CSV Run Logging
    # -----------------------
    log_file = MODELS_DIR / "train_runs.csv"
    write_header = not log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "run_time",
                "train_table",
                "roc_auc",
                "log_loss",
                "baseline_hr_rate",
                "top10_hr_rate",
                "top1_hr_rate",
                "top10_lift",
                "top1_lift",
                "features",
            ])

        writer.writerow([
            run_time,
            train_path.name,
            metrics["roc_auc"],
            metrics["log_loss"],
            metrics["test_hr_rate"],
            extra["top10_hr_rate"],
            extra["top1_hr_rate"],
            extra["top10_lift"],
            extra["top1_lift"],
            len(feature_cols),
        ])

@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]
    extra: dict


def latest_train_table() -> Path:
    """
    Preference order:
      1. 2021-2025 multi-season file (produced by build_features_multi_season.py)
      2. 2024 full-season file (legacy)
      3. Most-recently-modified train_table_*.parquet
    """
    multi = PROCESSED_DIR / "train_table_2021_2025_full.parquet"
    if multi.exists():
        return multi

    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season

    files = sorted(
        PROCESSED_DIR.glob("train_table_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed/")
    return files[0]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def calendar_split(
    df: pd.DataFrame,
    test_start: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cut = pd.Timestamp(test_start)
    train = df[df["game_date"] < cut].sort_values("game_date").reset_index(drop=True)
    test  = df[df["game_date"] >= cut].sort_values("game_date").reset_index(drop=True)
    return train, test


def pct_time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Empirical Bayes shrinkage
# ---------------------------------------------------------------------------
# Prior PA counts — how many "league-average" PAs to blend in before
# trusting a player's observed rate.  Higher = more aggressive shrinkage
# for cold-start players.
#
# Formula: shrunk_rate = (observed_events + prior_pa * league_avg_rate)
#                        / (observed_pa + prior_pa)
#
# This prevents a player with 2 PA and 1 HR showing up as a 50% HR rate.
# ---------------------------------------------------------------------------
_BATTER_PRIOR_PA  = 50   # stronger prior for batters (more noisy at short windows)
_PITCHER_PRIOR_PA = 75   # pitchers face more batters; signal stabilises quicker
_LEAGUE_RATES = {
    "hr":      0.033,   # ~3.3% HR rate per PA league-wide
    "barrel":  0.073,   # ~7.3% barrel rate
    "hardhit": 0.380,   # ~38% hard-hit rate (EV ≥ 95)
    "fb":      0.350,   # ~35% fly-ball rate (LA 20-40°)
    "k":       0.227,   # ~22.7% strikeout rate
    "bb":      0.083,   # ~8.3% walk rate
}


def _shrink_rate(
    rate_col: pd.Series,
    pa_col: pd.Series,
    league_rate: float,
    prior_pa: int,
) -> pd.Series:
    """
    Apply empirical Bayes shrinkage to a rate column.

    rate_col   : observed rate (NaN where pa < threshold)
    pa_col     : observed PA count
    league_rate: league-average rate for this metric
    prior_pa   : strength of the prior in PA units
    """
    # Reconstruct observed event count from rate × PA
    # Where rate is NaN (cold-start / not enough PA), treat as 0 observed events
    obs_pa     = pa_col.fillna(0).clip(lower=0)
    obs_events = rate_col.fillna(0) * obs_pa
    shrunk = (obs_events + prior_pa * league_rate) / (obs_pa + prior_pa)
    return shrunk


def apply_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw rate columns with empirical-Bayes-shrunk versions.

    Called on the full DataFrame before training so both train and test
    sets are shrunk consistently.

    Only replaces columns that are present — safe to call on legacy tables
    that are missing some columns.
    """
    df = df.copy()

    # ── Batter 14-day window ────────────────────────────────────────────────
    batter_14_pairs = [
        ("b_hr_rate_14",      "b_pa_14",  "hr",      _BATTER_PRIOR_PA),
        ("b_barrel_rate_14",  "b_pa_14",  "barrel",  _BATTER_PRIOR_PA),
        ("b_hardhit_rate_14", "b_pa_14",  "hardhit", _BATTER_PRIOR_PA),
        ("b_fb_rate_14",      "b_pa_14",  "fb",      _BATTER_PRIOR_PA),
        ("b_k_rate_14",       "b_pa_14",  "k",       _BATTER_PRIOR_PA),
        ("b_bb_rate_14",      "b_pa_14",  "bb",      _BATTER_PRIOR_PA),
    ]
    for rate_col, pa_col, metric, prior in batter_14_pairs:
        if rate_col in df.columns and pa_col in df.columns:
            df[rate_col] = _shrink_rate(
                df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior
            )

    # ── Batter season window ────────────────────────────────────────────────
    batter_szn_pairs = [
        ("b_hr_rate_szn",      "b_pa_szn",  "hr",      _BATTER_PRIOR_PA),
        ("b_barrel_rate_szn",  "b_pa_szn",  "barrel",  _BATTER_PRIOR_PA),
        ("b_hardhit_rate_szn", "b_pa_szn",  "hardhit", _BATTER_PRIOR_PA),
        ("b_fb_rate_szn",      "b_pa_szn",  "fb",      _BATTER_PRIOR_PA),
        ("b_k_rate_szn",       "b_pa_szn",  "k",       _BATTER_PRIOR_PA),
        ("b_bb_rate_szn",      "b_pa_szn",  "bb",      _BATTER_PRIOR_PA),
    ]
    for rate_col, pa_col, metric, prior in batter_szn_pairs:
        if rate_col in df.columns and pa_col in df.columns:
            df[rate_col] = _shrink_rate(
                df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior
            )

    # ── Pitcher 30-day window ───────────────────────────────────────────────
    pitcher_30_pairs = [
        ("p_hr_allowed_rate_30",      "p_pa_30",  "hr",      _PITCHER_PRIOR_PA),
        ("p_barrel_allowed_rate_30",  "p_pa_30",  "barrel",  _PITCHER_PRIOR_PA),
        ("p_hardhit_allowed_rate_30", "p_pa_30",  "hardhit", _PITCHER_PRIOR_PA),
        ("p_fb_allowed_rate_30",      "p_pa_30",  "fb",      _PITCHER_PRIOR_PA),
        ("p_k_rate_30",               "p_pa_30",  "k",       _PITCHER_PRIOR_PA),
        ("p_bb_rate_30",              "p_pa_30",  "bb",      _PITCHER_PRIOR_PA),
    ]
    for rate_col, pa_col, metric, prior in pitcher_30_pairs:
        if rate_col in df.columns and pa_col in df.columns:
            df[rate_col] = _shrink_rate(
                df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior
            )

    # ── Pitcher season window ───────────────────────────────────────────────
    pitcher_szn_pairs = [
        ("p_hr_allowed_rate_szn",      "p_pa_szn",  "hr",      _PITCHER_PRIOR_PA),
        ("p_barrel_allowed_rate_szn",  "p_pa_szn",  "barrel",  _PITCHER_PRIOR_PA),
        ("p_hardhit_allowed_rate_szn", "p_pa_szn",  "hardhit", _PITCHER_PRIOR_PA),
        ("p_fb_allowed_rate_szn",      "p_pa_szn",  "fb",      _PITCHER_PRIOR_PA),
        ("p_k_rate_szn",               "p_pa_szn",  "k",       _PITCHER_PRIOR_PA),
        ("p_bb_rate_szn",              "p_pa_szn",  "bb",      _PITCHER_PRIOR_PA),
    ]
    for rate_col, pa_col, metric, prior in pitcher_szn_pairs:
        if rate_col in df.columns and pa_col in df.columns:
            df[rate_col] = _shrink_rate(
                df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior
            )

    # ── Re-derive edge features from shrunk rates ───────────────────────────
    # The edge features in the table were computed from raw rates at build
    # time. Now that we've shrunk the underlying rates, we recompute the
    # edges so they reflect the corrected signal rather than noisy raw diffs.
    def _edge(a: str, b: str) -> pd.Series:
        if a in df.columns and b in df.columns:
            return df[a] - df[b]
        return pd.Series(np.nan, index=df.index)

    df["ev_edge_14_30"]      = _edge("b_ev_mean_14",      "p_ev_allowed_mean_30")
    df["hardhit_edge_14_30"] = _edge("b_hardhit_rate_14", "p_hardhit_allowed_rate_30")
    df["fb_edge_14_30"]      = _edge("b_fb_rate_14",      "p_fb_allowed_rate_30")
    df["barrel_edge_14_30"]  = _edge("b_barrel_rate_14",  "p_barrel_allowed_rate_30")
    df["hr_rate_edge_14_30"] = _edge("b_hr_rate_14",      "p_hr_allowed_rate_30")
    df["k_rate_edge_14_30"]  = _edge("b_k_rate_14",       "p_k_rate_30")
    df["bb_rate_edge_14_30"] = _edge("b_bb_rate_14",      "p_bb_rate_30")

    if "b_k_rate_14" in df.columns and "p_k_rate_30" in df.columns:
        df["k_rate_interaction_14_30"]  = df["b_k_rate_14"]  * df["p_k_rate_30"]
        df["bb_rate_interaction_14_30"] = df["b_bb_rate_14"] * df["p_bb_rate_30"]
        df["contact_pressure_14_30"]    = (1 - df["b_k_rate_14"]) * (1 - df["p_k_rate_30"])
        df["discipline_balance_14_30"]  = (
            (df["b_bb_rate_14"] - df["b_k_rate_14"]) -
            (df["p_bb_rate_30"] - df["p_k_rate_30"])
        )

    for hand in ("L", "R"):
        b_hr   = f"b_hr_rate_14_vs{hand}"
        p_hr   = f"p_hr_allowed_rate_30_vs{hand}"
        b_hard = f"b_hardhit_rate_14_vs{hand}"
        p_hard = f"p_hardhit_allowed_rate_30_vs{hand}"
        b_bar  = f"b_barrel_rate_14_vs{hand}"
        p_bar  = f"p_barrel_allowed_rate_30_vs{hand}"
        if all(c in df.columns for c in [b_hr, p_hr, b_hard, p_hard, b_bar, p_bar]):
            df[f"hr_rate_edge_14_30_vs{hand}"]  = _edge(b_hr,   p_hr)
            df[f"hardhit_edge_14_30_vs{hand}"]  = _edge(b_hard, p_hard)
            df[f"barrel_edge_14_30_vs{hand}"]   = _edge(b_bar,  p_bar)

    return df


def train_baseline(train_path: Path) -> TrainResult:
    df = pd.read_parquet(train_path)

    # ------------------------------------------------------------------
    # Apply empirical Bayes shrinkage before anything else.
    # This corrects cold-start rate inflation and re-derives edge features
    # from the corrected rates.
    # ------------------------------------------------------------------
    print("Applying empirical Bayes shrinkage to rate features ...")
    df = apply_shrinkage(df)

    # ------------------------------------------------------------------
    # Feature columns
    #
    # KEY CHANGE: PA count columns (b_pa_14, b_pa_szn, p_pa_30, p_pa_szn)
    # are intentionally EXCLUDED from the model features.
    #
    # Why: PA counts are a proxy for "is this player a full-time starter"
    # which correlates with skill but is not a matchup-specific edge.
    # Including them caused the model to rank players by playing time
    # (AUC 0.623) rather than by true HR talent in this matchup.
    #
    # After shrinkage all rate features already encode data reliability
    # (sparse windows get shrunk toward league average), so PA counts
    # carry no additional information the model needs.
    #
    # Lineup context features (batting_order_pos, expected_pa_today)
    # replace PA counts as the "opportunity" signal — they are matchup-
    # specific and directly encode how many AB a player will get today.
    # ------------------------------------------------------------------
    feature_cols = [
        # ── Batter (14d) — RATE ONLY, no raw PA ────────────────────────
        "b_hr_rate_14",
        "b_barrel_rate_14",
        "b_ev_mean_14",
        "b_la_mean_14",
        "b_hardhit_rate_14",
        "b_fb_rate_14",
        "b_k_rate_14",
        "b_bb_rate_14",

        # ── Batter (7d trend) ────────────────────────────────────────────
        "b_ev_mean_7",
        "b_hardhit_rate_7",
        "b_ev_trend",
        "b_hardhit_trend",
        "b_barrel_trend",
        "b_hr_trend",

        # ── Batter (season) — RATE ONLY, no raw PA ──────────────────────
        "b_hr_rate_szn",
        "b_barrel_rate_szn",
        "b_ev_mean_szn",
        "b_la_mean_szn",
        "b_hardhit_rate_szn",
        "b_fb_rate_szn",

        # ── Batter platoon splits (14d) ──────────────────────────────────
        "b_hr_rate_14_vsL",
        "b_hr_rate_14_vsR",
        "b_hardhit_rate_14_vsL",
        "b_hardhit_rate_14_vsR",
        "b_barrel_rate_14_vsL",
        "b_barrel_rate_14_vsR",

        # ── Batter platoon splits (season) ───────────────────────────────
        "b_hr_rate_szn_vsL",
        "b_hr_rate_szn_vsR",
        "b_hardhit_rate_szn_vsL",
        "b_hardhit_rate_szn_vsR",
        "b_barrel_rate_szn_vsL",
        "b_barrel_rate_szn_vsR",

        # ── Batter home/away splits ──────────────────────────────────────
        "b_hr_rate_home",
        "b_hr_rate_away",
        "b_hr_rate_home_edge",
        "b_hardhit_rate_home",
        "b_hardhit_rate_away",
        "b_barrel_rate_home",
        "b_barrel_rate_away",

        # ── Game context ─────────────────────────────────────────────────
        "is_home_game",
        "same_hand_matchup",

        # ── Pitcher allowed (30d) — RATE ONLY, no raw PA ────────────────
        "p_hr_allowed_rate_30",
        "p_ev_allowed_mean_30",
        "p_hardhit_allowed_rate_30",
        "p_fb_allowed_rate_30",
        "p_barrel_allowed_rate_30",
        "p_k_rate_30",
        "p_bb_rate_30",

        # ── Pitcher platoon splits (30d) ─────────────────────────────────
        "p_hr_allowed_rate_30_vsL",
        "p_hr_allowed_rate_30_vsR",
        "p_hardhit_allowed_rate_30_vsL",
        "p_hardhit_allowed_rate_30_vsR",
        "p_barrel_allowed_rate_30_vsL",
        "p_barrel_allowed_rate_30_vsR",

        # ── Pitcher allowed (season) — RATE ONLY, no raw PA ─────────────
        "p_hr_allowed_rate_szn",
        "p_ev_allowed_mean_szn",
        "p_hardhit_allowed_rate_szn",
        "p_fb_allowed_rate_szn",
        "p_barrel_allowed_rate_szn",

        # ── Edge features (batter 14d vs pitcher 30d) ────────────────────
        "ev_edge_14_30",
        "hardhit_edge_14_30",
        "fb_edge_14_30",
        "barrel_edge_14_30",
        "hr_rate_edge_14_30",
        "k_rate_edge_14_30",
        "bb_rate_edge_14_30",
        "k_rate_interaction_14_30",
        "bb_rate_interaction_14_30",
        "contact_pressure_14_30",
        "discipline_balance_14_30",

        # ── Platoon-aware edge features ──────────────────────────────────
        "hr_rate_edge_14_30_vsL",
        "hr_rate_edge_14_30_vsR",
        "hardhit_edge_14_30_vsL",
        "hardhit_edge_14_30_vsR",
        "barrel_edge_14_30_vsL",
        "barrel_edge_14_30_vsR",

        # ── Weather ──────────────────────────────────────────────────────
        "temp_f",
        "wind_hr_impact",
        "wind_out_strong",
        "wind_in_strong",
        "temp_above_75",
        "temp_above_85",
        "is_indoor",

        # ── Park ─────────────────────────────────────────────────────────
        "park_factor_hr",

        # ── Lineup / opportunity context ─────────────────────────────────
        # These replace raw PA counts as the "how many AB will this player
        # get today" signal.  batting_order_pos encodes both opportunity
        # (slots 1-4 get ~1 extra AB over a 9-game vs slot 9) and
        # lineup quality (managers put their best hitters at the top).
        # expected_pa_today is a continuous version of the same signal.
        # relief_pa_pct discounts matchup features when the batter faced
        # mostly relievers rather than the listed starter.
        "batting_order_pos",
        "is_top_of_order",
        "expected_pa_today",
        "relief_pa_pct",

        # ── Pitcher velocity ─────────────────────────────────────────────
        "p_fb_velo_30",
        "p_fb_pct_30",
        "p_offspeed_pct_30",
        "p_fb_velo_trend",

        # ── Days rest ────────────────────────────────────────────────────
        "b_days_rest",
        "p_days_rest",
        "p_is_short_rest",
    ]

    # Filter to only columns that actually exist in the dataset
    available = set(df.columns)
    missing_features = [c for c in feature_cols if c not in available]
    if missing_features:
        print(f"⚠️  Dropping {len(missing_features)} feature(s) not found in table:")
        for c in missing_features:
            print(f"     - {c}")
    feature_cols = [c for c in feature_cols if c in available]

    required_cols = feature_cols + ["hr_hit", "game_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training table: {missing}")

    # ------------------------------------------------------------------
    # Calendar split: train=2021-2024, test=2025
    # ------------------------------------------------------------------
    train_df, test_df = calendar_split(df, test_start=TEST_START_DATE)

    if len(test_df) == 0:
        print(
            f"⚠️  No rows found on/after {TEST_START_DATE}. "
            "Falling back to percentage-based split (last 20%)."
        )
        train_df, test_df = pct_time_split(df, test_size=0.2)
    else:
        print(
            f"Calendar split: train={len(train_df):,} rows "
            f"({train_df['game_date'].min().date()} → {train_df['game_date'].max().date()})  "
            f"test={len(test_df):,} rows "
            f"({test_df['game_date'].min().date()} → {test_df['game_date'].max().date()})"
        )

    train_start = str(train_df["game_date"].min().date())
    train_end   = str(train_df["game_date"].max().date())
    test_start  = str(test_df["game_date"].min().date())
    test_end    = str(test_df["game_date"].max().date())

    # Split train into core + calibration (time-aware, percentage-based within train)
    train_core_df, calib_df = pct_time_split(train_df, test_size=0.2)

    X_train_core = train_core_df[feature_cols].fillna(0.0)
    y_train_core = train_core_df["hr_hit"].astype(int)

    X_calib = calib_df[feature_cols].fillna(0.0)
    y_calib = calib_df["hr_hit"].astype(int)

    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["hr_hit"].astype(int)

    print(f"\nFeature count: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    base_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    base_pipeline.fit(X_train_core, y_train_core)
    p_test_lr_raw = base_pipeline.predict_proba(X_test)[:, 1]
    roc_lr_raw = float(roc_auc_score(y_test, p_test_lr_raw))
    print(f"LogReg (raw) ROC-AUC: {roc_lr_raw:.3f}")

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=10.0,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1,
    )
    from lightgbm import early_stopping, log_evaluation
    lgbm.fit(
        X_train_core, y_train_core,
        eval_set=[(X_calib, y_calib)],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    print(f"LightGBM best iteration: {lgbm.best_iteration_}")
    p_test_lgbm_raw = lgbm.predict_proba(X_test)[:, 1]
    roc_lgbm_raw = float(roc_auc_score(y_test, p_test_lgbm_raw))
    print(f"LightGBM (raw) ROC-AUC: {roc_lgbm_raw:.3f}")

    # ------------------------------------------------------------------
    # Calibrate both
    # ------------------------------------------------------------------
    calibrated_lr = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method="sigmoid",
        cv=None,
    )
    calibrated_lr.fit(X_calib, y_calib)

    calibrated_lgbm = CalibratedClassifierCV(
        estimator=FrozenEstimator(lgbm),
        method="sigmoid",
        cv=None,
    )
    calibrated_lgbm.fit(X_calib, y_calib)

    # ------------------------------------------------------------------
    # Choose best model by raw ROC-AUC
    # ------------------------------------------------------------------
    if roc_lgbm_raw > roc_lr_raw:
        print("Chosen model: LightGBM")
        model = calibrated_lgbm
        p_test = calibrated_lgbm.predict_proba(X_test)[:, 1]
        chosen_name = "lightgbm_calibrated"

        # Print top 20 feature importances
        importances = pd.Series(lgbm.feature_importances_, index=feature_cols)
        print("\nTop 20 features by importance:")
        print(importances.sort_values(ascending=False).head(20).to_string())
    else:
        print("Chosen model: LogReg")
        model = calibrated_lr
        p_test = calibrated_lr.predict_proba(X_test)[:, 1]
        chosen_name = "logreg_calibrated"

        # Print top 20 LR coefficients by absolute value
        lr_model = base_pipeline.named_steps["lr"]
        coefs = pd.Series(np.abs(lr_model.coef_[0]), index=feature_cols)
        print("\nTop 20 features by |coefficient|:")
        print(coefs.sort_values(ascending=False).head(20).to_string())

    # ------------------------------------------------------------------
    # Lift / bucket stats
    # ------------------------------------------------------------------
    q90 = np.quantile(p_test, 0.90)
    top10_mask = p_test >= q90
    top10_hr_rate = float(y_test[top10_mask].mean())

    q99 = np.quantile(p_test, 0.99)
    top1_mask = p_test >= q99
    top1_hr_rate = float(y_test[top1_mask].mean())

    baseline = float(y_test.mean())
    avg_pred = float(p_test.mean())

    extra = {
        "top10_hr_rate": top10_hr_rate,
        "top1_hr_rate": top1_hr_rate,
        "top1_count": int(top1_mask.sum()),
        "top1_avg_b_pa_14": float(X_test.loc[top1_mask, "b_pa_14"].mean()) if "b_pa_14" in X_test.columns else float("nan"),
        "top1_avg_p_pa_30": float(X_test.loc[top1_mask, "p_pa_30"].mean()) if "p_pa_30" in X_test.columns else float("nan"),
        "max_pred_prob": float(p_test.max()),
        "top10_lift": (top10_hr_rate / baseline) if baseline > 0 else float("nan"),
        "top1_lift":  (top1_hr_rate  / baseline) if baseline > 0 else float("nan"),
        "avg_minus_base_pp": (avg_pred - baseline) * 100.0,
    }

    metrics_dict = {
        "train_rows":    int(len(train_df)),
        "test_rows":     int(len(test_df)),
        "train_start":   train_start,
        "train_end":     train_end,
        "test_start":    test_start,
        "test_end":      test_end,
        "test_hr_rate":  float(y_test.mean()),
        "avg_pred_prob": float(p_test.mean()),
        "log_loss":      float(log_loss(y_test, p_test, labels=[0, 1])),
        "roc_auc":       float(roc_auc_score(y_test, p_test)),
    }

    # ------------------------------------------------------------------
    # Save model — bundle includes shrinkage flag so predict.py knows
    # to apply the same transformation at inference time.
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / f"hr_model_{chosen_name}_2021_2025.joblib"
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "apply_shrinkage": True,   # signal to predict.py
    }, model_path)

    return TrainResult(
        model_path=model_path,
        metrics=metrics_dict,
        feature_cols=feature_cols,
        extra=extra,
    )


if __name__ == "__main__":
    train_path = latest_train_table()
    result = train_baseline(train_path)
    print_summary(train_path, result.model_path, result.feature_cols, result.metrics, result.extra)

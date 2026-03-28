from __future__ import annotations

import csv
import glob
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
MODELS_DIR    = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_START_DATE = "2026-03-26"

# Minimum number of matched (prediction, outcome) pairs needed before the
# isotonic recalibration step is attempted. Below this threshold the fit
# is too noisy to be useful and we skip it silently.
# At ~150 batter-games per day this is reached after roughly 3-4 weeks.
_MIN_RECAL_SAMPLES = 500

# Current season year — recalibration only pulls data from this year's
# prediction CSVs so it doesn't mix in stale prior-season probs.
_RECAL_SEASON = 2026


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def f3(x: float) -> str:
    return f"{x:.3f}"

def f2(x: float) -> str:
    return f"{x:.2f}"

def fmt_int(x: int) -> str:
    return f"{x:,}"


def print_summary(
    train_path: Path,
    model_path: Path,
    feature_cols: list[str],
    metrics: dict,
    extra: dict,
) -> None:
    run_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = model_path.name

    logger.info("=" * 30)
    logger.info("MLB HR MODEL — TRAIN SUMMARY")
    logger.info("=" * 30)
    logger.info("Run time: %s", run_time)
    logger.info("Training table: %s", train_path.name)
    logger.info(
        "Date range: train=%s->%s  test=%s->%s",
        metrics["train_start"], metrics["train_end"],
        metrics["test_start"],  metrics["test_end"],
    )
    logger.info(
        "Rows: train=%s  test=%s",
        fmt_int(metrics["train_rows"]), fmt_int(metrics["test_rows"]),
    )
    logger.info("Split type: %s", metrics.get("split_type", "calendar"))
    logger.info("Test HR rate (baseline): %s", pct(metrics["test_hr_rate"]))
    logger.info("Performance (test):")
    logger.info("  ROC-AUC:   %s", f3(metrics["roc_auc"]))
    logger.info("  Log loss:  %s", f3(metrics["log_loss"]))
    logger.info("  Avg pred:  %s", pct(metrics["avg_pred_prob"]))
    logger.info("  Max pred:  %s", f3(extra["max_pred_prob"]))
    logger.info("Lift checks (test):")
    logger.info(
        "  Top 10%% HR rate: %s  (%.2fx baseline)",
        pct(extra["top10_hr_rate"]), extra["top10_lift"],
    )
    logger.info(
        "  Top 1%%  HR rate: %s  (%.2fx baseline)  (n=%s)",
        pct(extra["top1_hr_rate"]), extra["top1_lift"], fmt_int(extra["top1_count"]),
    )
    logger.info("  Top 1%%  avg b_pa_14: %s", f2(extra["top1_avg_b_pa_14"]))
    logger.info("  Top 1%%  avg p_pa_30: %s", f2(extra["top1_avg_p_pa_30"]))
    logger.info("Calibration delta: %+.2f pp", extra["avg_minus_base_pp"])
    logger.info("Features: %d", len(feature_cols))
    logger.info("Model saved: %s", model_name)

    recal = metrics.get("recal_samples")
    if recal:
        logger.info(
            "Isotonic recalibration: %d samples | MAE %.2fpp → %.2fpp (Δ%+.2fpp)",
            recal,
            metrics.get("recal_mae_before", float("nan")),
            metrics.get("recal_mae_after",  float("nan")),
            metrics.get("recal_improvement", float("nan")),
        )
    else:
        logger.info(
            "Isotonic recalibration: skipped "
            "(need %d+ matched samples from %d season predictions)",
            _MIN_RECAL_SAMPLES, _RECAL_SEASON,
        )

    logger.info("=" * 30)

    log_file = MODELS_DIR / "train_runs.csv"
    write_header = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_time", "train_table", "split_type", "roc_auc", "log_loss",
                "baseline_hr_rate", "top10_hr_rate", "top1_hr_rate",
                "top10_lift", "top1_lift", "features",
                "recal_samples", "recal_mae_before", "recal_mae_after",
            ])
        writer.writerow([
            run_time, train_path.name,
            metrics.get("split_type", "calendar"),
            metrics["roc_auc"], metrics["log_loss"], metrics["test_hr_rate"],
            extra["top10_hr_rate"], extra["top1_hr_rate"],
            extra["top10_lift"], extra["top1_lift"],
            len(feature_cols),
            metrics.get("recal_samples", ""),
            metrics.get("recal_mae_before", ""),
            metrics.get("recal_mae_after", ""),
        ])


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict
    feature_cols: list[str]
    extra: dict


def latest_train_table() -> Path:
    multi = PROCESSED_DIR / "train_table_2021_2026_full.parquet"
    if multi.exists():
        return multi
    multi_legacy = PROCESSED_DIR / "train_table_2021_2025_full.parquet"
    if multi_legacy.exists():
        return multi_legacy
    season = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    if season.exists():
        return season
    files = [
        p for p in PROCESSED_DIR.glob("train_table_*.parquet")
        if p.stat().st_size > 10_000
    ]
    if not files:
        raise FileNotFoundError("No non-empty train_table_*.parquet found in data/processed/")
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
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
    cut   = pd.Timestamp(test_start)
    train = df[df["game_date"] < cut].sort_values("game_date").reset_index(drop=True)
    test  = df[df["game_date"] >= cut].sort_values("game_date").reset_index(drop=True)
    return train, test


def pct_time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df  = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Empirical Bayes shrinkage
# ---------------------------------------------------------------------------

_BATTER_PRIOR_PA  = 50
_PITCHER_PRIOR_PA = 75
_LEAGUE_RATES = {
    "hr":      0.033,
    "barrel":  0.073,
    "hardhit": 0.380,
    "fb":      0.350,
    "k":       0.227,
    "bb":      0.083,
}


def _shrink_rate(
    rate_col: pd.Series,
    pa_col: pd.Series,
    league_rate: float,
    prior_pa: int,
) -> pd.Series:
    obs_pa     = pa_col.fillna(0).clip(lower=0)
    obs_events = rate_col.fillna(0) * obs_pa
    return (obs_events + prior_pa * league_rate) / (obs_pa + prior_pa)


def apply_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw rate columns with empirical-Bayes-shrunk versions."""
    df = df.copy()

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
            df[rate_col] = _shrink_rate(df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior)

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
            df[rate_col] = _shrink_rate(df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior)

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
            df[rate_col] = _shrink_rate(df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior)

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
            df[rate_col] = _shrink_rate(df[rate_col], df[pa_col], _LEAGUE_RATES[metric], prior)

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


# ---------------------------------------------------------------------------
# Isotonic recalibration helpers
# ---------------------------------------------------------------------------

def _load_season_predictions(season: int) -> pd.DataFrame:
    """
    Load all final-pass prediction CSVs for the given season year.
    Skips morning files — we only want confirmed-lineup predictions.
    Returns empty DataFrame if nothing found.
    """
    pattern = str(PREDICTIONS_DIR / f"predictions_{season}-*.csv")
    files   = sorted(glob.glob(pattern))
    dfs     = []
    for f in files:
        stem  = Path(f).stem.replace("predictions_", "")
        parts = stem.split("_")
        sfx   = parts[1] if len(parts) > 1 else "manual"
        if sfx == "morning":
            continue
        try:
            df = pd.read_csv(f)
            df["game_date"] = pd.Timestamp(parts[0])
            dfs.append(df)
        except Exception as e:
            logger.warning("Could not load prediction CSV %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def _load_season_outcomes(dates: list[str]) -> pd.DataFrame:
    """
    Fetch actual HR outcomes from Statcast for the given dates.
    Returns empty DataFrame on any error so recalibration is safely skipped.
    """
    if not dates:
        return pd.DataFrame(columns=["game_date", "batter", "hr_hit"])
    try:
        from src.data_sources.statcast import fetch_statcast_events
        events = fetch_statcast_events(
            start_date=min(dates),
            end_date=max(dates),
            columns=["game_date", "batter", "events", "game_type"],
            regular_season_only=True,
        ).df
        events["game_date"] = pd.to_datetime(events["game_date"])
        events["is_hr"] = (events["events"] == "home_run").fillna(False).astype(int)
        outcomes = (
            events
            .groupby(["game_date", "batter"], as_index=False)
            .agg(hr_hit=("is_hr", "max"))
        )
        date_set = {pd.to_datetime(d) for d in dates}
        return outcomes[outcomes["game_date"].isin(date_set)]
    except Exception as e:
        logger.warning("Statcast outcome fetch failed (recalibration skipped): %s", e)
        return pd.DataFrame(columns=["game_date", "batter", "hr_hit"])


def _build_recal_dataset(season: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Join this season's predictions to actual Statcast outcomes.
    Returns (y_pred, y_true) arrays, or (None, None) if not enough data.
    """
    preds = _load_season_predictions(season)
    if preds.empty or "hr_prob" not in preds.columns:
        logger.info(
            "Recalibration: no %d prediction CSVs found — skipping.", season
        )
        return None, None

    dates    = preds["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()
    outcomes = _load_season_outcomes(dates)
    if outcomes.empty:
        return None, None

    preds["batter"]    = pd.to_numeric(preds["batter"],    errors="coerce").astype("Int64")
    outcomes["batter"] = pd.to_numeric(outcomes["batter"], errors="coerce").astype("Int64")

    merged = preds.merge(
        outcomes.rename(columns={"hr_hit": "actual_hr"}),
        on=["game_date", "batter"],
        how="inner",
    ).dropna(subset=["actual_hr", "hr_prob"])

    n = len(merged)
    if n < _MIN_RECAL_SAMPLES:
        logger.info(
            "Recalibration: only %d matched samples (need %d) — skipping. "
            "Will activate automatically once enough %d data accumulates.",
            n, _MIN_RECAL_SAMPLES, season,
        )
        return None, None

    logger.info(
        "Recalibration dataset: %d samples | %d dates | HR rate=%.4f",
        n, len(dates), merged["actual_hr"].mean(),
    )
    return merged["hr_prob"].astype(float).values, merged["actual_hr"].astype(int).values


def _fit_isotonic_recalibrator(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> tuple[IsotonicRegression, dict]:
    """
    Fit an isotonic regression on (model_prob → actual_hr) pairs.
    Returns the fitted recalibrator and a dict of before/after MAE stats.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    y_recal = iso.predict(y_pred)

    # Calibration error in probability buckets
    df = pd.DataFrame({"pred": y_pred, "recal": y_recal, "true": y_true})
    df["bucket"] = pd.qcut(df["pred"], q=8, duplicates="drop")
    cal = df.groupby("bucket", observed=True).agg(
        actual=("true",  "mean"),
        before=("pred",  "mean"),
        after=("recal",  "mean"),
    )
    mae_before = float((cal["before"] - cal["actual"]).abs().mean() * 100)
    mae_after  = float((cal["after"]  - cal["actual"]).abs().mean() * 100)

    stats = {
        "recal_samples":     len(y_pred),
        "recal_mae_before":  round(mae_before, 3),
        "recal_mae_after":   round(mae_after, 3),
        "recal_improvement": round(mae_before - mae_after, 3),
    }

    logger.info(
        "Isotonic recalibration: MAE %.2fpp → %.2fpp (improvement %+.2fpp)",
        mae_before, mae_after, mae_before - mae_after,
    )

    if mae_after > mae_before:
        logger.warning(
            "Recalibration increased MAE — not enough data yet for a stable fit. "
            "Saving anyway; improvement will grow as season data accumulates."
        )

    return iso, stats


class _IsotonicWrappedModel:
    """
    Thin wrapper that applies a fitted isotonic recalibrator on top of an
    existing model's predict_proba output. Saved inside the joblib bundle
    so predict.py requires zero changes — it just calls predict_proba() as normal.
    """
    def __init__(self, base_model, recalibrator: IsotonicRegression):
        self.base_model   = base_model
        self.recalibrator = recalibrator

    def predict_proba(self, X) -> np.ndarray:
        base_probs  = self.base_model.predict_proba(X)[:, 1]
        recal_probs = self.recalibrator.predict(base_probs)
        return np.column_stack([1 - recal_probs, recal_probs])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_baseline(train_path: Path) -> TrainResult:
    df = pd.read_parquet(train_path)

    logger.info("Applying empirical Bayes shrinkage to rate features ...")
    df = apply_shrinkage(df)

    feature_cols = [
        "b_hr_rate_14", "b_barrel_rate_14", "b_ev_mean_14", "b_la_mean_14",
        "b_hardhit_rate_14", "b_fb_rate_14", "b_k_rate_14", "b_bb_rate_14",
        "b_ev_mean_7", "b_hardhit_rate_7",
        "b_ev_trend", "b_hardhit_trend", "b_barrel_trend", "b_hr_trend",
        "b_hr_rate_szn", "b_barrel_rate_szn", "b_ev_mean_szn", "b_la_mean_szn",
        "b_hardhit_rate_szn", "b_fb_rate_szn",
        "b_hr_rate_14_vsL", "b_hr_rate_14_vsR",
        "b_hardhit_rate_14_vsL", "b_hardhit_rate_14_vsR",
        "b_barrel_rate_14_vsL", "b_barrel_rate_14_vsR",
        "b_hr_rate_szn_vsL", "b_hr_rate_szn_vsR",
        "b_hardhit_rate_szn_vsL", "b_hardhit_rate_szn_vsR",
        "b_barrel_rate_szn_vsL", "b_barrel_rate_szn_vsR",
        "b_hr_rate_home", "b_hr_rate_away", "b_hr_rate_home_edge",
        "b_hardhit_rate_home", "b_hardhit_rate_away",
        "b_barrel_rate_home", "b_barrel_rate_away",
        "is_home_game", "same_hand_matchup",
        "p_hr_allowed_rate_30", "p_ev_allowed_mean_30",
        "p_hardhit_allowed_rate_30", "p_fb_allowed_rate_30",
        "p_barrel_allowed_rate_30", "p_k_rate_30", "p_bb_rate_30",
        "p_hr_allowed_rate_30_vsL", "p_hr_allowed_rate_30_vsR",
        "p_hardhit_allowed_rate_30_vsL", "p_hardhit_allowed_rate_30_vsR",
        "p_barrel_allowed_rate_30_vsL", "p_barrel_allowed_rate_30_vsR",
        "p_hr_allowed_rate_szn", "p_ev_allowed_mean_szn",
        "p_hardhit_allowed_rate_szn", "p_fb_allowed_rate_szn",
        "p_barrel_allowed_rate_szn",
        "ev_edge_14_30", "hardhit_edge_14_30", "fb_edge_14_30",
        "barrel_edge_14_30", "hr_rate_edge_14_30",
        "k_rate_edge_14_30", "bb_rate_edge_14_30",
        "k_rate_interaction_14_30", "bb_rate_interaction_14_30",
        "contact_pressure_14_30", "discipline_balance_14_30",
        "hr_rate_edge_14_30_vsL", "hr_rate_edge_14_30_vsR",
        "hardhit_edge_14_30_vsL", "hardhit_edge_14_30_vsR",
        "barrel_edge_14_30_vsL", "barrel_edge_14_30_vsR",
        "temp_f", "wind_hr_impact", "wind_out_strong", "wind_in_strong",
        "temp_above_75", "temp_above_85", "is_indoor",
        "park_factor_hr",
        "batting_order_pos", "is_top_of_order", "expected_pa_today", "relief_pa_pct",
        "p_fb_velo_30", "p_fb_pct_30", "p_offspeed_pct_30", "p_fb_velo_trend",
        "b_days_rest", "p_days_rest", "p_is_short_rest",
    ]

    available = set(df.columns)
    missing_features = [c for c in feature_cols if c not in available]
    if missing_features:
        logger.warning(
            "Dropping %d feature(s) not found in table: %s",
            len(missing_features), missing_features,
        )
    feature_cols = [c for c in feature_cols if c in available]

    required_cols = feature_cols + ["hr_hit", "game_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training table: {missing}")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    train_df, test_df = calendar_split(df, test_start=TEST_START_DATE)
    split_type = "calendar"

    if len(test_df) == 0:
        logger.warning(
            "No rows found on/after %s — falling back to 80/20 percentage split.",
            TEST_START_DATE,
        )
        train_df, test_df = pct_time_split(df, test_size=0.2)
        split_type = "pct_80_20_fallback"

    elif len(train_df) == 0:
        logger.warning(
            "Train set is empty — all %d rows are on/after %s. "
            "Falling back to an 80/20 chronological split within available data. "
            "ROC-AUC will be optimistic — this model is for pipeline "
            "validation only. Set TEST_MODE=False and rebuild for real training.",
            len(df), TEST_START_DATE,
        )
        train_df, test_df = pct_time_split(df, test_size=0.2)
        split_type = "pct_80_20_fallback_empty_train"

    else:
        logger.info(
            "Calendar split: train=%d rows (%s -> %s)  test=%d rows (%s -> %s)",
            len(train_df),
            train_df["game_date"].min().date(), train_df["game_date"].max().date(),
            len(test_df),
            test_df["game_date"].min().date(), test_df["game_date"].max().date(),
        )

    train_start = str(train_df["game_date"].min().date())
    train_end   = str(train_df["game_date"].max().date())
    test_start  = str(test_df["game_date"].min().date())
    test_end    = str(test_df["game_date"].max().date())

    train_core_df, calib_df = pct_time_split(train_df, test_size=0.2)

    X_train_core = train_core_df[feature_cols].fillna(0.0)
    y_train_core = train_core_df["hr_hit"].astype(int)
    X_calib      = calib_df[feature_cols].fillna(0.0)
    y_calib      = calib_df["hr_hit"].astype(int)
    X_test       = test_df[feature_cols].fillna(0.0)
    y_test       = test_df["hr_hit"].astype(int)

    logger.info("Feature count: %d", len(feature_cols))
    logger.info(
        "Split sizes — train_core: %d  calib: %d  test: %d",
        len(X_train_core), len(X_calib), len(X_test),
    )

    # Logistic Regression
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    base_pipeline.fit(X_train_core, y_train_core)
    p_test_lr_raw = base_pipeline.predict_proba(X_test)[:, 1]
    roc_lr_raw    = float(roc_auc_score(y_test, p_test_lr_raw))
    logger.info("LogReg (raw) ROC-AUC: %.3f", roc_lr_raw)

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=31,
        min_child_samples=300, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=10.0,
        min_split_gain=0.01, random_state=42, verbosity=-1,
    )
    lgbm.fit(
        X_train_core, y_train_core,
        eval_set=[(X_calib, y_calib)],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    logger.info("LightGBM best iteration: %d", lgbm.best_iteration_)
    p_test_lgbm_raw = lgbm.predict_proba(X_test)[:, 1]
    roc_lgbm_raw    = float(roc_auc_score(y_test, p_test_lgbm_raw))
    logger.info("LightGBM (raw) ROC-AUC: %.3f", roc_lgbm_raw)

    # Sigmoid calibration (layer 1 — same as before)
    calibrated_lr = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline), method="sigmoid", cv=None,
    )
    calibrated_lr.fit(X_calib, y_calib)

    calibrated_lgbm = CalibratedClassifierCV(
        estimator=FrozenEstimator(lgbm), method="sigmoid", cv=None,
    )
    calibrated_lgbm.fit(X_calib, y_calib)

    # Choose best model
    if roc_lgbm_raw > roc_lr_raw:
        logger.info("Chosen model: LightGBM")
        model       = calibrated_lgbm
        p_test      = calibrated_lgbm.predict_proba(X_test)[:, 1]
        chosen_name = "lightgbm_calibrated"
        importances = pd.Series(lgbm.feature_importances_, index=feature_cols)
        logger.info("Top 20 features by importance:\n%s",
                    importances.sort_values(ascending=False).head(20).to_string())
    else:
        logger.info("Chosen model: LogReg")
        model       = calibrated_lr
        p_test      = calibrated_lr.predict_proba(X_test)[:, 1]
        chosen_name = "logreg_calibrated"
        lr_model    = base_pipeline.named_steps["lr"]
        coefs       = pd.Series(np.abs(lr_model.coef_[0]), index=feature_cols)
        logger.info("Top 20 features by |coefficient|:\n%s",
                    coefs.sort_values(ascending=False).head(20).to_string())

    # ------------------------------------------------------------------
    # Isotonic recalibration (layer 2) — uses this season's live outcomes
    #
    # Runs automatically every night after training. Skips cleanly if
    # there aren't enough matched prediction→outcome pairs yet (early
    # season). Once _MIN_RECAL_SAMPLES is reached, the fitted isotonic
    # mapping is wrapped around the base model and saved into the same
    # .joblib bundle — predict.py requires zero changes.
    # ------------------------------------------------------------------
    recal_stats: dict = {}

    logger.info(
        "Attempting isotonic recalibration on %d season data ...", _RECAL_SEASON
    )
    y_pred_recal, y_true_recal = _build_recal_dataset(_RECAL_SEASON)

    if y_pred_recal is not None:
        iso, recal_stats = _fit_isotonic_recalibrator(y_pred_recal, y_true_recal)
        # Wrap the base model — predict.py calls predict_proba() unchanged
        model = _IsotonicWrappedModel(model, iso)
        logger.info("Isotonic recalibration applied to saved model.")
    else:
        logger.info("Isotonic recalibration skipped — base model saved as-is.")

    # ------------------------------------------------------------------
    # Lift / bucket stats (computed on base model probs, pre-recalibration,
    # so they're comparable across nightly runs regardless of recal status)
    # ------------------------------------------------------------------
    q90           = np.quantile(p_test, 0.90)
    top10_mask    = p_test >= q90
    top10_hr_rate = float(y_test[top10_mask].mean())

    q99          = np.quantile(p_test, 0.99)
    top1_mask    = p_test >= q99
    top1_hr_rate = float(y_test[top1_mask].mean())

    baseline = float(y_test.mean())
    avg_pred = float(p_test.mean())

    extra = {
        "top10_hr_rate":    top10_hr_rate,
        "top1_hr_rate":     top1_hr_rate,
        "top1_count":       int(top1_mask.sum()),
        "top1_avg_b_pa_14": float(X_test.loc[top1_mask, "b_pa_14"].mean()) if "b_pa_14" in X_test.columns else float("nan"),
        "top1_avg_p_pa_30": float(X_test.loc[top1_mask, "p_pa_30"].mean()) if "p_pa_30" in X_test.columns else float("nan"),
        "max_pred_prob":    float(p_test.max()),
        "top10_lift":       (top10_hr_rate / baseline) if baseline > 0 else float("nan"),
        "top1_lift":        (top1_hr_rate  / baseline) if baseline > 0 else float("nan"),
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
        "split_type":    split_type,
        **recal_stats,  # recal_samples, recal_mae_before, recal_mae_after, recal_improvement
    }

    # Save — single file, same name as before, predict.py unchanged
    model_path = MODELS_DIR / f"hr_model_{chosen_name}_2021_2026.joblib"
    joblib.dump({
        "model":           model,
        "feature_cols":    feature_cols,
        "apply_shrinkage": True,
    }, model_path)
    logger.info("Model saved: %s", model_path.name)

    return TrainResult(
        model_path=model_path,
        metrics=metrics_dict,
        feature_cols=feature_cols,
        extra=extra,
    )


if __name__ == "__main__":
    configure_logging()
    train_path = latest_train_table()
    logger.info("Training from: %s", train_path.name)
    result = train_baseline(train_path)
    print_summary(train_path, result.model_path, result.feature_cols, result.metrics, result.extra)

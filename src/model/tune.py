"""
LightGBM hyperparameter tuning with Optuna.

Standalone — does NOT modify train.py. Reads the same train table and uses
the same feature column list, then runs an Optuna study to optimize
inner-validation top-5% lift. Outputs:

  models/optuna_trials.csv     — every trial's params + metrics
  models/optuna_study.pkl      — pickled study, resumable on next invocation

The 30-day holdout (last HOLDOUT_DAYS of the parquet, matching train.py) is
sliced off and NEVER seen during tuning. The outer test set (last 20% of
the remaining data) is also not seen during tuning. The objective only sees
an inner train/val split derived from the train portion.

Usage
    python -m src.model.tune                        # 100 trials (default)
    python -m src.model.tune --n-trials 50          # smaller study
    python -m src.model.tune --resume               # resume from saved study

Reasoning for objective = top-5% lift (not ROC-AUC)
---------------------------------------------------
top-5% lift is the metric we monetize. Recency-weighting experiments showed
that improving ROC-AUC and improving top-5% can decouple — tuning to the
wrong target is exactly how we burned time chasing a false win.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from optuna.integration import LightGBMPruningCallback

from src.logging_config import configure_logging
from src.model.train import (
    HOLDOUT_DAYS,
    apply_shrinkage,
    calendar_split,
    pct_time_split,
    TEST_START_DATE,
    latest_train_table,
)

logger = logging.getLogger(__name__)
logging.getLogger("optuna").setLevel(logging.WARNING)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
TRIALS_CSV    = MODELS_DIR / "optuna_trials.csv"
STUDY_PKL     = MODELS_DIR / "optuna_study.pkl"

DEFAULT_N_TRIALS = 100
RANDOM_SEED      = 42

# Mirror of train.py's feature_cols (kept here as a literal so tune.py stays
# fully decoupled from train.py's internals). Any change in train.py must
# be mirrored here. The disabled hp_ump_hr_factor entry stays disabled.
FEATURE_COLS: list[str] = [
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
    "batting_order_pos", "is_top_of_order", "expected_pa_today",
    "p_fb_velo_30", "p_fb_pct_30", "p_offspeed_pct_30", "p_fb_velo_trend",
    "b_days_rest", "p_days_rest", "p_is_short_rest",
    "b_pull_air_rate_14", "b_pull_air_rate_szn",
    "b_pull_air_hr_rate_14", "b_pull_air_hr_rate_szn",
    "b_hr_rate_vs_fb_14",  "b_hr_rate_vs_fb_szn",
    "b_hr_rate_vs_brk_14", "b_hr_rate_vs_brk_szn",
    "b_hr_rate_vs_os_14",  "b_hr_rate_vs_os_szn",
    "p_fb_usage_30", "p_brk_usage_30", "p_os_usage_30",
    "p_hr_rate_vs_fb_30", "p_hr_rate_vs_brk_30", "p_hr_rate_vs_os_30",
    "matchup_fb_30", "matchup_brk_30", "matchup_os_30",
    "matchup_best_30",
    "pull_x_park",
    "p_pitches_last_start", "p_ip_last_start", "p_is_opener", "p_workload_score",
    "p_fatigue_index", "p_workload_x_rest",
    "opener_x_hardhit", "opener_x_barrel",
    "b_iso_14", "b_iso_szn", "b_iso_career",
    "b_iso_career_vsL", "b_iso_career_vsR",
    "iso_x_park", "iso_x_wind",
    "b_hr_rate_career_vsL", "b_hr_rate_career_vsR",
    "b_hardhit_career_vsL", "b_hardhit_career_vsR",
    "b_platoon_hr_edge", "platoon_edge_x_iso",
    "b_hr_rate_blend_vsL", "b_hr_rate_blend_vsR",
    "b_sweet_spot_rate_14", "b_sweet_spot_rate_30", "b_sweet_spot_rate_szn",
    "sweet_spot_x_park", "sweet_spot_x_wind", "sweet_spot_x_poor_command",
    "b_pa_30", "b_hr_rate_30", "b_barrel_rate_30", "b_ev_mean_30",
    "b_hardhit_rate_30", "b_k_rate_30", "b_bb_rate_30",
    "hr_rate_edge_30_30", "hardhit_edge_30_30", "barrel_edge_30_30",
    "p_command_30", "p_command_szn", "p_kbb_ratio_30",
    "barrel_x_poor_command", "hr30_x_poor_command",
    "p_spin_rate_fb_30", "p_extension_30", "p_pfx_z_fb_30",
    "p_pfx_x_fb_30", "p_stuff_score_30",
    "barrel_x_weak_stuff", "sweet_spot_x_weak_stuff",
    "b_hr_last_7d", "b_games_since_last_hr", "b_ev_hot_flag",
    "b_contact_hot_flag", "b_hr_streak", "b_avg_ev_7d",
    "hot_x_park", "hot_x_iso", "hr7d_x_park",
    "t_hr_rate_14", "t_hr_rate_szn", "t_hardhit_rate_14", "t_ev_mean_14",
    "team_hr_x_batter_barrel", "team_hr_x_sweet_spot",
    "park_pull_factor", "pull_x_park_direction",
]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "num_leaves":         {"type": "int",   "low": 15,    "high": 127},
    "learning_rate":      {"type": "float", "low": 0.01,  "high": 0.1,  "log": True},
    "feature_fraction":   {"type": "float", "low": 0.5,   "high": 1.0},
    "bagging_fraction":   {"type": "float", "low": 0.5,   "high": 1.0},
    "bagging_freq":       {"type": "int",   "low": 0,     "high": 7},
    "min_child_samples":  {"type": "int",   "low": 10,    "high": 200},
    "reg_alpha":          {"type": "float", "low": 1e-8,  "high": 10.0, "log": True},
    "reg_lambda":         {"type": "float", "low": 1e-8,  "high": 10.0, "log": True},
    "max_depth":          {"type": "int",   "low": -1,    "high": 12},
    "n_estimators":       {"type": "int",   "low": 200,   "high": 2000},
}


def _suggest(trial: optuna.Trial, name: str) -> float | int:
    spec = SEARCH_SPACE[name]
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _topk_lift(y_true: np.ndarray, y_prob: np.ndarray, pct: int) -> tuple[float, int]:
    """Top-N% HR rate / overall HR rate, plus the count of picks."""
    baseline = float(y_true.mean())
    if baseline <= 0 or len(y_true) == 0:
        return float("nan"), 0
    q = np.quantile(y_prob, 1 - pct / 100.0)
    mask = y_prob >= q
    if not mask.any():
        return float("nan"), 0
    rate = float(y_true[mask].mean())
    return rate / baseline, int(mask.sum())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inner_split(train_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Replicate train.py's preprocessing up through inner-split, but stop
    BEFORE the test set is touched and BEFORE the holdout is touched.

    Returns (inner_train_df, inner_val_df, feature_cols_present).
    """
    df = pd.read_parquet(train_path)
    logger.info("Loaded %d rows from %s", len(df), train_path.name)

    df = apply_shrinkage(df)

    df["game_date"] = pd.to_datetime(df["game_date"])
    holdout_cut = df["game_date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    df          = df[df["game_date"] <= holdout_cut].copy()
    logger.info("Holdout sliced off (>%s). Working set: %d rows.",
                holdout_cut.date(), len(df))

    train_df, _test_df = calendar_split(df, test_start=TEST_START_DATE)
    if len(_test_df) == 0:
        train_df, _test_df = pct_time_split(df, test_size=0.2)
    logger.info("Outer test sliced off — never seen by Optuna. Train portion: %d rows.",
                len(train_df))

    # Inner 80/20 chronological split
    inner_train_df, inner_val_df = pct_time_split(train_df, test_size=0.2)
    logger.info(
        "Inner split — train=%d (%s -> %s)  val=%d (%s -> %s)",
        len(inner_train_df),
        inner_train_df["game_date"].min().date(), inner_train_df["game_date"].max().date(),
        len(inner_val_df),
        inner_val_df["game_date"].min().date(), inner_val_df["game_date"].max().date(),
    )

    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Dropping %d missing feature(s): %s", len(missing), missing)

    return inner_train_df, inner_val_df, feature_cols_present


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    feature_cols: list[str],
):
    """Build an objective closure with the splits + feature_cols pre-bound."""
    X_tr = inner_train_df[feature_cols].fillna(0.0)
    y_tr = inner_train_df["hr_hit"].astype(int).values
    X_va = inner_val_df[feature_cols].fillna(0.0)
    y_va = inner_val_df["hr_hit"].astype(int).values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":      "binary",
            # AUC for both early stopping and the pruner so its direction
            # ("higher is better") aligns with the study direction
            # ("maximize"). The objective returned to Optuna is still
            # inner-val top-5% lift; AUC is just the proxy for pruning.
            "metric":         "auc",
            "verbosity":      -1,
            "boosting_type":  "gbdt",
            "random_state":   RANDOM_SEED,
            "n_jobs":         -1,
            "num_leaves":         _suggest(trial, "num_leaves"),
            "learning_rate":      _suggest(trial, "learning_rate"),
            "feature_fraction":   _suggest(trial, "feature_fraction"),
            "bagging_fraction":   _suggest(trial, "bagging_fraction"),
            "bagging_freq":       _suggest(trial, "bagging_freq"),
            "min_child_samples":  _suggest(trial, "min_child_samples"),
            "reg_alpha":          _suggest(trial, "reg_alpha"),
            "reg_lambda":         _suggest(trial, "reg_lambda"),
            "max_depth":          _suggest(trial, "max_depth"),
            "n_estimators":       _suggest(trial, "n_estimators"),
        }
        # LGBMClassifier maps these names; bagging requires bagging_freq>0.
        # When freq=0 LGBM ignores bagging_fraction, which is fine.

        model = LGBMClassifier(**params)
        pruning_cb = LightGBMPruningCallback(trial, "auc")
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(period=-1),
                    pruning_cb,
                ],
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Some param combos are infeasible (e.g., bagging incompatible
            # with subsample disabled). Treat as a failed trial.
            logger.warning("Trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned()

        p_va = model.predict_proba(X_va)[:, 1]

        try:
            auc = float(roc_auc_score(y_va, p_va))
        except Exception:
            auc = float("nan")
        try:
            ll  = float(log_loss(y_va, p_va, labels=[0, 1]))
        except Exception:
            ll  = float("nan")
        try:
            brier = float(brier_score_loss(y_va, p_va))
        except Exception:
            brier = float("nan")

        top1_lift, top1_n  = _topk_lift(y_va, p_va, 1)
        top5_lift, top5_n  = _topk_lift(y_va, p_va, 5)
        top10_lift, top10_n = _topk_lift(y_va, p_va, 10)

        trial.set_user_attr("inner_val_roc_auc",   auc)
        trial.set_user_attr("inner_val_log_loss",  ll)
        trial.set_user_attr("inner_val_brier",     brier)
        trial.set_user_attr("inner_val_top1_lift", top1_lift)
        trial.set_user_attr("inner_val_top1_n",    top1_n)
        trial.set_user_attr("inner_val_top5_lift", top5_lift)
        trial.set_user_attr("inner_val_top5_n",    top5_n)
        trial.set_user_attr("inner_val_top10_lift", top10_lift)
        trial.set_user_attr("inner_val_top10_n",    top10_n)
        trial.set_user_attr("best_iteration",      int(getattr(model, "best_iteration_", -1) or -1))

        return float(top5_lift) if not np.isnan(top5_lift) else 0.0

    return objective


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _boundary_check(best_params: dict) -> list[tuple[str, str, float, str]]:
    """Flag any best param sitting at or near the search-space boundary."""
    issues: list[tuple[str, str, float, str]] = []
    for name, spec in SEARCH_SPACE.items():
        v = best_params.get(name)
        if v is None:
            continue
        lo, hi = spec["low"], spec["high"]
        if spec["type"] == "int":
            if v == lo:
                issues.append((name, "low", v, f"hit lower bound ({lo})"))
            elif v == hi:
                issues.append((name, "high", v, f"hit upper bound ({hi})"))
        else:
            # Float: 5% of (log-scaled) range from each bound
            if spec.get("log"):
                lo_log, hi_log = np.log10(lo), np.log10(hi)
                v_log = np.log10(v) if v > 0 else lo_log
                margin = 0.05 * (hi_log - lo_log)
                if v_log <= lo_log + margin:
                    issues.append((name, "low",  v, f"within 5% of lower bound (log) {lo:.2g}"))
                elif v_log >= hi_log - margin:
                    issues.append((name, "high", v, f"within 5% of upper bound (log) {hi:.2g}"))
            else:
                margin = 0.05 * (hi - lo)
                if v <= lo + margin:
                    issues.append((name, "low",  v, f"within 5% of lower bound {lo}"))
                elif v >= hi - margin:
                    issues.append((name, "high", v, f"within 5% of upper bound {hi}"))
    return issues


def report_results(study: optuna.Study) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No completed trials — nothing to report.")
        return

    logger.info("=" * 78)
    logger.info("OPTUNA TUNING RESULTS")
    logger.info("=" * 78)
    logger.info("Completed trials: %d", len(completed))
    logger.info("Pruned trials:    %d",
                sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED))
    logger.info("Failed trials:    %d",
                sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL))

    completed.sort(key=lambda t: t.value if t.value is not None else -1, reverse=True)

    # Top 5 trials
    logger.info("")
    logger.info("TOP 5 TRIALS BY INNER-VAL TOP-5%% LIFT")
    logger.info("-" * 78)
    for rank, t in enumerate(completed[:5], 1):
        ua = t.user_attrs
        logger.info(
            "[#%d] trial %d  top5_lift=%.4f  top1_lift=%.4f  top10_lift=%.4f  "
            "ROC-AUC=%.4f  Brier=%.5f  best_iter=%d",
            rank, t.number,
            t.value or float("nan"),
            ua.get("inner_val_top1_lift", float("nan")),
            ua.get("inner_val_top10_lift", float("nan")),
            ua.get("inner_val_roc_auc",   float("nan")),
            ua.get("inner_val_brier",     float("nan")),
            ua.get("best_iteration", -1),
        )
        for k in SEARCH_SPACE:
            logger.info("    %-22s = %s", k, t.params.get(k))

    # Param importance
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning("Could not compute param importances: %s", e)
        importances = {}
    if importances:
        logger.info("")
        logger.info("PARAM IMPORTANCES (Optuna's get_param_importances)")
        logger.info("-" * 78)
        for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
            logger.info("    %-22s  %.4f", k, v)

    # Boundary check on best trial
    issues = _boundary_check(completed[0].params)
    logger.info("")
    logger.info("SEARCH-SPACE BOUNDARY CHECK on best trial")
    logger.info("-" * 78)
    if issues:
        for name, side, val, msg in issues:
            logger.info("    [WIDEN] %-22s = %-12s  %s", name, val, msg)
        logger.info("Recommend widening the flagged bounds and re-running before applying.")
    else:
        logger.info("    All best-trial params are interior to the search space.")

    logger.info("")
    logger.info("Trials CSV: %s", TRIALS_CSV)
    logger.info("Study pkl : %s", STUDY_PKL)
    logger.info("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Total number of trials (default {DEFAULT_N_TRIALS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing optuna_study.pkl if present")
    args = parser.parse_args()

    configure_logging()

    train_path = latest_train_table()
    inner_train_df, inner_val_df, feature_cols = load_inner_split(train_path)
    logger.info("Feature count: %d", len(feature_cols))

    study: optuna.Study
    if args.resume and STUDY_PKL.exists():
        study = joblib.load(STUDY_PKL)
        n_done = len(study.trials)
        logger.info("Resumed study: %d trials already completed.", n_done)
    else:
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=20)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"hr_lgbm_top5_lift_{datetime.now().strftime('%Y%m%d')}",
        )
        n_done = 0

    n_remaining = max(0, args.n_trials - n_done)
    if n_remaining == 0:
        logger.info("Study already has %d trials — nothing to do.", n_done)
    else:
        logger.info("Running %d trials (target total = %d)", n_remaining, args.n_trials)

    objective = make_objective(inner_train_df, inner_val_df, feature_cols)

    # Save study + CSV after every trial so an interrupt doesn't lose work.
    def save_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        try:
            joblib.dump(study, STUDY_PKL)
            df = study.trials_dataframe()
            df.to_csv(TRIALS_CSV, index=False)
        except Exception as e:
            logger.warning("Could not persist study/trials csv: %s", e)

    if n_remaining > 0:
        study.optimize(
            objective,
            n_trials=n_remaining,
            callbacks=[save_callback],
            show_progress_bar=False,
            gc_after_trial=True,
        )

    report_results(study)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path
import glob
import joblib
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from pybaseball.playerid_lookup import playerid_reverse_lookup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Maximum IDs to resolve in a single playerid_reverse_lookup call.
_LOOKUP_CHUNK_SIZE = 500


def latest_train_table() -> Path:
    """
    Preference order:
      1. 2021-2025 multi-season file
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
        glob.glob(str(PROCESSED_DIR / "train_table_*.parquet")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No train_table_*.parquet found in data/processed.")
    return Path(files[0])


def load_model():
    """
    Load the best available model.  Preference order:
      1. 2021-2025 LightGBM calibrated
      2. 2021-2025 LogReg calibrated
      3. Legacy 2024 models
    """
    candidates = [
        MODELS_DIR / "hr_model_lightgbm_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2021_2025.joblib",
        MODELS_DIR / "hr_model_lightgbm_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_calibrated_2024.joblib",
        MODELS_DIR / "hr_model_logreg_edges_calibrated_2024.joblib",
    ]
    for path in candidates:
        if path.exists():
            print(f"Loading model: {path.name}")
            bundle = joblib.load(path)
            return bundle["model"], bundle["feature_cols"], bundle.get("apply_shrinkage", False)

    raise FileNotFoundError(
        "No trained model found in models/. Run src/model/train.py first."
    )


def _build_id_name_map(ids: list[int]) -> dict[int, str]:
    """
    Resolve a list of MLBAM IDs to full names using playerid_reverse_lookup.
    De-dupes across both batter and pitcher columns to minimise HTTP calls.
    """
    if not ids:
        return {}

    unique_ids = list(set(ids))
    mapping: dict[int, str] = {}

    for start in range(0, len(unique_ids), _LOOKUP_CHUNK_SIZE):
        chunk = unique_ids[start : start + _LOOKUP_CHUNK_SIZE]
        try:
            look = playerid_reverse_lookup(chunk, key_type="mlbam")
            look["full_name"] = (
                look["name_first"].fillna("") + " " + look["name_last"].fillna("")
            ).str.strip()
            mapping.update(
                dict(zip(look["key_mlbam"].astype(int), look["full_name"]))
            )
        except Exception as exc:
            print(f"  ⚠️  playerid_reverse_lookup failed for chunk: {exc}")

    return mapping


def add_player_names(
    df: pd.DataFrame,
    id_name_map: dict[int, str],
    id_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Map MLBAM IDs to display names using a pre-built lookup dict."""
    df[out_col] = df[id_col].astype("Int64").map(
        lambda x: id_name_map.get(int(x)) if pd.notna(x) else None
    )
    return df


def _format_american_odds(odds_val) -> str:
    """Format American odds for display, handling NaN gracefully."""
    try:
        v = int(odds_val)
        return f"{v:+d}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "  n/a"


def _format_pct(val, decimals: int = 1) -> str:
    """Format a 0-1 probability as a percentage string."""
    try:
        return f"{float(val)*100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "   n/a"


def _format_edge(val) -> str:
    """Format edge as signed percentage points."""
    try:
        pp = float(val) * 100
        sign = "+" if pp >= 0 else ""
        return f"{sign}{pp:.1f}pp"
    except (TypeError, ValueError):
        return "    n/a"


def print_ranked_table(ranked: pd.DataFrame, has_odds: bool) -> None:
    """
    Pretty-print the top HR candidates.

    Without odds:  batter | pitcher | model_prob | lineup pos
    With odds:     above + market_over_price | market_fair_prob | edge
    """
    TOP_N = 20

    if has_odds:
        # Sort by edge (model edge over market) for value-focused view.
        # Separate batters with/without odds lines so no-line players
        # don't pollute the edge-sorted view.
        has_line   = ranked[ranked["edge"].notna()].copy()
        no_line    = ranked[ranked["edge"].isna()].copy()

        has_line = has_line.sort_values("edge", ascending=False)
        no_line  = no_line.sort_values("hr_prob", ascending=False)

        print(f"\n{'='*85}")
        print(f"  TOP HR CANDIDATES WITH ODDS EDGE")
        print(f"{'='*85}")
        print(
            f"  {'Batter':<24} {'Pitcher':<22} "
            f"{'Model':>7} {'Market':>8} {'Fair':>7} {'Edge':>8}  {'Book':>4}"
        )
        print(f"  {'-'*24} {'-'*22} {'-'*7} {'-'*8} {'-'*7} {'-'*8}  {'-'*4}")

        shown = 0
        for _, row in has_line.head(TOP_N).iterrows():
            batter  = str(row.get("batter_name", ""))[:24]
            pitcher = str(row.get("pitcher_name", ""))[:22]
            model   = _format_pct(row.get("hr_prob"))
            mkt     = _format_american_odds(row.get("market_over_price"))
            fair    = _format_pct(row.get("market_fair_prob"))
            edge    = _format_edge(row.get("edge"))
            book    = str(row.get("odds_bookmaker", ""))[:4].upper()
            print(
                f"  {batter:<24} {pitcher:<22} "
                f"{model:>7} {mkt:>8} {fair:>7} {edge:>8}  {book:<4}"
            )
            shown += 1

        remaining = TOP_N - shown
        if remaining > 0 and len(no_line) > 0:
            print(f"\n  — No line available (model rank only) —")
            for _, row in no_line.head(remaining).iterrows():
                batter  = str(row.get("batter_name", ""))[:24]
                pitcher = str(row.get("pitcher_name", ""))[:22]
                model   = _format_pct(row.get("hr_prob"))
                pos     = row.get("batting_order_pos", 0)
                print(f"  {batter:<24} {pitcher:<22} {model:>7}   (slot {pos})")

        print(f"{'='*85}")
        print(
            "  Edge = model probability minus fair market probability (vig-removed).\n"
            "  Positive edge = model more bullish than market.\n"
            "  Market line is 0.5 HRs over/under unless noted."
        )

    else:
        # Original display — no odds available
        print(f"\n{'='*70}")
        print(f"  TOP HR CANDIDATES (no odds data)")
        print(f"{'='*70}")
        print(
            f"  {'Batter':<26} {'Pitcher':<24} "
            f"{'Model':>7}  {'Slot':>4}  {'xPA':>5}"
        )
        print(f"  {'-'*26} {'-'*24} {'-'*7}  {'-'*4}  {'-'*5}")

        top = ranked.sort_values("hr_prob", ascending=False).head(TOP_N)
        for _, row in top.iterrows():
            batter  = str(row.get("batter_name", ""))[:26]
            pitcher = str(row.get("pitcher_name", ""))[:24]
            model   = _format_pct(row.get("hr_prob"))
            pos     = int(row.get("batting_order_pos", 0))
            xpa     = f"{float(row.get('expected_pa_today', 0)):.1f}"
            print(f"  {batter:<26} {pitcher:<24} {model:>7}  {pos:>4}  {xpa:>5}")

        print(f"{'='*70}")


if __name__ == "__main__":
    model, feature_cols, apply_shrinkage = load_model()
    train_path = latest_train_table()

    df = pd.read_parquet(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Apply the same empirical Bayes shrinkage used at training time.
    if apply_shrinkage:
        from src.model.train import apply_shrinkage as _apply_shrinkage
        print("Applying empirical Bayes shrinkage ...")
        df = _apply_shrinkage(df)

    target_date = df["game_date"].max()
    today_df = df[df["game_date"] == target_date].copy()

    # Keep only feature columns that exist in the current table
    missing = [c for c in feature_cols if c not in today_df.columns]
    if missing:
        print(f"⚠️  {len(missing)} feature(s) missing at inference — filling with 0:")
        for c in missing:
            print(f"     - {c}")
            today_df[c] = 0.0

    X = today_df[feature_cols].fillna(0.0)
    today_df["hr_prob"] = model.predict_proba(X)[:, 1]

    # Resolve player names
    all_ids: list[int] = []
    for col in ("batter", "pitcher"):
        if col in today_df.columns:
            all_ids.extend(today_df[col].dropna().astype(int).unique().tolist())
    id_name_map = _build_id_name_map(all_ids)

    today_df = add_player_names(today_df, id_name_map, "batter",  "batter_name")
    if "pitcher" in today_df.columns:
        today_df = add_player_names(today_df, id_name_map, "pitcher", "pitcher_name")

    ranked = today_df.sort_values("hr_prob", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Odds enrichment — gracefully skipped if key is missing or API is down
    # ------------------------------------------------------------------
    date_str = str(target_date.date())
    has_odds = False

    odds_api_key = os.environ.get("ODDS_API_KEY")
    if odds_api_key:
        try:
            from src.data_sources.odds import enrich_predictions_with_odds
            ranked = enrich_predictions_with_odds(
                ranked,
                date_str,
                api_key=odds_api_key,
            )
            has_odds = ranked["edge"].notna().any()
        except Exception as e:
            print(f"⚠️  Odds enrichment failed: {e}")
    else:
        print(
            "ℹ️  No ODDS_API_KEY found — running without odds data.\n"
            "   Set the env var to enable edge detection:\n"
            "     export ODDS_API_KEY=your_key_here"
        )

    print(f"\nPredictions for {date_str}:")
    print_ranked_table(ranked, has_odds=has_odds)

    # ------------------------------------------------------------------
    # Save today's predictions to CSV for later odds-vs-outcome analysis
    # ------------------------------------------------------------------
    save_cols = ["batter_name", "pitcher_name", "hr_prob", "batter", "pitcher",
                 "batting_order_pos", "expected_pa_today"]
    if has_odds:
        save_cols += ["market_over_price", "market_fair_prob", "edge", "odds_bookmaker"]

    out_dir = PROJECT_ROOT / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{date_str}.csv"
    ranked[[c for c in save_cols if c in ranked.columns]].to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")

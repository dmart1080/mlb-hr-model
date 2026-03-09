from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events
from src.data_sources.mlb_schedule import fetch_rosters_for_games, enrich_labels_with_roster
from src.features.build_labels import build_batter_game_labels
from src.features.park_factors import get_park_factors, DEFAULT_PARK_FACTOR
from src.features.build_features_common import (
    FASTBALL_TYPES,
    OFFSPEED_TYPES,
    MIN_PA_BATTER_SZN,
    MIN_PA_PITCHER_SZN,
    MIN_PA_WINDOW,
    _safe_mean,
    _is_barrel,
)
from src.features.build_features_fast import (
    precompute_batter_windows_fast,
    precompute_pitcher_windows_fast,
    precompute_pitcher_velo_fast,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_WEATHER_MAX_WORKERS = 20


@dataclass(frozen=True)
class FeaturesBuildResult:
    features_df: pd.DataFrame
    output_path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()


def _date_minus_days(d: pd.Timestamp, days: int) -> pd.Timestamp:
    return d - pd.Timedelta(days=days)


def _load_and_clean_events(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_statcast_events(
        start_date=start_date,
        end_date=end_date,
        columns=[
            "game_date", "game_pk", "at_bat_number",
            "batter", "pitcher", "events",
            "home_team", "launch_speed", "launch_angle",
            "p_throws", "stand",
            "release_speed", "pitch_type",
        ],
    ).df.copy()

    raw = raw.convert_dtypes(dtype_backend="numpy_nullable")
    raw["game_date"]     = pd.to_datetime(raw["game_date"])
    raw["p_throws"]      = raw["p_throws"].astype("string").str.upper().str.strip()
    raw["stand"]         = raw["stand"].astype("string").str.upper().str.strip()
    raw["pitch_type"]    = raw["pitch_type"].astype("string").str.upper().str.strip()
    raw["release_speed"] = pd.to_numeric(raw["release_speed"], errors="coerce")

    pitches_df = (
        raw.copy()
        .sort_values("game_date")
        .dropna(subset=["game_date"])
        .reset_index(drop=True)
    )
    pitches_df["game_date"] = pitches_df["game_date"].astype("datetime64[ns]")

    raw["is_hr"] = (raw["events"] == "home_run").fillna(False).astype("int8")

    pa_df = (
        raw
        .sort_values(["game_pk", "batter", "game_date"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .agg({
            "game_date":    "first",
            "batter":       "last",
            "pitcher":      "last",
            "home_team":    "first",
            "events":       "last",
            "is_hr":        "max",
            "launch_speed": "mean",
            "launch_angle": "mean",
            "p_throws":     "last",
            "stand":        "last",
        })
    )

    ev_str = pa_df["events"].astype("string")
    pa_df["is_so"] = ev_str.str.contains("strikeout", na=False).astype("int8")
    pa_df["is_bb"] = (ev_str == "walk").fillna(False).astype("int8")

    ev = pd.to_numeric(pa_df["launch_speed"], errors="coerce")
    la = pd.to_numeric(pa_df["launch_angle"],  errors="coerce")

    pa_df["is_barrel"]    = _is_barrel(ev, la).astype("int8")
    pa_df["launch_speed"] = ev
    pa_df["launch_angle"] = la

    pa_df = (
        pa_df
        .sort_values("game_date")
        .dropna(subset=["game_date"])
        .reset_index(drop=True)
    )
    pa_df["game_date"] = pa_df["game_date"].astype("datetime64[ns]")

    return pitches_df, pa_df


# ---------------------------------------------------------------------------
# Days-rest helpers
# ---------------------------------------------------------------------------

def _compute_days_rest(pa_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    batter_dates  = {k: sorted(g.unique()) for k, g in pa_df.groupby("batter")["game_date"]}
    pitcher_dates = {k: sorted(g.unique()) for k, g in pa_df.groupby("pitcher")["game_date"]}

    def _gap(player_id, game_date, lookup):
        dates = lookup.get(player_id, [])
        prior = [d for d in dates if d < game_date]
        return float((game_date - max(prior)).days) if prior else np.nan

    labels = labels.copy()
    labels["b_days_rest"] = [
        _gap(int(r.batter), r.game_date, batter_dates)
        for r in labels.itertuples(index=False)
    ]
    labels["p_days_rest"] = [
        _gap(int(r.pitcher_id) if pd.notna(r.pitcher_id) else -1,
             r.game_date, pitcher_dates)
        for r in labels.itertuples(index=False)
    ]
    labels["p_is_short_rest"] = (labels["p_days_rest"] <= 3).astype("int8")
    labels["b_days_rest"] = labels["b_days_rest"].fillna(4.0)
    labels["p_days_rest"] = labels["p_days_rest"].fillna(4.0)
    return labels


# ---------------------------------------------------------------------------
# Parallelised weather fetch
# ---------------------------------------------------------------------------

def fetch_weather_for_games_fast(
    game_pks: list[int],
    home_teams: dict[int, str],
    *,
    force_refresh: bool = False,
    game_dates: dict[int, str] | None = None,
    max_workers: int = _WEATHER_MAX_WORKERS,
) -> pd.DataFrame:
    """
    Parallel replacement for weather.fetch_weather_for_games.
    Only un-cached games are fetched concurrently; cached reads stay in the
    main thread.  Removes the sequential sleep(0.05)/game dead time.
    """
    import json
    from src.data_sources.weather import (
        fetch_game_weather, _cache_path, _cache_is_stale,
        _neutral_weather, INDOOR_PARKS,
    )

    unique_pks = list(dict.fromkeys(game_pks))
    game_dates = game_dates or {}

    cached_pks = [
        gp for gp in unique_pks
        if not _cache_is_stale(_cache_path(gp), game_date=game_dates.get(gp))
        and not force_refresh
    ]
    fetch_pks = [gp for gp in unique_pks if gp not in set(cached_pks)]

    results: dict[int, dict] = {}

    for gp in cached_pks:
        with open(_cache_path(gp)) as f:
            results[gp] = json.load(f)

    if fetch_pks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_pk = {
                pool.submit(
                    fetch_game_weather,
                    gp,
                    force_refresh=True,
                    game_date=game_dates.get(gp),
                ): gp
                for gp in fetch_pks
            }
            done = 0
            for future in as_completed(future_to_pk):
                gp = future_to_pk[future]
                try:
                    results[gp] = future.result()
                except Exception as e:
                    results[gp] = _neutral_weather(gp, error=str(e))
                done += 1
                if done % 100 == 0:
                    print(f"    Weather: {done}/{len(fetch_pks)} games fetched ...")
        print(f"    Weather: {len(fetch_pks)}/{len(fetch_pks)} games fetched ...")

    rows = []
    for gp in unique_pks:
        r = results[gp].copy()
        home_team = home_teams.get(gp, "")
        r["is_indoor"] = int(home_team in INDOOR_PARKS)
        if r["is_indoor"]:
            r["wind_speed_mph"] = 0.0
            r["wind_hr_factor"] = 0.0
            r["wind_hr_impact"] = 0.0
            r["temp_f"]         = 72.0
        rows.append(r)

    df = pd.DataFrame(rows)
    df["temp_above_75"]   = (df["temp_f"] > 75).astype("int8")
    df["temp_above_85"]   = (df["temp_f"] > 85).astype("int8")
    df["wind_out_strong"] = ((df["wind_hr_factor"] > 0) & (df["wind_speed_mph"] >= 10)).astype("int8")
    df["wind_in_strong"]  = ((df["wind_hr_factor"] < 0) & (df["wind_speed_mph"] >= 10)).astype("int8")

    return df[[
        "game_pk", "temp_f", "wind_speed_mph", "wind_hr_factor",
        "wind_hr_impact", "is_indoor", "temp_above_75", "temp_above_85",
        "wind_out_strong", "wind_in_strong",
    ]]


# ---------------------------------------------------------------------------
# Relief PA pct
# ---------------------------------------------------------------------------

def _compute_relief_pa_pct(pa_df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of each batter's PAs in a game against non-starters.

    Uses the per-batter starter_pitcher_id from enrich_labels_with_roster,
    which already resolves the correct opposing starter for each batter.
    We join on (game_pk, batter) so each batter is compared against their
    own opposing starter — not a single game-level starter that would be
    wrong for half the batters.

    Must be called AFTER enrich_labels_with_roster.
    """
    labels = labels.copy()
    labels["game_pk"] = labels["game_pk"].astype(int)
    labels["batter"]  = labels["batter"].astype(int)

    pa = pa_df.copy()
    pa["game_pk"] = pa["game_pk"].astype(int)
    pa["batter"]  = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)

    if "starter_pitcher_id" in labels.columns:
        # Per-batter starter: join on (game_pk, batter) so each batter
        # gets their own opposing starter, not a shared game-level value
        batter_starter = (
            labels[["game_pk", "batter", "starter_pitcher_id"]]
            .dropna(subset=["starter_pitcher_id"])
            .copy()
        )
        batter_starter["starter_pitcher_id"] = batter_starter["starter_pitcher_id"].astype(int)
        pa = pa.merge(batter_starter, on=["game_pk", "batter"], how="left")
        pa["is_relief_pa"] = np.where(
            pa["starter_pitcher_id"].isna(),
            0,
            (pa["pitcher"] != pa["starter_pitcher_id"]).astype("boolean").fillna(False).astype(int),
        )
    else:
        # Fallback: first pitcher by at_bat_number as game-level starter
        first_pitcher = (
            pa.sort_values(["game_pk", "at_bat_number"])
            .groupby("game_pk")["pitcher"]
            .first()
            .rename("starter_pitcher")
            .reset_index()
        )
        pa = pa.merge(first_pitcher, on="game_pk", how="left")
        pa["is_relief_pa"] = (
            (pa["pitcher"] != pa["starter_pitcher"])
            .astype("boolean").fillna(False).astype(int)
        )

    relief_pct = (
        pa.groupby(["game_pk", "batter"])
        .agg(relief_pa_pct=("is_relief_pa", "mean"))
        .reset_index()
    )

    if "relief_pa_pct" in labels.columns:
        labels = labels.drop(columns=["relief_pa_pct"])
    labels = labels.merge(relief_pct, on=["game_pk", "batter"], how="left")
    labels["relief_pa_pct"] = labels["relief_pa_pct"].fillna(0.0)
    return labels


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def _add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _edge(a: str, b: str) -> pd.Series:
        return df[a] - df[b]

    df["ev_edge_14_30"]             = _edge("b_ev_mean_14",       "p_ev_allowed_mean_30")
    df["hardhit_edge_14_30"]        = _edge("b_hardhit_rate_14",  "p_hardhit_allowed_rate_30")
    df["fb_edge_14_30"]             = _edge("b_fb_rate_14",       "p_fb_allowed_rate_30")
    df["barrel_edge_14_30"]         = _edge("b_barrel_rate_14",   "p_barrel_allowed_rate_30")
    df["hr_rate_edge_14_30"]        = _edge("b_hr_rate_14",       "p_hr_allowed_rate_30")
    df["k_rate_edge_14_30"]         = _edge("b_k_rate_14",        "p_k_rate_30")
    df["bb_rate_edge_14_30"]        = _edge("b_bb_rate_14",       "p_bb_rate_30")

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

    if "wind_hr_impact" in df.columns:
        df["hardhit_x_wind"] = df["b_hardhit_rate_14"] * df["wind_hr_impact"]
        df["barrel_x_wind"]  = df["b_barrel_rate_14"]  * df["wind_hr_impact"]

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features_for_range(start_date: str, end_date: str) -> FeaturesBuildResult:
    start_dt = _to_date(start_date)
    end_dt   = _to_date(end_date)
    history_start = _date_minus_days(start_dt, 60)

    print(f"  Loading events {history_start.date()} → {end_dt.date()} ...")
    pitches_df, pa_df = _load_and_clean_events(
        start_date=history_start.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )

    game_pk_to_home = (
        pa_df.dropna(subset=["home_team"])
        .drop_duplicates(subset=["game_pk"])
        .set_index("game_pk")["home_team"]
        .to_dict()
    )

    batter_team_lookup: dict = {}  # populated after roster enrichment below

    target_pa = pa_df[pa_df["game_date"].between(start_dt, end_dt)]
    labels = build_batter_game_labels(target_pa)
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    labels["pitcher_id"] = pd.to_numeric(
        labels.get("starter_id", labels["pitcher_mode"]).fillna(labels["pitcher_mode"]),
        errors="coerce",
    ).astype("Int64")

    pitcher_hand_lookup = (
        pa_df.dropna(subset=["pitcher", "p_throws"])
        .groupby("pitcher")["p_throws"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )
    batter_hand_lookup = (
        pa_df.dropna(subset=["batter", "stand"])
        .groupby("batter")["stand"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        .to_dict()
    )

    labels["pitcher_hand"] = labels["pitcher_id"].map(
        lambda x: pitcher_hand_lookup.get(int(x)) if pd.notna(x) else None
    )
    labels["batter_hand"] = labels["batter"].map(batter_hand_lookup)

    print("  Computing days rest ...")
    labels = _compute_days_rest(pa_df, labels)

    # ------------------------------------------------------------------
    # Roster enrichment
    # ------------------------------------------------------------------
    target_game_pks = labels["game_pk"].dropna().astype(int).unique().tolist()
    print(f"  Fetching rosters/batting orders for {len(target_game_pks):,} games ...")
    starters_df, batting_df = fetch_rosters_for_games(target_game_pks)
    # ------------------------------------------------------------------
    # Build batter_team_lookup from batting_df["team_side"] — the only
    # reliable source. pa_df["home_team"] is the game's home team for ALL
    # rows regardless of which side the batter plays for, so the old modal
    # heuristic assigned every batter to the home team.
    # ------------------------------------------------------------------
    if not batting_df.empty and "team_side" in batting_df.columns:
        home_games_per_batter = (
            pa_df[pa_df["game_date"].between(start_dt, end_dt)]
            .groupby("batter")["home_team"]
            .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) > 0 else None)
            .to_dict()
        )
        bd = batting_df.copy()
        bd["game_pk"] = bd["game_pk"].astype(int)
        bd["batter"]  = bd["batter"].astype(int)
        for row in bd.itertuples(index=False):
            home_abbr = game_pk_to_home.get(row.game_pk)
            if home_abbr is None:
                continue
            if row.team_side == "home":
                batter_team_lookup[row.batter] = home_abbr
            else:
                t = home_games_per_batter.get(row.batter)
                if t and t != home_abbr:
                    batter_team_lookup[row.batter] = t

    # Now that batter_team_lookup is populated, assign batter_team to labels
    # so _resolve_starter can pick the correct opposing starter per batter.
    labels["batter_team"] = labels["batter"].map(batter_team_lookup)

    labels = enrich_labels_with_roster(labels, starters_df, batting_df, game_pk_to_home)

    # relief_pa_pct — must come after roster enrichment so starter_pitcher_id exists
    labels = _compute_relief_pa_pct(target_pa, labels)

    # ------------------------------------------------------------------
    # Weather — parallelised
    # ------------------------------------------------------------------
    game_dates_map = (
        labels.drop_duplicates("game_pk")
        .set_index("game_pk")["game_date"]
        .apply(lambda d: str(pd.Timestamp(d).date()))
        .to_dict()
    )
    print(f"  Fetching weather for {len(target_game_pks):,} games ...")
    weather_df = fetch_weather_for_games_fast(
        target_game_pks,
        game_pk_to_home,
        game_dates=game_dates_map,
    )
    weather_df["game_pk"] = weather_df["game_pk"].astype(int)

    # ------------------------------------------------------------------
    # Batter windows — vectorised
    # ------------------------------------------------------------------
    n_b = labels[["batter", "game_date"]].drop_duplicates().shape[0]
    print(f"  Precomputing batter windows for {n_b:,} (batter, date) pairs ...")
    batter_stats = precompute_batter_windows_fast(
        pa_df,
        labels[["batter", "game_date", "pitcher_hand"]],
        batter_team_lookup=batter_team_lookup,
        game_pk_home_lookup=game_pk_to_home,
        label_game_pks=labels[["batter", "game_date", "game_pk"]],
    )

    # ------------------------------------------------------------------
    # Pitcher windows — vectorised
    # ------------------------------------------------------------------
    starter_col = (
        "starter_pitcher_id" if "starter_pitcher_id" in labels.columns else "pitcher_id"
    )
    pitcher_need = (
        labels[[starter_col, "game_date", "batter_hand"]]
        .rename(columns={starter_col: "pitcher"})
        .dropna(subset=["pitcher"])
        .copy()
    )
    pitcher_need["pitcher"] = pitcher_need["pitcher"].astype(int)
    n_p = pitcher_need[["pitcher", "game_date"]].drop_duplicates().shape[0]

    print(f"  Precomputing pitcher PA windows for {n_p:,} (pitcher, date) pairs ...")
    pitcher_stats = precompute_pitcher_windows_fast(pa_df, pitcher_need)

    print(f"  Precomputing pitcher velo windows for {n_p:,} (pitcher, date) pairs ...")
    pitcher_velo = precompute_pitcher_velo_fast(pitches_df, pitcher_need)

    # ------------------------------------------------------------------
    # Merge everything
    # ------------------------------------------------------------------
    merge_pitcher_col = starter_col
    features_df = (
        labels
        .merge(batter_stats,  on=["batter", "game_date"], how="left")
        .merge(
            pitcher_stats.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(
            pitcher_velo.rename(columns={"pitcher": merge_pitcher_col}),
            on=[merge_pitcher_col, "game_date"], how="left",
        )
        .merge(weather_df, on="game_pk", how="left")
    )

    # Park factor
    features_df["home_team"] = features_df["game_pk"].map(game_pk_to_home)
    # Clean up string columns — replace float NaNs with None so
    # pyarrow can serialise the object column to parquet.
    import math as _math
    for _str_col in ("batter_team", "home_team", "pitcher_hand", "batter_hand"):
        if _str_col in features_df.columns:
            features_df[_str_col] = features_df[_str_col].apply(
                lambda v: None if (
                    v is None
                    or (isinstance(v, float) and _math.isnan(v))
                    or str(v) in ("nan", "None", "<NA>", "0.0", "0")
                ) else str(v)
            )
    season_year = int(pd.to_datetime(features_df["game_date"]).dt.year.mode().iloc[0])
    pf_map = get_park_factors(season=season_year)
    features_df["park_factor_hr"] = (
        features_df["home_team"].map(pf_map).fillna(DEFAULT_PARK_FACTOR / 100.0)
    )

    features_df = _add_edge_features(features_df)

    # Tidy pitcher column
    if "starter_pitcher_id" in features_df.columns and "pitcher" not in features_df.columns:
        features_df = features_df.rename(columns={"starter_pitcher_id": "pitcher"})
    elif "starter_pitcher_id" in features_df.columns:
        features_df = features_df.drop(columns=["starter_pitcher_id"], errors="ignore")

    features_df["game_date"] = pd.to_datetime(features_df["game_date"]).dt.date
    features_df["hr_hit"]    = features_df["hr_hit"].astype(int)

    non_stat_cols = {
        "game_date", "game_pk", "batter", "pitcher", "home_team",
        "batter_team",  # string — must not be zero-filled
        "hr_hit", "pitcher_hand", "batter_hand", "pitcher_mode",
        "starter_id", "starter_pitcher_id",
    }
    stat_cols = [c for c in features_df.columns if c not in non_stat_cols]
    features_df[stat_cols] = features_df[stat_cols].fillna(0.0)

    out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
    features_df.to_parquet(out_path, index=False)

    return FeaturesBuildResult(features_df=features_df, output_path=out_path)


if __name__ == "__main__":
    result = build_features_for_range("2024-03-20", "2024-10-01")
    full_path = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    result.features_df.to_parquet(full_path, index=False)
    print(f"\nSaved: {full_path}")
    print(f"Rows: {len(result.features_df):,}")
    print("\nLabel HR rate:", result.features_df["hr_hit"].mean())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    m = v.mean()
    return 0.0 if pd.isna(m) else float(m)

def _safe_rate_bool(mask) -> float:
    if not isinstance(mask, pd.Series):
        return 0.0 if pd.isna(mask) else float(mask)

    v = mask.astype("boolean")
    m = v.mean(skipna=True)
    return 0.0 if pd.isna(m) else float(m)


from src.data_sources.statcast import fetch_statcast_events
from src.features.build_labels import build_batter_game_labels
from src.features.park_factors import HR_PARK_FACTOR, DEFAULT_PARK_FACTOR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeaturesBuildResult:
    features_df: pd.DataFrame
    output_path: Path


def _to_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d).normalize()


def _date_minus_days(d: pd.Timestamp, days: int) -> pd.Timestamp:
    return d - pd.Timedelta(days=days)


def build_features_for_range(start_date: str, end_date: str) -> FeaturesBuildResult:
    start_dt = _to_date(start_date)
    end_dt = _to_date(end_date)

    history_start = _date_minus_days(start_dt, 60)

    events = fetch_statcast_events(
        start_date=history_start.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        columns=[
            "game_date",
            "game_pk",
            "at_bat_number",
            "batter",
            "pitcher",
            "events",
            "home_team",
            "launch_speed",
            "launch_angle",
        ],
    ).df.copy()


    # Force numpy-backed dtypes to avoid slow pyarrow boolean indexing
    events = events.convert_dtypes(dtype_backend="numpy_nullable")

    events["game_date"] = pd.to_datetime(events["game_date"])
    events["is_hr"] = (events["events"] == "home_run").fillna(False).astype("int8")

    # -------------------------------
    # COLLAPSE TO PLATE-APPEARANCE LEVEL  (ADD THIS)
    # -------------------------------
    # Statcast often comes pitch-level. We want 1 row per PA for correct "PA" counts.
    events = (
        events
        .sort_values(["game_pk", "batter", "game_date"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .agg({
            "game_date": "first",
            "batter": "last",
            "pitcher": "last",
            "home_team": "first",
            "events": "last",
            "is_hr": "max",
            "launch_speed": "max",
            "launch_angle": "max",
        })
    )

    events = events.sort_values("game_date").reset_index(drop=True)
    events["game_date"] = pd.to_datetime(events["game_date"]).astype("datetime64[ns]")
    events = events.dropna(subset=["game_date"])

    # Manual barrel approximation using EV + LA (safe with missing values) AFTER collapse
    if ("launch_speed" in events.columns) and ("launch_angle" in events.columns):
        ev = pd.to_numeric(events["launch_speed"], errors="coerce")
        la = pd.to_numeric(events["launch_angle"], errors="coerce")

        is_barrel_bool = (ev >= 95) & (la.between(20, 35))
        events["is_barrel"] = is_barrel_bool.fillna(False).astype("int8")
    else:
        events["is_barrel"] = 0

    # Pre-split by batter/pitcher to avoid scanning full events table every loop
    events_by_batter = {k: g for k, g in events.groupby("batter", sort=False)}
    events_by_pitcher = {k: g for k, g in events.groupby("pitcher", sort=False)}


    # Fast lookup: game_pk → home_team
    game_pk_to_home = (
        events.dropna(subset=["home_team"])
        .drop_duplicates(subset=["game_pk"])
        .set_index("game_pk")["home_team"]
        .to_dict()
    )

    # Create labels
    target_events = events[
        (events["game_date"] >= start_dt) &
        (events["game_date"] <= end_dt)
    ]

    labels = build_batter_game_labels(target_events)
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    feature_rows = []

    for row in labels.itertuples(index=False):
        game_date = row.game_date
        batter_id = row.batter
        pitcher_id = row.pitcher_mode
        hr_hit = row.hr_hit
        game_pk = row.game_pk

        as_of = game_date - pd.Timedelta(days=1)

        # ---------------- Park Factor ----------------
        home_team = game_pk_to_home.get(game_pk)
        park_factor_hr = HR_PARK_FACTOR.get(home_team, DEFAULT_PARK_FACTOR) / 100.0

        # ---------------- Batter 14-day ----------------
        b_14_start = as_of - pd.Timedelta(days=14)

        batter_all = events_by_batter.get(batter_id)
        if batter_all is None:
            batter_14 = events.iloc[0:0]
        else:
            batter_14 = batter_all[batter_all["game_date"].between(b_14_start + pd.Timedelta(days=1), as_of)]


        b_pa_14 = len(batter_14)
        b_hr_14 = int(batter_14["is_hr"].sum())
        b_hr_rate_14 = (b_hr_14 / b_pa_14) if b_pa_14 > 0 else 0.0
        
        b_barrels_14 = int(batter_14["is_barrel"].sum())
        b_barrel_rate_14 = (b_barrels_14 / b_pa_14) if b_pa_14 > 0 else 0.0
        
        # ---------- Batter contact-quality (14d) ----------
        b_ev_14 = pd.to_numeric(batter_14["launch_speed"], errors="coerce")
        b_la_14 = pd.to_numeric(batter_14["launch_angle"], errors="coerce")

        b_ev_mean_14 = _safe_mean(batter_14["launch_speed"]) if b_pa_14 > 0 else 0.0
        b_la_mean_14 = _safe_mean(batter_14["launch_angle"]) if b_pa_14 > 0 else 0.0

        b_hardhit_rate_14 = _safe_rate_bool(b_ev_14 >= 95) if b_pa_14 > 0 else 0.0
        b_fb_rate_14 = _safe_rate_bool((b_la_14 >= 20) & (b_la_14 <= 40)) if b_pa_14 > 0 else 0.0

        # ---------------- Batter Season ----------------
        if batter_all is None:
            batter_szn = events.iloc[0:0]
        else:
            season_start = pd.Timestamp(as_of.year, 3, 1)
            batter_szn = batter_all[batter_all["game_date"].between(season_start, as_of)]



        b_pa_szn = len(batter_szn)
        b_hr_szn = int(batter_szn["is_hr"].sum())
        b_hr_rate_szn = (b_hr_szn / b_pa_szn) if b_pa_szn > 0 else 0.0
        
        # ---------- Batter contact-quality (season) ----------
        b_ev_szn = pd.to_numeric(batter_szn["launch_speed"], errors="coerce")
        b_la_szn = pd.to_numeric(batter_szn["launch_angle"], errors="coerce")

        b_ev_mean_szn = _safe_mean(batter_szn["launch_speed"]) if b_pa_szn > 0 else 0.0
        b_la_mean_szn = _safe_mean(batter_szn["launch_angle"]) if b_pa_szn > 0 else 0.0

        b_hardhit_rate_szn = _safe_rate_bool((b_ev_szn >= 95).mean()) if b_pa_szn > 0 else 0.0
        b_fb_rate_szn = _safe_rate_bool((b_la_szn >= 20) & (b_la_szn <= 40)) if b_pa_szn > 0 else 0.0

        b_barrels_szn = int(batter_szn["is_barrel"].sum())
        b_barrel_rate_szn = (b_barrels_szn / b_pa_szn) if b_pa_szn > 0 else 0.0


        # ---------------- Pitcher 30-day ----------------
        p_30_start = as_of - pd.Timedelta(days=30)
        pitcher_id = int(row.pitcher_mode) if pd.notna(row.pitcher_mode) else None

        pitcher_all = events_by_pitcher.get(pitcher_id) if pitcher_id is not None else None
        if pitcher_all is None:
            pitcher_30 = events.iloc[0:0]
        else:
            pitcher_30 = pitcher_all[pitcher_all["game_date"].between(p_30_start + pd.Timedelta(days=1), as_of)]


        p_pa_30 = len(pitcher_30)
        p_hr_allowed_30 = int(pitcher_30["is_hr"].sum())
        p_hr_allowed_rate_30 = (
            p_hr_allowed_30 / p_pa_30
        ) if p_pa_30 > 0 else 0.0
       
        # ---------- Pitcher contact-quality allowed (30d) ----------
        p_ev_30 = pd.to_numeric(pitcher_30["launch_speed"], errors="coerce")
        p_la_30 = pd.to_numeric(pitcher_30["launch_angle"], errors="coerce")

        p_ev_allowed_mean_30 = _safe_mean(pitcher_30["launch_speed"]) if p_pa_30 > 0 else 0.0
        p_hardhit_allowed_rate_30 = _safe_rate_bool(p_ev_30 >= 95) if p_pa_30 > 0 else 0.0
        p_fb_allowed_rate_30 = _safe_rate_bool((p_la_30 >= 20) & (p_la_30 <= 40)) if p_pa_30 > 0 else 0.0

        p_barrels_allowed_30 = int(pitcher_30["is_barrel"].sum())
        p_barrel_allowed_rate_30 = (p_barrels_allowed_30 / p_pa_30) if p_pa_30 > 0 else 0.0


        # ---------------- Pitcher Season ----------------
        if pitcher_all is None:
            pitcher_szn = events.iloc[0:0]
        else:
            season_start = pd.Timestamp(as_of.year, 3, 1)
            pitcher_szn = pitcher_all[pitcher_all["game_date"].between(season_start, as_of)]


        p_pa_szn = len(pitcher_szn)
        p_hr_allowed_szn = int(pitcher_szn["is_hr"].sum())
        p_hr_allowed_rate_szn = (
            p_hr_allowed_szn / p_pa_szn
        ) if p_pa_szn > 0 else 0.0

        # ---------- Pitcher contact-quality allowed (season) ----------
        p_ev_szn = pd.to_numeric(pitcher_szn["launch_speed"], errors="coerce")
        p_la_szn = pd.to_numeric(pitcher_szn["launch_angle"], errors="coerce")

        p_ev_allowed_mean_szn = _safe_mean(pitcher_szn["launch_speed"]) if p_pa_szn > 0 else 0.0
        p_hardhit_allowed_rate_szn = _safe_rate_bool(p_ev_szn >= 95) if p_pa_szn > 0 else 0.0
        p_fb_allowed_rate_szn = _safe_rate_bool((p_la_szn >= 20) & (p_la_szn <= 40)) if p_pa_szn > 0 else 0.0

        p_barrels_allowed_szn = int(pitcher_szn["is_barrel"].sum())
        p_barrel_allowed_rate_szn = (p_barrels_allowed_szn / p_pa_szn) if p_pa_szn > 0 else 0.0

        # Interaction / "edge" features (batter minus pitcher allowed)
        ev_edge_14_30 = b_ev_mean_14 - p_ev_allowed_mean_30
        hardhit_edge_14_30 = b_hardhit_rate_14 - p_hardhit_allowed_rate_30
        fb_edge_14_30 = b_fb_rate_14 - p_fb_allowed_rate_30
        barrel_edge_14_30 = b_barrel_rate_14 - p_barrel_allowed_rate_30
        hr_rate_edge_14_30 = b_hr_rate_14 - p_hr_allowed_rate_30

        feature_rows.append(
            {
                "game_date": game_date.date(),
                "game_pk": game_pk,
                "batter": batter_id,
                "pitcher": int(pitcher_id) if pd.notna(pitcher_id) else None,
                "home_team": home_team,
                "park_factor_hr": park_factor_hr,
                "hr_hit": int(hr_hit),

                "b_pa_14": b_pa_14,
                "b_hr_14": b_hr_14,
                "b_hr_rate_14": b_hr_rate_14,

                "b_pa_szn": b_pa_szn,
                "b_hr_szn": b_hr_szn,
                "b_hr_rate_szn": b_hr_rate_szn,

                "p_pa_30": p_pa_30,
                "p_hr_allowed_30": p_hr_allowed_30,
                "p_hr_allowed_rate_30": p_hr_allowed_rate_30,

                "p_pa_szn": p_pa_szn,
                "p_hr_allowed_szn": p_hr_allowed_szn,
                "p_hr_allowed_rate_szn": p_hr_allowed_rate_szn,
                "b_barrel_rate_14": b_barrel_rate_14,

                "b_ev_mean_14": b_ev_mean_14,
                "b_la_mean_14": b_la_mean_14,
                "b_hardhit_rate_14": b_hardhit_rate_14,
                "b_fb_rate_14": b_fb_rate_14,

                "b_ev_mean_szn": b_ev_mean_szn,
                "b_la_mean_szn": b_la_mean_szn,
                "b_hardhit_rate_szn": b_hardhit_rate_szn,
                "b_fb_rate_szn": b_fb_rate_szn,
                "b_barrel_rate_szn": b_barrel_rate_szn,

                "p_ev_allowed_mean_30": p_ev_allowed_mean_30,
                "p_hardhit_allowed_rate_30": p_hardhit_allowed_rate_30,
                "p_fb_allowed_rate_30": p_fb_allowed_rate_30,
                "p_barrel_allowed_rate_30": p_barrel_allowed_rate_30,

                "p_ev_allowed_mean_szn": p_ev_allowed_mean_szn,
                "p_hardhit_allowed_rate_szn": p_hardhit_allowed_rate_szn,
                "p_fb_allowed_rate_szn": p_fb_allowed_rate_szn,
                "p_barrel_allowed_rate_szn": p_barrel_allowed_rate_szn,

                "ev_edge_14_30": ev_edge_14_30,    
                "hardhit_edge_14_30": hardhit_edge_14_30,
                "fb_edge_14_30": fb_edge_14_30,
                "barrel_edge_14_30": barrel_edge_14_30,
                "hr_rate_edge_14_30": hr_rate_edge_14_30,

            }
        )

    features_df = pd.DataFrame(feature_rows)

    out_path = PROCESSED_DIR / f"train_table_{start_date}_to_{end_date}.parquet"
    features_df.to_parquet(out_path, index=False)

    return FeaturesBuildResult(
        features_df=features_df,
        output_path=out_path
    )


if __name__ == "__main__":
    result = build_features_for_range("2024-03-20", "2024-10-01")
    full_path = PROCESSED_DIR / "train_table_2024_full_season.parquet"
    result.features_df.to_parquet(full_path, index=False)

    print(f"Saved features to: {full_path}")
    print(f"Rows: {len(result.features_df):,}")
    print(result.features_df.head(5))
    print("\nLabel HR rate:", result.features_df["hr_hit"].mean())

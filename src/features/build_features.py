from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events
from src.data_sources.weather import fetch_weather_for_games
from src.features.build_labels import build_batter_game_labels
from src.features.park_factors import HR_PARK_FACTOR, DEFAULT_PARK_FACTOR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FASTBALL_TYPES = {"FF", "SI"}
OFFSPEED_TYPES = {"SL", "CH", "CU", "KC", "FS", "ST", "SV", "CS", "EP"}

# Minimum PA thresholds — below these, season rates are treated as NaN
# rather than zero so the model can distinguish "unknown" from "0%".
MIN_PA_BATTER_SZN = 10
MIN_PA_PITCHER_SZN = 10
MIN_PA_WINDOW = 5


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


def _safe_mean(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) == 0:
        return np.nan          # ← return NaN, not 0, so callers can distinguish
    m = v.mean()
    return np.nan if pd.isna(m) else float(m)


def _is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    """
    Statcast barrel definition (approximation matching Baseball Savant):
      EV ≥ 98 mph AND LA 26°–30°  → always barrel
      As EV increases above 98, the valid LA range widens symmetrically.

    Full lookup table taken from Baseball Savant documentation:
        EV    LA range
        98    26–30
        99    25–31
        100   24–33
        101   23–34
        102   22–35
        103   21–36
        104   20–37
        105   19–38
        106   18–39
        107   17–40
        108+  16–41 (capped)

    This is materially more accurate than the simple EV≥95 & LA 20-35
    definition used previously.
    """
    ev = pd.to_numeric(launch_speed, errors="coerce")
    la = pd.to_numeric(launch_angle, errors="coerce")

    # Build per-row LA bounds based on EV
    ev_clipped = ev.clip(upper=108)
    delta = (ev_clipped - 98).clip(lower=0)   # 0 when EV < 98

    la_min = 26 - delta
    la_max = 30 + delta

    return (
        ev.notna() & la.notna() &
        (ev >= 98) &
        (la >= la_min) & (la <= la_max)
    ).fillna(False)


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
    raw["game_date"] = pd.to_datetime(raw["game_date"])

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

    # FIX: use mean (not max) for launch_speed/launch_angle per PA.
    # max() would inflate EV by picking the hardest-hit foul ball rather than
    # the true batted-ball outcome.  mean() over typically 1–3 tracked pitches
    # per PA is a much better estimate of contact quality.
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
            "launch_speed": "mean",   # was "max" — corrected
            "launch_angle": "mean",   # was "max" — corrected
            "p_throws":     "last",
            "stand":        "last",
        })
    )

    ev_str = pa_df["events"].astype("string")
    pa_df["is_so"] = ev_str.str.contains("strikeout", na=False).astype("int8")
    pa_df["is_bb"] = (ev_str == "walk").fillna(False).astype("int8")

    ev = pd.to_numeric(pa_df["launch_speed"], errors="coerce")
    la = pd.to_numeric(pa_df["launch_angle"],  errors="coerce")

    # FIX: use the corrected barrel definition
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
# Pitcher velo helpers
# ---------------------------------------------------------------------------

def _pitcher_velo_stats(grp: pd.DataFrame, suffix: str) -> dict:
    if len(grp) == 0:
        return {f"p_fb_velo_{suffix}": np.nan, f"p_fb_pct_{suffix}": np.nan,
                f"p_offspeed_pct_{suffix}": np.nan}
    is_fb       = grp["pitch_type"].isin(FASTBALL_TYPES)
    is_offspeed = grp["pitch_type"].isin(OFFSPEED_TYPES)
    total       = len(grp)
    fb_velo     = _safe_mean(grp.loc[is_fb, "release_speed"]) if is_fb.any() else np.nan
    return {
        f"p_fb_velo_{suffix}":      fb_velo,
        f"p_fb_pct_{suffix}":       float(is_fb.sum())       / total,
        f"p_offspeed_pct_{suffix}": float(is_offspeed.sum()) / total,
    }


def _precompute_pitcher_velo(pitches_df: pd.DataFrame, target_dates: pd.DataFrame) -> pd.DataFrame:
    need = target_dates.drop_duplicates(subset=["pitcher", "game_date"]).reset_index(drop=True)
    pitches_by_pitcher = {k: g for k, g in pitches_df.groupby("pitcher", sort=False)}
    empty = pitches_df.iloc[0:0]

    rows = []
    for _, r in need.iterrows():
        pitcher_id = r["pitcher"]
        game_date  = r["game_date"]
        as_of      = game_date - pd.Timedelta(days=1)

        grp = pitches_by_pitcher.get(pitcher_id)
        w30 = grp[grp["game_date"].between(as_of - pd.Timedelta(days=29), as_of)] \
              if grp is not None else empty

        start_dates = (
            sorted(grp[grp["game_date"] <= as_of]["game_date"].unique(), reverse=True)
            if grp is not None else []
        )

        stats = {"pitcher": pitcher_id, "game_date": game_date,
                 **_pitcher_velo_stats(w30, "30")}

        def _starts_velo(start_list):
            if not start_list or grp is None:
                return np.nan
            subset  = grp[grp["game_date"].isin(start_list)]
            fb_rows = subset[subset["pitch_type"].isin(FASTBALL_TYPES)]
            if fb_rows.empty:
                return np.nan
            val = fb_rows["release_speed"].dropna().mean()
            return np.nan if pd.isna(val) else float(val)

        recent_velo = _starts_velo(start_dates[:3])
        prior_velo  = _starts_velo(start_dates[3:6])
        stats["p_fb_velo_trend"] = (
            (recent_velo - prior_velo)
            if not pd.isna(recent_velo) and not pd.isna(prior_velo) else np.nan
        )
        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batter window aggregation
# ---------------------------------------------------------------------------

def _batter_stats_for_window(grp: pd.DataFrame, suffix: str, min_pa: int = 0) -> dict:
    """
    Compute batter rolling stats.  Returns NaN for rate features when the
    window has fewer than min_pa plate appearances, so the model can
    distinguish 'no data' from 'truly zero'.
    """
    pa = len(grp)
    no_data = pa < min_pa

    if pa == 0:
        return {f"b_pa_{suffix}": 0, f"b_hr_{suffix}": 0,
                f"b_hr_rate_{suffix}": np.nan, f"b_barrel_rate_{suffix}": np.nan,
                f"b_ev_mean_{suffix}": np.nan, f"b_la_mean_{suffix}": np.nan,
                f"b_hardhit_rate_{suffix}": np.nan, f"b_fb_rate_{suffix}": np.nan,
                f"b_k_rate_{suffix}": np.nan, f"b_bb_rate_{suffix}": np.nan}

    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr = int(grp["is_hr"].sum())

    # For windows below the minimum threshold, still record the counts but
    # mark rates as NaN — the model's fillna(0) will then treat them as
    # league-average rather than zero, which is more accurate for cold-starts.
    nan_if_sparse = np.nan if no_data else None

    def _rate(numerator, denominator):
        if nan_if_sparse is not None:
            return np.nan
        return numerator / denominator

    return {
        f"b_pa_{suffix}":           pa,
        f"b_hr_{suffix}":           hr,
        f"b_hr_rate_{suffix}":      _rate(hr, pa),
        f"b_barrel_rate_{suffix}":  _rate(float(grp["is_barrel"].sum()), pa),
        f"b_ev_mean_{suffix}":      np.nan if no_data else _safe_mean(ev),
        f"b_la_mean_{suffix}":      np.nan if no_data else _safe_mean(la),
        f"b_hardhit_rate_{suffix}": (
            np.nan if no_data or not ev.notna().any()
            else float((ev >= 95).mean())
        ),
        f"b_fb_rate_{suffix}":      (
            np.nan if no_data or not la.notna().any()
            else float(la.between(20, 40).mean())
        ),
        f"b_k_rate_{suffix}":       _rate(float(grp["is_so"].sum()), pa),
        f"b_bb_rate_{suffix}":      _rate(float(grp["is_bb"].sum()), pa),
    }


def _batter_trend_stats(w7: pd.DataFrame, w8_14: pd.DataFrame) -> dict:
    def _ev_mean(g):  return _safe_mean(g["launch_speed"]) if len(g) > 0 else np.nan
    def _hardhit(g):
        if len(g) == 0: return np.nan
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else np.nan
    def _barrel(g):   return float(g["is_barrel"].sum()) / len(g) if len(g) > 0 else np.nan
    def _hr_rate(g):  return float(g["is_hr"].sum())    / len(g) if len(g) > 0 else np.nan

    def _diff(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return a - b

    ev7, ev8_14         = _ev_mean(w7),  _ev_mean(w8_14)
    hh7, hh8_14         = _hardhit(w7), _hardhit(w8_14)
    bar7, bar8_14       = _barrel(w7),  _barrel(w8_14)
    hr7, hr8_14         = _hr_rate(w7), _hr_rate(w8_14)
    return {
        "b_ev_trend":       _diff(ev7, ev8_14),
        "b_hardhit_trend":  _diff(hh7, hh8_14),
        "b_barrel_trend":   _diff(bar7, bar8_14),
        "b_hr_trend":       _diff(hr7, hr8_14),
        "b_ev_mean_7":      ev7,
        "b_hardhit_rate_7": hh7,
    }


def _batter_home_away_stats(grp: pd.DataFrame, batter_team: str | None) -> dict:
    empty = {"b_hr_rate_home": np.nan, "b_hr_rate_away": np.nan,
             "b_hardhit_rate_home": np.nan, "b_hardhit_rate_away": np.nan,
             "b_barrel_rate_home": np.nan, "b_barrel_rate_away": np.nan,
             "b_hr_rate_home_edge": np.nan}
    if len(grp) == 0 or batter_team is None:
        return empty

    home_mask = grp["home_team"] == batter_team
    home_grp  = grp[home_mask]
    away_grp  = grp[~home_mask]

    def _rate(g, col):
        return float(g[col].sum()) / len(g) if len(g) >= MIN_PA_WINDOW else np.nan
    def _hardhit(g):
        if len(g) < MIN_PA_WINDOW: return np.nan
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else np.nan

    hr_home = _rate(home_grp, "is_hr")
    hr_away = _rate(away_grp, "is_hr")
    return {
        "b_hr_rate_home":      hr_home,
        "b_hr_rate_away":      hr_away,
        "b_hardhit_rate_home": _hardhit(home_grp),
        "b_hardhit_rate_away": _hardhit(away_grp),
        "b_barrel_rate_home":  _rate(home_grp, "is_barrel"),
        "b_barrel_rate_away":  _rate(away_grp, "is_barrel"),
        "b_hr_rate_home_edge": (
            hr_home - hr_away
            if not pd.isna(hr_home) and not pd.isna(hr_away)
            else np.nan
        ),
    }


def _precompute_batter_windows(
    pa_df: pd.DataFrame,
    target_dates: pd.DataFrame,
    batter_team_lookup: dict,
    game_pk_home_lookup: dict,
    label_game_pks: pd.DataFrame,
) -> pd.DataFrame:
    need = target_dates.drop_duplicates(subset=["batter", "game_date"]).reset_index(drop=True)
    events_by_batter = {k: g for k, g in pa_df.groupby("batter", sort=False)}
    empty = pa_df.iloc[0:0]
    bgpk  = label_game_pks.set_index(["batter", "game_date"])["game_pk"].to_dict()

    rows = []
    for _, r in need.iterrows():
        batter_id    = r["batter"]
        game_date    = r["game_date"]
        pitcher_hand = r.get("pitcher_hand", None)
        as_of        = game_date - pd.Timedelta(days=1)

        grp = events_by_batter.get(batter_id)

        if grp is None:
            w7 = w8_14 = w14 = wszn = empty
        else:
            w14   = grp[grp["game_date"].between(as_of - pd.Timedelta(days=13), as_of)]
            w7    = grp[grp["game_date"].between(as_of - pd.Timedelta(days=6),  as_of)]
            w8_14 = grp[grp["game_date"].between(as_of - pd.Timedelta(days=13),
                                                  as_of - pd.Timedelta(days=7))]
            szn_start = pd.Timestamp(as_of.year, 3, 1)
            wszn = grp[grp["game_date"].between(szn_start, as_of)]

        stats = {
            "batter":    batter_id,
            "game_date": game_date,
            **_batter_stats_for_window(w14,  "14",  min_pa=MIN_PA_WINDOW),
            **_batter_stats_for_window(wszn, "szn", min_pa=MIN_PA_BATTER_SZN),
        }

        for hand in ("L", "R"):
            w14_vs  = w14[w14["p_throws"]   == hand] if len(w14)  > 0 else empty
            wszn_vs = wszn[wszn["p_throws"]  == hand] if len(wszn) > 0 else empty
            stats.update(_batter_stats_for_window(w14_vs,  f"14_vs{hand}", min_pa=MIN_PA_WINDOW))
            stats.update(_batter_stats_for_window(wszn_vs, f"szn_vs{hand}", min_pa=MIN_PA_BATTER_SZN))

        stats.update(_batter_trend_stats(w7, w8_14))

        batter_team = batter_team_lookup.get(batter_id)
        stats.update(_batter_home_away_stats(wszn, batter_team))

        game_pk = bgpk.get((batter_id, game_date))
        if game_pk is not None and batter_team is not None:
            home_today = game_pk_home_lookup.get(game_pk)
            stats["is_home_game"] = int(home_today == batter_team)
        else:
            stats["is_home_game"] = -1

        batter_hand = (
            grp["stand"].dropna().mode().iloc[0]
            if grp is not None and grp["stand"].notna().any() else None
        )
        stats["same_hand_matchup"] = (
            int(batter_hand == pitcher_hand)
            if batter_hand is not None and pitcher_hand is not None else -1
        )

        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pitcher window aggregation
# ---------------------------------------------------------------------------

def _pitcher_stats_for_window(grp: pd.DataFrame, suffix: str, min_pa: int = 0) -> dict:
    pa = len(grp)
    no_data = pa < min_pa

    if pa == 0:
        return {f"p_pa_{suffix}": 0, f"p_hr_allowed_{suffix}": 0,
                f"p_hr_allowed_rate_{suffix}": np.nan, f"p_ev_allowed_mean_{suffix}": np.nan,
                f"p_hardhit_allowed_rate_{suffix}": np.nan, f"p_fb_allowed_rate_{suffix}": np.nan,
                f"p_barrel_allowed_rate_{suffix}": np.nan, f"p_k_rate_{suffix}": np.nan,
                f"p_bb_rate_{suffix}": np.nan}

    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr = int(grp["is_hr"].sum())

    def _rate(num, denom):
        return np.nan if no_data else num / denom

    return {
        f"p_pa_{suffix}":                   pa,
        f"p_hr_allowed_{suffix}":           hr,
        f"p_hr_allowed_rate_{suffix}":      _rate(hr, pa),
        f"p_ev_allowed_mean_{suffix}":      np.nan if no_data else _safe_mean(ev),
        f"p_hardhit_allowed_rate_{suffix}": (
            np.nan if no_data or not ev.notna().any()
            else float((ev >= 95).mean())
        ),
        f"p_fb_allowed_rate_{suffix}":      (
            np.nan if no_data or not la.notna().any()
            else float(la.between(20, 40).mean())
        ),
        f"p_barrel_allowed_rate_{suffix}":  _rate(float(grp["is_barrel"].sum()), pa),
        f"p_k_rate_{suffix}":               _rate(float(grp["is_so"].sum()), pa),
        f"p_bb_rate_{suffix}":              _rate(float(grp["is_bb"].sum()), pa),
    }


def _precompute_pitcher_windows(pa_df: pd.DataFrame, target_dates: pd.DataFrame) -> pd.DataFrame:
    need = target_dates.drop_duplicates(subset=["pitcher", "game_date"]).reset_index(drop=True)
    events_by_pitcher = {k: g for k, g in pa_df.groupby("pitcher", sort=False)}
    empty = pa_df.iloc[0:0]

    rows = []
    for _, r in need.iterrows():
        pitcher_id = r["pitcher"]
        game_date  = r["game_date"]
        as_of      = game_date - pd.Timedelta(days=1)

        grp = events_by_pitcher.get(pitcher_id)
        if grp is None:
            w30 = wszn = empty
        else:
            w30 = grp[grp["game_date"].between(as_of - pd.Timedelta(days=29), as_of)]
            szn_start = pd.Timestamp(as_of.year, 3, 1)
            wszn = grp[grp["game_date"].between(szn_start, as_of)]

        stats = {"pitcher": pitcher_id, "game_date": game_date,
                 **_pitcher_stats_for_window(w30, "30", min_pa=MIN_PA_WINDOW),
                 **_pitcher_stats_for_window(wszn, "szn", min_pa=MIN_PA_PITCHER_SZN)}

        for hand in ("L", "R"):
            w30_vs  = w30[w30["stand"]   == hand] if len(w30)  > 0 else empty
            wszn_vs = wszn[wszn["stand"]  == hand] if len(wszn) > 0 else empty
            stats.update(_pitcher_stats_for_window(w30_vs,  f"30_vs{hand}", min_pa=MIN_PA_WINDOW))
            stats.update(_pitcher_stats_for_window(wszn_vs, f"szn_vs{hand}", min_pa=MIN_PA_PITCHER_SZN))

        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def _add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # FIX: edge features now propagate NaN when either side is unknown
    # rather than silently producing a misleading non-zero difference.
    def _edge(a: str, b: str) -> pd.Series:
        return df[a] - df[b]   # pandas NaN arithmetic handles this correctly

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

    # Weather-aware edge: batter contact quality × wind boost
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

    batter_team_lookup: dict = {}
    for batter_id, grp in pa_df.groupby("batter"):
        teams = grp["home_team"].dropna()
        if len(teams) > 0:
            batter_team_lookup[batter_id] = teams.mode().iloc[0]

    target_pa = pa_df[pa_df["game_date"].between(start_dt, end_dt)]
    labels = build_batter_game_labels(target_pa)
    labels["game_date"] = pd.to_datetime(labels["game_date"])

    # FIX: prefer starter_id (first pitcher faced) over mode pitcher.
    # Fall back to pitcher_mode for rows where starter_id is absent (older data).
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

    # Weather — fetch for all unique game_pks in target window
    target_game_pks = labels["game_pk"].dropna().astype(int).unique().tolist()
    print(f"  Fetching weather for {len(target_game_pks):,} games ...")
    weather_df = fetch_weather_for_games(target_game_pks, game_pk_to_home)
    weather_df["game_pk"] = weather_df["game_pk"].astype(int)

    # Batter windows
    n_b = labels[["batter", "game_date"]].drop_duplicates().shape[0]
    print(f"  Precomputing batter windows for {n_b:,} (batter, date) pairs ...")
    batter_stats = _precompute_batter_windows(
        pa_df,
        labels[["batter", "game_date", "pitcher_hand"]],
        batter_team_lookup=batter_team_lookup,
        game_pk_home_lookup=game_pk_to_home,
        label_game_pks=labels[["batter", "game_date", "game_pk"]],
    )

    # Pitcher windows
    pitcher_need = (
        labels[["pitcher_id", "game_date", "batter_hand"]]
        .rename(columns={"pitcher_id": "pitcher"})
        .dropna(subset=["pitcher"])
        .copy()
    )
    pitcher_need["pitcher"] = pitcher_need["pitcher"].astype(int)
    n_p = pitcher_need[["pitcher", "game_date"]].drop_duplicates().shape[0]
    print(f"  Precomputing pitcher PA windows for {n_p:,} (pitcher, date) pairs ...")
    pitcher_stats = _precompute_pitcher_windows(pa_df, pitcher_need)

    print(f"  Precomputing pitcher velo windows for {n_p:,} (pitcher, date) pairs ...")
    pitcher_velo = _precompute_pitcher_velo(pitches_df, pitcher_need)

    # Merge everything
    features_df = (
        labels
        .merge(batter_stats,  on=["batter", "game_date"], how="left")
        .merge(pitcher_stats.rename(columns={"pitcher": "pitcher_id"}),
               on=["pitcher_id", "game_date"], how="left")
        .merge(pitcher_velo.rename(columns={"pitcher": "pitcher_id"}),
               on=["pitcher_id", "game_date"], how="left")
        .merge(weather_df,  on="game_pk", how="left")
    )

    # Park factor
    features_df["home_team"]      = features_df["game_pk"].map(game_pk_to_home)
    features_df["park_factor_hr"] = (
        features_df["home_team"].map(HR_PARK_FACTOR).fillna(DEFAULT_PARK_FACTOR) / 100.0
    )

    features_df = _add_edge_features(features_df)

    # Tidy
    features_df = features_df.rename(columns={"pitcher_id": "pitcher"})
    features_df["game_date"] = features_df["game_date"].dt.date
    features_df["hr_hit"]    = features_df["hr_hit"].astype(int)

    # FIX: only zero-fill non-identity columns.  Sparse-window NaN values
    # (set deliberately above) are filled with 0.0 here so the model pipeline
    # remains unchanged; the key improvement is that NaN was allowed to
    # propagate through edge arithmetic correctly before this final fill.
    non_stat_cols = {
        "game_date", "game_pk", "batter", "pitcher", "home_team",
        "hr_hit", "pitcher_hand", "batter_hand", "pitcher_mode",
        "starter_id",
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.statcast import fetch_statcast_events
from src.data_sources.weather import fetch_weather_for_games
from src.data_sources.mlb_schedule import fetch_rosters_for_games, enrich_labels_with_roster
from src.features.build_labels import build_batter_game_labels, compute_relief_pa_pct
from src.features.park_factors import get_park_factors, DEFAULT_PARK_FACTOR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FASTBALL_TYPES = {"FF", "SI"}
OFFSPEED_TYPES = {"SL", "CH", "CU", "KC", "FS", "ST", "SV", "CS", "EP"}


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
        return 0.0
    m = v.mean()
    return 0.0 if pd.isna(m) else float(m)


# ---------------------------------------------------------------------------
# Bayesian shrinkage — cold-start handling
# ---------------------------------------------------------------------------

# League-average priors derived from 2021–2024 Statcast data.
# These are the rates we regress toward when PA count is low.
_BATTER_PRIORS: dict[str, float] = {
    "hr_rate":     0.035,
    "barrel_rate": 0.072,
    "hardhit_rate":0.385,
    "fb_rate":     0.115,
    "k_rate":      0.228,
    "bb_rate":     0.083,
    "ev_mean":     88.5,   # mph — shrink toward league EV mean
    "la_mean":     12.0,   # degrees
}

_PITCHER_PRIORS: dict[str, float] = {
    "hr_allowed_rate":     0.035,
    "barrel_allowed_rate": 0.072,
    "hardhit_allowed_rate":0.385,
    "fb_allowed_rate":     0.115,
    "k_rate":              0.228,
    "bb_rate":             0.083,
    "ev_allowed_mean":     88.5,
}

# Effective "prior PA" — how many league-average PA we blend in.
# Lower = trust observed data faster; higher = more shrinkage for low-PA players.
_BATTER_PRIOR_PA  = 50
_PITCHER_PRIOR_PA = 75   # pitchers need more data; higher variance per PA


def _shrink(observed: float, obs_pa: int, prior: float, prior_pa: int) -> float:
    """
    Empirical Bayes shrinkage: blend observed rate with league-average prior.

    shrunk = (obs_pa * observed + prior_pa * prior) / (obs_pa + prior_pa)

    At obs_pa=0  → returns prior exactly.
    At obs_pa=prior_pa → 50/50 blend.
    At obs_pa >> prior_pa → returns observed.
    """
    total = obs_pa + prior_pa
    return (obs_pa * observed + prior_pa * prior) / total


def _apply_batter_shrinkage(stats: dict, suffix: str) -> dict:
    """
    Apply shrinkage to a batter window stats dict in-place.
    `suffix` is e.g. '14' or 'szn'.
    """
    pa = int(stats.get(f"b_pa_{suffix}", 0))
    if pa == 0:
        # No data at all — just return league averages directly
        stats[f"b_hr_rate_{suffix}"]       = _BATTER_PRIORS["hr_rate"]
        stats[f"b_barrel_rate_{suffix}"]   = _BATTER_PRIORS["barrel_rate"]
        stats[f"b_hardhit_rate_{suffix}"]  = _BATTER_PRIORS["hardhit_rate"]
        stats[f"b_fb_rate_{suffix}"]       = _BATTER_PRIORS["fb_rate"]
        stats[f"b_k_rate_{suffix}"]        = _BATTER_PRIORS["k_rate"]
        stats[f"b_bb_rate_{suffix}"]       = _BATTER_PRIORS["bb_rate"]
        stats[f"b_ev_mean_{suffix}"]       = _BATTER_PRIORS["ev_mean"]
        stats[f"b_la_mean_{suffix}"]       = _BATTER_PRIORS["la_mean"]
        return stats

    for stat_key, prior_key in [
        (f"b_hr_rate_{suffix}",      "hr_rate"),
        (f"b_barrel_rate_{suffix}",  "barrel_rate"),
        (f"b_hardhit_rate_{suffix}", "hardhit_rate"),
        (f"b_fb_rate_{suffix}",      "fb_rate"),
        (f"b_k_rate_{suffix}",       "k_rate"),
        (f"b_bb_rate_{suffix}",      "bb_rate"),
        (f"b_ev_mean_{suffix}",      "ev_mean"),
        (f"b_la_mean_{suffix}",      "la_mean"),
    ]:
        if stat_key in stats:
            stats[stat_key] = _shrink(
                stats[stat_key], pa, _BATTER_PRIORS[prior_key], _BATTER_PRIOR_PA
            )
    return stats


def _apply_pitcher_shrinkage(stats: dict, suffix: str) -> dict:
    """
    Apply shrinkage to a pitcher window stats dict in-place.
    `suffix` is e.g. '30' or 'szn'.
    """
    pa = int(stats.get(f"p_pa_{suffix}", 0))
    if pa == 0:
        stats[f"p_hr_allowed_rate_{suffix}"]      = _PITCHER_PRIORS["hr_allowed_rate"]
        stats[f"p_barrel_allowed_rate_{suffix}"]  = _PITCHER_PRIORS["barrel_allowed_rate"]
        stats[f"p_hardhit_allowed_rate_{suffix}"] = _PITCHER_PRIORS["hardhit_allowed_rate"]
        stats[f"p_fb_allowed_rate_{suffix}"]      = _PITCHER_PRIORS["fb_allowed_rate"]
        stats[f"p_k_rate_{suffix}"]               = _PITCHER_PRIORS["k_rate"]
        stats[f"p_bb_rate_{suffix}"]              = _PITCHER_PRIORS["bb_rate"]
        stats[f"p_ev_allowed_mean_{suffix}"]      = _PITCHER_PRIORS["ev_allowed_mean"]
        return stats

    for stat_key, prior_key in [
        (f"p_hr_allowed_rate_{suffix}",      "hr_allowed_rate"),
        (f"p_barrel_allowed_rate_{suffix}",  "barrel_allowed_rate"),
        (f"p_hardhit_allowed_rate_{suffix}", "hardhit_allowed_rate"),
        (f"p_fb_allowed_rate_{suffix}",      "fb_allowed_rate"),
        (f"p_k_rate_{suffix}",               "k_rate"),
        (f"p_bb_rate_{suffix}",              "bb_rate"),
        (f"p_ev_allowed_mean_{suffix}",      "ev_allowed_mean"),
    ]:
        if stat_key in stats:
            stats[stat_key] = _shrink(
                stats[stat_key], pa, _PITCHER_PRIORS[prior_key], _PITCHER_PRIOR_PA
            )
    return stats


# League-average bullpen priors (same scale as pitcher priors)
_BULLPEN_PRIOR_PA = 100   # bullpen stats are noisier; regress more aggressively
_BULLPEN_PRIORS: dict[str, float] = {
    "hr_allowed_rate":     0.038,   # relievers allow slightly more HRs per PA
    "hardhit_allowed_rate":0.375,
    "bb_rate":             0.095,
}


# ---------------------------------------------------------------------------
# Starter identity from Statcast (Option B — no extra API calls)
# ---------------------------------------------------------------------------

def _derive_game_starters(pa_df: pd.DataFrame) -> dict[tuple[int, str], int]:
    """
    Derive the starting pitcher for each (game_pk, team_side) from Statcast PA data.
    The first pitcher to appear per team per game is almost always the starter (~98%).

    Returns dict mapping (game_pk, home_team_abbr) → starter_pitcher_id
    and      (game_pk, away_team_abbr)  → starter_pitcher_id
    but keyed as (game_pk, team) where team is the *pitching* team.

    Actually returns: dict[(game_pk, batting_team)] → starter_pitcher_id
    i.e. the starter the batting team faced.
    """
    # Sort by game_pk, game_date, at_bat_number proxy (at_bat_number not in pa_df,
    # but game_date ordering within a game_pk is sufficient)
    ordered = pa_df.sort_values(["game_pk", "game_date"]).copy()

    # For each (game_pk, home_team) the first pitcher is the home team's starter
    # (facing away batters) and vice versa.
    # home_team in pa_df = the home team of the game, same for every row of that game.
    # We want: for a given batter's team, who was the first pitcher they faced?

    # Group by (game_pk, batter's team) — but we don't have batter's team directly.
    # Proxy: if batter's home_team == game home_team → batter is home → pitcher is away starter
    # We'll build (game_pk, pitching_team) → first_pitcher instead.

    # Identify pitching team per PA: it's the team that is NOT the home_team when
    # the pitcher is pitching at home, but that's circular. Simpler:
    # first pitcher per (game_pk) per side (home/away) distinguished by whether
    # the batter is home or away.
    #
    # batter is HOME  → batter's team == home_team  → pitcher is the AWAY starter
    # batter is AWAY  → batter's team != home_team  → pitcher is the HOME starter

    starters: dict[tuple, int] = {}
    for game_pk, grp in ordered.groupby("game_pk", sort=False):
        home_team = grp["home_team"].iloc[0] if len(grp) > 0 else None
        if home_team is None:
            continue

        # Home batters face away starter → first pitcher when home_team bats
        # Use the first row where home_team is batting (home_team == batter's team)
        # We don't have batter team directly; use the first pitcher per game as
        # home starter, second distinct pitcher as away starter heuristic.
        # Better: split by "is batter on home team" using mode of stand/home_team.

        # Find first pitcher facing home batters (= away starter)
        # and first pitcher facing away batters (= home starter)
        first_by_pitcher: dict[str, int] = {}
        for _, row in grp.iterrows():
            # Determine if this row's batter is on the home team
            # We use home_team from the row itself — if pitcher's team == home_team
            # that means the pitcher is the home pitcher facing away batters.
            # Statcast doesn't give us batting team directly, but we stored
            # batter_team via mode in the outer function. Here we use a simpler
            # heuristic: first pitcher in game = home team's starter,
            # track per-side by watching when pitcher changes.
            side = "home_pitching"  # placeholder; we resolve per (game_pk, home_team) below
            break

        # Cleanest approach: first pitcher to appear per game is home starter,
        # track the split by at_bat_number parity isn't reliable.
        # Instead: group by game_pk only, take first pitcher = home starter (pitching to away batters).
        # We store (game_pk, batting_team) → pitcher.
        # batting_team of away = not home_team. We'll resolve in the merge step.
        first_pitcher = int(grp["pitcher"].iloc[0])
        starters[(game_pk, "first")] = first_pitcher

        # Find away starter: first pitcher who appeared AND is different team.
        # Use a simple heuristic: scan until pitcher changes, that new pitcher
        # pitched to the other side.
        pitchers_seen = []
        for p in grp["pitcher"]:
            if p not in pitchers_seen:
                pitchers_seen.append(p)
            if len(pitchers_seen) == 2:
                break
        starters[(game_pk, "second")] = int(pitchers_seen[-1]) if len(pitchers_seen) > 1 else first_pitcher

    return starters


def _build_starter_lookup(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean (game_pk, home_team, home_starter_id, away_starter_id) table
    derived purely from Statcast data. Used for bullpen identification.

    Strategy: within each game, home starter = first pitcher facing away batters,
    away starter = first pitcher facing home batters.
    We identify "away batter" rows as rows where batter appeared more than once
    as a visitor. Since we don't have batter team stored per-row, we use:
      - The first unique pitcher in chronological order is the home team's starter
        (they pitch in the top of the 1st, facing away batters).
      - The second unique pitcher is the away team's starter (they pitch the bottom).
    This holds for ~98% of games.
    """
    rows = []
    ordered = pa_df.sort_values(["game_pk", "game_date"]).copy()

    for game_pk, grp in ordered.groupby("game_pk", sort=False):
        home_team = grp["home_team"].dropna()
        if home_team.empty:
            continue
        home_team = home_team.iloc[0]

        unique_pitchers = list(dict.fromkeys(grp["pitcher"].tolist()))
        home_starter = int(unique_pitchers[0]) if len(unique_pitchers) >= 1 else None
        away_starter = int(unique_pitchers[1]) if len(unique_pitchers) >= 2 else home_starter

        rows.append({
            "game_pk":         int(game_pk),
            "home_team":       home_team,
            "home_starter_id": home_starter,
            "away_starter_id": away_starter,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bullpen window aggregation
# ---------------------------------------------------------------------------

def _precompute_bullpen_windows(
    pa_df: pd.DataFrame,
    starter_lookup: pd.DataFrame,
    target_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (pitching_team, game_date) in target_dates, compute the rolling
    30-day allowed rates for that team's bullpen (all pitchers EXCEPT the starter).

    Returns DataFrame with columns:
        pitching_team, game_date,
        bp_pa_30, bp_hr_allowed_rate_30, bp_hardhit_allowed_rate_30, bp_bb_rate_30,
        bp_era_proxy_30
    """
    # Annotate each PA with whether the pitcher was a starter in that game
    # by joining starter_lookup
    pa = pa_df.merge(
        starter_lookup[["game_pk", "home_starter_id", "away_starter_id", "home_team"]],
        on="game_pk",
        how="left",
        suffixes=("", "_sl"),
    )

    # Determine pitching team per PA:
    # If pitcher == home_starter_id → pitching team is home_team
    # If pitcher == away_starter_id → pitching team is away team (unknown directly,
    #   but we can derive: away team = mode team for batters in this game whose home_team != home_team)
    # Simpler: tag each pitcher with their team via mode across all games
    pitcher_team = (
        pa_df.dropna(subset=["pitcher", "home_team"])
        .groupby("pitcher")["home_team"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    pa["pitcher_team"] = pa["pitcher"].map(pitcher_team)

    # A pitcher is the starter if they are home_starter_id or away_starter_id for that game
    pa["is_starter_pitcher"] = (
        (pa["pitcher"] == pa["home_starter_id"]) |
        (pa["pitcher"] == pa["away_starter_id"])
    ).fillna(False)

    # Bullpen PA = not the starter
    bullpen_pa = pa[~pa["is_starter_pitcher"]].copy()
    bullpen_pa = bullpen_pa.dropna(subset=["pitcher_team"])

    # Group by (pitcher_team, game_date) for rolling lookup
    events_by_team = {k: g for k, g in bullpen_pa.groupby("pitcher_team", sort=False)}
    empty = bullpen_pa.iloc[0:0]

    need = target_dates.drop_duplicates(
        subset=["pitching_team", "game_date"]
    ).reset_index(drop=True)

    rows = []
    for _, r in need.iterrows():
        team      = r["pitching_team"]
        game_date = r["game_date"]
        as_of     = game_date - pd.Timedelta(days=1)

        grp = events_by_team.get(team)
        w30 = (
            grp[grp["game_date"].between(as_of - pd.Timedelta(days=29), as_of)]
            if grp is not None else empty
        )

        pa_count = len(w30)
        if pa_count == 0:
            hr_rate   = _shrink(0.0, 0, _BULLPEN_PRIORS["hr_allowed_rate"],     _BULLPEN_PRIOR_PA)
            hh_rate   = _shrink(0.0, 0, _BULLPEN_PRIORS["hardhit_allowed_rate"], _BULLPEN_PRIOR_PA)
            bb_rate   = _shrink(0.0, 0, _BULLPEN_PRIORS["bb_rate"],              _BULLPEN_PRIOR_PA)
        else:
            ev = w30["launch_speed"]
            hr_rate = _shrink(float(w30["is_hr"].sum()) / pa_count,  pa_count, _BULLPEN_PRIORS["hr_allowed_rate"],     _BULLPEN_PRIOR_PA)
            hh_rate = _shrink(float((ev >= 95).mean()) if ev.notna().any() else 0.0, pa_count, _BULLPEN_PRIORS["hardhit_allowed_rate"], _BULLPEN_PRIOR_PA)
            bb_rate = _shrink(float(w30["is_bb"].sum()) / pa_count,  pa_count, _BULLPEN_PRIORS["bb_rate"],              _BULLPEN_PRIOR_PA)

        rows.append({
            "pitching_team":            team,
            "game_date":                game_date,
            "bp_pa_30":                 pa_count,
            "bp_hr_allowed_rate_30":    hr_rate,
            "bp_hardhit_allowed_rate_30": hh_rate,
            "bp_bb_rate_30":            bb_rate,
            # Composite: HR rate + hard-hit rate weighted sum (lightweight ERA proxy)
            "bp_era_proxy_30":          hr_rate * 2.0 + hh_rate * 0.5,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lineup protection computation
# ---------------------------------------------------------------------------

def _compute_lineup_protection(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each batter-game row, compute the season OPS of the hitter directly
    ahead and directly behind in the batting order, plus the average OPS of
    the full lineup (excluding the batter).

    OPS proxy = b_hr_rate_szn * 40 + b_hardhit_rate_szn * 0.5 + b_bb_rate_szn
    (scaled to approximate OPS range without needing hits/doubles data)

    Uses the batting_order_pos and season stats already present in features_df.
    """
    df = features_df.copy()

    # Build per-game batting order table from rows that have a valid position
    has_order = df["batting_order_pos"].between(1, 9)

    order_df = df.loc[has_order, [
        "game_pk", "batter", "batting_order_pos",
        "b_hr_rate_szn", "b_hardhit_rate_szn", "b_bb_rate_szn",
    ]].copy()

    # OPS proxy — higher is better hitter
    order_df["ops_proxy"] = (
        order_df["b_hr_rate_szn"]      * 40.0 +
        order_df["b_hardhit_rate_szn"] *  0.5 +
        order_df["b_bb_rate_szn"]
    )

    # League-average OPS proxy for fallback
    league_avg_ops = float(order_df["ops_proxy"].mean()) if len(order_df) > 0 else 1.5

    # Build lookup: (game_pk, batting_order_pos) → ops_proxy
    order_lookup = (
        order_df
        .set_index(["game_pk", "batting_order_pos"])["ops_proxy"]
        .to_dict()
    )

    # Full lineup ops per game (for lineup_ops_context)
    lineup_ops = (
        order_df.groupby("game_pk")["ops_proxy"]
        .mean()
        .to_dict()
    )

    def _get_ops(game_pk, pos):
        # Wrap around 1–9
        wrapped = ((pos - 1) % 9) + 1
        return order_lookup.get((game_pk, wrapped), league_avg_ops)

    protection_ahead  = []
    protection_behind = []
    lineup_context    = []

    for _, row in df.iterrows():
        gp  = row["game_pk"]
        pos = int(row["batting_order_pos"])

        if pos == 0:
            # Unknown order — use league average
            protection_ahead.append(league_avg_ops)
            protection_behind.append(league_avg_ops)
            lineup_context.append(league_avg_ops)
            continue

        protection_ahead.append(_get_ops(gp, pos - 1))
        protection_behind.append(_get_ops(gp, pos + 1))

        # lineup_ops_context: exclude this batter
        game_ops = lineup_ops.get(gp, league_avg_ops)
        n = order_df[order_df["game_pk"] == gp].shape[0]
        if n > 1:
            batter_ops = order_lookup.get((gp, pos), league_avg_ops)
            excl_ops = (game_ops * n - batter_ops) / (n - 1)
        else:
            excl_ops = league_avg_ops
        lineup_context.append(excl_ops)

    df["protection_ops_ahead"]  = protection_ahead
    df["protection_ops_behind"] = protection_behind
    df["lineup_ops_context"]    = lineup_context

    return df


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
            "launch_speed": "max",
            "launch_angle": "max",
            "p_throws":     "last",
            "stand":        "last",
        })
    )

    ev_str = pa_df["events"].astype("string")
    pa_df["is_so"] = ev_str.str.contains("strikeout", na=False).astype("int8")
    pa_df["is_bb"] = (ev_str == "walk").fillna(False).astype("int8")

    ev = pd.to_numeric(pa_df["launch_speed"], errors="coerce")
    la = pd.to_numeric(pa_df["launch_angle"],  errors="coerce")
    pa_df["is_barrel"]    = ((ev >= 95) & (la.between(20, 35))).fillna(False).astype("int8")
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

    # Use starter_pitcher_id if available, fall back to pitcher_id
    pitcher_col = "starter_pitcher_id" if "starter_pitcher_id" in labels.columns else "pitcher_id"
    labels["p_days_rest"] = [
        _gap(
            int(getattr(r, pitcher_col)) if pd.notna(getattr(r, pitcher_col)) else -1,
            r.game_date,
            pitcher_dates,
        )
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
        return {f"p_fb_velo_{suffix}": 0.0, f"p_fb_pct_{suffix}": 0.0,
                f"p_offspeed_pct_{suffix}": 0.0}
    is_fb       = grp["pitch_type"].isin(FASTBALL_TYPES)
    is_offspeed = grp["pitch_type"].isin(OFFSPEED_TYPES)
    total       = len(grp)
    fb_velo     = _safe_mean(grp.loc[is_fb, "release_speed"]) if is_fb.any() else 0.0
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
            if not pd.isna(recent_velo) and not pd.isna(prior_velo) else 0.0
        )
        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batter window aggregation
# ---------------------------------------------------------------------------

def _batter_stats_for_window(grp: pd.DataFrame, suffix: str) -> dict:
    pa = len(grp)
    if pa == 0:
        return {f"b_pa_{suffix}": 0, f"b_hr_{suffix}": 0,
                f"b_hr_rate_{suffix}": 0.0, f"b_barrel_rate_{suffix}": 0.0,
                f"b_ev_mean_{suffix}": 0.0, f"b_la_mean_{suffix}": 0.0,
                f"b_hardhit_rate_{suffix}": 0.0, f"b_fb_rate_{suffix}": 0.0,
                f"b_k_rate_{suffix}": 0.0, f"b_bb_rate_{suffix}": 0.0}
    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr = int(grp["is_hr"].sum())
    return {
        f"b_pa_{suffix}":           pa,
        f"b_hr_{suffix}":           hr,
        f"b_hr_rate_{suffix}":      hr / pa,
        f"b_barrel_rate_{suffix}":  float(grp["is_barrel"].sum()) / pa,
        f"b_ev_mean_{suffix}":      _safe_mean(ev),
        f"b_la_mean_{suffix}":      _safe_mean(la),
        f"b_hardhit_rate_{suffix}": float((ev >= 95).mean()) if ev.notna().any() else 0.0,
        f"b_fb_rate_{suffix}":      float(la.between(20, 40).mean()) if la.notna().any() else 0.0,
        f"b_k_rate_{suffix}":       float(grp["is_so"].sum()) / pa,
        f"b_bb_rate_{suffix}":      float(grp["is_bb"].sum()) / pa,
    }


def _batter_trend_stats(w7: pd.DataFrame, w8_14: pd.DataFrame) -> dict:
    def _ev_mean(g):  return _safe_mean(g["launch_speed"]) if len(g) > 0 else 0.0
    def _hardhit(g):
        if len(g) == 0: return 0.0
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else 0.0
    def _barrel(g):   return float(g["is_barrel"].sum()) / len(g) if len(g) > 0 else 0.0
    def _hr_rate(g):  return float(g["is_hr"].sum())    / len(g) if len(g) > 0 else 0.0
    return {
        "b_ev_trend":       _ev_mean(w7)  - _ev_mean(w8_14),
        "b_hardhit_trend":  _hardhit(w7) - _hardhit(w8_14),
        "b_barrel_trend":   _barrel(w7)  - _barrel(w8_14),
        "b_hr_trend":       _hr_rate(w7) - _hr_rate(w8_14),
        "b_ev_mean_7":      _ev_mean(w7),
        "b_hardhit_rate_7": _hardhit(w7),
    }


def _batter_home_away_stats(grp: pd.DataFrame, batter_team: str | None) -> dict:
    empty = {"b_hr_rate_home": 0.0, "b_hr_rate_away": 0.0,
             "b_hardhit_rate_home": 0.0, "b_hardhit_rate_away": 0.0,
             "b_barrel_rate_home": 0.0, "b_barrel_rate_away": 0.0,
             "b_hr_rate_home_edge": 0.0}
    if len(grp) == 0 or batter_team is None:
        return empty

    home_mask = grp["home_team"] == batter_team
    home_grp  = grp[home_mask]
    away_grp  = grp[~home_mask]

    def _rate(g, col): return float(g[col].sum()) / len(g) if len(g) > 0 else 0.0
    def _hardhit(g):
        if len(g) == 0: return 0.0
        ev = g["launch_speed"]
        return float((ev >= 95).mean()) if ev.notna().any() else 0.0

    hr_home = _rate(home_grp, "is_hr")
    hr_away = _rate(away_grp, "is_hr")
    return {
        "b_hr_rate_home":      hr_home,
        "b_hr_rate_away":      hr_away,
        "b_hardhit_rate_home": _hardhit(home_grp),
        "b_hardhit_rate_away": _hardhit(away_grp),
        "b_barrel_rate_home":  _rate(home_grp, "is_barrel"),
        "b_barrel_rate_away":  _rate(away_grp, "is_barrel"),
        "b_hr_rate_home_edge": hr_home - hr_away,
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

        w14_stats  = _batter_stats_for_window(w14,  "14")
        wszn_stats = _batter_stats_for_window(wszn, "szn")
        _apply_batter_shrinkage(w14_stats,  "14")
        _apply_batter_shrinkage(wszn_stats, "szn")

        stats = {
            "batter":    batter_id,
            "game_date": game_date,
            **w14_stats,
            **wszn_stats,
        }

        for hand in ("L", "R"):
            w14_vs  = w14[w14["p_throws"]   == hand] if len(w14)  > 0 else empty
            wszn_vs = wszn[wszn["p_throws"]  == hand] if len(wszn) > 0 else empty
            stats.update(_batter_stats_for_window(w14_vs,  f"14_vs{hand}"))
            stats.update(_batter_stats_for_window(wszn_vs, f"szn_vs{hand}"))

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

def _pitcher_stats_for_window(grp: pd.DataFrame, suffix: str) -> dict:
    pa = len(grp)
    if pa == 0:
        return {f"p_pa_{suffix}": 0, f"p_hr_allowed_{suffix}": 0,
                f"p_hr_allowed_rate_{suffix}": 0.0, f"p_ev_allowed_mean_{suffix}": 0.0,
                f"p_hardhit_allowed_rate_{suffix}": 0.0, f"p_fb_allowed_rate_{suffix}": 0.0,
                f"p_barrel_allowed_rate_{suffix}": 0.0, f"p_k_rate_{suffix}": 0.0,
                f"p_bb_rate_{suffix}": 0.0}
    ev = grp["launch_speed"]
    la = grp["launch_angle"]
    hr = int(grp["is_hr"].sum())
    return {
        f"p_pa_{suffix}":                   pa,
        f"p_hr_allowed_{suffix}":           hr,
        f"p_hr_allowed_rate_{suffix}":      hr / pa,
        f"p_ev_allowed_mean_{suffix}":      _safe_mean(ev),
        f"p_hardhit_allowed_rate_{suffix}": float((ev >= 95).mean()) if ev.notna().any() else 0.0,
        f"p_fb_allowed_rate_{suffix}":      float(la.between(20, 40).mean()) if la.notna().any() else 0.0,
        f"p_barrel_allowed_rate_{suffix}":  float(grp["is_barrel"].sum()) / pa,
        f"p_k_rate_{suffix}":               float(grp["is_so"].sum()) / pa,
        f"p_bb_rate_{suffix}":              float(grp["is_bb"].sum()) / pa,
    }


def _precompute_pitcher_windows(pa_df: pd.DataFrame, target_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pitcher rolling windows. For each (pitcher, game_date) we build:
      - 30-day window  (all PA allowed)
      - season window  (all PA allowed)
      - handedness splits for both windows

    This is now called with the STARTER pitcher id only, so the stats
    reflect starter performance rather than a mix of starter + bullpen.
    """
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

        w30_stats  = _pitcher_stats_for_window(w30,  "30")
        wszn_stats = _pitcher_stats_for_window(wszn, "szn")
        _apply_pitcher_shrinkage(w30_stats,  "30")
        _apply_pitcher_shrinkage(wszn_stats, "szn")

        stats = {"pitcher": pitcher_id, "game_date": game_date,
                 **w30_stats,
                 **wszn_stats}

        for hand in ("L", "R"):
            w30_vs  = w30[w30["stand"]   == hand] if len(w30)  > 0 else empty
            wszn_vs = wszn[wszn["stand"]  == hand] if len(wszn) > 0 else empty
            stats.update(_pitcher_stats_for_window(w30_vs,  f"30_vs{hand}"))
            stats.update(_pitcher_stats_for_window(wszn_vs, f"szn_vs{hand}"))

        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def _add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ev_edge_14_30"]             = df["b_ev_mean_14"]      - df["p_ev_allowed_mean_30"]
    df["hardhit_edge_14_30"]        = df["b_hardhit_rate_14"] - df["p_hardhit_allowed_rate_30"]
    df["fb_edge_14_30"]             = df["b_fb_rate_14"]      - df["p_fb_allowed_rate_30"]
    df["barrel_edge_14_30"]         = df["b_barrel_rate_14"]  - df["p_barrel_allowed_rate_30"]
    df["hr_rate_edge_14_30"]        = df["b_hr_rate_14"]      - df["p_hr_allowed_rate_30"]
    df["k_rate_edge_14_30"]         = df["b_k_rate_14"]       - df["p_k_rate_30"]
    df["bb_rate_edge_14_30"]        = df["b_bb_rate_14"]      - df["p_bb_rate_30"]
    df["k_rate_interaction_14_30"]  = df["b_k_rate_14"]       * df["p_k_rate_30"]
    df["bb_rate_interaction_14_30"] = df["b_bb_rate_14"]      * df["p_bb_rate_30"]
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
            df[f"hr_rate_edge_14_30_vs{hand}"]  = df[b_hr]   - df[p_hr]
            df[f"hardhit_edge_14_30_vs{hand}"]  = df[b_hard] - df[p_hard]
            df[f"barrel_edge_14_30_vs{hand}"]   = df[b_bar]  - df[p_bar]

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

    # ------------------------------------------------------------------
    # Enrich labels with actual starter + batting order from MLB API
    # ------------------------------------------------------------------
    target_game_pks = labels["game_pk"].dropna().astype(int).unique().tolist()
    print(f"  Fetching starters + batting orders for {len(target_game_pks):,} games ...")
    starters_df, batting_df = fetch_rosters_for_games(target_game_pks)

    # batter_team needed to resolve which starter the batter faces
    labels["batter_team"] = labels["batter"].map(batter_team_lookup)

    labels = enrich_labels_with_roster(
        labels,
        starters_df,
        batting_df,
        game_pk_to_home,
    )

    # ------------------------------------------------------------------
    # Pitcher identity: prefer starter_pitcher_id, fall back to pitcher_mode
    # ------------------------------------------------------------------
    labels["pitcher_id"] = labels["starter_pitcher_id"].combine_first(
        pd.to_numeric(labels["pitcher_mode"], errors="coerce").astype("Int64")
    )

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
    # Compute relief_pa_pct now that starter_pitcher_id is resolved
    # ------------------------------------------------------------------
    labels = compute_relief_pa_pct(labels, target_pa)

    # Weather
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

    # Pitcher windows — now uses starter pitcher ids
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

    # ------------------------------------------------------------------
    # Bullpen windows — derived from Statcast, no extra API calls
    # ------------------------------------------------------------------
    print("  Deriving game starters from Statcast for bullpen identification ...")
    starter_lookup = _build_starter_lookup(pa_df)

    # For each label row we need the opposing team's bullpen stats.
    # pitching_team = the team pitching TO this batter.
    # If batter is home → pitching team is away. We derive away team from
    # game_pk_to_home: any team that appears in pa_df for that game that
    # is NOT the home_team.
    game_pk_to_away: dict[int, str] = {}
    for game_pk, home in game_pk_to_home.items():
        away_rows = pa_df[
            (pa_df["game_pk"] == game_pk) & (pa_df["home_team"] != home)
        ]
        if not away_rows.empty:
            away_team = away_rows["home_team"].mode()
            # home_team col always stores the game's home team — away team
            # is derived from pitcher_team lookup instead
            pass

    # Better: build game_pk → (home_team, away_team) from starter_lookup
    # then assign pitching_team per label row
    team_lookup_df = starter_lookup[["game_pk", "home_team"]].copy()

    # Derive away team from pa_df: for each game_pk find pitchers and their teams
    pitcher_team_map = (
        pa_df.dropna(subset=["pitcher", "home_team"])
        .groupby("pitcher")["home_team"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )

    # For starters: home starter's team = home_team, away starter's team = other
    def _away_team_for_game(game_pk: int) -> str | None:
        row = starter_lookup[starter_lookup["game_pk"] == game_pk]
        if row.empty:
            return None
        home_starter = row.iloc[0]["away_starter_id"]  # away starter pitches at home
        return pitcher_team_map.get(home_starter)

    game_pk_to_away = {
        int(r["game_pk"]): pitcher_team_map.get(r["away_starter_id"])
        for _, r in starter_lookup.iterrows()
        if r["away_starter_id"] is not None
    }

    # pitching_team for each label row
    labels_copy = labels.copy()
    labels_copy["game_pk"] = labels_copy["game_pk"].astype(int)
    labels_copy["pitching_team"] = labels_copy.apply(
        lambda r: (
            game_pk_to_away.get(r["game_pk"])
            if str(r.get("batter_team", "")) == str(game_pk_to_home.get(r["game_pk"], ""))
            else game_pk_to_home.get(r["game_pk"])
        ),
        axis=1,
    )

    bullpen_need = (
        labels_copy[["pitching_team", "game_date"]]
        .dropna(subset=["pitching_team"])
        .drop_duplicates()
        .copy()
    )
    n_bp = len(bullpen_need)
    print(f"  Precomputing bullpen windows for {n_bp:,} (team, date) pairs ...")
    bullpen_stats = _precompute_bullpen_windows(pa_df, starter_lookup, bullpen_need)

    # Merge everything
    features_df = (
        labels_copy
        .merge(batter_stats,  on=["batter", "game_date"], how="left")
        .merge(pitcher_stats.rename(columns={"pitcher": "pitcher_id"}),
               on=["pitcher_id", "game_date"], how="left")
        .merge(pitcher_velo.rename(columns={"pitcher": "pitcher_id"}),
               on=["pitcher_id", "game_date"], how="left")
        .merge(weather_df, on="game_pk", how="left")
        .merge(bullpen_stats, on=["pitching_team", "game_date"], how="left")
    )

    # Interaction: how much does bullpen quality matter given relief exposure?
    features_df["bp_hr_rate_x_relief_pct"] = (
        features_df["bp_hr_allowed_rate_30"] * features_df["relief_pa_pct"]
    )

    # Park factor — fetched dynamically per season, cached locally
    features_df["home_team"] = features_df["game_pk"].map(game_pk_to_home)
    season_year = start_dt.year
    print(f"  Fetching park factors for {season_year} ...")
    pf_map = get_park_factors(season=season_year)
    features_df["park_factor_hr"] = (
        features_df["home_team"]
        .map(pf_map)
        .fillna(DEFAULT_PARK_FACTOR / 100.0)
    )

    features_df = _add_edge_features(features_df)

    # ------------------------------------------------------------------
    # Lineup protection — computed after batter season stats are merged
    # ------------------------------------------------------------------
    print("  Computing lineup protection features ...")
    features_df = _compute_lineup_protection(features_df)

    # Tidy
    features_df = features_df.rename(columns={"pitcher_id": "pitcher"})
    features_df["game_date"] = features_df["game_date"].dt.date
    features_df["hr_hit"]    = features_df["hr_hit"].astype(int)

    non_stat_cols = {
        "game_date", "game_pk", "batter", "pitcher", "home_team",
        "hr_hit", "pitcher_hand", "batter_hand", "pitcher_mode",
        "starter_pitcher_id", "batter_team", "pitching_team",
        # batting order raw cols (kept but not stat-filled)
        "batting_order", "is_starter_batter", "team_side",
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

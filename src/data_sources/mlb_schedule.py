from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SCHEDULE_CACHE = CACHE_DIR / "schedule"
SCHEDULE_CACHE.mkdir(parents=True, exist_ok=True)

# Max parallel workers for roster fetches.
_MAX_WORKERS = 20
_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _game_cache_path(game_pk: int) -> Path:
    return SCHEDULE_CACHE / f"game_{game_pk}.json"


def _date_cache_path(date_str: str) -> Path:
    return SCHEDULE_CACHE / f"schedule_{date_str}.json"


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 15) -> dict:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Probable starters: fetch by date (unchanged — called once per date, not hot path)
# ---------------------------------------------------------------------------

def fetch_probable_starters_for_date(
    date_str: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch probable starters for all games on a given date.
    """
    cache = _date_cache_path(date_str)

    if cache.exists() and not force_refresh:
        with open(cache) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    )
    try:
        data = _get(url)
    except Exception as e:
        print(f"  [mlb_schedule] Failed to fetch schedule for {date_str}: {e}")
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team",
                                     "home_starter_id", "away_starter_id"])

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk   = int(game["gamePk"])
            home_team = game["teams"]["home"]["team"].get("abbreviation", "")
            away_team = game["teams"]["away"]["team"].get("abbreviation", "")
            home_prob = game["teams"]["home"].get("probablePitcher", {})
            away_prob = game["teams"]["away"].get("probablePitcher", {})

            rows.append({
                "game_pk":          game_pk,
                "home_team":        home_team,
                "away_team":        away_team,
                "home_starter_id":  int(home_prob["id"]) if home_prob else None,
                "away_starter_id":  int(away_prob["id"]) if away_prob else None,
            })

    with open(cache, "w") as f:
        json.dump(rows, f)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-game live feed: actual starter + batting order
# ---------------------------------------------------------------------------

def fetch_game_roster(
    game_pk: int,
    *,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch actual starter and batting orders for a single game from the
    MLB live feed endpoint.

    Returns a dict:
    {
        "game_pk": int,
        "home_starter_id": int | None,
        "away_starter_id": int | None,
        "batting_orders": {
            batter_mlbam_id: {
                "batting_order": int,
                "is_starter":    bool,
                "team_side":     "home" | "away",
            },
            ...
        }
    }
    """
    cache = _game_cache_path(game_pk)

    if cache.exists() and not force_refresh:
        with open(cache) as f:
            return json.load(f)

    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    last_err = None
    data = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            data = _get(url)
            break
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(2)
    else:
        print(f"  [mlb_schedule] Failed to fetch game {game_pk} after {_MAX_RETRIES} attempts: {last_err}")
        return _empty_roster(game_pk)

    box = data.get("liveData", {}).get("boxscore", {})
    batting_orders: dict[int, dict] = {}
    starters: dict[str, int | None] = {"home": None, "away": None}

    for side in ("home", "away"):
        team_box = box.get("teams", {}).get(side, {})
        players  = team_box.get("players", {})

        for player_key, player_data in players.items():
            pid         = player_data.get("person", {}).get("id")
            game_status = player_data.get("gameStatus", {})
            order_raw   = player_data.get("battingOrder")

            if pid is None:
                continue

            if order_raw is not None:
                try:
                    order_int = int(str(order_raw).strip()) // 100
                except (ValueError, TypeError):
                    order_int = 0

                is_starter = game_status.get("isOnBench", True) is False

                batting_orders[int(pid)] = {
                    "batting_order": order_int,
                    "is_starter":    is_starter,
                    "team_side":     side,
                }

        pitchers = team_box.get("pitchers", [])
        if pitchers:
            starters[side] = int(pitchers[0])

    result = {
        "game_pk":         game_pk,
        "home_starter_id": starters["home"],
        "away_starter_id": starters["away"],
        "batting_orders":  batting_orders,
    }

    with open(cache, "w") as f:
        json.dump(result, f)

    return result


def _empty_roster(game_pk: int) -> dict:
    return {
        "game_pk":         game_pk,
        "home_starter_id": None,
        "away_starter_id": None,
        "batting_orders":  {},
    }


# ---------------------------------------------------------------------------
# Bulk fetch: starters + batting orders for a list of game_pks — PARALLELIZED
# ---------------------------------------------------------------------------

def fetch_rosters_for_games(
    game_pks: list[int],
    *,
    force_refresh: bool = False,
    max_workers: int = _MAX_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch actual starters and batting orders for a list of game_pks in parallel.

    Returns
    -------
    starters_df : DataFrame
        Columns: game_pk, home_starter_id, away_starter_id

    batting_df : DataFrame
        Columns: game_pk, batter, batting_order, is_starter_batter, team_side
    """
    unique_pks = list(dict.fromkeys(game_pks))

    # Split cached vs uncached
    cached_pks = [gp for gp in unique_pks if _game_cache_path(gp).exists() and not force_refresh]
    fetch_pks  = [gp for gp in unique_pks if gp not in set(cached_pks)]

    rosters: dict[int, dict] = {}

    # Load cached entries
    for gp in cached_pks:
        with open(_game_cache_path(gp)) as f:
            rosters[gp] = json.load(f)

    # Fetch uncached in parallel
    if fetch_pks:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_pk = {
                pool.submit(fetch_game_roster, gp, force_refresh=force_refresh): gp
                for gp in fetch_pks
            }
            for future in as_completed(future_to_pk):
                gp = future_to_pk[future]
                try:
                    rosters[gp] = future.result()
                except Exception as e:
                    print(f"  [mlb_schedule] game {gp} failed: {e}")
                    rosters[gp] = _empty_roster(gp)
                completed += 1
                if completed % 100 == 0:
                    print(f"    Rosters: {completed}/{len(fetch_pks)} games fetched ...")

        print(f"    Rosters: {len(fetch_pks)}/{len(fetch_pks)} games fetched ...")

    # Assemble output DataFrames in original order
    starter_rows: list[dict] = []
    batting_rows: list[dict] = []

    for gp in unique_pks:
        roster = rosters[gp]
        starter_rows.append({
            "game_pk":         roster["game_pk"],
            "home_starter_id": roster["home_starter_id"],
            "away_starter_id": roster["away_starter_id"],
        })
        for batter_id, info in roster["batting_orders"].items():
            batting_rows.append({
                "game_pk":           roster["game_pk"],
                "batter":            int(batter_id),
                "batting_order":     info["batting_order"],
                "is_starter_batter": int(info["is_starter"]),
                "team_side":         info["team_side"],
            })

    starters_df = pd.DataFrame(starter_rows) if starter_rows else pd.DataFrame(
        columns=["game_pk", "home_starter_id", "away_starter_id"]
    )
    batting_df = pd.DataFrame(batting_rows) if batting_rows else pd.DataFrame(
        columns=["game_pk", "batter", "batting_order", "is_starter_batter", "team_side"]
    )

    return starters_df, batting_df


# ---------------------------------------------------------------------------
# Convenience: resolve starter for each batter-game row (unchanged)
# ---------------------------------------------------------------------------

_EXPECTED_PA = {
    1: 4.7, 2: 4.5, 3: 4.4, 4: 4.3,
    5: 4.1, 6: 4.0, 7: 3.9, 8: 3.8, 9: 3.7,
}
_EXPECTED_PA_DEFAULT = 4.1


def enrich_labels_with_roster(
    labels: pd.DataFrame,
    starters_df: pd.DataFrame,
    batting_df: pd.DataFrame,
    game_pk_home_lookup: dict[int, str],
) -> pd.DataFrame:
    """
    Merge starter and batting order information into the labels DataFrame.
    """
    labels = labels.copy()
    labels["game_pk"] = labels["game_pk"].astype(int)

    starters = starters_df.copy()
    starters["game_pk"] = starters["game_pk"].astype(int)

    home_lookup = {int(k): v for k, v in game_pk_home_lookup.items()}

    starter_map: dict[int, tuple] = {}
    for _, r in starters.iterrows():
        starter_map[int(r["game_pk"])] = (r["home_starter_id"], r["away_starter_id"])

    def _resolve_starter(row) -> int | None:
        gp  = row["game_pk"]
        bat = str(row.get("batter_team", ""))
        home = home_lookup.get(gp, "")
        s = starters[starters["game_pk"] == gp]
        if s.empty:
            return None
        s = s.iloc[0]
        if bat == home:
            return s["away_starter_id"]
        else:
            return s["home_starter_id"]

    labels["starter_pitcher_id"] = labels.apply(_resolve_starter, axis=1)

    fallback_mask = labels["starter_pitcher_id"].isna()
    if fallback_mask.any() and "pitcher_mode" in labels.columns:
        labels.loc[fallback_mask, "starter_pitcher_id"] = (
            pd.to_numeric(labels.loc[fallback_mask, "pitcher_mode"], errors="coerce")
        )

    labels["starter_pitcher_id"] = (
        pd.to_numeric(labels["starter_pitcher_id"], errors="coerce").astype("Int64")
    )

    batting = batting_df.copy()
    if not batting.empty:
        batting["game_pk"] = batting["game_pk"].astype(int)
        batting["batter"]  = batting["batter"].astype(int)
        batting = batting[["game_pk", "batter", "batting_order", "is_starter_batter"]]
        labels = labels.merge(batting, on=["game_pk", "batter"], how="left")
    else:
        labels["batting_order"]     = pd.NA
        labels["is_starter_batter"] = pd.NA

    labels["batting_order_pos"] = (
        pd.to_numeric(labels["batting_order"], errors="coerce")
        .fillna(0)
        .clip(0, 9)
        .astype(int)
    )
    labels["is_top_of_order"] = (
        labels["batting_order_pos"].between(1, 4).astype("int8")
    )
    labels["expected_pa_today"] = labels["batting_order_pos"].map(
        lambda p: _EXPECTED_PA.get(p, _EXPECTED_PA_DEFAULT)
    )

    if "relief_pa_pct" not in labels.columns:
        labels["relief_pa_pct"] = 0.0

    return labels


if __name__ == "__main__":
    starters_df, batting_df = fetch_rosters_for_games([745456, 745457])
    print("Starters:")
    print(starters_df)
    print("\nBatting orders (first 10):")
    print(batting_df.head(10))

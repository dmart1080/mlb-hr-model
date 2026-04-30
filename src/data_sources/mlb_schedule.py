from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SCHEDULE_CACHE = CACHE_DIR / "schedule"
SCHEDULE_CACHE.mkdir(parents=True, exist_ok=True)

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
# Probable starters
# ---------------------------------------------------------------------------

def fetch_probable_starters_for_date(
    date_str: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch probable starters for all games on a given date."""
    cache = _date_cache_path(date_str)

    if cache.exists() and not force_refresh:
        with open(cache) as f:
            data = json.load(f)
        logger.debug("Schedule cache hit for %s", date_str)
        return pd.DataFrame(data)

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    )
    try:
        data = _get(url)
    except Exception as e:
        logger.error("Failed to fetch schedule for %s: %s", date_str, e)
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team",
                                     "home_starter_id", "away_starter_id",
                                     "game_datetime"])

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
                # ISO UTC start time — needed to disambiguate doubleheader
                # games when joining odds (Odds API returns commence_time
                # in the same format).
                "game_datetime":    game.get("gameDate"),
            })

    df = pd.DataFrame(rows)

    # --- Rotowire fallback: fill any None starter IDs ---
    missing_home = df["home_starter_id"].isna().sum()
    missing_away = df["away_starter_id"].isna().sum()
    if missing_home > 0 or missing_away > 0:
        logger.info(
            "MLB API missing starters: %d home, %d away — trying Rotowire fallback",
            missing_home, missing_away,
        )
        try:
            from src.data_sources.rotowire_starters import enrich_schedule_with_rotowire
            df = enrich_schedule_with_rotowire(df, date_str, force_refresh=force_refresh)
        except Exception as e:
            logger.warning("Rotowire fallback failed: %s", e)
    else:
        df["starter_source"] = "mlb_api"

    with open(cache, "w") as f:
        json.dump(df.to_dict("records"), f)

    logger.debug("Fetched %d games for %s", len(df), date_str)
    return df


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
            logger.warning(
                "Roster fetch failed game_pk=%d attempt=%d/%d: %s",
                game_pk, attempt, _MAX_RETRIES, e,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(2)
    else:
        logger.error(
            "Roster fetch exhausted retries game_pk=%d: %s", game_pk, last_err
        )
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

                # FIX 2: isOnBench=True means the player IS on the bench
                # (substitute), so is_starter = NOT isOnBench.
                # The previous default of True meant that when the key was
                # absent we incorrectly treated the player as a bench player.
                # Default to False (assume starter) when the key is missing.
                is_on_bench = game_status.get("isOnBench", False)
                is_starter  = not is_on_bench

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
# Bulk fetch — parallelised
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

    cached_pks = [gp for gp in unique_pks if _game_cache_path(gp).exists() and not force_refresh]
    fetch_pks  = [gp for gp in unique_pks if gp not in set(cached_pks)]

    logger.info(
        "Roster fetch: %d games total (%d cached, %d to fetch)",
        len(unique_pks), len(cached_pks), len(fetch_pks),
    )

    rosters: dict[int, dict] = {}

    for gp in cached_pks:
        with open(_game_cache_path(gp)) as f:
            rosters[gp] = json.load(f)

    if fetch_pks:
        completed = 0
        failed: list[int] = []
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
                    logger.error("Roster future failed game_pk=%d: %s", gp, e)
                    rosters[gp] = _empty_roster(gp)
                    failed.append(gp)
                completed += 1
                if completed % 100 == 0:
                    logger.info("Rosters: %d/%d fetched ...", completed, len(fetch_pks))

        logger.info("Rosters: %d/%d fetched", len(fetch_pks), len(fetch_pks))
        if failed:
            logger.warning(
                "Roster fetch failed for %d game(s): %s", len(failed), failed[:10]
            )

    # Assemble output DataFrames
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
# Convenience: enrich labels with roster data
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
    """Merge starter and batting order information into the labels DataFrame."""
    labels = labels.copy()
    labels["game_pk"] = labels["game_pk"].astype(int)

    starters = starters_df.copy()
    starters["game_pk"] = starters["game_pk"].astype(int)

    home_lookup = {int(k): v for k, v in game_pk_home_lookup.items()}

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
    fallback_count = fallback_mask.sum()
    if fallback_count > 0 and "pitcher_mode" in labels.columns:
        logger.debug(
            "starter_pitcher_id missing for %d rows — falling back to pitcher_mode",
            fallback_count,
        )
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
        logger.warning("batting_df is empty — batting_order features will be missing")
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
    from src.logging_config import configure_logging
    configure_logging()

    starters_df, batting_df = fetch_rosters_for_games([745456, 745457])
    print("Starters:")
    print(starters_df)
    print("\nBatting orders (first 10):")
    print(batting_df.head(10))

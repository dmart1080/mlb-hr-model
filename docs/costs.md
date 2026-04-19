# Operating Costs

Running totals and per-call math for the services that power this model.
Recalculate whenever cron cadence, game count, or market list changes.

## Summary

| Service | Plan | Monthly cost | Notes |
|---|---|---|---|
| Hetzner VPS (CX22, Ashburn) | Shared 2 vCPU / 2 GB RAM | ~$4 / mo | Hosts cron + Python env |
| The Odds API | Free tier (500 credits / mo) | $0 | ~16 credits per pass — see below |
| MLB Stats API | Free | $0 | Schedule, lineups, live weather, venue |
| pybaseball / Statcast | Free | $0 | Backtest outcomes for recap |
| Discord webhook | Free | $0 | Pick & recap delivery |
| **Total** | | **~$4 / mo** | |

## Odds API credit math

Endpoints we hit per "fresh pull" (one wave with `--force-refresh`):

| Call | Quantity | Credits per call | Total |
|---|---|---|---|
| `GET /v4/sports/baseball_mlb/events` | 1 | 1 | 1 |
| `GET /v4/sports/baseball_mlb/events/{id}/odds` (`regions=us`, `markets=batter_home_runs`) | ~15 games | 1 per region×market | ~15 |
| **Per pass** | | | **~16** |

See [src/data_sources/odds.py:126](../src/data_sources/odds.py#L126) and [:152](../src/data_sources/odds.py#L152).
Remaining credits are logged at DEBUG from the `x-requests-remaining` response header.

### Caching

`_CACHE_TTL_HOURS = 12` in [src/data_sources/odds.py:67](../src/data_sources/odds.py#L67).
Cached responses cost 0 credits. `--force-refresh` bypasses cache.

**First-wave-only refresh:** the scheduler force-refreshes the first final pass
of each day and passes `--no-force-refresh` to predict for every subsequent
wave. Waves 2+ reuse the 12h cache so they cost 0 credits. See
[src/scheduler.py](../src/scheduler.py) `--auto-final` branch.

### Monthly burn (current cron)

Crontab on VPS:

- `*/30 * * * * ... --auto-final` → fires each wave as due; first wave refreshes odds, waves 2+ use cache
- `*/15 * * * * ... CLV_ENABLED=true --auto-clv` → once/day, latest wave only, deduped

| Activity | Credits/day | Credits/month |
|---|---|---|
| First final pass of day (force refresh) | ~16 | ~480 |
| Subsequent waves (cache hit) | 0 | 0 |
| CLV snapshot (once, pre-last-wave) | ~16 | ~480 |
| **Total** | **~32** | **~960** |

**Still over the 500/mo free tier with CLV on.** Realistic options:
- **CLV on**: ~960/mo → needs the $30 20k tier, OR skip CLV on some days
- **CLV off**: ~480/mo → fits free tier with thin margin

### Upgrade break-even

The $30/mo 20k-credit tier covers current usage with ~8× headroom. Decision
rule: upgrade only if closing-line-value data (after ~30 days of CLV tracking)
shows sustained positive CLV. Until then, the free tier + cache + CLV dedupe
is intentional discipline, not a bottleneck.

## Free services — rate limits

- **MLB Stats API** (schedule, lineups, weather, venue): no published limit. Used
  in every pass. Cached in-process.
- **Statcast (pybaseball)**: scraped; occasional 429s on heavy backtests. Only hit
  by `weekly_recap.py` and training, not the live pipeline.
- **Discord webhook**: 30 messages / minute per webhook. We send <10/day.

## What could push costs up

- Adding more markets (e.g. hits, total bases): each additional market doubles
  the per-event cost.
- Adding more regions (eu, uk): each adds a credit per event.
- Running `--auto-final` at higher cadence than `*/30`.
- Disabling the 12h cache or shortening its TTL.
- Re-enabling per-wave CLV (was ~1,920/mo extra — kept disabled via
  latest-wave-only dedupe).

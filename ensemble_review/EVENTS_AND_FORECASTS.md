# Hydrological events and forecast setup

The Task tab offers three analysis choices. Daily discharge forecasting reports
continuous discharge metrics. Statistical extremes retain the existing
single-day high-flow threshold evaluation. Hydrological event evaluation adds
the definitions below. It does not claim that a discharge event is a confirmed
flood, landslide, or other observed impact.

## Event definitions

Thresholds are calculated separately for each target basin from observations
in the training date range. The default high-flow, low-flow, and daily-rise
quantiles are 0.80, 0.20, and 0.95. Daily falls use their own 0.95
training-period quantile. These are configurable in the Task tab.

| Event | Definition | Severity reported |
| --- | --- | --- |
| Sustained high flow | Consecutive days above the high-flow threshold, for at least the configured minimum duration | Sum of discharge above the threshold, in mm |
| Low-flow drought | Consecutive days below the low-flow threshold, for at least the configured minimum duration | Sum of discharge deficits, in mm |
| Rapid daily rise | Consecutive daily changes greater than the training-period rise threshold | Sum of changes above the threshold, in mm/day |
| Rapid daily fall | Consecutive daily decreases greater than the training-period fall threshold | Sum of decreases above the threshold, in mm/day |
| Repeated high-flow pulses | Two or more high-flow spells separated by no more than the configured gap | Total excess flow of the component spells, in mm |
| Regional concurrence | At least the configured number of selected basins above their own high-flow thresholds on the same days, for the minimum duration | Sum of affected basin-days |
| Regional low flow | At least the configured number of selected basins below their own low-flow thresholds on the same days, for the minimum duration | Sum of affected basin-days |

Events are split at missing dates. Results are reported per seed and basin, with
an `all` row for regional concurrence. They include observed and predicted
event counts, event-day counts, total severity, and overlap-based event
precision and recall. A precision or recall with no corresponding events is
reported as undefined. Rapid changes start on the second test
day because a preceding test-day prediction is needed to calculate the rise.

These thresholds are exploratory. A percentile is not a local flood danger
level or ecological minimum flow. Persisted events and impacts need their own
observations and independently selected thresholds.

## One-day and rolling forecasts

The default one-day setup preserves the existing evaluation: each test-day
window uses observed discharge up to the preceding day. Rolling mode produces
non-overlapping blocks of the configured horizon, seven days by default. The
first day of each block uses observed discharge available at its issue date.
For subsequent days, predicted discharge replaces observed discharge in all
input-window positions after that issue date. Observed discharge becomes
available again when the next block begins. `predictions.csv` includes
`issue_date` and `lead_day` to distinguish the two modes and compare lead times.
The Results metrics view and saved manifest report NSE, KGE, and RMSE by lead
day; `metrics.csv` also includes `lead_1`, `lead_2`, and so on when rolling mode
is used.
Event scores in rolling mode are calculated from the stitched sequence of
non-overlapping blocks. An event that crosses a block boundary is therefore
assessed using predictions from two issue dates, and the next block may use
newly observed discharge available at that date.

Rolling mode uses **historically observed meteorological and optional predictor
values** for days after issue. It is therefore a *perfect-weather hindcast* of
the discharge rollout, not an operational forecast issued without future
weather information. It reuses the trained one-day model; the model is not
separately trained for each lead time. To make an operational forecast, supply
weather forecasts available at each issue date and evaluate the whole
weather-to-discharge chain by lead time and event.

# Wind power and day-ahead prices in Denmark

**Question.** How does a growing share of renewable power change wholesale
day-ahead prices in Denmark: lower average prices, changes in volatility, and a
falling market value (capture rate) of renewables?

Mini-project for the EA course: build a simple model, run scenarios, compare
with the literature. This repository covers steps 1-3 (data, indicators, model).
Step 4 (comparison with the literature) uses `results/model_literature_check.csv`.

## How to run

Python 3.10+ with `pandas numpy scipy matplotlib requests`
(`pip install -r requirements.txt`; Anaconda already has them). From the project folder:

```bash
python src/fetch_data.py      # 1a. download raw data (about 10 min, once)
python src/clean.py           # 1b. build data/processed/dk_hourly.csv.gz
python src/indicators.py      # 2.  indicators + figures 1-6
python src/model.py 2019      # 3.  model: calibration, validation, scenarios, figures 7-9
```

The argument of `model.py` is the reference year whose weather and load are
used in the scenarios (2019: last calm year before the crisis; try 2024 too).

## Data (Energi Data Service, Energinet, free, CC BY 4.0)

| Dataset | Content | Period |
|---|---|---|
| `Elspotprices` | hourly day-ahead prices, DK1, DK2 and neighbours | 1 Jul 1999 to 30 Sep 2025 |
| `DayAheadPrices` | successor, 15-min prices (averaged to hours) | from 1 Oct 2025 |
| `ElectricityBalanceNonv` | hourly load, onshore/offshore wind, solar | from 1 Jan 2017 (checked on the API); `clean.py` prints the coverage |

The balance data are non-validated SCADA data (Energinet warns they may have gaps);
`clean.py` prints the coverage per year.

## Step 2 - indicators (`src/indicators.py`)

Per area and year (and month): wind and wind+solar share of load; average price;
intraday volatility (mean of the daily std of the 24 hourly prices) and its
relative version (std / mean); wind and solar capture rate
(output-weighted price / average price); number of negative-price hours.
Tables in `results/indicators_*.csv`, figures `fig1` to `fig6`.

## Step 3 - model B: toy merit order with one interconnector (`src/model.py`)

Residual load share `x_t = (load − wind − solar) / mean load`.
Neighbour price `P_t` = observed neighbour price averaged by month and hour of day
(exogenous: keeps fuel level and daily profile, drops the weather signal shared with Denmark).

```
domestic merit order:  f(z) = max(p_floor, P_t · (1 + b · (z − r0)))
market coupling, interconnector capacity k:
   x_t − k > r0  ->  p_t = f(x_t − k)    (import line full, price above neighbours)
   x_t + k < r0  ->  p_t = f(x_t + k)    (export line full, price below neighbours)
   otherwise     ->  p_t = P_t           (coupled)
```

| Parameter | Meaning |
|---|---|
| `b` | steepness of the Danish merit order |
| `r0` | residual load at which Danish cost equals the neighbour price |
| `k` | interconnection + flexibility, as a share of mean load |
| `p_floor` | lowest bid (negative with subsidies / must-run) |

Calibrated by least absolute error on hourly prices 2015-2019 (load and wind data
start in 2017, so in practice 2017-2019), validated on
2020-2025 against a naive benchmark (`P_t` alone). Scenarios scale the wind and
solar output of the reference year to shares from 5% to 120% of load.
Sensitivity (DK1): interconnection `k`, price floor, slope `b`.

Outputs: `results/model_*.csv`, figures `fig7_model_validation`, `fig8_scenarios`,
`fig9_sensitivity_DK1`.

## Testing without internet

`tests/make_fake_raw.py` writes synthetic raw files built from known parameters
(b = 1.2, r0 = 0.6, k = 0.35). The calibration recovers them
(b ≈ 1.19, r0 ≈ 0.60, k ≈ 0.35), which checks the code, not the economics.

```bash
# Windows PowerShell: $env:DKW_OUT="tests/fake"   |  bash: export DKW_OUT=tests/fake
python tests/make_fake_raw.py; python src/clean.py; python src/indicators.py; python src/model.py
```

## Known limits of the model (to discuss in step 4)

- Neighbours are exogenous: more Danish wind does not lower neighbour prices
  (Stiewe et al. 2024 find strong cross-border cannibalisation).
- One aggregated interconnector and fixed parameters: no grid expansion,
  no investment or retirement, no storage dynamics, no hydro reservoir behaviour.
- Linear merit order scaled by the neighbour price: fuel and CO2 enter only through `P_t`.
- Wind and solar are scaled with the reference-year profiles, so the
  geographical spread of a larger fleet is ignored.
- 2022 crisis and the 2025 move to 15-minute prices are not modelled explicitly.

Source: Energinet (www.energidataservice.dk).

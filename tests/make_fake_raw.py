"""Generate synthetic raw files with the same layout as Energi Data Service.

Only used to test the pipeline without internet access. The numbers are made up.
Usage (from the project root):
    DKW_OUT=tests/fake python tests/make_fake_raw.py
    DKW_OUT=tests/fake python src/clean.py   (then indicators.py, model.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import RAW  # noqa: E402

rng = np.random.default_rng(0)
YEARS = range(2013, 2026)


def wind_series(n, cf):
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.97 * z[t - 1] + rng.normal(0, 0.25)
    return np.clip(cf + 0.25 * np.tanh(z), 0, 1)


for y in YEARS:
    t = pd.date_range(f"{y}-01-01", f"{y + 1}-01-01", freq="h", tz="UTC", inclusive="left")
    n = len(t)
    h = t.hour.to_numpy()
    season = 1 + 0.15 * np.cos(2 * np.pi * (t.day_of_year.to_numpy() - 15) / 365)
    fuel = 35 * (1 + 0.04 * (y - 2013)) * (3.5 if y == 2022 else 1.8 if y in (2021, 2023) else 1)
    prices, balance = [], []
    common_wind = wind_series(n, 0.33)
    for area, scale, wcap in [("DK1", 1.0, 0.8), ("DK2", 0.6, 0.45)]:
        load = 2300 * scale * season * (1 + 0.15 * np.sin(np.pi * (h - 6) / 12).clip(0))
        cap = 2500 * scale * wcap * (1 + 0.08 * (y - 2013))
        wind = cap * np.clip(common_wind + rng.normal(0, 0.05, n), 0, 1)
        solar = 50 * scale * (y - 2012) ** 1.6 * np.clip(np.sin(np.pi * (h - 5) / 14), 0, None)
        x = (load - wind - solar) / load.mean()
        p_neigh = fuel * season * (1 + 0.1 * np.sin(np.pi * (h - 6) / 12)) + rng.normal(0, 3, n)
        p = np.where(x < 0.25, np.maximum(-10, p_neigh * (1 + 1.2 * (x - 0.25))),
                     np.where(x > 0.95, p_neigh * (1 + 1.2 * (x - 0.95)), p_neigh))
        p = p + rng.normal(0, 4, n)
        prices.append(pd.DataFrame({"HourUTC": t.tz_localize(None), "PriceArea": area,
                                    "SpotPriceDKK": p * 7.45, "SpotPriceEUR": p}))
        balance.append(pd.DataFrame({"HourUTC": t.tz_localize(None), "PriceArea": area,
                                     "TotalLoad": load, "OnshoreWindPower": wind * 0.6,
                                     "OffshoreWindPower": wind * 0.4, "SolarPower": solar}))
    for nb in ["NO2", "SE3", "SE4", "DE"]:
        pn = fuel * season * (1 + 0.1 * np.sin(np.pi * (h - 6) / 12)) + rng.normal(0, 5, n)
        prices.append(pd.DataFrame({"HourUTC": t.tz_localize(None), "PriceArea": nb,
                                    "SpotPriceDKK": pn * 7.45, "SpotPriceEUR": pn}))
    pd.concat(prices).to_csv(RAW / f"Elspotprices_{y}.csv", index=False)
    pd.concat(balance).to_csv(RAW / f"ElectricityBalanceNonv_{y}.csv", index=False)
print(f"Fake raw files written to {RAW}")

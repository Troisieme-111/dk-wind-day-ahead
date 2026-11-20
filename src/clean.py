"""Step 1b - build one clean hourly table for DK1 and DK2.

Output columns (data/processed/dk_hourly.csv.gz):
  time_utc, area, price (EUR/MWh), load, wind_on, wind_off, wind, solar (MWh/h),
  p_neigh (mean hourly price of the neighbouring areas, EUR/MWh)

From 1 October 2025 day-ahead prices are quarter-hourly; they are averaged to
hours so the series stays homogeneous.
"""
import glob

import numpy as np
import pandas as pd

from config import RAW, HOURLY_FILE, AREAS, NEIGHBOURS, EUR_PER_DKK


def _read(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / pattern)))
    if not files:
        raise FileNotFoundError(f"No raw files {pattern} in {RAW}. Run fetch_data.py first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _utc_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if c.upper().endswith("UTC"))


def _find(df: pd.DataFrame, exact: str, keyword: str):
    if exact in df.columns:
        return exact
    hits = [c for c in df.columns if keyword.lower() in c.lower()]
    return hits[0] if hits else None


def load_prices() -> pd.DataFrame:
    """Hourly prices for all areas, long format: time_utc, area, price."""
    sp = _read("Elspotprices_*.csv")
    sp["price"] = sp["SpotPriceEUR"].fillna(sp["SpotPriceDKK"] * EUR_PER_DKK)
    sp = sp.rename(columns={_utc_col(sp): "time_utc", "PriceArea": "area"})
    parts = [sp[["time_utc", "area", "price"]]]

    try:
        da = _read("DayAheadPrices_*.csv")
        eur = next(c for c in da.columns if "EUR" in c.upper())
        dkk = next((c for c in da.columns if "DKK" in c.upper()), None)
        da["price"] = da[eur]
        if dkk:
            da["price"] = da["price"].fillna(da[dkk] * EUR_PER_DKK)
        da = da.rename(columns={_utc_col(da): "time_utc", "PriceArea": "area"})
        da["time_utc"] = pd.to_datetime(da["time_utc"]).dt.floor("h")
        da = da.groupby(["time_utc", "area"], as_index=False)["price"].mean()
        parts.append(da)
    except FileNotFoundError:
        print("   (no DayAheadPrices files: series stops on 2025-09-30)")

    p = pd.concat(parts, ignore_index=True)
    p["time_utc"] = pd.to_datetime(p["time_utc"])
    return (p.dropna(subset=["price"])
             .drop_duplicates(["time_utc", "area"], keep="last"))


def load_balance() -> pd.DataFrame:
    """Hourly load, wind and solar for DK1/DK2."""
    b = _read("ElectricityBalanceNonv_*.csv")
    cols = {
        "load": _find(b, "TotalLoad", "load"),
        "wind_on": _find(b, "OnshoreWindPower", "onshore"),
        "wind_off": _find(b, "OffshoreWindPower", "offshore"),
        "solar": _find(b, "SolarPower", "solar"),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise KeyError(f"Columns not found in ElectricityBalanceNonv: {missing}; got {list(b.columns)}")
    b = b.rename(columns={_utc_col(b): "time_utc", "PriceArea": "area",
                          **{v: k for k, v in cols.items()}})
    b["time_utc"] = pd.to_datetime(b["time_utc"]).dt.floor("h")
    b = b[b["area"].isin(AREAS)]
    b = b.groupby(["time_utc", "area"], as_index=False)[list(cols)].mean()
    for c in ["wind_on", "wind_off", "solar"]:
        b[c] = b[c].fillna(0).clip(lower=0)
    b["wind"] = b["wind_on"] + b["wind_off"]
    return b


def build() -> pd.DataFrame:
    prices = load_prices()
    wide = prices.pivot_table(index="time_utc", columns="area", values="price")

    frames = []
    for area in AREAS:
        neigh = [a for a in NEIGHBOURS[area] if a in wide.columns]
        p_neigh = wide[neigh].mean(axis=1) if neigh else pd.Series(np.nan, index=wide.index)
        if "SYSTEM" in wide.columns:
            p_neigh = p_neigh.fillna(wide["SYSTEM"])
        f = pd.DataFrame({"price": wide[area], "p_neigh": p_neigh})
        f["area"] = area
        frames.append(f.reset_index())
    df = pd.concat(frames, ignore_index=True)

    df = df.merge(load_balance(), on=["time_utc", "area"], how="left")
    df = df.dropna(subset=["price"]).sort_values(["area", "time_utc"])
    df = df[["time_utc", "area", "price", "p_neigh", "load",
             "wind_on", "wind_off", "wind", "solar"]]
    return df


def report(df: pd.DataFrame):
    """Coverage per year: hours with a price, and share with load/wind data."""
    d = df.assign(year=df["time_utc"].dt.year)
    rep = d.groupby(["area", "year"]).agg(
        hours=("price", "size"),
        with_balance=("load", lambda s: s.notna().mean()),
        mean_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
    ).round(2)
    print(rep.to_string())
    return rep


if __name__ == "__main__":
    df = build()
    df.to_csv(HOURLY_FILE, index=False)
    print(f"Saved {len(df)} rows -> {HOURLY_FILE}")
    report(df)

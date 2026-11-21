"""Step 1a - download raw data from Energi Data Service (free, no key).

Datasets (https://www.energidataservice.dk):
  Elspotprices            hourly day-ahead prices, all areas, 1999-07 -> 2025-09-30
  DayAheadPrices          day-ahead prices (15-min), successor from 2025-10-01
  ElectricityBalanceNonv  hourly load, wind (on/offshore), solar, exchange, DK1/DK2

One CSV per dataset and year is written to data/raw/. Files already present
are skipped, so the script can be re-run after an interruption.

Usage:  python src/fetch_data.py            (all years)
        python src/fetch_data.py 2015 2025  (a range of years)
"""
import re
import sys
import time

import pandas as pd
import requests

from config import RAW, FIRST_YEAR, LAST_YEAR

API = "https://api.energidataservice.dk/dataset/{name}"
DATASETS = {
    # name: (first date available, last date used)
    "Elspotprices": ("1999-07-01", "2025-10-01"),
    "DayAheadPrices": ("2025-10-01", None),
    "ElectricityBalanceNonv": ("1999-07-01", None),
}


def fetch(name: str, start: str, end: str) -> pd.DataFrame:
    """Download all records of `name` between start (incl.) and end (excl.), DK time."""
    params = {"start": start, "end": end, "limit": 0}
    for attempt in range(8):
        try:
            r = requests.get(API.format(name=name), params=params, timeout=120)
            if r.status_code == 429:          # rate limit: the block can last ~5 min
                m = re.search(r"(\d+) seconds", r.text)
                wait = int(m.group(1)) + 5 if m else 60
                print(f"   rate limited, waiting {wait} s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return pd.DataFrame(r.json().get("records", []))
        except requests.RequestException as e:
            print(f"   retry {attempt + 1} after error: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {name} {start}->{end}")


def main(y0: int, y1: int):
    for name, (first, last) in DATASETS.items():
        print(f"\n== {name}")
        for year in range(y0, y1 + 1):
            start, end = f"{year}-01-01", f"{year + 1}-01-01"
            start = max(start, first)
            if last:
                end = min(end, last)
            if start >= end:
                continue
            out = RAW / f"{name}_{year}.csv"
            if out.exists():
                print(f"   {year}: already downloaded")
                continue
            df = fetch(name, start, end)
            if df.empty:
                print(f"   {year}: no data")
                continue
            df.to_csv(out, index=False)
            print(f"   {year}: {len(df):>7} rows  columns={list(df.columns)}")
            time.sleep(0.5)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    y0 = args[0] if args else FIRST_YEAR
    y1 = args[1] if len(args) > 1 else LAST_YEAR
    main(y0, y1)

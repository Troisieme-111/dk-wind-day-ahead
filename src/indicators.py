"""Step 2 - indicators and descriptive figures.

For each area and year (and month):
  wind_share   = sum(wind) / sum(load)            (VRE share adds solar)
  mean_price   = time-average day-ahead price
  vol_intraday = mean over days of the std of the 24 hourly prices
  cv_intraday  = vol_intraday / mean_price (relative volatility)
  capture_wind = (sum(price*wind)/sum(wind)) / mean_price   (value factor)
  neg_hours    = number of hours with price < 0
"""
import numpy as np
import pandas as pd

from config import HOURLY_FILE, RESULTS, FIGURES, AREAS, COLORS, setup_matplotlib


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(HOURLY_FILE, parse_dates=["time_utc"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    local = df["time_utc"].dt.tz_convert("Europe/Copenhagen")
    df["date"] = local.dt.date
    df["year"] = local.dt.year
    df["hour"] = local.dt.hour
    df["month"] = local.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    return df


def metrics(g: pd.DataFrame, price: str = "price") -> pd.Series:
    """Indicators for one group of hours (one area-year or area-month)."""
    p = g[price]
    mean_p = p.mean()
    vol = g.groupby("date")[price].std().mean()
    ok = g["load"].notna()
    w, l, s = g.loc[ok, "wind"], g.loc[ok, "load"], g.loc[ok, "solar"]
    cap = (p[ok] * w).sum() / w.sum() / p[ok].mean() if w.sum() > 0 else np.nan
    cap_s = (p[ok] * s).sum() / s.sum() / p[ok].mean() if s.sum() > 0 else np.nan
    return pd.Series({
        "hours": len(g),
        "wind_share": w.sum() / l.sum() if ok.any() else np.nan,
        "vre_share": (w.sum() + s.sum()) / l.sum() if ok.any() else np.nan,
        "mean_price": mean_p,
        "vol_intraday": vol,
        "cv_intraday": vol / mean_p if mean_p > 0 else np.nan,
        "capture_wind": cap,
        "capture_solar": cap_s,
        "neg_hours": int((p < 0).sum()),
    })


def table(df: pd.DataFrame, by: str, price: str = "price") -> pd.DataFrame:
    return (df.groupby(["area", by]).apply(metrics, price=price, include_groups=False)
              .reset_index())


# ---------------------------------------------------------------- figures
def _per_area(ax, t, col, label=None, **kw):
    for a in AREAS:
        s = t[t["area"] == a]
        ax.plot(s["year"], s[col], marker="o", color=COLORS[a], label=a, **kw)


def figures(yearly: pd.DataFrame, monthly: pd.DataFrame, df: pd.DataFrame):
    plt = setup_matplotlib()
    y = yearly[yearly["hours"] > 8000]          # complete years only

    # 1. Penetration
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _per_area(ax, y, "wind_share")
    for a in AREAS:
        s = y[y["area"] == a]
        ax.plot(s["year"], s["vre_share"], ls="--", color=COLORS[a], lw=1.5)
    ax.set_ylabel("share of load")
    ax.set_title("Wind share (solid) and wind + solar share (dashed)", loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend()
    fig.savefig(FIGURES / "fig1_penetration.png"); plt.close(fig)

    # 2. Price level
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _per_area(ax, y, "mean_price")
    ax.axvspan(2021.5, 2022.5, color=COLORS["grey"], alpha=0.12)
    ax.text(2022, ax.get_ylim()[1] * 0.95, "2022 crisis", ha="center",
            va="top", color=COLORS["muted"], fontsize=8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Average day-ahead price", loc="left")
    ax.legend()
    fig.savefig(FIGURES / "fig2_price_level.png"); plt.close(fig)

    # 3. Volatility: absolute and relative, two panels (no dual axis)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    _per_area(axs[0], y, "vol_intraday")
    axs[0].set_ylabel("EUR/MWh")
    axs[0].set_title("Intraday volatility (std of 24 hourly prices)", loc="left", fontsize=10)
    _per_area(axs[1], y, "cv_intraday")
    axs[1].set_ylabel("std / mean")
    axs[1].set_title("Relative intraday volatility", loc="left", fontsize=10)
    axs[0].legend()
    fig.savefig(FIGURES / "fig3_volatility.png"); plt.close(fig)

    # 4. Capture rate vs wind share
    fig, ax = plt.subplots(figsize=(7, 4.2))
    m = monthly[(monthly["mean_price"] > 5) & monthly["capture_wind"].between(0, 1.5)]
    for a in AREAS:
        s = m[m["area"] == a]
        ax.scatter(s["wind_share"], s["capture_wind"], s=10, color=COLORS[a], alpha=0.25)
        s = y[y["area"] == a].dropna(subset=["capture_wind"])
        ax.plot(s["wind_share"], s["capture_wind"], marker="o", color=COLORS[a], label=f"{a} (yearly)")
        for _, r in s.iloc[[0, -1]].iterrows():
            ax.annotate(int(r["year"]), (r["wind_share"], r["capture_wind"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8, color=COLORS["muted"])
    ax.axhline(1, color=COLORS["grey"], lw=1)
    ax.set_xlabel("wind share of load")
    ax.set_ylabel("wind capture rate")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Wind capture rate vs wind share (dots: months)", loc="left")
    ax.legend()
    fig.savefig(FIGURES / "fig4_capture_rate.png"); plt.close(fig)

    # 5. Negative-price hours
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.4
    for i, a in enumerate(AREAS):
        s = y[y["area"] == a]
        ax.bar(s["year"] + (i - 0.5) * width, s["neg_hours"], width=width - 0.04,
               color=COLORS[a], label=a)
    ax.set_ylabel("hours per year")
    ax.set_title("Hours with a negative day-ahead price", loc="left")
    ax.legend()
    fig.savefig(FIGURES / "fig5_negative_hours.png"); plt.close(fig)

    # 6. Why a merit-order model: price gap to neighbours vs residual load
    d = df[(df["area"] == "DK1") & df["load"].notna()].copy()
    d["p_neigh_m"] = d.groupby("month")["p_neigh"].transform("mean")
    d["x"] = (d["load"] - d["wind"] - d["solar"]) / d.groupby("year")["load"].transform("mean")
    d["rel"] = d["price"] / d["p_neigh_m"]
    d = d[d["p_neigh_m"] > 5]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    hb = ax.hexbin(d["x"], d["rel"].clip(-1, 3), gridsize=60, cmap="Blues", mincnt=3, bins="log")
    fig.colorbar(hb, ax=ax, label="hours (log)")
    ax.set_xlabel("residual load / mean load  (load − wind − solar)")
    ax.set_ylabel("price / monthly neighbour price")
    ax.set_title("DK1: price falls below neighbours when residual load is low", loc="left")
    fig.savefig(FIGURES / "fig6_price_vs_residual.png"); plt.close(fig)


if __name__ == "__main__":
    df = load_hourly()
    yearly = table(df, "year")
    monthly = table(df, "month")
    yearly.to_csv(RESULTS / "indicators_yearly.csv", index=False)
    monthly.to_csv(RESULTS / "indicators_monthly.csv", index=False)
    print(yearly.round(3).to_string(index=False))
    figures(yearly, monthly, df)
    print(f"Figures saved in {FIGURES}")

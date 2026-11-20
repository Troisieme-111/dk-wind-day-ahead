"""Step 3 - Model B: a toy merit-order model with one interconnector.

Idea
----
Denmark is a small area coupled to large neighbours (Nordic hydro, Germany).
Wind and solar have zero marginal cost, so what sets the Danish price is the
*residual load* x_t = (load - wind - solar) / mean load, met by thermal
plants and by trade.

Equations (one hour t, P_t = neighbour price averaged by month and hour of day)
-------------------------------------------------------------------------------
  Domestic supply curve (merit order), linear in residual load with a floor:
      f(z) = max( p_floor , P_t * (1 + b * (z - r0)) )
  r0 is the residual load at which domestic cost equals the neighbour price,
  b the steepness of the merit order, p_floor the lowest bid (negative when
  subsidised producers keep running at negative prices).

  Interconnector of capacity k (as a share of mean load) - two-node market coupling:
      if x_t - k > r0 :  p_t = f(x_t - k)   import line full, price above neighbours
      if x_t + k < r0 :  p_t = f(x_t + k)   export line full, price below neighbours
      otherwise       :  p_t = P_t          coupled, Denmark takes the neighbour price

Four parameters (b, r0, k, p_floor) are fitted on hourly prices 2015-2019
and tested on 2020-2025. Scenarios then scale wind and solar output of a
reference year to reach a target share and record the three outcomes of the
research question (price level, volatility, capture rate) plus negative hours.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (RESULTS, FIGURES, AREAS, COLORS, CALIBRATION_YEARS,
                    VALIDATION_YEARS, setup_matplotlib)
from indicators import load_hourly, table, metrics

PARAM_NAMES = ["b", "r0", "k", "p_floor"]
X0 = np.array([1.0, 0.6, 0.4, -5.0])


# ------------------------------------------------------------------ core
def supply(z, P, b, r0, p_floor):
    return np.maximum(p_floor, P * (1 + b * (z - r0)))


def price(x, P, b, r0, k, p_floor):
    """Hourly price from residual load share x and neighbour price P (arrays)."""
    k = max(k, 0.0)
    p = np.array(P, dtype=float, copy=True)
    imp = x - k > r0
    exp_ = x + k < r0
    p[imp] = supply(x[imp] - k, P[imp], b, r0, p_floor)
    p[exp_] = supply(x[exp_] + k, P[exp_], b, r0, p_floor)
    return p


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add model inputs: residual-load share x and monthly neighbour price P."""
    d = df[df["load"].notna() & df["p_neigh"].notna()].copy()
    d["mean_load"] = d.groupby(["area", "year"])["load"].transform("mean")
    d["x"] = (d["load"] - d["wind"] - d["solar"]) / d["mean_load"]
    # Neighbour price averaged by month and hour of day: keeps the fuel level and
    # the typical daily profile, but removes the hour-to-hour weather signal that
    # Danish wind shares with its neighbours (avoids circularity).
    d["P"] = d.groupby(["area", "month", "hour"])["p_neigh"].transform("mean")
    return d


def run(d: pd.DataFrame, theta) -> np.ndarray:
    return price(d["x"].to_numpy(), d["P"].to_numpy(), *theta)


# ------------------------------------------------------------ calibration
def calibrate(d: pd.DataFrame) -> np.ndarray:
    """Least absolute error on hourly prices (robust to price spikes)."""
    cal = d[d["year"].isin(CALIBRATION_YEARS)]
    y = cal["price"].to_numpy()

    def loss(theta):
        if theta[0] <= 0 or theta[2] < 0:
            return 1e9
        return np.mean(np.abs(run(cal, theta) - y))

    best = None
    for k0 in (0.2, 0.4, 0.8):                       # a few starts
        r = minimize(loss, X0 * [1, 1, 0, 1] + [0, 0, k0, 0], method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def fit_stats(d: pd.DataFrame, years) -> dict:
    s = d[d["year"].isin(years)]
    e = s["p_model"] - s["price"]
    base = s["P"] - s["price"]                       # naive: neighbour monthly price
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAE_naive": np.abs(base).mean(),
            "corr": np.corrcoef(s["p_model"], s["price"])[0, 1]}


# -------------------------------------------------------------- scenarios
def scenario_curve(ref: pd.DataFrame, theta, shares) -> pd.DataFrame:
    """Scale wind+solar of a reference year to each target VRE share."""
    base_share = (ref["wind"].sum() + ref["solar"].sum()) / ref["load"].sum()
    rows = []
    for s in shares:
        a = s / base_share
        r = ref.copy()
        r["wind"], r["solar"] = ref["wind"] * a, ref["solar"] * a
        r["x"] = (r["load"] - r["wind"] - r["solar"]) / r["mean_load"]
        r["p_model"] = run(r, theta)
        m = metrics(r, price="p_model")
        m["target_share"] = s
        m["rel_price"] = m["mean_price"] / r["P"].mean()
        rows.append(m)
    return pd.DataFrame(rows)


def sensitivity(ref, theta, shares):
    """Vary one parameter at a time around the calibrated value."""
    b, r0, k, pf = theta
    cases = {
        "interconnection k": [("k = 0 (isolated)", [b, r0, 0, pf]),
                              ("k x 0.5", [b, r0, k * 0.5, pf]),
                              ("k fitted", theta),
                              ("k x 2", [b, r0, k * 2, pf])],
        "price floor": [("floor = 0 (no subsidy)", [b, r0, k, 0.0]),
                        ("floor fitted", theta),
                        ("floor = -50", [b, r0, k, -50.0])],
        "merit-order slope b": [("b x 0.5", [b * 0.5, r0, k, pf]),
                                ("b fitted", theta),
                                ("b x 1.5", [b * 1.5, r0, k, pf])],
    }
    out = []
    for group, lst in cases.items():
        for label, th in lst:
            c = scenario_curve(ref, np.array(th, dtype=float), shares)
            c["group"], c["case"] = group, label
            out.append(c)
    return pd.concat(out, ignore_index=True)


def literature_check(ref, theta, area):
    """Model analogues of two published numbers (see README, step 4)."""
    share = (ref["wind"].sum() + ref["solar"].sum()) / ref["load"].sum()
    c = scenario_curve(ref, theta, [share * 0.99, share * 1.01, share - 0.05, share + 0.05])
    # elasticity of the mean price to wind output (Rintamaki et al. 2017: about -0.07)
    elas = np.log(c["mean_price"][1] / c["mean_price"][0]) / np.log(1.01 / 0.99)
    # capture-rate slope, points per percentage point (Stiewe et al. 2024: -0.62)
    slope = (c["capture_wind"][3] - c["capture_wind"][2]) / 10 * 100
    return {"area": area, "vre_share": share, "price_elasticity_to_vre": elas,
            "capture_slope_pts_per_pp": slope}


# ---------------------------------------------------------------- figures
def fig_validation(obs, mod, plt):
    cols = [("mean_price", "Average price (EUR/MWh)"), ("vol_intraday", "Intraday volatility (EUR/MWh)"),
            ("capture_wind", "Wind capture rate"), ("neg_hours", "Negative-price hours")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (c, title) in zip(axs.flat, cols):
        for a in AREAS:
            o, m = obs[obs["area"] == a], mod[mod["area"] == a]
            ax.plot(o["year"], o[c], color=COLORS[a], marker="o", label=f"{a} observed")
            ax.plot(m["year"], m[c], color=COLORS[a], ls="--", marker="s", mfc="white",
                    label=f"{a} model")
        ax.axvspan(min(CALIBRATION_YEARS) - 0.5, max(CALIBRATION_YEARS) + 0.5,
                   color=COLORS["grey"], alpha=0.10)
        ax.set_title(title, loc="left", fontsize=10)
    axs[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Model vs observed (shaded: calibration years)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_model_validation.png"); plt.close(fig)


def fig_scenarios(curves, plt, ref_year):
    cols = [("rel_price", "Average price / neighbour price"),
            ("vol_intraday", "Intraday volatility (EUR/MWh)"),
            ("capture_wind", "Wind capture rate"), ("neg_hours", "Negative-price hours")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (c, title) in zip(axs.flat, cols):
        for a in AREAS:
            s = curves[curves["area"] == a]
            ax.plot(s["target_share"], s[c], color=COLORS[a], label=a)
            ax.axvline(s["actual_share"].iloc[0], color=COLORS[a], lw=1, ls=":")
        ax.set_title(title, loc="left", fontsize=10)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for ax in axs[1]:
        ax.set_xlabel("wind + solar share of load")
    axs[0, 0].legend()
    fig.suptitle(f"Scenarios: wind and solar scaled up, weather and load of {ref_year} "
                 "(dotted: actual share)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_scenarios.png"); plt.close(fig)


def fig_sensitivity(sens, plt, area, ref_year):
    groups = sens["group"].unique()
    fig, axs = plt.subplots(2, len(groups), figsize=(12, 6.5), sharex=True)
    shades = ["#9ec5f0", "#5a9be3", "#2a78d6", "#0d3f80"]
    for j, g in enumerate(groups):
        s = sens[sens["group"] == g]
        for i, case in enumerate(s["case"].unique()):
            c = s[s["case"] == case]
            col = shades[i + (4 - s["case"].nunique())]
            axs[0, j].plot(c["target_share"], c["capture_wind"], color=col, label=case)
            axs[1, j].plot(c["target_share"], c["rel_price"], color=col)
        axs[0, j].set_title(g, loc="left", fontsize=10)
        axs[0, j].legend(fontsize=8)
        axs[1, j].set_xlabel("wind + solar share of load")
        axs[1, j].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axs[0, 0].set_ylabel("wind capture rate")
    axs[1, 0].set_ylabel("average price / neighbour price")
    fig.suptitle(f"Sensitivity, {area}, reference year {ref_year}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig9_sensitivity_{area}.png"); plt.close(fig)


# ------------------------------------------------------------------- main
def main(ref_year: int = 2019):
    plt = setup_matplotlib()
    d = prepare(load_hourly())
    shares = np.round(np.arange(0.05, 1.21, 0.05), 2)

    params, stats, curves, checks, sims = [], [], [], [], []
    for a in AREAS:
        da = d[d["area"] == a].copy()
        theta = calibrate(da)
        da["p_model"] = run(da, theta)
        sims.append(da)
        params.append({"area": a, **dict(zip(PARAM_NAMES, theta))})
        for label, yrs in [("calibration", CALIBRATION_YEARS), ("validation", VALIDATION_YEARS)]:
            stats.append({"area": a, "period": label, **fit_stats(da, yrs)})

        ref = da[da["year"] == ref_year]
        c = scenario_curve(ref, theta, shares)
        c["area"] = a
        c["actual_share"] = (ref["wind"].sum() + ref["solar"].sum()) / ref["load"].sum()
        curves.append(c)
        checks.append(literature_check(ref, theta, a))
        if a == "DK1":
            sens = sensitivity(ref, theta, shares)
            sens.to_csv(RESULTS / "model_sensitivity_DK1.csv", index=False)
            fig_sensitivity(sens, plt, a, ref_year)

    params, stats = pd.DataFrame(params), pd.DataFrame(stats)
    curves, checks = pd.concat(curves), pd.DataFrame(checks)
    sim = pd.concat(sims)
    obs = table(sim, "year", price="price")
    mod = table(sim, "year", price="p_model")
    obs, mod = obs[obs["hours"] > 8000], mod[mod["hours"] > 8000]

    params.to_csv(RESULTS / "model_parameters.csv", index=False)
    stats.to_csv(RESULTS / "model_fit.csv", index=False)
    curves.to_csv(RESULTS / "model_scenarios.csv", index=False)
    checks.to_csv(RESULTS / "model_literature_check.csv", index=False)
    obs.merge(mod, on=["area", "year"], suffixes=("_obs", "_model")).to_csv(
        RESULTS / "model_vs_observed_yearly.csv", index=False)

    fig_validation(obs, mod, plt)
    fig_scenarios(curves, plt, ref_year)

    pd.set_option("display.width", 140)
    print("Parameters\n", params.round(3).to_string(index=False))
    print("\nFit (EUR/MWh)\n", stats.round(2).to_string(index=False))
    print("\nComparison with literature\n", checks.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2019)

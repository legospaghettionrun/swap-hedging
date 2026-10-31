import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import backtest as bt
from src.changes import daily_changes_bp
from src.portfolio import simulate_book
from src.run_backtest import table, TRAIN, TEST

# (spread multiplier, impact multiplier) applied to the base cost model
# NOTE: scaling both by the same factor would only rescale the frontier (after
# re-tuning lambda), so the scenarios below change the MIX of linear vs convex cost.
SCENARIOS = {
    "base": (1.0, 1.0),
    "mostly linear": (1.0, 0.1),     # spread-heavy: the optimiser can't see this part
    "nearly pure linear": (1.0, 0.01),
    "mostly convex": (0.2, 5.0),     # impact-heavy: lumpy trades are punished
}


def run_scenario(dy, book, cov):
    pol = {"no hedge": bt.p_none(), "min-var daily": bt.p_minvar()}
    for lim in [5e3, 10e3, 25e3, 50e3, 100e3, 200e3, 400e3]:
        pol[f"threshold {lim / 1e3:.0f}k"] = bt.p_threshold(lim)
    for lam in 10.0 ** np.arange(4, 13):          # wide grid: C changes by scenario
        pol[f"cost-aware {lam:.0e}"] = bt.p_optimal(lam)
    return {n: bt.run(p, dy, book, cov) for n, p in pol.items()}


def matched_saving(tab):
    """Average cost saving of cost-aware vs threshold at the SAME risk level."""
    thr = tab[tab.index.str.startswith("threshold")]
    opt = tab[tab.index.str.startswith("cost-aware")].sort_values("risk $k/day")
    x, y = opt["risk $k/day"].values, opt["cost $k/yr"].values
    out = []
    for _, row in thr.iterrows():
        r, k = row["risk $k/day"], row["cost $k/yr"]
        if x.min() <= r <= x.max() and k > 0:
            out.append(1 - np.interp(r, x, y) / k)
    return (100 * np.mean(out), len(out)) if out else (np.nan, 0)


def main():
    dy = daily_changes_bp()
    book = simulate_book(dy.index)
    cov = bt.ewma_cov_path(dy)
    base_s, base_i = bt.SPREAD, bt.IMPACT

    rows, test_tables = [], {}
    for name, (ms, mi) in SCENARIOS.items():
        bt.set_costs(base_s * ms, base_i * mi)
        res = run_scenario(dy, book, cov)
        tr, te = table(res, TRAIN), table(res, TEST)
        test_tables[name] = te
        s_tr, _ = matched_saving(tr)
        s_te, n_te = matched_saving(te)
        rows.append({
            "scenario": name,
            "min-var cost $k/yr (test)": te.loc["min-var daily", "cost $k/yr"],
            "saving vs threshold, train %": s_tr,
            "saving vs threshold, test %": s_te,
            "points compared (test)": n_te,
        })
    bt.set_costs(base_s, base_i)

    summary = pd.DataFrame(rows).set_index("scenario").round(1)
    pd.set_option("display.width", 140)
    print("Cost-aware vs threshold: average cost saving at matched risk")
    print("(positive = cost-aware cheaper; negative = threshold cheaper)\n")
    print(summary.to_string())
    summary.to_csv("results/cost_sensitivity.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (name, te) in zip(axes, test_tables.items()):
        thr = te[te.index.str.startswith("threshold")].sort_values("cost $k/yr")
        opt = te[te.index.str.startswith("cost-aware")].sort_values("cost $k/yr")
        mv = te.loc["min-var daily"]
        ax.plot(thr["cost $k/yr"], thr["risk $k/day"], "o-", label="threshold")
        ax.plot(opt["cost $k/yr"], opt["risk $k/day"], "s-", label="cost-aware")
        ax.scatter(mv["cost $k/yr"], mv["risk $k/day"], marker="*", s=180,
                   color="k", label="min-var daily", zorder=3)
        ax.set_ylim(0, 6 * mv["risk $k/day"])
        ax.set_title(name)
        ax.set_xlabel("cost ($k/yr)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("risk ($k/day)")
    axes[0].legend()
    fig.suptitle("Test period 2020-2026: does the frontier ranking survive other cost models?")
    plt.tight_layout()
    plt.savefig("results/cost_sensitivity.png", dpi=150)
    print("\nsaved results/cost_sensitivity.png and results/cost_sensitivity.csv")


if __name__ == "__main__":
    main()

import pandas as pd
from src.changes import daily_changes_bp
from src.portfolio import simulate_book
from src.backtest import (ewma_cov_path, run, p_none, p_minvar,
                          p_threshold, p_optimal)

TRAIN = slice(None, "2019-12-31")
TEST = slice("2020-01-01", None)


def table(results, period):
    base = results["no hedge"].loc[period]
    rows = {}
    for name, res in results.items():
        r = res.loc[period]
        yrs = len(r) / 252
        rows[name] = {
            "risk $k/day": r.gross.std() / 1e3,
            "var reduction %": 100 * (1 - r.gross.var() / base.gross.var()),
            "cost $k/yr": r.cost.sum() / 1e3 / yrs,
            "traded $M/yr": r.traded.sum() / 1e6 / yrs,
        }
    return pd.DataFrame(rows).T.round(1)


def main():
    dy = daily_changes_bp()
    book = simulate_book(dy.index)
    cov = ewma_cov_path(dy)

    policies = {"no hedge": p_none(), "min-var daily": p_minvar()}
    for lim in [10e3, 25e3, 50e3, 100e3, 200e3]:
        policies[f"threshold {lim / 1e3:.0f}k"] = p_threshold(lim)
    for lam in [1e5, 1e6, 1e7, 1e8, 1e9]:
        policies[f"cost-aware {lam:.0e}"] = p_optimal(lam)

    results = {n: run(p, dy, book, cov) for n, p in policies.items()}

    pd.set_option("display.width", 120)
    for label, period in [("TRAIN 2010-2019", TRAIN), ("TEST 2020-2026", TEST)]:
        t = table(results, period)
        print(f"\n=== {label} ===")
        print(t.to_string())
        t.to_csv(f"results/summary_{label.split()[0].lower()}.csv")


if __name__ == "__main__":
    main()

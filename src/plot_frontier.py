import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = [("train", "Train 2010-2019"), ("test", "Test 2020-2026")]
    for ax, (name, title) in zip(axes, panels):
        t = pd.read_csv(f"results/summary_{name}.csv", index_col=0)
        base = t.loc["no hedge", "risk $k/day"]
        thr = t[t.index.str.startswith("threshold")]
        opt = t[t.index.str.startswith("cost-aware")]
        ax.plot(thr["cost $k/yr"], thr["risk $k/day"], "o-", label="threshold")
        ax.plot(opt["cost $k/yr"], opt["risk $k/day"], "s-", label="cost-aware")
        for lab, row in thr.iterrows():
            ax.annotate(lab.replace("threshold ", ""), (row["cost $k/yr"], row["risk $k/day"]),
                        textcoords="offset points", xytext=(5, -10), fontsize=7)
        for lab, row in opt.iterrows():
            ax.annotate("lam=" + lab.replace("cost-aware ", ""),
                        (row["cost $k/yr"], row["risk $k/day"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
        mv = t.loc["min-var daily"]
        ax.scatter(mv["cost $k/yr"], mv["risk $k/day"], marker="*", s=200,
                   color="k", label="min-var daily", zorder=3)
        ax.set_title(f"{title} (unhedged: {base:.0f}k/day, off-chart)")
        ax.set_xlabel("transaction cost ($k / year)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("residual risk ($k/day, std of daily P&L)")
    axes[0].legend()
    fig.suptitle("Hedging frontier: closer to the bottom-left is better")
    plt.tight_layout()
    plt.savefig("results/frontier.png", dpi=150)
    print("saved results/frontier.png")


if __name__ == "__main__":
    main()

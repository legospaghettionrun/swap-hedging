import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.changes import daily_changes_bp


def pca(dy):
    cov = np.cov(dy.values.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # sign conventions so the shapes read the same way every run
    if vecs[:, 0].sum() < 0:
        vecs[:, 0] *= -1                                  # level: positive
    if vecs[-1, 1] - vecs[0, 1] < 0:
        vecs[:, 1] *= -1                                  # slope: 30Y minus 2Y > 0
    if vecs[0, 2] + vecs[3, 2] - vecs[1, 2] - vecs[2, 2] < 0:
        vecs[:, 2] *= -1                                  # curvature: wings minus belly > 0
    return cov, vals / vals.sum(), vecs


def main():
    dy = daily_changes_bp()
    print(f"{len(dy)} daily changes, {dy.index[0].date()} to {dy.index[-1].date()}")

    cov, ratio, vecs = pca(dy)
    print("\nexplained variance by component:")
    for i, r in enumerate(ratio, 1):
        print(f"  PC{i}: {r:.1%}")

    print("\nloadings (rows = tenor, cols = PC1..PC4):")
    for t, row in zip(dy.columns, vecs):
        print(f"  {t:>3}: " + "  ".join(f"{x:+.2f}" for x in row))

    print("\ndaily vol of changes by year (bp):")
    print(dy.groupby(dy.index.year).std().round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = ["PC1 level", "PC2 slope", "PC3 curvature"]
    for i in range(3):
        ax[0].plot(dy.columns, vecs[:, i], marker="o", label=names[i])
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_title("Principal component loadings")
    ax[0].legend()
    ax[1].bar(range(1, 5), ratio * 100)
    ax[1].set_title("Variance explained (%)")
    ax[1].set_xticks(range(1, 5))
    ax[1].set_xticklabels([f"PC{i}" for i in range(1, 5)])
    plt.tight_layout()
    plt.savefig("results/pca.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4))
    dy["10Y"].rolling(60).std().plot(ax=ax)
    ax.set_title("Rolling 60-day volatility of 10Y daily changes (bp)")
    plt.tight_layout()
    plt.savefig("results/vol_regimes.png", dpi=150)
    print("\nsaved results/pca.png and results/vol_regimes.png")


if __name__ == "__main__":
    main()

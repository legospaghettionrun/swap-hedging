import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.changes import daily_changes_bp
from src.portfolio import simulate_book, unhedged_pnl


def main():
    dy = daily_changes_bp()
    book = simulate_book(dy.index)
    pnl = unhedged_pnl(book, dy)

    print("book DV01 ($/bp) summary:")
    print(book.describe().loc[["mean", "std", "min", "max"]].round(0).to_string())

    print(f"\nunhedged daily P&L: mean ${pnl.mean():,.0f}, std ${pnl.std():,.0f}")
    print(f"worst day ${pnl.min():,.0f} on {pnl.idxmin().date()}")
    print(f"best day  ${pnl.max():,.0f} on {pnl.idxmax().date()}")
    print("\ndaily P&L std by year ($k):")
    print((pnl.groupby(pnl.index.year).std() / 1e3).round(0).to_string())

    fig, ax = plt.subplots(figsize=(11, 4))
    (pnl.cumsum() / 1e6).plot(ax=ax)
    ax.set_title("Cumulative unhedged P&L ($M, illustrative book)")
    plt.tight_layout()
    plt.savefig("results/unhedged_pnl.png", dpi=150)
    print("\nsaved results/unhedged_pnl.png")


if __name__ == "__main__":
    main()

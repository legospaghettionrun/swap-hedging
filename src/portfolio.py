import numpy as np
import pandas as pd

BUCKETS = ["2Y", "5Y", "10Y", "30Y"]

# H[i, j] = DV01 ($/bp) of ONE contract of instrument j in bucket i (illustrative)
H = np.array([
    [400.0,  40.0,   0.0,    0.0],
    [20.0,  450.0,  60.0,    0.0],
    [0.0,    15.0, 800.0,  100.0],
    [0.0,     0.0,  30.0, 1800.0],
])

BOOK_MEAN = np.array([20e3, 40e3, 60e3, 30e3])   # $/bp, illustrative


def simulate_book(index, seed=42, phi=0.98, shock=0.07):
    """DV01 held at the close of each day.

    log(DV01 / mean) follows a mean-reverting AR(1), so DV01 stays positive.
    Stationary std of the log is about shock / sqrt(1 - phi^2) = 0.35.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((len(index), 4))
    for t in range(1, len(index)):
        x[t] = phi * x[t - 1] + rng.normal(scale=shock, size=4)
    return pd.DataFrame(BOOK_MEAN * np.exp(x), index=index, columns=BUCKETS)


def unhedged_pnl(book, dy):
    """Day t P&L uses the DV01 held at the close of day t-1."""
    return -(book.shift(1) * dy).sum(axis=1).dropna()

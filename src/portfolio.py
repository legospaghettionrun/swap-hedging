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


def simulate_book(index, seed=42, phi=0.98, shock=1.5e3):
    """DV01 held at the close of each day (mean-reverting around BOOK_MEAN)."""
    rng = np.random.default_rng(seed)
    d = np.zeros((len(index), 4))
    d[0] = BOOK_MEAN
    for t in range(1, len(index)):
        d[t] = BOOK_MEAN + phi * (d[t - 1] - BOOK_MEAN) + rng.normal(scale=shock, size=4)
    return pd.DataFrame(d, index=index, columns=BUCKETS)


def unhedged_pnl(book, dy):
    """Day t P&L uses the DV01 held at the close of day t-1."""
    return -(book.shift(1) * dy).sum(axis=1).dropna()

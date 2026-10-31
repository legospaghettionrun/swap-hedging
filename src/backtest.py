import numpy as np
import pandas as pd
from src.portfolio import H

# Hedge instruments: 2Y, 5Y, 10Y futures only (no 30Y future), so the 30Y
# bucket can only be hedged through correlation -> residual risk exists.
INSTR = [0, 1, 2]
HS = H[:, INSTR]                    # 4 buckets x 3 instruments
DV = np.diag(H)[INSTR]              # DV01 per contract in its own bucket

# Illustrative transaction costs (state these as assumptions!)
SPREAD = 0.5                        # $ per $/bp of DV01 traded (linear)
IMPACT = 1e-5                       # $ per ($/bp)^2 of DV01 traded (quadratic)
C = np.diag(IMPACT * DV ** 2)       # quadratic cost, per contract^2
WARMUP = 60


def trade_cost(q):
    x = np.abs(q) * DV              # DV01 traded per instrument
    return SPREAD * x.sum() + IMPACT * (x ** 2).sum()


def ewma_cov_path(dy, lam=0.94, warmup=WARMUP):
    """path[t] = EWMA covariance estimate using data through day t only."""
    X = dy.values
    path = np.full((len(X), 4, 4), np.nan)
    S = X[:warmup].T @ X[:warmup] / warmup
    path[warmup - 1] = S
    for t in range(warmup, len(X)):
        S = lam * S + (1 - lam) * np.outer(X[t], X[t])
        path[t] = S
    return path


def risk(r, S):
    return float(np.sqrt(max(r @ S @ r, 0.0)))


def minvar_trade(r0, S):
    """Trade that minimises residual variance, ignoring costs."""
    return -np.linalg.solve(HS.T @ S @ HS, HS.T @ S @ r0)


# ---- policies: f(book_dv01, position, cov) -> trade in contracts
def p_none():
    return lambda b, pos, S: np.zeros(len(INSTR))


def p_minvar():
    return lambda b, pos, S: minvar_trade(b + HS @ pos, S)


def p_threshold(limit):
    def f(b, pos, S):
        r0 = b + HS @ pos
        q = minvar_trade(r0, S)
        excess = risk(r0, S) - risk(r0 + HS @ q, S)   # $/day of risk we would remove
        return q if excess > limit else np.zeros(len(INSTR))
    return f


def p_optimal(lam):
    def f(b, pos, S):
        r0 = b + HS @ pos
        A = HS.T @ S @ HS + lam * C
        return -np.linalg.solve(A, HS.T @ S @ r0)
    return f


def run(policy, dy, book, cov_path, warmup=WARMUP):
    """Decide at the close of day i-1 (using data up to i-1), earn P&L on day i."""
    X, B = dy.values, book.values
    pos = np.zeros(len(INSTR))
    rows = []
    for i in range(warmup, len(X)):
        b, S = B[i - 1], cov_path[i - 1]
        q = policy(b, pos, S)
        cost = trade_cost(q)
        pos = pos + q
        r = b + HS @ pos
        rows.append((-(r @ X[i]), cost, float((np.abs(q) * DV).sum())))
    return pd.DataFrame(rows, index=dy.index[warmup:],
                        columns=["gross", "cost", "traded"])

from src.data import load_yields

START = "2010-01-04"


def daily_changes_bp(path="data/fredgraph.csv", start=START):
    y = load_yields(path).loc[start:].dropna()
    y.columns = ["2Y", "5Y", "10Y", "30Y"]
    return y.diff().dropna() * 100.0   # yields are in %, so x100 = bp

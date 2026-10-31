import pandas as pd

TENORS = ["DGS2", "DGS5", "DGS10", "DGS30"]


def load_yields(path="data/fredgraph.csv"):
    df = pd.read_csv(path, parse_dates=["observation_date"],
                     index_col="observation_date")
    df = df[TENORS].apply(pd.to_numeric, errors="coerce")
    return df


if __name__ == "__main__":
    df = load_yields()
    print("rows in file:", len(df))
    print("\nfirst valid date per tenor:")
    print(df.apply(lambda s: s.first_valid_index()))
    print("\nlast valid date per tenor:")
    print(df.apply(lambda s: s.last_valid_index()))
    print("\nmissing values per tenor since 2006-02-09:")
    print(df.loc["2006-02-09":].isna().sum())
    print("\nrows with all four tenors:", len(df.dropna()))
    print("\nlast 3 rows:")
    print(df.tail(3))

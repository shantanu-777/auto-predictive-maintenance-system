# src/data_processing/preprocessing.py
import pandas as pd
from pathlib import Path

def compute_rul_train(df):
    # For each unit, RUL = max_cycle - current_cycle
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df

def compute_rul_test(df_test, rul_series):
    """
    The test set is truncated before failure. The provided rul_series gives the true RUL
    for the last cycle of each unit in the test set (order corresponds to unit id order).
    We'll append RUL values backwards across cycles.
    """
    df = df_test.copy()
    # ensure unit order aligned with rul_series by unique sorted units
    units = sorted(df["unit"].unique())
    last_cycle = df.groupby("unit")["cycle"].max().reindex(units)
    # build mapping unit -> RUL at last observed cycle
    rul_at_last = dict(zip(units, rul_series.tolist()))
    # for each row: RUL = rul_at_last[unit] + (last_cycle[unit] - cycle)
    df["last_cycle"] = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["unit"].map(rul_at_last) + (df["last_cycle"] - df["cycle"])
    df = df.drop(columns=["last_cycle"])
    return df

def clip_rul(df, max_rul=130):
    df = df.copy()
    df["RUL_clipped"] = df["RUL"].clip(upper=max_rul)
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", default="data/processed/nasa_cmapss")
    p.add_argument("--out_dir", default="data/processed/for_model")
    args = p.parse_args()

    proc = Path(args.proc_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for train_csv in proc.glob("*_train.csv"):
        ds = train_csv.stem.split("_")[0]
        df_train = pd.read_csv(train_csv)
        df_train = compute_rul_train(df_train)
        df_train = clip_rul(df_train, max_rul=130)
        df_train.to_csv(out / f"{ds}_train_labeled.csv", index=False)

        test_csv = proc / f"{ds}_test.csv"
        rul_csv = proc / f"{ds}_rul.csv"
        if test_csv.exists() and rul_csv.exists():
            df_test = pd.read_csv(test_csv)
            rul = pd.read_csv(rul_csv, header=None).squeeze()
            df_test = compute_rul_test(df_test, rul)
            df_test = clip_rul(df_test, max_rul=130)
            df_test.to_csv(out / f"{ds}_test_labeled.csv", index=False)

    print("Processed and labeled data saved to", out)

# src/data_processing/feature_engineering.py
import pandas as pd
from pathlib import Path

SENSOR_COL_PREFIX = "sensor_"

def rolling_stat_features(df, window=30, sensor_cols=None):
    df = df.copy()
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith(SENSOR_COL_PREFIX)]
    out = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        # compute rolling features per sensor
        rolled = g[sensor_cols].rolling(window=window, min_periods=1).agg(["mean", "std", "min", "max"])
        # flatten multiindex
        rolled.columns = [f"{c[0]}_{c[1]}" for c in rolled.columns]
        tmp = pd.concat([g.reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

def create_features_for_dataset(in_csv, out_csv, window=30):
    df = pd.read_csv(in_csv)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df_feat = rolling_stat_features(df, window=window, sensor_cols=sensor_cols)
    df_feat.to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="data/processed/for_model")
    p.add_argument("--out_dir", default="data/processed/features")
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for csv in Path(args.in_dir).glob("*_labeled.csv"):
        out = Path(args.out_dir) / csv.name.replace("_labeled", "_features")
        create_features_for_dataset(csv, out, window=args.window)
    print("Feature files created in", args.out_dir)

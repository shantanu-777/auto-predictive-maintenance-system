# src/data_processing/save_norm_stats.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def compute_and_save_norm(proc_dir="data/processed/for_model", dataset="FD001", out_dir="models/sequence"):
    proc_dir = Path(proc_dir)
    train_csv = proc_dir / f"{dataset}_train_labeled.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found.")
    df = pd.read_csv(train_csv)
    # choose same features as dataset class
    feature_cols = [c for c in df.columns if (c.startswith("op_setting") or c.startswith("sensor_") or "_mean" in c or "_std" in c or "_min" in c or "_max" in c)]
    arr = df[feature_cols].to_numpy(dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    out = {"mean": mean.tolist(), "std": std.tolist(), "feature_order": feature_cols}
    out_path = Path(out_dir) / dataset
    out_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path / "norm_stats.joblib")
    # also save feature_order as features.joblib if not present
    try:
        joblib.dump(feature_cols, out_path / "features.joblib")
    except Exception:
        pass
    print("Saved norm_stats.joblib to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", default="data/processed/for_model")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--out_dir", default="models/sequence")
    args = parser.parse_args()
    compute_and_save_norm(args.proc_dir, args.dataset, args.out_dir)

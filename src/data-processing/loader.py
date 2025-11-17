# src/data_processing/loader.py
import pandas as pd
from pathlib import Path

COLS = [
    "unit", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 23)]  # 26 total columns: 5 + 21 sensors? adjust if 26 sensors

def read_cmapss_txt(path: Path):
    """
    Read a CMAPSS-style whitespace text file into a DataFrame.
    """
    df = pd.read_csv(path, sep='\s+', header=None)
    # If the file has 26 columns, adjust sensor naming accordingly:
    if df.shape[1] == 26:
        sensor_count = 26 - 5
        cols = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, sensor_count+1)]
    else:
        # fallback to generic naming
        cols = [f"c{i}" for i in range(df.shape[1])]
    df.columns = cols
    return df

def read_all_raw(raw_dir: str):
    """
    Expects files like: train_FD001.txt, test_FD001.txt, RUL_FD001.txt (or similar).
    Returns dict: {'FD001': {'train': df, 'test': df, 'rul': df_rul}, ...}
    """
    raw = Path(raw_dir)
    datasets = {}
    for txt in raw.glob("*.txt"):
        name = txt.stem.lower()
        # simple detection
        if "train" in name:
            key = name.split("train")[-1].strip("_.")
            ds = datasets.setdefault(key.upper(), {})
            ds['train'] = read_cmapss_txt(txt)
        elif "test" in name:
            key = name.split("test")[-1].strip("_.")
            ds = datasets.setdefault(key.upper(), {})
            ds['test'] = read_cmapss_txt(txt)
        elif "rul" in name:
            key = name.split("rul")[-1].strip("_.")
            ds = datasets.setdefault(key.upper(), {})
            rul = pd.read_csv(txt, header=None).squeeze()
            ds['rul'] = rul
    return datasets

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw/nasa_cmapss")
    p.add_argument("--out_dir", default="data/processed/nasa_cmapss")
    args = p.parse_args()

    datasets = read_all_raw(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds_name, parts in datasets.items():
        if 'train' in parts:
            parts['train'].to_csv(out / f"{ds_name}_train.csv", index=False)
        if 'test' in parts:
            parts['test'].to_csv(out / f"{ds_name}_test.csv", index=False)
        if 'rul' in parts:
            parts['rul'].to_csv(out / f"{ds_name}_rul.csv", index=False, header=False)
    print("Saved processed raw -> CSV files to", out)

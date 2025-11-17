# src/models/sequence_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class SequenceRULDataset(Dataset):
    """
    Creates sliding-window sequences from a labeled CMAPSS dataset CSV.
    Expects columns: unit, cycle, RUL (or RUL_clipped), and feature cols (sensor_*, op_setting*, and engineered features).
    Produces (sequence, target) where target is the RUL at the last timestep of the sequence (sequence-to-one).
    """
    def __init__(self, csv_path, seq_len=50, sensor_prefix="sensor_", feature_cols=None, target_col="RUL_clipped", normalize=True):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.seq_len = int(seq_len)
        self.target_col = target_col if target_col in self.df.columns else "RUL"
        if feature_cols is None:
            # pick sensible features
            self.feature_cols = [c for c in self.df.columns if (c.startswith("op_setting") or c.startswith(sensor_prefix) or "_mean" in c or "_std" in c or "_min" in c or "_max" in c)]
            # remove any accidental target columns
            self.feature_cols = [c for c in self.feature_cols if c not in (self.target_col, "RUL")]
        else:
            self.feature_cols = feature_cols
        # prepare units grouped arrays
        self.units = []
        for unit, g in self.df.groupby("unit"):
            g = g.sort_values("cycle").reset_index(drop=True)
            arr = g[self.feature_cols].to_numpy(dtype=np.float32)
            targ = g[self.target_col].to_numpy(dtype=np.float32)
            self.units.append({"features": arr, "target": targ, "unit": int(unit)})
        # build index map of (unit_idx, start_idx) for windows
        self.index_map = []
        for u_idx, u in enumerate(self.units):
            L = u["features"].shape[0]
            if L < 1:
                continue
            # allow shorter sequences at the start (pad later) OR skip — we will skip very short sequences here
            for end in range(0, L):
                start = max(0, end - self.seq_len + 1)
                # only keep windows where length == seq_len OR allow shorter: keep both options
                if (end - start + 1) == self.seq_len:
                    self.index_map.append((u_idx, start, end))
        # normalization (fit on whole dataset)
        self.normalize = normalize
        if self.normalize and len(self.units) > 0:
            all_feats = np.vstack([u["features"] for u in self.units])
            self.mean = all_feats.mean(axis=0, keepdims=False)
            self.std = all_feats.std(axis=0, keepdims=False) + 1e-8
        else:
            self.mean = np.zeros(len(self.feature_cols), dtype=np.float32)
            self.std = np.ones(len(self.feature_cols), dtype=np.float32)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        u_idx, start, end = self.index_map[idx]
        u = self.units[u_idx]
        seq = u["features"][start:end+1]  # shape (seq_len, n_features)
        # normalize
        seq = (seq - self.mean) / self.std
        target = u["target"][end]  # RUL at last timestep
        # if seq shorter than seq_len (should not happen in current map) pad at front with zeros
        if seq.shape[0] < self.seq_len:
            pad_len = self.seq_len - seq.shape[0]
            pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])
        return {"sequence": seq.astype(np.float32), "target": np.float32(target), "unit": u["unit"]}

    def get_feature_names(self):
        return list(self.feature_cols)

# src/api/utils.py

import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from typing import List
import re
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "auto-pm-lstm"
MODEL_STAGE = None
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

MODEL_BASE = Path("models/sequence/FD001")
FEATURES_PATH = MODEL_BASE / "features.joblib"
NORM_PATH = MODEL_BASE / "norm_stats.joblib"

SEQ_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fallback defaults
HIDDEN_DIM_DEFAULT = 64
NUM_LAYERS_DEFAULT = 2
DROPOUT_DEFAULT = 0.1
BIDIR_DEFAULT = False


# ============================================================
# Basic LSTM wrapper
# ============================================================

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM_DEFAULT, num_layers=NUM_LAYERS_DEFAULT,
                 dropout=DROPOUT_DEFAULT, bidirectional=BIDIR_DEFAULT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


# ============================================================
# LOAD FEATURES
# ============================================================

if FEATURES_PATH.exists():
    FEATURES = joblib.load(FEATURES_PATH)
else:
    if NORM_PATH.exists():
        tmp = joblib.load(NORM_PATH)
        FEATURES = tmp.get("feature_order", [])
    else:
        raise FileNotFoundError("features.joblib missing")

INPUT_DIM = len(FEATURES)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

if NORM_PATH.exists():
    stats = joblib.load(NORM_PATH)
    MEAN = np.array(stats.get("mean", []), dtype=np.float32)
    STD = np.array(stats.get("std", []), dtype=np.float32)
else:
    MEAN = None
    STD = None
    print("[WARN] No normalization stats found")


# ============================================================
# MODEL LOADER (MLflow → checkpoint inference → fallback)
# ============================================================

def infer_checkpoint_arch(sd):
    """Infer hidden_dim, input_dim, num_layers, bidirectional from checkpoint keys."""
    hidden = None
    inp = None
    layers = 1
    bidir = False

    for k, v in sd.items():
        if k.endswith("lstm.weight_ih_l0"):
            hidden = v.shape[0] // 4
            inp = v.shape[1]
        if k.endswith("lstm.weight_ih_l1"):
            layers = 2
        if "reverse" in k:
            bidir = True

    return hidden, inp, layers, bidir


def load_checkpoint_compatible(path):
    print(f"[INFO] Loading local checkpoint: {path}")
    state = torch.load(path, map_location="cpu")
    sd = state["model_state"] if isinstance(state, dict) and "model_state" in state else state

    # infer architecture
    hid, inp, layers, bidir = infer_checkpoint_arch(sd)
    print(f"[INFO] Inferred → input_dim={inp}, hidden_dim={hid}, layers={layers}, bidir={bidir}")

    # build model
    model = LSTMRegressor(inp, hid, layers, DROPOUT_DEFAULT, bidir)

    # patch head.* → fc.*
    mapped = {}
    for k, v in sd.items():
        if k.startswith("head.0."):
            new_k = k.replace("head.0.", "fc.0.")
        elif k.startswith("head.3."):
            new_k = k.replace("head.3.", "fc.2.")
        else:
            new_k = k
        mapped[new_k] = v

    # Replace fc with Sequential(Linear -> ReLU -> Linear)
    import torch.nn as nn
    mid = sd["head.0.weight"].shape[0] if "head.0.weight" in sd else hid // 2
    model.fc = nn.Sequential(
        nn.Linear(hid, mid),
        nn.ReLU(),
        nn.Linear(mid, 1),
    )

    # load weights tolerant
    res = model.load_state_dict(mapped, strict=False)
    print("[INFO] Loaded checkpoint with strict=False")
    if res.missing_keys:
        print("[WARN] missing:", res.missing_keys)
    if res.unexpected_keys:
        print("[WARN] unexpected:", res.unexpected_keys)

    return model.to(DEVICE)


def load_model():
    """
    Try MLflow → fallback to checkpoint but adapt architecture.
    """
    # 1) MLflow attempt
    try:
        print(f"[INFO] Loading from MLflow registry: {MODEL_NAME}")
        versions = client.get_latest_versions(MODEL_NAME)
        version = versions[0].version
        uri = f"models:/{MODEL_NAME}/{version}"
        print("[INFO] MLflow URI:", uri)
        model = mlflow.pytorch.load_model(uri)
        print("[INFO] Loaded from MLflow")
        return model.to(DEVICE)
    except Exception as e:
        print("[WARN] MLflow load failed:", e)

    # 2) checkpoint fallback (compatible)
    ckpt = Path("models/sequence/FD001/lstm_best.pth")
    if ckpt.exists():
        return load_checkpoint_compatible(ckpt)

    raise RuntimeError("No valid MLflow model or checkpoint found.")


# load model immediately
model = load_model()


# ============================================================
# NORMALIZATION + PREDICTION
# ============================================================

def normalize_sequence_by_saved_stats(seq):
    if MEAN is None:
        return seq
    return (seq - MEAN) / STD


def predict_rul(sequence: List[List[float]]) -> float:
    seq = np.array(sequence, dtype=np.float32)

    if seq.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} features, got {seq.shape[1]}")

    if seq.shape[0] < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - seq.shape[0], INPUT_DIM), np.float32)
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-SEQ_LEN:]

    seq = normalize_sequence_by_saved_stats(seq)
    t = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(t).cpu().numpy().item()

    return float(pred)

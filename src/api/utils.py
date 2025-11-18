# src/api/utils.py
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import List

from src.Models.lstm_gru import LSTMRegressor  # adjust import if needed

# ---- CONFIG ----
MODEL_BASE = Path("models/sequence/FD001")
FEATURES_PATH = MODEL_BASE / "features.joblib"
NORM_PATH = MODEL_BASE / "norm_stats.joblib"
CKPT_PATH = MODEL_BASE / "lstm_best.pth"
SEQ_LEN = 50  # must match training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- LOAD feature order ----
if FEATURES_PATH.exists():
    FEATURES = joblib.load(FEATURES_PATH)
else:
    # fallback: try to infer from norm_stats
    if NORM_PATH.exists():
        tmp = joblib.load(NORM_PATH)
        FEATURES = tmp.get("feature_order", [])
    else:
        raise FileNotFoundError("features.joblib not found in models/sequence/FD001")

INPUT_DIM = len(FEATURES)

# ---- LOAD norm stats ----
if NORM_PATH.exists():
    norm = joblib.load(NORM_PATH)
    MEAN = np.array(norm.get("mean", []), dtype=np.float32)
    STD = np.array(norm.get("std", []), dtype=np.float32)
    # if feature_order provided, ensure length matches
    if len(MEAN) != INPUT_DIM:
        # try fallback: if mean is dict with keys, or feature mismatch, raise a warning
        print("[WARN] norm_stats length does not match FEATURES length; continuing but be cautious.")
else:
    # fallback -> no normalization (not recommended)
    MEAN = None
    STD = None
    print("[WARN] norm_stats.joblib not found. Inference will not normalize inputs.")

# ---- BUILD model & load checkpoint ----
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False

model = LSTMRegressor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(DEVICE)

if CKPT_PATH.exists():
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
else:
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

model.eval()

# ---- Utilities ----
def normalize_sequence_by_saved_stats(seq: np.ndarray) -> np.ndarray:
    """seq shape (T, F)"""
    if MEAN is None or STD is None:
        return seq
    # ensure order: seq should be in the order of FEATURES; caller must pass correct ordering
    seq = (seq - MEAN) / STD
    return seq

def predict_rul(sequence: List[List[float]]) -> float:
    """
    sequence: Python list of lists (T x F) in the same feature order as FEATURES
    """
    seq = np.array(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("sequence must be 2D list T x F")
    if seq.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} features (order: {FEATURES[:5]}...), got {seq.shape[1]}")

    # pad/trim to SEQ_LEN
    if seq.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - seq.shape[0]
        pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])
    elif seq.shape[0] > SEQ_LEN:
        seq = seq[-SEQ_LEN:]

    seq = normalize_sequence_by_saved_stats(seq)
    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, T, F)
    with torch.no_grad():
        pred = model(seq_tensor).cpu().numpy().item()
    return float(pred)

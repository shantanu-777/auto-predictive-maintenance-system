# src/api/utils.py
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import List
from src.Models.lstm_gru import LSTMRegressor
# ----- Model Loading ----- -----

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("models/sequence/FD001")
FEATURES_PATH = MODEL_DIR / "features.joblib"
CHECKPOINT_PATH = MODEL_DIR / "lstm_best.pth"

# load feature names
FEATURES = joblib.load(FEATURES_PATH)
INPUT_DIM = len(FEATURES)
SEQ_LEN = 50  # must match training


# model hyperparams must match training
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False

# build model
model = LSTMRegressor(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL
).to(DEVICE)

# load checkpoint
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)

model.eval()

# ----- Normalization -----
# For now: compute mean/std from stored feature values (consistent with notebook)
# Ideally: save mean/std during training.
def compute_mean_std():
    # training-based normalization is best — for now, recompute from features.joblib path
    # You can update this later with exact mean/std saved from training
    return 0, 1   # placeholder (model was trained with per-batch normalization)

MEAN = None
STD = None

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize sequence using training statistics."""
    if MEAN is None:
        return seq
    return (seq - MEAN) / STD

# ----- Prediction -----
def predict_rul(sequence: List[List[float]]) -> float:
    """
    sequence: Python list of lists (T x F)
    """
    seq = np.array(sequence, dtype=np.float32)
    
    # ensure correct feature count
    if seq.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} features, got {seq.shape[1]}")

    # pad/trim to model sequence length
    SEQ_LEN = 50   # must match training
    if seq.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - seq.shape[0]
        pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])
    elif seq.shape[0] > SEQ_LEN:
        seq = seq[-SEQ_LEN:]

    seq = normalize_sequence(seq)

    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, T, F)

    with torch.no_grad():
        pred = model(seq_tensor).cpu().numpy().item()

    return float(pred)

# src/api/utils.py

import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from typing import List

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "auto-pm-lstm"      # MLflow registry name
MODEL_STAGE = "None"             # or "Production", "Staging"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Feature + normalization metadata (these remain local)
MODEL_BASE = Path("models/sequence/FD001")
FEATURES_PATH = MODEL_BASE / "features.joblib"
NORM_PATH = MODEL_BASE / "norm_stats.joblib"
SEQ_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default architecture hyperparameters (used for local fallback)
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
BIDIRECTIONAL = False

class LSTMRegressor(nn.Module):
    """Simple LSTM regressor used for local checkpoint fallback."""
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, bidirectional: bool = BIDIRECTIONAL):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden*dirs)
        last = out[:, -1, :]            # take last time step
        out = self.fc(last)             # (batch, 1)
        return out.squeeze(-1)          # (batch,)

# ============================================================
# ============================================================
# LOAD FEATURE ORDER
# ============================================================

if FEATURES_PATH.exists():
    FEATURES = joblib.load(FEATURES_PATH)
else:
    if NORM_PATH.exists():
        tmp = joblib.load(NORM_PATH)
        FEATURES = tmp.get("feature_order", [])
    else:
        raise FileNotFoundError("features.joblib not found.")

INPUT_DIM = len(FEATURES)


# ============================================================
# LOAD NORMALIZATION STATS
# ============================================================

if NORM_PATH.exists():
    norm = joblib.load(NORM_PATH)
    MEAN = np.array(norm.get("mean", []), dtype=np.float32)
    STD = np.array(norm.get("std", []), dtype=np.float32)

    if len(MEAN) != INPUT_DIM:
        print("[WARN] norm_stats length mismatch with FEATURES.")
else:
    MEAN = None
    STD = None
    print("[WARN] norm_stats.joblib missing — proceeding without normalization.")


# ============================================================
# LOAD MODEL FROM MLFLOW MODEL REGISTRY (with LOCAL FALLBACK)
# ============================================================

def load_model_from_registry_or_local():
    """
    Try to load model from MLflow Model Registry (models:/NAME/VERSION or stage).
    If that fails (artifacts missing or OSError), fallback to local checkpoint path models/sequence/FD001/lstm_best.pth.
    Returns a PyTorch model on DEVICE.
    """
    # First try MLflow registry
    try:
        print(f"[INFO] Attempting to load model from MLflow Registry: {MLFLOW_TRACKING_URI} (name: {MODEL_NAME}, stage: {MODEL_STAGE})")
        # try to use stage first, then any latest version
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                version = versions[0].version
            else:
                versions = client.get_latest_versions(MODEL_NAME)
                version = versions[0].version
            model_uri = f"models:/{MODEL_NAME}/{version}"
            print(f"[INFO] Loading model from MLflow URI: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri).to(DEVICE)
            model.eval()
            print(f"[INFO] Loaded model from MLflow: {model_uri}")
            return model
        except Exception as inner_e:
            print("[WARN] MLflow registry load failed (will attempt local fallback). Reason:", inner_e)
    except Exception as e:
        print("[WARN] MLflow interaction failed (will attempt local fallback). Reason:", e)

    # FALLBACK: local checkpoint
    local_ckpt = Path("models/sequence/FD001/lstm_best.pth")
    try:
        if local_ckpt.exists():
            print(f"[INFO] Loading model from local checkpoint: {local_ckpt}")
            # instantiate architecture (same as training)
            model_local = LSTMRegressor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(DEVICE)
            state = torch.load(local_ckpt, map_location=DEVICE)
            if "model_state" in state:
                model_local.load_state_dict(state["model_state"])
            else:
                model_local.load_state_dict(state)
            model_local.eval()
            print("[INFO] Loaded model from local checkpoint successfully.")
            return model_local
        else:
            raise FileNotFoundError(f"Local checkpoint not found at {local_ckpt}")
    except Exception as fallback_err:
        # Both registry and local failed — surface clear error
        print("[ERROR] Failed to load model from MLflow registry and local checkpoint.")
        raise RuntimeError(f"Model load failed. MLflow error: {inner_e if 'inner_e' in locals() else 'n/a'}. Local fallback error: {fallback_err}")

# Replace model = load_model_from_registry() with:
model = load_model_from_registry_or_local()


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_sequence_by_saved_stats(seq: np.ndarray) -> np.ndarray:
    """Normalize T x F sequence using saved stats."""
    if MEAN is None or STD is None:
        return seq
    return (seq - MEAN) / STD


# ============================================================
# PREDICT RUL
# ============================================================

def predict_rul(sequence: List[List[float]]) -> float:
    """
    sequence: Python list of lists (T x F)
    features MUST match the order of FEATURES loaded from joblib.
    """
    seq = np.array(sequence, dtype=np.float32)

    if seq.ndim != 2:
        raise ValueError("Input sequence must be 2D (T x F).")

    if seq.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} features, got {seq.shape[1]}.")

    # pad or trim
    if seq.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - seq.shape[0]
        seq = np.vstack([np.zeros((pad_len, INPUT_DIM), dtype=np.float32), seq])
    else:
        seq = seq[-SEQ_LEN:]

    # normalize
    seq = normalize_sequence_by_saved_stats(seq)

    # convert to tensor
    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # (1, T, F)

    # inference
    with torch.no_grad():
        pred = model(seq_tensor).cpu().numpy().item()

    return float(pred)

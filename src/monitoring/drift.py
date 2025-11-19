# src/monitoring/drift.py
import numpy as np

def compute_input_stats(sequence_np):
    """
    sequence_np: numpy array shape (T, F)
    Returns mean and std across all values (scalar) for simple drift tracking.
    """
    seq = np.asarray(sequence_np, dtype=np.float32)
    return float(np.mean(seq)), float(np.std(seq))

def check_output_drift(pred, history_vals, z_thresh=3.0):
    """
    Very simple z-score check: given history_vals (list or array of past preds),
    return True if pred is > z_thresh away from history mean.
    """
    hist = np.asarray(history_vals, dtype=np.float32)
    if hist.size < 10:
        return False
    mu = hist.mean()
    sigma = hist.std()
    if sigma < 1e-6:
        return False
    z = abs((pred - mu) / sigma)
    return z > z_thresh

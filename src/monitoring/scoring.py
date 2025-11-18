# src/monitoring/scoring.py
from typing import Dict, Any
import math

def compute_health_score(predicted_rul: float,
                         max_rul: float = 130.0,
                         total_anomalies: int = 0,
                         anomaly_capacity: int = 10,
                         alpha: float = 0.7) -> Dict[str, Any]:
    """
    Compute health score between 0 and 100.
    - predicted_rul: RUL predicted by model (cycles)
    - max_rul: normalization maximum for RUL (same as RUL clipping)
    - total_anomalies: total anomalies detected for the engine
    - anomaly_capacity: anomalies count at which penalty saturates
    - alpha: weight of anomaly penalty (0=no penalty, 1=full)
    Returns dict with 'health_score' and components.
    """
    # clamp
    pr = max(0.0, min(predicted_rul, max_rul))
    rul_score = pr / float(max_rul)  # 0..1

    # anomaly penalty normalized 0..1
    anom_norm = min(float(total_anomalies) / float(max(1, anomaly_capacity)), 1.0)
    # final score
    raw = 100.0 * rul_score * (1.0 - alpha * anom_norm)
    score = max(0.0, min(100.0, raw))

    components = {
        "predicted_rul": float(predicted_rul),
        "rul_score": float(rul_score),
        "total_anomalies": int(total_anomalies),
        "anom_norm": float(anom_norm),
        "alpha": float(alpha),
        "raw_score": float(raw)
    }
    return {"health_score": round(score, 3), "components": components}

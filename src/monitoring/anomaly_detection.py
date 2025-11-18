# src/monitoring/anomaly_detection.py
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

def compute_rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling z-score (centered) for a pandas Series."""
    roll_mean = series.rolling(window=window, min_periods=1, center=False).mean()
    roll_std = series.rolling(window=window, min_periods=1, center=False).std().replace(0, 1e-8)
    return (series - roll_mean) / roll_std

def compute_rolling_mad_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Robust z-score using rolling MAD (median absolute deviation)."""
    roll_med = series.rolling(window=window, min_periods=1).median()
    mad = (series - roll_med).abs().rolling(window=window, min_periods=1).median().replace(0, 1e-8)
    # constant 1.4826 scales MAD to be comparable to std under normality
    z = 0.6745 * (series - roll_med) / mad
    return z

def detect_spikes(series: pd.Series, diff_thresh: float = None) -> pd.Series:
    """Detect spikes by looking at absolute first differences larger than threshold.
    If diff_thresh is None, use 3x std of diffs.
    Returns boolean Series."""
    diffs = series.diff().abs().fillna(0.0)
    if diff_thresh is None:
        diff_thresh = max(1e-8, 3 * diffs.rolling(window=30, min_periods=1).std().median())
    return diffs > diff_thresh

def sensor_anomalies_for_unit(df_unit: pd.DataFrame, sensor_cols: List[str],
                              window: int = 30, z_thresh: float = 3.0, mad_thresh: float = 3.0,
                              detect_spike: bool = True) -> pd.DataFrame:
    """For a single unit (engine) DataFrame, compute anomaly flags for each sensor.
    Returns a copy of df_unit with extra columns: {sensor}_z, {sensor}_mad_z, {sensor}_anomaly (bool).
    """
    df = df_unit.sort_values("cycle").reset_index(drop=True).copy()
    for s in sensor_cols:
        z = compute_rolling_zscore(df[s], window=window)
        madz = compute_rolling_mad_zscore(df[s], window=window)
        spike = detect_spikes(df[s]) if detect_spike else pd.Series(False, index=df.index)
        # anomaly if any method flags
        anomaly_flag = (z.abs() > z_thresh) | (madz.abs() > mad_thresh) | spike
        df[f"{s}_z"] = z
        df[f"{s}_madz"] = madz
        df[f"{s}_spike"] = spike
        df[f"{s}_anomaly"] = anomaly_flag
    return df

def aggregate_anomaly_summary(df_flagged: pd.DataFrame, sensor_cols: List[str]) -> Dict[str, Any]:
    """Summarize anomalies for a unit: counts per sensor, total anomalies, last anomaly cycle per sensor."""
    summary = {"total_anomalies": 0, "per_sensor": {}, "recent_anomalies": {}}
    total = 0
    for s in sensor_cols:
        col = f"{s}_anomaly"
        if col in df_flagged.columns:
            count = int(df_flagged[col].sum())
            total += count
            summary["per_sensor"][s] = count
            # last cycle where anomaly True
            df_idx = df_flagged[df_flagged[col]]
            if not df_idx.empty:
                summary["recent_anomalies"][s] = int(df_idx.iloc[-1]["cycle"])
            else:
                summary["recent_anomalies"][s] = None
    summary["total_anomalies"] = int(total)
    return summary

def detect_anomalies_batch(df: pd.DataFrame, sensor_cols: List[str],
                           window: int = 30, z_thresh: float = 3.0, mad_thresh: float = 3.0) -> Dict[int, Dict[str, Any]]:
    """Run anomaly detection for each unit in df. Returns mapping unit -> summary dict and flagged DataFrame per unit."""
    units = {}
    for unit, g in df.groupby("unit"):
        flagged = sensor_anomalies_for_unit(g, sensor_cols=sensor_cols, window=window, z_thresh=z_thresh, mad_thresh=mad_thresh)
        summary = aggregate_anomaly_summary(flagged, sensor_cols)
        units[int(unit)] = {"flagged_df": flagged, "summary": summary}
    return units

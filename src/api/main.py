# src/api/main.py
import datetime
import time
import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import io
import joblib
import numpy as np

from .models import SequenceInput, PredictionResponse
from .utils import predict_rul, INPUT_DIM, SEQ_LEN  # ensure utils exposes these names

from src.monitoring.anomaly_detection import detect_anomalies_batch
from src.monitoring.scoring import compute_health_score
from src.monitoring.reporting import generate_score_pdf
from src.monitoring.inference_logger import log_inference, fetch_recent
from src.monitoring.drift import compute_input_stats

from mlflow.tracking import MlflowClient
import mlflow

# -------------------------
# App + static reports dir
# -------------------------
app = FastAPI(title="Automotive Predictive Maintenance API - Extended", version="1.1")

analysis_dir = Path("analysis")
analysis_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(analysis_dir)), name="reports")

# -------------------------
# MLflow + model config
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "auto-pm-lstm")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "None")  # e.g., "Production" or "Staging" or "None"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# -------------------------
# Root
# -------------------------
@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_rul_api(data: SequenceInput):
    """
    Accepts:
      data.sequence : list[list[float]] (T x F), feature order must match FEATURES used in training.
      optional: data.unit (if provided by client)
    Returns:
      predicted_rul and used_sequence_length
    """
    # run inference and measure latency
    t0 = time.time()
    pred = predict_rul(data.sequence)
    latency_ms = (time.time() - t0) * 1000.0

    # compute simple input stats for drift logging
    try:
        seq_np = np.array(data.sequence, dtype=np.float32)
        input_mean, input_std = compute_input_stats(seq_np)
    except Exception:
        input_mean, input_std = None, None

    # resolve model version (best-effort)
    model_version = "unknown"
    try:
        versions = mlflow_client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            model_version = str(versions[0].version)
        else:
            versions = mlflow_client.get_latest_versions(MODEL_NAME)
            if versions:
                model_version = str(versions[0].version)
    except Exception:
        # ignore errors retrieving from MLflow
        pass

    # unit if provided
    unit = getattr(data, "unit", None) if hasattr(data, "unit") else None

    # log inference
    try:
        log_inference(
            unit=unit,
            dataset=None,
            model_name=MODEL_NAME,
            model_version=model_version,
            latency_ms=latency_ms,
            prediction=float(pred),
            input_meta={"mean": input_mean, "std": input_std},
            notes=None,
        )
    except Exception as e:
        print("[WARN] inference log failed:", e)

    return PredictionResponse(predicted_rul=float(pred), used_sequence_length=len(data.sequence))


# -------------------------
# Model version endpoint
# -------------------------
@app.get("/model/version")
def model_version():
    try:
        versions = mlflow_client.get_latest_versions(MODEL_NAME)
        if not versions:
            return {"model_name": MODEL_NAME, "version": None}
        latest = versions[0]
        return {"model_name": MODEL_NAME, "version": latest.version, "stage": latest.current_stage}
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Recent inference logs endpoint
# -------------------------
@app.get("/inference_logs")
def recent_inferences(limit: int = 100):
    try:
        rows = fetch_recent(limit=limit)
        keys = ["id", "ts_iso", "unit", "dataset", "model_name", "model_version", "latency_ms", "prediction"]
        result = [dict(zip(keys, r)) for r in rows]
        return {"count": len(result), "rows": result}
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Score endpoint (upload CSV -> anomalies, RUL, health score, PDF)
# -------------------------
@app.post("/score")
async def score_upload(
    file: UploadFile = File(...),
    dataset: str = Form(None),
    z_window: int = Form(30),
    z_thresh: float = Form(3.0),
    mad_thresh: float = Form(3.0),
    anomaly_capacity: int = Form(10),
    alpha: float = Form(0.7),
) -> JSONResponse:
    """
    Accepts a CSV file upload (containing unit, cycle, sensor_* columns).
    Runs per-unit prediction (last cycle) + anomaly detection + health scoring.
    Returns JSON with per-unit summaries and paths to saved artifacts (CSV + PDF).
    """
    # read CSV
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to parse CSV: {e}")

    # basic checks
    if "unit" not in df.columns or "cycle" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'unit' and 'cycle' columns.")

    # determine sensor columns
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        # allow falling back to any numeric columns except unit/cycle/RUL
        sensor_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("unit", "cycle", "RUL", "RUL_clipped")
        ]

    # run anomaly detection
    anomalies = detect_anomalies_batch(
        df, sensor_cols=sensor_cols, window=z_window, z_thresh=z_thresh, mad_thresh=mad_thresh
    )

    # prepare analysis directory for this run
    analysis_root = Path("analysis")
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = analysis_root / f"score_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    anomalies_map_for_report: Dict[int, Dict[str, Any]] = {}

    # iterate units
    for unit, info in anomalies.items():
        flagged_df = info.get("flagged_df")
        summary = info.get("summary", {})  # total_anomalies, per_sensor, recent_anomalies

        if flagged_df is None or flagged_df.empty:
            # skip
            continue

        # select features
        feat_cols = [
            c
            for c in flagged_df.columns
            if (c.startswith("op_setting") or c.startswith("sensor_") or "_mean" in c or "_std" in c or "_min" in c or "_max" in c)
        ]
        if len(feat_cols) == 0:
            feat_cols = [
                c
                for c in flagged_df.select_dtypes(include=[np.number]).columns
                if c not in ("unit", "cycle", "RUL", "RUL_clipped")
            ]

        seq = flagged_df.sort_values("cycle")[feat_cols].to_numpy(dtype=float)

        # ensure correct input dim: try to reorder features if mismatch
        if seq.shape[1] != INPUT_DIM:
            try:
                feat_order = joblib.load("models/sequence/FD001/features.joblib")
                seq = flagged_df.sort_values("cycle")[feat_order].to_numpy(dtype=float)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Feature dimension mismatch for unit {unit}: expected {INPUT_DIM}, got {seq.shape[1]}",
                )

        # pad/trim to SEQ_LEN
        if seq.shape[0] < SEQ_LEN:
            pad_len = SEQ_LEN - seq.shape[0]
            pad = np.zeros((pad_len, seq.shape[1]), dtype=float)
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-SEQ_LEN:]

        # predict
        try:
            pred = predict_rul(seq.tolist())
        except Exception:
            pred = None

        total_anoms = int(summary.get("total_anomalies", 0))
        scoring = compute_health_score(
            predicted_rul=float(pred) if pred is not None else 0.0,
            max_rul=130.0,
            total_anomalies=total_anoms,
            anomaly_capacity=int(anomaly_capacity),
            alpha=float(alpha),
        )

        results.append(
            {
                "unit": int(unit),
                "predicted_rul": float(pred) if pred is not None else None,
                "total_anomalies": total_anoms,
                "per_sensor_anomalies": summary.get("per_sensor", {}),
                "recent_anomalies": summary.get("recent_anomalies", {}),
                "health_score": scoring.get("health_score"),
                "health_components": scoring.get("components"),
            }
        )

        # save flagged csv for debugging / archival
        try:
            flagged_df.to_csv(run_dir / f"unit_{unit}_flagged.csv", index=False)
        except Exception:
            pass

        anomalies_map_for_report[int(unit)] = {"flagged_df": flagged_df, "summary": summary}

    # sort by health ascending (worst first), placing None at the end
    results = sorted(results, key=lambda x: (x["health_score"] is None, x["health_score"]))

    # save scored CSV
    df_results = pd.DataFrame(results)
    csv_path = run_dir / f"{(dataset or 'dataset')}_scored_{run_ts}.csv"
    try:
        df_results.to_csv(csv_path, index=False)
    except Exception:
        pass

    # generate PDF report
    pdf_path = None
    report_url = None
    try:
        pdf_path = generate_score_pdf({"units": results}, anomalies_map_for_report, run_dir, dataset_tag=(dataset or "dataset"), top_k=5)
        report_url = f"/reports/{pdf_path.name}" if pdf_path is not None else None
    except Exception:
        pdf_path = None
        report_url = None

    return JSONResponse({"dataset": dataset, "units": results, "csv_path": str(csv_path), "report_url": report_url})

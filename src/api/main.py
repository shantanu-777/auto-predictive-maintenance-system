# src/api/main.py
import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import pandas as pd
import io
import joblib
import numpy as np
from .models import SequenceInput, PredictionResponse
from .utils import predict_rul, INPUT_DIM, SEQ_LEN  # ensure utils exposes these names
from src.monitoring.anomaly_detection import detect_anomalies_batch
from src.monitoring.scoring import compute_health_score
import os
from pathlib import Path
from src.monitoring.reporting import generate_score_pdf

app = FastAPI(title="Automotive Predictive Maintenance API - Extended", version="1.1")
from fastapi.staticfiles import StaticFiles
app.mount("/reports", StaticFiles(directory="analysis"), name="reports")


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_rul_api(data: SequenceInput):
    pred = predict_rul(data.sequence)
    return PredictionResponse(predicted_rul=pred, used_sequence_length=len(data.sequence))


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
    Returns JSON with per-unit summaries.
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

    # prepare analysis directory
    analysis_root = Path("analysis")
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = analysis_root / f"score_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    anomalies_map_for_report: Dict[int, Dict[str, Any]] = {}

    for unit, info in anomalies.items():
        flagged_df = info.get("flagged_df")
        summary = info.get("summary", {})  # total_anomalies, per_sensor, recent_anomalies

        if flagged_df is None or flagged_df.empty:
            # nothing to analyze for this unit
            continue

        # build sequence for last cycles using same logic as dashboard/utils
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

    # sort by health ascending (worst first)
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

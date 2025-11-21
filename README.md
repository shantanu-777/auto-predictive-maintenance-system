Auto Predictive Maintenance System

A complete end-to-end predictive maintenance pipeline for sequence-based RUL (Remaining Useful Life) prediction, anomaly detection, engine health scoring, monitoring, and reporting — built with FastAPI, PyTorch, MLflow, and Streamlit.


<img width="1912" height="950" alt="Screenshot 2025-11-17 115322" src="https://github.com/user-attachments/assets/1d225605-2f93-4bfb-84d1-1f16cb14a2b6" />

<img width="1907" height="930" alt="Screenshot 2025-11-17 115355" src="https://github.com/user-attachments/assets/76c4dd9a-1559-4b8d-a424-d383884eb32b" />

This system demonstrates a full ML engineering workflow:
data → features → sequence model → registry → API → dashboard → scoring → monitoring → PDF reports.

<img width="1536" height="1024" alt="ChatGPT Image Nov 21, 2025, 10_23_03 AM" src="https://github.com/user-attachments/assets/da21042d-5210-4d7c-8a1a-58ed24356010" />


🚀 Key Features

Complete MLOps workflow using MLflow (tracking + model registry)

LSTM-based sequence model trained on CMAPSS-style datasets

FastAPI inference server with robust fallback (MLflow → local checkpoint)

Automated anomaly detection (Z-score + MAD)

Engine health scoring system

Streamlit dashboard

Real-time predictions

Anomaly visualization

Health scoring

PDF report downloads

Monitoring & inference logs

PDF report generator for scored datasets

Production-style inference logging (SQLite)



📁 Repository Structure
📦 auto-predictive-maintenance-system
│
├── data/
│   ├── raw/                # raw datasets (ignored)
│   └── processed/          # feature-engineered data
│
├── models/
│   ├── sequence/FD001/     # model checkpoints, features, norm stats
│
├── mlruns/                 # MLflow experiment + model registry (ignored)
│
├── src/
│   ├── api/                # FastAPI inference service
│   ├── data_processing/    # preprocessing & feature engineering
│   ├── Models/             # LSTM, datasets, XGBoost, training scripts
│   └── monitoring/         # drift, anomaly, logging, report generator
│
├── dashboards/
│   └── app.py              # Streamlit dashboard
│
├── analysis/               # Generated reports (PDF + CSV)
├── notebooks/              # Training & research notebooks
│
├── requirements.txt
└── README.md


⚙️ Setup & Installation

Run all commands from the repository root.

1. Create & activate virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install dependencies
pip install -r requirements.txt

📊 Data Pipeline

The complete preprocessing pipeline includes:

Outlier handling

Normalization

Rolling-window statistics

Feature engineering (mean/std/min/max)

Sequence creation for LSTM

Dataset splitting for FD001–FD004

Scripts (located in src/data_processing/):

python src/data_processing/loader.py
python src/data_processing/preprocessing.py
python src/data_processing/feature_engineering.py


Processed datasets stored in:

data/processed/for_model/

🤖 Model Training
1. Train XGBoost baseline
python src/models/xgboost_train.py --proc_dir data/processed/for_model --datasets FD001

2. Train LSTM model
python src/models/train_sequence.py --proc_dir data/processed/for_model --dataset FD001 --seq_len 50 --batch_size 64 --epochs 40 --use_cuda


Resulting artifacts stored in:

models/sequence/FD001/

🧾 Register the Model (MLflow)

Set env variables:

$env:MLFLOW_TRACKING_URI = "file:./mlruns"
$env:PYTHONPATH = "$PWD\src"


Register:

python src/monitoring/register_model.py `
  --model-name auto-pm-lstm `
  --model-path models/sequence/FD001/lstm_best.pth `
  --features-path models/sequence/FD001/features.joblib `
  --seq-len 50

⚡ Run FastAPI Inference Server
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
$env:PYTHONPATH = "$PWD\src"
python -m uvicorn src.api.main:app --reload --port 8000


API will load:

MLflow model (if available)

OR fallback to local checkpoint (lstm_best.pth)

Main endpoints:
Endpoint	Function
/predict	Predict RUL for a single sequence
/score	Upload dataset → anomalies + RUL + health scores + PDF
/inference_logs	Retrieve inference history
/model/version	Get model registry version
📈 Run Streamlit Dashboard
streamlit run dashboards/app.py

Dashboard capabilities:

Explore test sets

Predict RUL for any unit

Upload datasets for full scoring

View anomaly charts

Download CSV + PDF reports

Monitor inference logs live

📝 Health Scoring System

Each engine receives a health score combining:

Predicted RUL

Total anomalies detected

Anomaly capacity threshold

Penalty weight (α)

Formula:

health = (RUL_score * (1 - α)) + (anomaly_penalty * α)

📄 PDF Reports

When scoring datasets, a PDF is generated containing:

Engine ranking

Anomaly timelines

Per-sensor anomaly stats

RUL predictions

Overall health distribution

PDF is stored in:

analysis/score_<timestamp>/


and served via:

/reports/<filename>.pdf

📡 Monitoring

The system logs every inference to:

mlruns/inference_logs.db


Logged fields:

id, timestamp, unit, dataset, model_name,
model_version, latency_ms, prediction, input_meta


The dashboard retrieves logs from:

GET /inference_logs

📊 MLflow Tracking

Launch MLflow UI:

mlflow ui --backend-store-uri file:./mlruns --port 5000


Access:

http://127.0.0.1:5000


Includes:

Experiment metrics

Model registry

Artifacts

Training runs

📦 Packaging
requirements.txt includes:

FastAPI

Uvicorn

PyTorch

XGBoost

NumPy / Pandas

Streamlit

Plotly

MLflow

Joblib

ReportLab

Requests

(Optional) add:

docker-compose.yml
Dockerfile
scripts/run_local.ps1

📚 Notebooks

Suggested notebooks:

notebooks/
 ├── 01_data_processing.ipynb
 ├── 02_model_training_xgboost.ipynb
 ├── 03_model_training_lstm.ipynb
 └── 04_inference_demo.ipynb

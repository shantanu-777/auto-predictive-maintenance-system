# dashboards/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import io
from pathlib import Path
from typing import Tuple, List

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Automotive Health Monitoring", layout="wide", initial_sidebar_state="expanded")

ROOT = Path.cwd()
PROC_DIR = ROOT / "data" / "processed" / "for_model"
API_URL = "http://127.0.0.1:8000/predict"   # change if your API is hosted elsewhere
API_BASE = API_URL.replace("/predict", "")
SEQ_LEN = 50  # must match model training

DARK_BG = "#0f1720"
CARD_BG = "#0b1220"
ACCENT = "#0ea5a0"  # teal accent

# ----------------------
# Helpers
# ----------------------
@st.cache_data
def list_datasets(proc_dir: Path) -> List[str]:
    files = list(proc_dir.glob("*_train_labeled.csv"))
    datasets = [p.stem.split("_")[0] for p in files]
    return sorted(datasets)

@st.cache_data
def load_dataset_csv(dataset: str, kind: str = "test") -> pd.DataFrame:
    path = PROC_DIR / f"{dataset}_{kind}_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)

def get_units(df: pd.DataFrame) -> List[int]:
    return sorted(df["unit"].unique())

def build_sequence_for_unit(df_unit: pd.DataFrame, last_n: int = SEQ_LEN) -> Tuple[np.ndarray, List[str]]:
    df_unit = df_unit.sort_values("cycle").reset_index(drop=True)
    feature_cols = [c for c in df_unit.columns if (c.startswith("op_setting") or c.startswith("sensor_") or "_mean" in c or "_std" in c or "_min" in c or "_max" in c)]
    seq = df_unit[feature_cols].to_numpy(dtype=np.float32)
    if seq.shape[0] < last_n:
        pad_len = last_n - seq.shape[0]
        pad = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-last_n:]
    return seq, feature_cols

def call_predict_api(sequence: np.ndarray) -> float:
    payload = {"sequence": sequence.tolist()}
    resp = requests.post(API_URL, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return float(data["predicted_rul"])

def compute_anomaly_flags(df_unit: pd.DataFrame, sensor_cols: List[str], z_thresh: float = 3.0) -> pd.DataFrame:
    # compute z-score per sensor using rolling mean/std and flag
    df = df_unit.copy().sort_values("cycle").reset_index(drop=True)
    for s in sensor_cols:
        df[f"{s}_z"] = (df[s] - df[s].rolling(30, min_periods=1).mean()) / (df[s].rolling(30, min_periods=1).std().replace(0, 1))
        df[f"{s}_anomaly"] = df[f"{s}_z"].abs() > z_thresh
    return df

# ----------------------
# Sidebar: Controls
# ----------------------
st.sidebar.markdown("<h2 style='color: white;'>Automotive Health Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Top-level page selector
page = st.sidebar.selectbox("Page", ["Main", "Monitoring"])

st.sidebar.markdown("---")

# Only show these dataset/model controls on Main page
if page == "Main":
    try:
        datasets = list_datasets(PROC_DIR)
    except Exception:
        datasets = []
    dataset = st.sidebar.selectbox("Select dataset", datasets, index=0 if datasets else -1)
    mode = st.sidebar.radio("Mode", ["Explore test set", "Upload CSV (simulate)"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model API**")
    st.sidebar.write(f"`{API_URL}`")
    if st.sidebar.button("Change API URL"):
        new = st.sidebar.text_input("New API URL", value=API_URL)
        if new:
            st.experimental_set_query_params(api=new)

# ----------------------
# Monitoring page
# ----------------------
if page == "Monitoring":
    st.title("System Monitoring")
    st.markdown("This page shows recent inference logs (latency, model version, prediction) collected by the API.")

    @st.cache_data(ttl=30)
    def fetch_logs(limit: int = 500):
        try:
            resp = requests.get(f"{API_BASE}/inference_logs", params={"limit": limit}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "count": 0, "rows": []}

    with st.spinner("Loading logs..."):
        logs_json = fetch_logs(limit=500)

    if "error" in logs_json:
        st.error(f"Could not fetch logs: {logs_json['error']}")
    else:
        rows = logs_json.get("rows", [])
        if not rows:
            st.info("No inference logs available yet (make some predictions to populate).")
        else:
            df_logs = pd.DataFrame(rows)
            # Normalize column types
            if "latency_ms" in df_logs.columns:
                df_logs["latency_ms"] = pd.to_numeric(df_logs["latency_ms"], errors="coerce")
            if "prediction" in df_logs.columns:
                df_logs["prediction"] = pd.to_numeric(df_logs["prediction"], errors="coerce")

            st.subheader("Recent Inference Logs")
            st.dataframe(df_logs, use_container_width=True)

            st.subheader("Latency (ms) — recent requests")
            st.line_chart(df_logs["latency_ms"].fillna(0))

            st.subheader("Prediction distribution (recent)")
            fig_pred = px.histogram(df_logs, x="prediction", nbins=30, title="Prediction distribution", template="plotly_dark")
            st.plotly_chart(fig_pred, use_container_width=True)

            st.subheader("Latency distribution")
            fig_lat = px.histogram(df_logs, x="latency_ms", nbins=50, title="Latency distribution (ms)", template="plotly_dark")
            st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")
    st.markdown("Tip: run predictions from the Main page (or via the API) to populate logs.")
    st.stop()  # stop here so Main UI isn't rendered below

# ----------------------
# Main layout (unchanged UI)
# ----------------------
st.markdown(f"<div style='background:{CARD_BG}; padding: 12px; border-radius: 8px'>"
            f"<h1 style='color: {ACCENT}; margin: 0'>Automotive Health Monitoring — {dataset}</h1>"
            f"<p style='color: #9AA5B1; margin: 0'>Sequence model RUL prediction · Streamlined demo</p>"
            f"</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Engine selection & Sensor view")
    if mode == "Explore test set":
        df_test = load_dataset_csv(dataset, "test")
        units = get_units(df_test)
        unit = st.selectbox("Select unit (engine)", units, index=0)
        df_unit = df_test[df_test["unit"] == unit].sort_values("cycle")
        seq, feature_cols = build_sequence_for_unit(df_unit, last_n=SEQ_LEN)
    else:
        uploaded = st.file_uploader("Upload CSV (must contain unit, cycle, sensor_* columns)", type=["csv"])
        if uploaded:
            df_uploaded = pd.read_csv(uploaded)
            uploaded_unit_ids = sorted(df_uploaded["unit"].unique())
            unit = uploaded_unit_ids[0]
            df_unit = df_uploaded[df_uploaded["unit"] == unit].sort_values("cycle")
            seq, feature_cols = build_sequence_for_unit(df_unit, last_n=SEQ_LEN)
        else:
            st.info("Upload a CSV to simulate telematics. Using a sample test unit for preview.")
            df_test = load_dataset_csv(dataset, "test")
            units = get_units(df_test)
            unit = units[0]
            df_unit = df_test[df_test["unit"] == unit].sort_values("cycle")
            seq, feature_cols = build_sequence_for_unit(df_unit, last_n=SEQ_LEN)

    # Sensor multi-plot - show first 6 sensors for clarity
    sensor_cols = [c for c in feature_cols if c.startswith("sensor_")]
    sensors_to_plot = sensor_cols[:6] if len(sensor_cols) > 6 else sensor_cols
    fig = go.Figure(layout=go.Layout(template="plotly_dark"))
    for s in sensors_to_plot:
        fig.add_trace(go.Scatter(x=df_unit["cycle"], y=df_unit[s], mode="lines+markers", name=s, hovertemplate="cycle %{x}<br>%{y:.3f}"))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # show RUL trend
    st.subheader("RUL trend (true)")
    fig2 = px.line(df_unit, x="cycle", y="RUL_clipped" if "RUL_clipped" in df_unit.columns else "RUL", title=f"Unit {unit} RUL over time", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Prediction panel")
    st.markdown("**Sequence length used:** " + str(SEQ_LEN))
    st.markdown(f"**Selected unit:** {unit}")
    btn = st.button("Predict RUL (send to API)")
    if btn:
        try:
            pred = call_predict_api(seq)
            st.metric(label="Predicted RUL (cycles)", value=f"{pred:.1f}")
        except Exception as e:
            st.error(f"API call failed: {e}")

    st.markdown("### Quick stats")
    st.write(df_unit.describe().T[["mean","std","min","max"]].head(8))

    # Anomaly detection
    st.subheader("Anomaly indicators (z-score)")
    z_thresh = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
    df_anom = compute_anomaly_flags(df_unit, sensor_cols, z_thresh=z_thresh)
    # show a small table of any recent anomalies
    anomalies = []
    for s in sensor_cols:
        if f"{s}_anomaly" in df_anom.columns:
            if df_anom[f"{s}_anomaly"].any():
                idx = df_anom[df_anom[f"{s}_anomaly"]].index[-1]
                anomalies.append((s, int(df_anom.loc[idx, "cycle"])))
    if anomalies:
        st.warning("Recent anomalies detected for sensors:")
        for s, cyc in anomalies:
            st.write(f"- **{s}** at cycle {cyc}")
    else:
        st.success("No anomalies detected (recent).")

# ----------------------
# Bottom section: unit-level prediction table and export
# ----------------------
st.markdown("---")
st.subheader("Batch predict multiple units (test set)")

if st.button("Run batch predictions for test set (last cycle per unit)"):
    df_test = load_dataset_csv(dataset, "test")
    units = get_units(df_test)
    results = []
    with st.spinner("Predicting..."):
        for u in units:
            df_u = df_test[df_test["unit"] == u].sort_values("cycle")
            seq_u, _ = build_sequence_for_unit(df_u, last_n=SEQ_LEN)
            try:
                pred = call_predict_api(seq_u)
            except Exception as e:
                pred = np.nan
            true_rul = float(df_u.iloc[-1]["RUL_clipped"] if "RUL_clipped" in df_u.columns else df_u.iloc[-1]["RUL"])
            results.append({"unit": int(u), "true_rul": true_rul, "pred_rul": pred})
    df_res = pd.DataFrame(results).sort_values("unit")
    st.dataframe(df_res)
    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv, file_name=f"{dataset}_batch_preds.csv", mime="text/csv")

# ----------------------
# Score Upload & Engine Health (BMW Industrial Pro)
# Insert this AFTER the batch predict section in dashboards/app.py
# ----------------------
st.markdown("---")
st.subheader("Upload & Score dataset — End-to-end health report")

with st.form("upload_score_form"):
    uploaded_file = st.file_uploader("Upload dataset CSV (unit, cycle, sensor_* columns)", type=["csv"], help="Upload a CSV to compute per-engine RUL, anomalies and health score")
    dataset_name = st.text_input("Dataset tag (optional)", value=dataset)
    z_window = st.number_input("Z-window (rolling)", value=30, min_value=5, max_value=400, step=1)
    z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
    mad_thresh = st.slider("MAD z threshold", 2.0, 5.0, 3.0, 0.1)
    anomaly_capacity = st.number_input("Anomaly capacity (saturation)", value=10, min_value=1, step=1)
    alpha = st.slider("Anomaly penalty weight (alpha)", 0.0, 1.0, 0.7, 0.05)
    submit = st.form_submit_button("Upload & Score")

score_results = None
if submit:
    if not uploaded_file:
        st.error("Please upload a CSV file first.")
    else:
        st.info("Uploading file and requesting score from API...")
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {
                "dataset": dataset_name or dataset,
                "z_window": int(z_window),
                "z_thresh": float(z_thresh),
                "mad_thresh": float(mad_thresh),
                "anomaly_capacity": int(anomaly_capacity),
                "alpha": float(alpha)
            }
            resp = requests.post(API_URL.replace("/predict", "/score"), files=files, data=data, timeout=180)
            resp.raise_for_status()
            score_results = resp.json()
            st.success("Scoring complete.")
        except Exception as e:
            st.exception(f"API call failed: {e}")
        # after score_results assigned
        report_url = score_results.get("report_url")
        csv_path = score_results.get("csv_path")

        if report_url:
            full_url = API_URL.replace("/predict","") + report_url  # API_URL is e.g. http://127.0.0.1:8000/predict
            # show download link/button
            st.markdown(f"[Download PDF report]({full_url})")
    

# Display results area
if score_results:
    units = score_results.get("units", [])
    if not units:
        st.warning("No units returned by the API.")
    else:
        df_units = pd.DataFrame(units)
        # basic normalization for display
        df_units["health_score"] = df_units["health_score"].astype(float)
        df_units["predicted_rul"] = df_units["predicted_rul"].astype(float)
        df_units["total_anomalies"] = df_units["total_anomalies"].astype(int)

        # KPIs
        worst = df_units.nsmallest(1, "health_score").iloc[0]
        avg_health = df_units["health_score"].mean()
        total_anoms = df_units["total_anomalies"].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric(label="Worst engine (health)", value=f"Unit {int(worst['unit'])} — {worst['health_score']}", delta=f"{int(worst['total_anomalies'])} anomalies")
        k2.metric(label="Avg health score (units)", value=f"{avg_health:.2f}")
        k3.metric(label="Total anomalies (all units)", value=f"{int(total_anoms)}")

        st.markdown("#### Engine ranking (worst → best)")
        # Expand per-sensor nested dicts into readable text
        def per_sensor_to_str(d):
            if not isinstance(d, dict):
                return ""
            return ", ".join([f"{k}:{v}" for k, v in d.items()])

        # convert nested dict columns to strings for table
        df_disp = df_units.copy()
        df_disp["per_sensor_anomalies"] = df_disp["per_sensor_anomalies"].apply(per_sensor_to_str)
        df_disp["recent_anomalies"] = df_disp["recent_anomalies"].apply(per_sensor_to_str)

        st.dataframe(df_disp.sort_values("health_score").reset_index(drop=True), use_container_width=True)

        # Bar chart: anomalies by unit
        st.markdown("#### Anomalies by unit")
        bar = px.bar(df_units.sort_values("total_anomalies", ascending=False), x="unit", y="total_anomalies",
                     color="health_score", color_continuous_scale="RdYlGn_r", title="Anomalies per unit (worst first)")
        bar.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(bar, use_container_width=True)

        # Download CSV
        csv_bytes = df_units.sort_values("health_score").to_csv(index=False).encode("utf-8")
        st.download_button("Download scored results", csv_bytes, file_name=f"{dataset}_scored_results.csv", mime="text/csv")

        # Expandable detail panels for top 5 worst engines
        st.markdown("#### Top 5 worst engines — detailed view")
        for row in df_units.sort_values("health_score").head(5).itertuples(index=False):
            unit_id = int(row.unit)
            with st.expander(f"Unit {unit_id} — Health {row.health_score} — {row.total_anomalies} anomalies"):
                st.write("Predicted RUL:", row.predicted_rul)
                st.write("Per-sensor anomalies:", row.per_sensor_anomalies)
                st.write("Recent anomalies (last cycle per sensor):", row.recent_anomalies)
                # Optionally show small timeseries snapshot for that unit from uploaded CSV
                try:
                    # if user uploaded file, we have uploaded_file bytes saved earlier
                    uploaded_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                    u_df = uploaded_df[uploaded_df["unit"] == unit_id].sort_values("cycle")
                    if not u_df.empty:
                        cols_to_plot = [c for c in u_df.columns if c.startswith("sensor_")][:6]
                        fig_small = px.line(u_df, x="cycle", y=cols_to_plot, title=f"Unit {unit_id} sensor snapshot", template="plotly_dark")
                        st.plotly_chart(fig_small, use_container_width=True)
                except Exception:
                    pass

# ----------------------
# Footer
# ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#93c5fd'>Automotive Health Monitoring · Demo · Built with Streamlit</div>", unsafe_allow_html=True)

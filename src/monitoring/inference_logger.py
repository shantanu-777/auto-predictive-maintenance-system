# src/monitoring/inference_logger.py
"""
Tiny SQLite-based inference logger for audit and metrics.
Fields: id, ts_iso, unit, dataset, model_name, model_version, latency_ms, prediction, input_hash (optional), notes
"""
import sqlite3
from pathlib import Path
import time
import json

DB_PATH = Path("mlruns") / "inference_logs.db"  # keep in mlruns folder
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inference_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_iso TEXT,
        unit INTEGER,
        dataset TEXT,
        model_name TEXT,
        model_version TEXT,
        latency_ms REAL,
        prediction REAL,
        input_meta TEXT,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_inference(unit, dataset, model_name, model_version, latency_ms, prediction, input_meta=None, notes=None):
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    cur.execute("""
    INSERT INTO inference_logs (ts_iso, unit, dataset, model_name, model_version, latency_ms, prediction, input_meta, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, unit, dataset, model_name, model_version, latency_ms, prediction, json.dumps(input_meta) if input_meta else None, notes))
    conn.commit()
    conn.close()

def fetch_recent(limit=100):
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, ts_iso, unit, dataset, model_name, model_version, latency_ms, prediction FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

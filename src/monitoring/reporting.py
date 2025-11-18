# src/monitoring/reporting.py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import io
import datetime

plt.style.use("dark_background")

def _make_cover_page(pp: PdfPages, dataset: str, total_units:int, avg_health:float, total_anoms:int):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")
    ax.text(0.5, 0.8, "Automotive Predictive Maintenance — Report", ha="center", fontsize=20, weight='bold')
    ax.text(0.5, 0.68, f"Dataset: {dataset}", ha="center", fontsize=12)
    ax.text(0.5, 0.64, f"Generated: {datetime.datetime.now().isoformat()}", ha="center", fontsize=10)
    ax.text(0.1, 0.46, f"Total units scored: {total_units}", fontsize=12)
    ax.text(0.1, 0.42, f"Average health score: {avg_health:.2f}", fontsize=12)
    ax.text(0.1, 0.38, f"Total anomalies: {total_anoms}", fontsize=12)
    pp.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _make_table_page(pp: PdfPages, df_ranking: pd.DataFrame):
    # Show top N table (first page)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Engine ranking (worst -> best)", fontsize=14)
    tbl = ax.table(cellText=df_ranking.values, colLabels=df_ranking.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    pp.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _make_unit_plot(pp: PdfPages, flagged_df: pd.DataFrame, unit:int, sensor_cols:List[str]):
    # Plot selected sensors and RUL trend
    fig, axs = plt.subplots(2, 1, figsize=(11, 8))
    df = flagged_df.sort_values("cycle").reset_index(drop=True)
    cycles = df["cycle"].values
    # sensors (up to 4)
    sensors = sensor_cols[:4] if len(sensor_cols) >= 4 else sensor_cols
    for s in sensors:
        axs[0].plot(cycles, df[s].values, label=s)
    axs[0].set_title(f"Unit {unit} - Sensors")
    axs[0].legend(loc='upper right', fontsize=8)
    # RUL if available
    if "RUL_clipped" in df.columns:
        axs[1].plot(cycles, df["RUL_clipped"].values, label="RUL_clipped", color='orange')
    elif "RUL" in df.columns:
        axs[1].plot(cycles, df["RUL"].values, label="RUL", color='orange')
    # overlay anomaly markers if present
    anom_cols = [c for c in df.columns if c.endswith("_anomaly")]
    if anom_cols:
        # compute union of anomaly cycles
        anom_mask = np.zeros_like(cycles, dtype=bool)
        for ac in anom_cols:
            anom_mask = anom_mask | df[ac].astype(bool).values
        axs[1].scatter(cycles[anom_mask], (df["RUL_clipped"].values if "RUL_clipped" in df.columns else df["RUL"].values)[anom_mask],
                       color='red', label='anomaly', s=12)
    axs[1].set_title("RUL & Anomalies")
    axs[1].legend(loc='upper right', fontsize=8)
    pp.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def generate_score_pdf(score_results: Dict[str, Any],
                       anomalies_map: Dict[int, Dict[str, Any]],
                       analysis_dir: Path,
                       dataset_tag: str = "dataset",
                       top_k: int = 5) -> Path:
    """
    score_results: JSON-like dict returned by /score (with 'units' list)
    anomalies_map: mapping unit -> {'flagged_df': DataFrame, 'summary': {...}}
    analysis_dir: folder to save output
    Returns: Path to pdf file
    """
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = analysis_dir / f"{dataset_tag}_scored_report_{timestamp}.pdf"

    units = score_results.get("units", [])
    if not units:
        # nothing to do
        raise ValueError("No units supplied to generate PDF")

    df_units = pd.DataFrame(units)
    total_units = len(df_units)
    avg_health = float(df_units["health_score"].mean())
    total_anoms = int(df_units["total_anomalies"].sum())

    # prepare ranking table for PDF (selected columns)
    df_rank = df_units[["unit", "predicted_rul", "total_anomalies", "health_score"]].sort_values("health_score").reset_index(drop=True)
    df_rank.columns = ["Unit", "Predicted RUL", "Total Anomalies", "Health Score"]

    with PdfPages(out_pdf) as pp:
        _make_cover_page(pp, dataset=dataset_tag, total_units=total_units, avg_health=avg_health, total_anoms=total_anoms)
        _make_table_page(pp, df_rank.head(40))  # includes up to 40 rows

        # Add per-unit pages for top_k worst
        worst_units = df_rank.sort_values("Health Score").head(top_k)["Unit"].tolist()
        for u in worst_units:
            # anomalies_map should contain flagged_df
            info = anomalies_map.get(int(u))
            if info is None:
                continue
            flagged = info.get("flagged_df")
            if flagged is None or flagged.empty:
                continue
            # choose sensor cols
            sensor_cols = [c for c in flagged.columns if c.startswith("sensor_")]
            _make_unit_plot(pp, flagged, unit=int(u), sensor_cols=sensor_cols)

    return out_pdf

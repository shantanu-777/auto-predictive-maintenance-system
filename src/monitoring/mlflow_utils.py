# src/monitoring/mlflow_utils.py
import mlflow
import os

def set_mlflow_experiment(experiment_name="auto-predictive-maintenance"):
    """
    Create or set MLflow experiment and return the experiment ID.
    """
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id if exp else None

def ensure_artifact_dir(path="mlruns"):
    os.makedirs(path, exist_ok=True)
    return path

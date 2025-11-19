# src/monitoring/register_model.py
"""
Register a PyTorch sequence model into MLflow Model Registry.

Usage:
  $ env MLFLOW_TRACKING_URI=file:./mlruns python src/monitoring/register_model.py \
      --model-name auto-pm-lstm --model-path models/sequence/FD001/lstm_best.pth

This will:
 - load the saved checkpoint
 - instantiate the model architecture (must match training)
 - log the model to MLflow (mlflow.pytorch.log_model)
 - register the model in MLflow Model Registry with the provided name
 - print the registered model version
"""
import argparse
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
import torch
import joblib
import numpy as np
from src.Models.lstm_gru import LSTMRegressor

def load_checkpoint(checkpoint_path: Path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    return state

def build_model(input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    return model

def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    model_path = Path(args.model_path)
    feat_path = Path(args.features_path) if args.features_path else (model_path.parent / "features.joblib")
    norm_path = Path(args.norm_path) if args.norm_path else (model_path.parent / "norm_stats.joblib")

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    features = joblib.load(feat_path)
    input_dim = len(features)
    print(f"[INFO] Input dim from features: {input_dim}")

    # load state
    state = load_checkpoint(model_path, device=args.device)

    # instantiate model and load weights
    model = build_model(input_dim=input_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
    model.load_state_dict(state)
    model.eval()

    sample_input = np.zeros((1, args.seq_len, input_dim), dtype=np.float32)  # (B, T, F)
    signature = None
    try:
        from mlflow.models.signature import infer_signature
        signature = infer_signature(sample_input, np.array([0.0]))
    except Exception as e:
        print("[WARN] Could not infer signature:", e)

    # log model in a new MLflow run then register
    with mlflow.start_run() as run:
        artifact_path = "model"
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path, signature=signature, input_example=sample_input)
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"[INFO] Model logged as {model_uri}")

        # register
        client = mlflow.tracking.MlflowClient()
        # Ensure the registered model exists; get_registered_model raises if not found
        try:
            client.get_registered_model(args.model_name)
            print(f"[INFO] Registered model '{args.model_name}' already exists.")
        except mlflow.exceptions.MlflowException:
            try:
                client.create_registered_model(args.model_name)
                print(f"[INFO] Created registered model '{args.model_name}'.")
            except Exception as e:
                print(f"[WARN] Could not create registered model: {e}")

        mv = client.create_model_version(name=args.model_name, source=f"{mlflow.get_artifact_uri()}/{artifact_path}", run_id=run_id)
        print(f"[INFO] Registered model version: {mv.version} (name: {args.model_name})")

        # (optionally promote to stage 'Production' manually via mlflow UI or here)
        print("Done. Use `mlflow ui` or MLflow server UI to manage model versions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default="file:./mlruns")
    parser.add_argument("--model-name", default="auto-pm-lstm")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--norm-path", default=None)
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    # adapt to mlflow env
    args.mlflow_uri = args.mlflow_uri
    main(args)

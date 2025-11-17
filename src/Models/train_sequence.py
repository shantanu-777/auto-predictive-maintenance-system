# src/models/train_sequence.py
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sequence_dataset import SequenceRULDataset
from lstm_gru import LSTMRegressor, GRURegressor
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_batch(batch):
    seqs = np.stack([b["sequence"] for b in batch]).astype(np.float32)
    targets = np.array([b["target"] for b in batch]).astype(np.float32)
    return torch.from_numpy(seqs), torch.from_numpy(targets)



def evaluate_numpy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds.extend(y_hat.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), evaluate_numpy(trues, preds), preds, trues

def valid_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * X.size(0)
            preds.extend(y_hat.detach().cpu().numpy().tolist())
            trues.extend(y.detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), evaluate_numpy(trues, preds), preds, trues

def main(args):
    set_seed(args.seed)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # load dataset csv (single dataset run per invocation)
    proc_dir = Path(args.proc_dir)
    ds = args.dataset.strip().upper()
    csv_path = proc_dir / f"{ds}_train_labeled.csv"
    test_csv = proc_dir / f"{ds}_test_labeled.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    dataset = SequenceRULDataset(csv_path, seq_len=args.seq_len, normalize=True)

    # split into train/val by units (avoid leakage): split index_map by unit idx
    # simpler: split index_map into train/val by proportion
    n = len(dataset)
    val_n = int(n * args.val_split)
    train_n = n - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b), drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b))


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    input_dim = len(dataset.get_feature_names())
    if args.model_type.lower().startswith("lstm"):
        model = LSTMRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
    else:
        model = GRURegressor(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    epochs_no_improve = 0

    with mlflow.start_run(run_name=f"{args.model_type}_{ds}_seq"):
        mlflow.log_params({
            "dataset": ds, "seq_len": args.seq_len, "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "lr": args.lr,
            "dropout": args.dropout, "bidirectional": args.bidirectional
        })
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics, _, _ = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics, val_preds, val_trues = valid_epoch(model, val_loader, criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            print(f"[E{epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_metrics['mae']:.4f}")

            # checkpointing based on val_loss
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                ckpt_dir = Path("models/sequence") / ds
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{args.model_type}_best.pth"
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
                mlflow.log_artifact(str(ckpt_path), artifact_path="models")
                # also save feature names
                feat_path = ckpt_dir / "features.joblib"
                joblib.dump(dataset.get_feature_names(), feat_path)
                mlflow.log_artifact(str(feat_path), artifact_path="models")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.early_stopping_patience:
                print("[INFO] Early stopping triggered.")
                break

        # final test evaluation using saved best model on test CSV
        # load best checkpoint
        best_ckpt = ckpt_dir / f"{args.model_type}_best.pth"
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])

        # build test dataset loader (all windows from test)
        test_dataset = SequenceRULDataset(test_csv, seq_len=args.seq_len, normalize=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b))

        _, test_metrics, test_preds, test_trues = valid_epoch(model, test_loader, criterion, device)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        print("[TEST] ", test_metrics)

        # save final model in models/sequence/<ds> folder
        final_dir = Path("models/sequence") / ds
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{args.model_type}_final.pth"
        torch.save({"model_state": model.state_dict(), "feature_names": dataset.get_feature_names()}, final_path)
        mlflow.log_artifact(str(final_path), artifact_path="models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", default="data/processed/for_model", help="Processed labeled data directory")
    parser.add_argument("--dataset", default="FD001", help="Dataset code: FD001 .. FD004")
    parser.add_argument("--model_type", default="lstm", help="lstm or gru")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow_uri", default="file:./mlruns")
    parser.add_argument("--experiment_name", default="auto-predictive-maintenance")
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    main(args)

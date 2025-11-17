# src/models/xgboost_train.py
import os
import joblib
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
from monitoring.mlflow_utils import set_mlflow_experiment

RANDOM_SEED = 42

def load_labeled_dataset(proc_dir, dataset="FD001"):
    """
    Loads train/test labeled csv for a dataset (e.g. FD001).
    Returns train_df, test_df (dataframes with RUL column).
    """
    proc_dir = Path(proc_dir)
    train_path = proc_dir / f"{dataset}_train_labeled.csv"
    test_path = proc_dir / f"{dataset}_test_labeled.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing files for {dataset} in {proc_dir}")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def last_cycle_aggregation(df):
    """
    Selects the last observed cycle row per unit. Keeps sensor & op settings + RUL.
    """
    df = df.copy()
    # keep rows where cycle == max cycle per unit
    last_idx = df.groupby("unit")["cycle"].idxmax()
    last = df.loc[last_idx].reset_index(drop=True)
    return last

def select_feature_columns(df):
    # pick op settings + sensor_* and any engineered features starting with sensor_
    features = [c for c in df.columns if (c.startswith("op_setting") or c.startswith("sensor_") or "_mean" in c or "_std" in c or "_min" in c or "_max" in c)]
    # drop obvious non-features
    features = [f for f in features if f not in ("RUL", "RUL_clipped")]
    return features

def train_xgb(X_train, y_train, X_val, y_val, params=None, num_boost_round=200):
    params = params or {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=25, verbose_eval=False)
    return model, params

def predict_xgb(model, X):
    d = xgb.DMatrix(X)
    return model.predict(d)

def evaluate(y_true, y_pred):
    """
    Returns dict with mae, rmse, r2.
    Uses sklearn.mean_squared_error without 'squared' kw to remain compatible.
    """
    # ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)           # returns MSE
    rmse = float(np.sqrt(mse))                         # RMSE = sqrt(MSE)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def save_feature_importance(model, feature_names, out_path):
    fmap = sorted(zip(feature_names, model.get_score(importance_type="gain").items()), key=lambda x: x[0]) if False else None
    # xgboost's get_score returns dict keyed by 'f0','f1' when trained from DMatrix without feature_names
    # Instead, use model.get_score and map indices to feature names when possible
    scores = model.get_score(importance_type="gain")
    fi = []
    for k, v in scores.items():
        # k like 'f12' -> index 12
        if k.startswith("f"):
            idx = int(k[1:])
            name = feature_names[idx] if idx < len(feature_names) else k
        else:
            name = k
        fi.append((name, v))
    fi_sorted = sorted(fi, key=lambda x: x[1], reverse=True)
    # save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fi_sorted, columns=["feature", "gain"]).to_csv(out_path, index=False)

def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    set_mlflow_experiment(args.experiment_name)

    datasets = args.datasets.split(",")
    for ds in datasets:
        ds = ds.strip().upper()
        print(f"[INFO] Processing dataset {ds}")
        train_df, test_df = load_labeled_dataset(args.proc_dir, dataset=ds)
        train_last = last_cycle_aggregation(train_df)
        test_last = last_cycle_aggregation(test_df)

        feature_cols = select_feature_columns(train_last)
        print(f"[INFO] {len(feature_cols)} feature columns detected")

        X = train_last[feature_cols]
        y = train_last["RUL_clipped"] if "RUL_clipped" in train_last.columns else train_last["RUL"]
        X_test = test_last[feature_cols]
        y_test = test_last["RUL_clipped"] if "RUL_clipped" in test_last.columns else test_last["RUL"]

        # train/val split from train set
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Start MLflow run
        with mlflow.start_run(run_name=f"xgb_baseline_{ds}") as run:
            # log params
            mlflow.log_param("dataset", ds)
            mlflow.log_param("num_train_units", int(X_tr.shape[0]))
            mlflow.log_param("num_val_units", int(X_val.shape[0]))
            mlflow.log_param("feature_count", int(len(feature_cols)))

            # train
            model, params = train_xgb(X_tr, y_tr, X_val, y_val, params=None, num_boost_round=args.num_boost_round)

            # predict & evaluate on validation
            val_preds = predict_xgb(model, X_val)
            val_metrics = evaluate(y_val, val_preds)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            # predict on test and log metrics
            test_preds = predict_xgb(model, X_test)
            test_metrics = evaluate(y_test, test_preds)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # save model artifact
            model_dir = Path("models/xgboost") / ds
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "xgboost_model.json"
            model.save_model(str(model_path))
            # also save a sklearn-compatible wrapper via joblib (optional)
            joblib.dump((model, feature_cols), model_dir / "xgb_model_and_features.joblib")

            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(model_dir / "xgb_model_and_features.joblib"), artifact_path="models")

            # feature importance
            fi_path = model_dir / "feature_importance.csv"
            save_feature_importance(model, feature_cols, fi_path)
            mlflow.log_artifact(str(fi_path), artifact_path="models")

            # log params used by xgboost
            for k, v in params.items():
                mlflow.log_param(f"xgb_{k}", v)

            print(f"[INFO] Finished run {run.info.run_id} for dataset {ds}")
            print("Validation metrics:", val_metrics)
            print("Test metrics:", test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", default="data/processed/for_model", help="Processed labeled data directory")
    parser.add_argument("--datasets", default="FD001", help="Comma-separated datasets to train on: e.g. FD001,FD002")
    parser.add_argument("--num_boost_round", type=int, default=300)
    parser.add_argument("--mlflow_uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", default="auto-predictive-maintenance", help="MLflow experiment name")
    args = parser.parse_args()
    main(args)

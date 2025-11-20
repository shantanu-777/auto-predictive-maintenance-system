from mlflow.tracking import MlflowClient

client = MlflowClient("file:./mlruns")
client.delete_model_version("auto-pm-lstm", 1)
print("Deleted auto-pm-lstm version 1")

from mlflow.tracking import MlflowClient
client = MlflowClient("file:./mlruns")

# List registered models and their latest versions
for rm in client.search_registered_models():
    print("Registered model:", rm.name)
    # rm.latest_versions is a list; show version + source URI
    for v in (rm.latest_versions or []):
        print("  version:", v.version, "source:", getattr(v, "source", None))
print("------ Done ------")

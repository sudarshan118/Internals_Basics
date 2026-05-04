import mlflow
from mlflow import MlflowClient
import json
import os

mlflow.set_tracking_uri("file:///C:/Users/sudarshan ps/Internals_Basics/MLOPs_Lab_CIE/mlruns")

client = MlflowClient()

# Get the experiment
experiment = client.get_experiment_by_name("steelforge-production-hours")
experiment_id = experiment.experiment_id

# Get all runs, find the best one by MAE
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.mae ASC"]
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_mae = best_run.data.metrics["mae"]
best_model_name = best_run.data.tags.get("mlflow.runName", "best_model")

print(f"Best run: {best_run_id} | Model: {best_model_name} | MAE: {best_mae}")

# Register the model
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "steelforge-production-hours-predictor"

result = mlflow.register_model(model_uri, registered_model_name)

version = result.version

print(f"Registered as version: {version}")

output = {
    "registered_model_name": registered_model_name,
    "version": int(version),
    "run_id": best_run_id,
    "source_metric": "mae",
    "source_metric_value": round(best_mae, 4)
}

os.makedirs("results", exist_ok=True)
with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ step3_s6.json saved")
print(json.dumps(output, indent=2))
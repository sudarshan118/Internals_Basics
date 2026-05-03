import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import mlflow
import json

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("freshbasket-delivery-time-min")

model_name = "freshbasket-delivery-time-min-predictor"

# Get runs
runs = mlflow.search_runs()

if runs.empty:
    print("❌ No runs found — re-run train.py first")
    exit()

# Sort by RMSE
runs = runs.sort_values("metrics.rmse")

best_run = runs.iloc[0]
run_id = best_run["run_id"]

model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, model_name)

output = {
    "registered_model_name": model_name,
    "version": result.version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": float(best_run["metrics.rmse"])
}

with open("../results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed")
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("file:///C:/Users/sudarshan ps/Internals_Basics/MLOPs_Lab_CIE/mlruns")

df = pd.read_csv("data/training_data.csv")
X = df.drop("production_hours", axis=1)
y = df["production_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("steelforge-production-hours")

results = []

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log params
        params = model.get_params()
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "run_id": run.info.run_id
        })

        print(f"{name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}, run_id: {run.info.run_id}")

        # Save best model artifact locally
        import pickle
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "steelforge-production-hours",
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ step1_s1.json saved")
print(json.dumps(output, indent=2))
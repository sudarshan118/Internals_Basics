import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set experiment
mlflow.set_experiment("freshbasket-delivery-time-min")

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

models = {
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # 🔥 IMPORTANT (FIX)
        mlflow.sklearn.log_model(model, "model")

        # Predict
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("team", "ml_engineering")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

# Select best model
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "freshbasket-delivery-time-min",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

# Save JSON
with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")
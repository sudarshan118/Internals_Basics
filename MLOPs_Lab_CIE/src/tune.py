import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
import numpy as np
import json
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid (as per question)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

mlflow.set_experiment("freshbasket-delivery-time-min")

with mlflow.start_run(run_name="tuning-freshbasket"):

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    output = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-freshbasket"
    }

    with open("../results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 completed")
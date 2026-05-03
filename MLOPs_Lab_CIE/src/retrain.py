import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
train = pd.read_csv("../data/training_data.csv")
new = pd.read_csv("../data/new_data.csv")

# Combine data
combined = pd.concat([train, new], ignore_index=True)

# Prepare features
X = combined.drop("delivery_time_min", axis=1)
y = combined["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Retrain model (same as best model)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate retrained model
preds = model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, preds))

# Load previous RMSE from Task 1 JSON
with open("../results/step1_s1.json", "r") as f:
    step1 = json.load(f)

champion_rmse = step1["best_metric_value"]

# Compare
improvement = champion_rmse - retrained_rmse

action = "promoted" if improvement >= 0.5 else "kept_champion"

# Output JSON
output = {
    "original_data_rows": len(train),
    "new_data_rows": len(new),
    "combined_data_rows": len(combined),
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

with open("../results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")
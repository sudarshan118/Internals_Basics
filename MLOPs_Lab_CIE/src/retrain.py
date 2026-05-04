import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json
import os
import pickle

mlflow.set_tracking_uri("file:///C:/Users/sudarshan ps/Internals_Basics/MLOPs_Lab_CIE/mlruns")

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

print(f"Original rows: {len(train_df)}, New rows: {len(new_df)}, Combined: {len(combined_df)}")

# Use same test split as Task 1 (original data only for fair comparison)
X_orig = train_df.drop("production_hours", axis=1)
y_orig = train_df["production_hours"]
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# Champion MAE from Task 1
champion_mae = 2.1942

# Retrain on combined data
X_combined = combined_df.drop("production_hours", axis=1)
y_combined = combined_df["production_hours"]
X_train_combined, _, y_train_combined, _ = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

retrained_model = Ridge(alpha=1.0)
retrained_model.fit(X_train_combined, y_train_combined)

# Evaluate on same test set as champion
retrained_preds = retrained_model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, retrained_preds)

improvement = champion_mae - retrained_mae
min_threshold = 0.5

if improvement >= min_threshold:
    action = "promoted"
    with open("models/Ridge_retrained.pkl", "wb") as f:
        pickle.dump(retrained_model, f)
    print("✅ Retrained model PROMOTED")
else:
    action = "kept_champion"
    print("⚠️ Champion kept — improvement insufficient")

print(f"Champion MAE: {champion_mae}, Retrained MAE: {round(retrained_mae, 4)}, Improvement: {round(improvement, 4)}")

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_mae": champion_mae,
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": min_threshold,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ step4_s8.json saved")
print(json.dumps(output, indent=2))
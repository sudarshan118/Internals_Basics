import argparse
import pickle
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size_tons", type=float, required=True)
parser.add_argument("--machine_count", type=float, required=True)
parser.add_argument("--alloy_grade", type=float, required=True)
parser.add_argument("--shift_type", type=float, required=True)
args = parser.parse_args()

model_path = os.path.join(os.path.dirname(__file__), "../models/Ridge.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

features = np.array([[args.batch_size_tons, args.machine_count, args.alloy_grade, args.shift_type]])
prediction = model.predict(features)[0]

print(f"Predicted production_hours: {round(prediction, 4)}")
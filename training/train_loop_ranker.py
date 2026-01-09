import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

df = pd.read_csv("loop_training_dataset.csv")
X = df.drop(columns=["weight","track_id","start_time","end_time"])
y = df["weight"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train)
Path("training/models").mkdir(exist_ok=True)
model.save_model("training/models/loop_ranker.json")

with open("training/models/features.json","w") as f:
    json.dump(X.columns.tolist(), f)

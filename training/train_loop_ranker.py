import pandas as pd
import xgboost as xgb
import json
from pathlib import Path


def _train_test_split_indices(n: int, test_size: float = 0.2, seed: int = 1337):
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx

df = pd.read_csv("loop_training_dataset.csv")
X = df.drop(columns=["weight","track_id","start_time","end_time"])
y = df["weight"]

train_idx, test_idx = _train_test_split_indices(len(df), test_size=0.2, seed=1337)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train)
Path("training/models").mkdir(exist_ok=True, parents=True)
model.save_model("training/models/loop_ranker.json")

with open("training/models/features.json","w") as f:
    json.dump(X.columns.tolist(), f)

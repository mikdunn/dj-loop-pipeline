from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

# Allow running as a script: `python training/train_loop_ranker_torch.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training.torch_ranker import StandardScaler, build_mlp, save_checkpoint


def _split_indices(n: int, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _as_float_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def main() -> int:
    p = argparse.ArgumentParser(description="Train PyTorch loop ranker (regression on weight).")
    p.add_argument("--dataset", default="loop_training_dataset.csv", help="CSV produced by dataset_builder")
    p.add_argument("--outdir", default="training/models", help="Output directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--hidden", default="256,128", help="Hidden layer sizes, comma-separated")
    p.add_argument("--dropout", type=float, default=0.10)
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    if "weight" not in df.columns:
        raise ValueError("Dataset must contain a 'weight' column")

    drop_cols = {"weight", "track_id", "start_time", "end_time"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    x_all = _as_float_matrix(df, feature_cols)
    y_all = df["weight"].to_numpy(dtype=np.float32)

    train_idx, test_idx = _split_indices(len(df), args.test_size, args.seed)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    hidden_sizes = tuple(int(s.strip()) for s in args.hidden.split(",") if s.strip())

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_mlp(x_train.shape[1], hidden_sizes=hidden_sizes, dropout=args.dropout).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).view(-1, 1))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).view(-1, 1))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_test = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        test_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                test_losses.append(float(loss.detach().cpu().item()))

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        test_mse = float(np.mean(test_losses)) if test_losses else float("nan")
        if test_mse < best_test:
            best_test = test_mse

        print(f"epoch {epoch:03d} | train_mse={train_mse:.6f} | test_mse={test_mse:.6f} | device={device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint with scaler + features embedded.
    ckpt_path = outdir / "loop_ranker.pt"
    save_checkpoint(
        str(ckpt_path),
        model=model.to("cpu"),
        features=feature_cols,
        scaler=scaler,
        meta={
            "arch": "mlp",
            "hidden_sizes": list(hidden_sizes),
            "dropout": float(args.dropout),
            "trained_epochs": int(args.epochs),
            "best_test_mse": float(best_test),
        },
    )

    # Also write features.json for compatibility with existing pipeline signature.
    with open(outdir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    print(f"Saved: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

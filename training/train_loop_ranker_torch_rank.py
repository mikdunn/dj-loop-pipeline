from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as a script: `python training/train_loop_ranker_torch_rank.py ...`
if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training.torch_ranker import StandardScaler, build_mlp, save_checkpoint


def _as_float_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _group_indices_by_track(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "track_id" not in df.columns:
        raise ValueError("Dataset must contain track_id")
    groups: Dict[str, List[int]] = {}
    for i, tid in enumerate(df["track_id"].astype(str).tolist()):
        groups.setdefault(tid, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _split_tracks(track_ids: List[str], test_size: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(track_ids)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_size))
    test_ids = ids[:n_test].tolist()
    train_ids = ids[n_test:].tolist()
    return train_ids, test_ids


def _soft_targets_from_weights(w: np.ndarray, beta: float) -> np.ndarray:
    # stable softmax(beta*w)
    w = w.astype(np.float32)
    z = beta * w
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(np.sum(e)) + 1e-8
    return (e / s).astype(np.float32)


def _subsample_keep_best(
    idx: np.ndarray,
    weights: np.ndarray,
    max_cands: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample candidates but always keep the best (highest-weight) row.

    This stabilizes training when some tracks have huge candidate sets.
    """
    if max_cands <= 0 or idx.size <= max_cands:
        return idx
    if idx.size == 0:
        return idx

    local_w = weights[idx]
    best_local = int(np.argmax(local_w))
    best_idx = int(idx[best_local])

    # Sample remaining without replacement from the other indices
    others = idx[idx != best_idx]
    k = max_cands - 1
    if others.size <= k:
        chosen = others
    else:
        chosen = rng.choice(others, size=k, replace=False)

    out = np.concatenate([np.asarray([best_idx], dtype=np.int64), chosen.astype(np.int64)])
    rng.shuffle(out)
    return out


def _precision_at_k(best_rank_1_indexed: int, k: int) -> float:
    return 1.0 if best_rank_1_indexed <= k else 0.0


def _mrr(best_rank_1_indexed: int) -> float:
    return 1.0 / float(best_rank_1_indexed)


def _evaluate_ranking(
    *,
    model,
    device,
    x_all: np.ndarray,
    weights: np.ndarray,
    groups: Dict[str, np.ndarray],
    track_list: List[str],
    ks: List[int],
    max_cands_per_track: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Evaluate ranking quality per track.

    We treat "best loop" as the max-weight candidate in each track.
    """
    import torch

    model.eval()

    ranks_1idx: List[int] = []
    p_at: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: List[float] = []

    with torch.no_grad():
        for tid in track_list:
            idx = groups.get(tid)
            if idx is None or idx.size == 0:
                continue

            # For evaluation speed, optionally cap candidates (keeping best)
            if max_cands_per_track is not None and max_cands_per_track > 0 and idx.size > max_cands_per_track:
                idx_eval = _subsample_keep_best(idx, weights, max_cands_per_track, rng)
            else:
                idx_eval = idx

            w = weights[idx_eval]
            best_pos = int(np.argmax(w))

            xb = torch.from_numpy(x_all[idx_eval]).to(device)
            scores = model(xb).view(-1).detach().cpu().numpy()
            order = np.argsort(-scores)  # descending

            # 1-indexed rank of the best candidate
            best_rank = int(np.where(order == best_pos)[0][0]) + 1
            ranks_1idx.append(best_rank)
            mrrs.append(_mrr(best_rank))
            for k in ks:
                p_at[k].append(_precision_at_k(best_rank, k))

    out: Dict[str, float] = {}
    out["num_tracks"] = float(len(ranks_1idx))
    out["mrr"] = float(np.mean(mrrs)) if mrrs else float("nan")
    for k in ks:
        out[f"p_at_{k}"] = float(np.mean(p_at[k])) if p_at[k] else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Train a PyTorch loop ranker optimized for best-loop-per-track ranking.\n\n"
            "This uses a listwise softmax ranking loss per track: it learns to score the best\n"
            "candidates higher than others, then you can rank all loops best->worst and\n"
            "apply a cutoff threshold during inference."
        )
    )
    p.add_argument("--dataset", default="loop_training_dataset.csv")
    p.add_argument("--outdir", default="training/models")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-tracks", type=int, default=8, help="How many tracks per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--hidden", default="256,128")
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument(
        "--objective",
        default="listwise_softmax",
        choices=["listwise_softmax"],
        help="Ranking objective. (Currently: listwise softmax KL / cross-entropy)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=8.0,
        help="Softmax temperature for turning weights into soft targets (higher = more peaky)",
    )
    p.add_argument(
        "--max-cands-per-track",
        type=int,
        default=512,
        help="Cap candidates per track for training speed (randomly sampled each epoch)",
    )
    p.add_argument(
        "--eval-max-cands-per-track",
        type=int,
        default=1024,
        help="Cap candidates per track during evaluation (keeping the best). Set 0 to disable.",
    )
    p.add_argument(
        "--metrics-k",
        default="1,3,5",
        help="Comma-separated K values for Precision@K (best-loop hit rate).",
    )
    p.add_argument("--mlflow", action="store_true", help="If set, log params/metrics/artifacts to MLflow")
    p.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (optional)")
    p.add_argument("--mlflow-experiment", default="dj-loop-pipeline", help="MLflow experiment name")
    p.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset CSV not found: "
            f"{dataset_path}.\n\n"
            "Build it first using:\n"
            "  python training/build_training_dataset_cli.py --audio-dir data/raw_audio --timestamps-json timestamps.json --out loop_training_dataset.csv\n\n"
            "(Adjust --audio-dir and --timestamps-json to your files.)"
        )

    df = pd.read_csv(dataset_path)
    if "weight" not in df.columns:
        raise ValueError("Dataset must contain a 'weight' column")

    drop_cols = {"weight", "track_id", "start_time", "end_time"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    x_all = _as_float_matrix(df, feature_cols)
    y_weight = df["weight"].to_numpy(dtype=np.float32)

    groups = _group_indices_by_track(df)
    all_track_ids = list(groups.keys())
    train_ids, test_ids = _split_tracks(all_track_ids, args.test_size, args.seed)

    if not train_ids:
        raise ValueError("No training tracks after split")

    train_row_idx = np.concatenate([groups[t] for t in train_ids if groups[t].size > 0])
    if train_row_idx.size == 0:
        raise ValueError("Training split has no rows")

    scaler = StandardScaler().fit(x_all[train_row_idx])
    x_all = scaler.transform(x_all)

    import torch

    # Reproducibility
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_sizes = tuple(int(s.strip()) for s in args.hidden.split(",") if s.strip())
    model = build_mlp(len(feature_cols), hidden_sizes=hidden_sizes, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ks = [int(s.strip()) for s in str(args.metrics_k).split(",") if s.strip()]
    ks = sorted({k for k in ks if k > 0}) or [1, 3, 5]

    eval_max = int(args.eval_max_cands_per_track)
    eval_max = None if eval_max <= 0 else eval_max

    def eval_tracks_listwise_loss(track_list: List[str]) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for tid in track_list:
                idx = groups[tid]
                if idx.size == 0:
                    continue
                xb = torch.from_numpy(x_all[idx]).to(device)
                wb = y_weight[idx]
                # targets
                pt = _soft_targets_from_weights(wb, args.beta)
                pt = torch.from_numpy(pt).to(device)

                scores = model(xb).view(-1)
                logp = torch.log_softmax(scores, dim=0)
                loss = -(pt * logp).sum()
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    rng = np.random.default_rng(args.seed)

    # Optional MLflow logging (no hard dependency)
    mlflow = None
    mlflow_run = None
    if bool(args.mlflow):
        try:
            import mlflow as _mlflow  # type: ignore

            mlflow = _mlflow
            if args.mlflow_uri:
                mlflow.set_tracking_uri(str(args.mlflow_uri))
            mlflow.set_experiment(str(args.mlflow_experiment))
            mlflow_run = mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_params(
                {
                    "dataset": str(args.dataset),
                    "epochs": int(args.epochs),
                    "batch_tracks": int(args.batch_tracks),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "test_size": float(args.test_size),
                    "seed": int(args.seed),
                    "hidden": str(args.hidden),
                    "dropout": float(args.dropout),
                    "beta": float(args.beta),
                    "max_cands_per_track": int(args.max_cands_per_track),
                    "eval_max_cands_per_track": int(args.eval_max_cands_per_track),
                    "objective": str(args.objective),
                    "metrics_k": str(args.metrics_k),
                }
            )
        except Exception as e:
            print(f"[warn] MLflow logging requested but unavailable: {e}")
            mlflow = None
            mlflow_run = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Shuffle tracks each epoch
        shuffled = train_ids.copy()
        rng.shuffle(shuffled)

        batch_losses = []
        for b0 in range(0, len(shuffled), args.batch_tracks):
            batch_tids = shuffled[b0 : b0 + args.batch_tracks]

            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            count = 0

            for tid in batch_tids:
                idx = groups[tid]
                if idx.size == 0:
                    continue

                # Subsample candidates for this track
                if args.max_cands_per_track and idx.size > args.max_cands_per_track:
                    idx = _subsample_keep_best(idx, y_weight, int(args.max_cands_per_track), rng)

                xb = torch.from_numpy(x_all[idx]).to(device)
                wb = y_weight[idx]
                pt = _soft_targets_from_weights(wb, args.beta)
                pt = torch.from_numpy(pt).to(device)

                scores = model(xb).view(-1)
                logp = torch.log_softmax(scores, dim=0)
                loss = -(pt * logp).sum()

                total_loss = total_loss + loss
                count += 1

            if count == 0:
                continue

            total_loss = total_loss / count
            total_loss.backward()
            opt.step()

            batch_losses.append(float(total_loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        test_loss = eval_tracks_listwise_loss(test_ids)

        train_rank = _evaluate_ranking(
            model=model,
            device=device,
            x_all=x_all,
            weights=y_weight,
            groups=groups,
            track_list=train_ids,
            ks=ks,
            max_cands_per_track=eval_max,
            rng=rng,
        )
        test_rank = _evaluate_ranking(
            model=model,
            device=device,
            x_all=x_all,
            weights=y_weight,
            groups=groups,
            track_list=test_ids,
            ks=ks,
            max_cands_per_track=eval_max,
            rng=rng,
        )

        msg = (
            f"epoch {epoch:03d} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f}"
            f" | train_mrr={train_rank['mrr']:.4f} | test_mrr={test_rank['mrr']:.4f}"
            f" | test_p@{ks[0]}={test_rank[f'p_at_{ks[0]}']:.4f} | device={device}"
        )
        print(msg)

        if mlflow is not None:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("train_mrr", float(train_rank["mrr"]), step=epoch)
            mlflow.log_metric("test_mrr", float(test_rank["mrr"]), step=epoch)
            for k in ks:
                mlflow.log_metric(f"train_p_at_{k}", float(train_rank[f"p_at_{k}"]), step=epoch)
                mlflow.log_metric(f"test_p_at_{k}", float(test_rank[f"p_at_{k}"]), step=epoch)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = outdir / "loop_ranker_rank.pt"
    save_checkpoint(
        str(ckpt_path),
        model=model.to("cpu").eval(),
        features=feature_cols,
        scaler=scaler,
        meta={
            "arch": "mlp_rank",
            "hidden_sizes": list(hidden_sizes),
            "dropout": float(args.dropout),
            "trained_epochs": int(args.epochs),
            "beta": float(args.beta),
            "objective": "listwise_softmax",
        },
    )

    with open(outdir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    if mlflow is not None:
        try:
            mlflow.log_artifact(str(ckpt_path))
            mlflow.log_artifact(str(outdir / "features.json"))
        except Exception as e:
            print(f"[warn] MLflow artifact logging failed: {e}")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(f"Saved: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from similarity import (  # type: ignore
    build_raw_similarity,
    collect_audio_files,
    compute_graph_quality,
    extract_loop_features,
    extract_pattern_sequence,
    fiedler_labels,
    sharpen_similarity,
)


ModelFamily = Literal["rf", "svm_rbf_calibrated", "voting_rf_svm", "stacking_rf_svm", "knn", "mlp"]
LeakageMode = Literal["transductive", "strict_train_only"]


def make_estimator(model_family: ModelFamily, random_state: int):
    if model_family == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )

    if model_family == "svm_rbf_calibrated":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=4.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_family == "knn":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance")),
            ]
        )

    if model_family == "mlp":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=400,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_family == "voting_rf_svm":
        rf = RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        svm = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=4.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return VotingClassifier(estimators=[("rf", rf), ("svm", svm)], voting="soft")

    if model_family == "stacking_rf_svm":
        rf = RandomForestClassifier(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        svm = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=4.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return StackingClassifier(
            estimators=[("rf", rf), ("svm", svm)],
            final_estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state),
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_family: {model_family}")


def _assign_test_labels_from_train(train_seq: List[np.ndarray], train_labels: np.ndarray, test_seq: List[np.ndarray]) -> np.ndarray:
    if not train_seq:
        raise RuntimeError("No train sequences available for strict_train_only label assignment")

    out: List[int] = []
    for seq in test_seq:
        # Build pairwise structure with [test] + train, then take nearest train sequence by distance.
        _, dist_local, _ = build_raw_similarity([seq] + train_seq)
        nearest_train_idx = int(np.argmin(dist_local[0, 1:]))
        out.append(int(train_labels[nearest_train_idx]))
    return np.asarray(out, dtype=int)


def _build_artifact_paths(out_dir: Path, mode: str, model_family: ModelFamily, leakage_mode: LeakageMode) -> Tuple[Path, Path, Path]:
    # Backward-compatible default names for existing downstream scripts.
    if model_family == "rf" and leakage_mode == "transductive":
        dataset_csv = out_dir / f"contact_train_dataset_{mode}.csv"
        model_path = out_dir / f"contact_rf_model_{mode}.joblib"
        report_path = out_dir / f"contact_train_report_{mode}.json"
        return dataset_csv, model_path, report_path

    suffix = f"{mode}_{model_family}_{leakage_mode}"
    dataset_csv = out_dir / f"contact_train_dataset_{suffix}.csv"
    model_path = out_dir / f"contact_model_{suffix}.joblib"
    report_path = out_dir / f"contact_train_report_{suffix}.json"
    return dataset_csv, model_path, report_path


def run_train(
    folder: Path,
    out_dir: Path,
    mode: str,
    max_files: int,
    n_bins: int,
    random_state: int = 42,
    model_family: ModelFamily = "rf",
    leakage_mode: LeakageMode = "transductive",
    test_size: float = 0.25,
) -> Path:
    files = collect_audio_files(folder)
    if max_files > 0:
        files = files[:max_files]
    if len(files) < 12:
        raise RuntimeError("Need at least 12 files for train/eval")

    feats_seq: List[np.ndarray] = []
    row_feats: List[Dict[str, float]] = []
    keep_files: List[Path] = []
    for p in files:
        seq = extract_pattern_sequence(p)
        feat = extract_loop_features(p)
        if seq is not None and feat is not None:
            feats_seq.append(seq)
            row_feats.append(feat)
            keep_files.append(p)

    if len(keep_files) < 12:
        raise RuntimeError("Too few valid files after sequence/feature extraction")

    rows = []
    for p, feat in zip(keep_files, row_feats):
        r = {"file": str(p), "label": -1}
        r.update(feat)
        rows.append(r)

    df = pd.DataFrame(rows)

    feature_cols = [c for c in df.columns if c not in {"file", "label"}]
    X = df[feature_cols]
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=random_state,
        shuffle=True,
    )

    yv = np.full(len(df), fill_value=-1, dtype=int)
    graph_metrics: Dict[str, float]

    if leakage_mode == "transductive":
        sim_raw, dist, ov = build_raw_similarity(feats_seq)
        sim = sharpen_similarity(sim_raw, dist, ov, mode=mode)
        graph_metrics = compute_graph_quality(sim)
        y_all = fiedler_labels(sim, n_bins=n_bins)
        yv = np.asarray(y_all, dtype=int)
    elif leakage_mode == "strict_train_only":
        train_seq = [feats_seq[int(i)] for i in idx_train]
        test_seq = [feats_seq[int(i)] for i in idx_test]

        sim_raw_train, dist_train, ov_train = build_raw_similarity(train_seq)
        sim_train = sharpen_similarity(sim_raw_train, dist_train, ov_train, mode=mode)
        graph_metrics = compute_graph_quality(sim_train)
        y_train_bins = np.asarray(fiedler_labels(sim_train, n_bins=n_bins), dtype=int)
        y_test_bins = _assign_test_labels_from_train(train_seq=train_seq, train_labels=y_train_bins, test_seq=test_seq)

        yv[idx_train] = y_train_bins
        yv[idx_test] = y_test_bins
    else:
        raise ValueError(f"Unsupported leakage_mode: {leakage_mode}")

    df["label"] = yv

    if int(pd.Series(yv).nunique()) < 2:
        raise RuntimeError("Label bins collapsed to one class; increase max_files or adjust mode")

    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_train = yv[idx_train]
    y_test = yv[idx_test]

    clf = make_estimator(model_family=model_family, random_state=random_state)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    report = {
        "mode": mode,
        "model_family": model_family,
        "leakage_mode": leakage_mode,
        "n_rows": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "random_state": int(random_state),
        "max_files": int(max_files),
        "n_bins": int(n_bins),
        "test_size": float(test_size),
        **graph_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv, model_path, report_path = _build_artifact_paths(
        out_dir=out_dir,
        mode=mode,
        model_family=model_family,
        leakage_mode=leakage_mode,
    )

    df.to_csv(dataset_csv, index=False)
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "mode": mode,
            "n_bins": n_bins,
            "model_family": model_family,
            "leakage_mode": leakage_mode,
            "test_size": float(test_size),
            "split_indices": {
                "train": [int(i) for i in idx_train.tolist()],
                "test": [int(i) for i in idx_test.tolist()],
            },
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved dataset: {dataset_csv}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(
        f"mode={mode}, model={model_family}, leakage={leakage_mode} | "
        f"macro_f1={report['macro_f1']:.4f}, weighted_f1={report['weighted_f1']:.4f}, accuracy={report['accuracy']:.4f}"
    )

    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train contact-similarity classifier with selectable sharpening mode")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out_dir", default="training/models/retrain_compare")
    ap.add_argument("--mode", choices=("none", "self_tuned_knn", "sinkhorn"), default="self_tuned_knn")
    ap.add_argument(
        "--model_family",
        choices=("rf", "svm_rbf_calibrated", "voting_rf_svm", "stacking_rf_svm", "knn", "mlp"),
        default="rf",
    )
    ap.add_argument("--leakage_mode", choices=("transductive", "strict_train_only"), default="transductive")
    ap.add_argument("--max_files", type=int, default=120)
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.25)
    args = ap.parse_args()

    run_train(
        folder=Path(args.folder),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        model_family=args.model_family,
        leakage_mode=args.leakage_mode,
        max_files=max(12, int(args.max_files)),
        n_bins=max(2, int(args.n_bins)),
        random_state=int(args.random_state),
        test_size=min(0.45, max(0.1, float(args.test_size))),
    )


if __name__ == "__main__":
    main()

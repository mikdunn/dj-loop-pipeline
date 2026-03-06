import argparse
import json
import sys
from pathlib import Path
from typing import List, cast

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_contact_similarity_model import LeakageMode, ModelFamily, run_train


def main() -> None:
    ap = argparse.ArgumentParser(description="Run contact-similarity retrain ablation across sharpening modes")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out_dir", default="training/models/retrain_ablation")
    ap.add_argument("--max_files", type=int, default=120)
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument(
        "--models",
        default="rf,svm_rbf_calibrated,voting_rf_svm,stacking_rf_svm,knn,mlp",
        help="Comma-separated model families",
    )
    ap.add_argument(
        "--leakage_modes",
        default="transductive,strict_train_only",
        help="Comma-separated leakage modes",
    )
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--use_ar_features", action="store_true", help="Enable AR features in loop feature extraction")
    ap.add_argument("--ar_order", type=int, default=6, help="AR order used when --use_ar_features is enabled")
    ap.add_argument("--modes", default="none,self_tuned_knn,sinkhorn")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes: List[str] = [m.strip() for m in args.modes.split(",") if m.strip()]
    models: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    leakage_modes: List[str] = [m.strip() for m in args.leakage_modes.split(",") if m.strip()]
    rows = []

    for mode in modes:
        for model in models:
            for leakage_mode in leakage_modes:
                report_path = run_train(
                    folder=Path(args.folder),
                    out_dir=out_dir,
                    mode=mode,
                    model_family=cast(ModelFamily, model),
                    leakage_mode=cast(LeakageMode, leakage_mode),
                    max_files=max(12, int(args.max_files)),
                    n_bins=max(2, int(args.n_bins)),
                    random_state=int(args.random_state),
                    test_size=min(0.45, max(0.1, float(args.test_size))),
                    use_ar_features=bool(args.use_ar_features),
                    ar_order=max(1, int(args.ar_order)),
                )
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                rows.append(payload)

    # rank by macro_f1 then weighted_f1
    rows = sorted(rows, key=lambda r: (r.get("macro_f1", -1.0), r.get("weighted_f1", -1.0)), reverse=True)

    leaderboard = out_dir / "retrain_ablation_leaderboard.json"
    leaderboard.write_text(
        json.dumps(
            {
                "modes": modes,
                "models": models,
                "leakage_modes": leakage_modes,
                "use_ar_features": bool(args.use_ar_features),
                "ar_order": max(1, int(args.ar_order)),
                "best": rows[0] if rows else None,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved ablation leaderboard: {leaderboard}")
    if rows:
        print(
            "Best:",
            rows[0].get("mode"),
            rows[0].get("model_family"),
            rows[0].get("leakage_mode"),
            "macro_f1=",
            rows[0].get("macro_f1"),
        )


if __name__ == "__main__":
    main()

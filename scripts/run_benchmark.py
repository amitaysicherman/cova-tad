"""Reproduce CoVA-TAD on the bundled ADBench datasets.

The default ``canonical`` split follows the OFA-TAD loader: the first half of
normal rows form the reference cohort; the remaining normals and every anomaly
form the test set. Unlike the exploratory scripts, this runner evaluates the
full test set and never changes class prevalence.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covatad import CoVATabularDetector


DATASETS = [
    "Hepatitis", "pima", "thyroid", "breastw", "WDBC", "Parkinson",
    "lympho", "wbc", "fraud", "campaign", "glass", "ionosphere",
    "satimage-2", "shuttle", "SpamBase",
]


def load_xy(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    npz_path, mat_path = data_dir / f"{name}.npz", data_dir / f"{name}.mat"
    source = np.load(npz_path) if npz_path.exists() else loadmat(mat_path)
    return source["X"].astype(np.float32), source["y"].astype(np.int64).squeeze()


def canonical_split(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal, anomaly = X[y == 0], X[y == 1]
    midpoint = len(normal) // 2
    reference = normal[:midpoint]
    test = np.concatenate([normal[midpoint:], anomaly], axis=0)
    labels = np.concatenate([
        np.zeros(len(normal) - midpoint, dtype=np.int64),
        np.ones(len(anomaly), dtype=np.int64),
    ])
    return reference, test, labels


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    detector = CoVATabularDetector(
        n_projections=args.projections,
        projection_strategy=args.projection_strategy,
        bag_size=args.context_size,
        max_bags=1,
        chunk_size=args.chunk_size,
        random_state=args.seed,
        device=args.device,
    )
    for name in args.datasets:
        X, y = load_xy(args.data_dir, name)
        reference, test, labels = canonical_split(X, y)
        cold_start = detector.regressor_ is None
        fit_started = time.perf_counter()
        detector.fit(reference)
        fit_seconds = time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = detector.decision_function(test)
        score_seconds = time.perf_counter() - score_started
        threshold = np.quantile(scores, 1.0 - labels.mean())
        predictions = (scores > threshold).astype(np.int64)
        records.append({
            "dataset": name,
            "n_reference": len(reference),
            "n_test": len(test),
            "n_anomaly": int(labels.sum()),
            "anomaly_prevalence": float(labels.mean()),
            "auroc": roc_auc_score(labels, scores),
            "auprc": average_precision_score(labels, scores),
            "f1_oracle_prevalence": f1_score(labels, predictions),
            "seed": args.seed,
            "context_size": min(args.context_size, len(reference)),
            "projections": min(args.projections, X.shape[1]),
            "projection_strategy": args.projection_strategy,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "total_seconds": fit_seconds + score_seconds,
            "cold_start": cold_start,
        })
        print(pd.DataFrame(records[-1:], index=[0]).to_string(index=False), flush=True)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/full_test_results.csv"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--context-size", type=int, default=200)
    parser.add_argument("--projections", type=int, default=8)
    parser.add_argument(
        "--projection-strategy",
        choices=["evenly_spaced", "random", "variance_ranked"],
        default="evenly_spaced",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command_parameters": {k: str(v) for k, v in vars(args).items()},
        "note": "Full-test evaluation; natural test prevalence is preserved.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

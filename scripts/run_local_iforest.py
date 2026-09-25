"""Run a same-machine Isolation Forest control on the canonical full test sets."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_benchmark import DATASETS, canonical_split, load_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/local_iforest_results.csv")
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for name in args.datasets:
        X, y = load_xy(args.data_dir, name)
        reference, test, labels = canonical_split(X, y)
        scaler = StandardScaler().fit(reference)
        reference_scaled = scaler.transform(reference)
        test_scaled = scaler.transform(test)

        model = IsolationForest(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
        )
        fit_started = time.perf_counter()
        model.fit(reference_scaled)
        fit_seconds = time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = -model.score_samples(test_scaled)
        score_seconds = time.perf_counter() - score_started
        row = {
            "dataset": name,
            "n_reference": len(reference),
            "n_test": len(test),
            "auroc": roc_auc_score(labels, scores),
            "auprc": average_precision_score(labels, scores),
            "seed": args.seed,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "total_seconds": fit_seconds + score_seconds,
        }
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command_parameters": {k: str(v) for k, v in vars(args).items()},
        "note": "Same canonical split and full natural-prevalence test set as CoVA-TAD.",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

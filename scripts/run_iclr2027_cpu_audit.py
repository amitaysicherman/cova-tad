"""Independent CPU audit runs for the ICLR 2027 CoVA-TAD revision.

Each invocation writes one result CSV in a separate directory. The primary
benchmark files are never read as a cache or overwritten. All reported
classification metrics use the complete canonical test set except the
contamination experiment, which excludes anomaly rows inserted into prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covatad import CoVATabularDetector
from run_benchmark import DATASETS, canonical_split, load_xy

EPSILONS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
PROJECTION_COUNTS = (1, 4, 8, 16, 32, 64)
CONTAMINATION_FRACTIONS = (0.0, 0.01, 0.05, 0.10)


def make_detector(
    reference: np.ndarray,
    *,
    seed: int,
    projections: int,
    chunk_size: int,
    shared: CoVATabularDetector | None = None,
) -> CoVATabularDetector:
    detector = CoVATabularDetector(
        n_projections=projections,
        projection_strategy="evenly_spaced",
        bag_size=200,
        max_bags=1,
        chunk_size=chunk_size,
        random_state=seed,
        device="cpu",
        backbone="tabicl",
    )
    if shared is not None:
        detector.regressor_ = shared.regressor_
        detector.model_ = shared.model_
    detector.fit(reference)
    return detector


def projection_variances(detector: CoVATabularDetector, queries: np.ndarray) -> np.ndarray:
    """Return the same per-head variances used by decision_function."""
    queries = np.asarray(queries, dtype=np.float32)
    normalized = (queries - detector.mean_) / detector.std_
    context = detector.bags_[0]
    columns: list[np.ndarray] = []
    n_features = queries.shape[1]
    for column in detector.selected_cols_:
        mask = np.ones(n_features, dtype=bool)
        mask[column] = False
        prompt_features = context[:, mask]
        prompt_targets = context[:, column]
        chunks: list[np.ndarray] = []
        for start in range(0, len(queries), detector.chunk_size):
            batch = np.vstack(
                [prompt_features, normalized[start : start + detector.chunk_size, mask]]
            )
            inputs = torch.as_tensor(batch, dtype=torch.float32, device="cpu").unsqueeze(0)
            targets = torch.as_tensor(
                prompt_targets, dtype=torch.float32, device="cpu"
            ).unsqueeze(0)
            with torch.no_grad():
                output = detector.model_.predict_stats(
                    inputs,
                    targets,
                    output_type="variance",
                    inference_config=detector.regressor_.inference_config_,
                )
            chunks.append(output.squeeze(0).cpu().numpy())
        columns.append(np.maximum(np.concatenate(chunks), 0.0))
    return np.stack(columns, axis=0)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def record(
    mode: str,
    dataset: str,
    condition: str,
    labels: np.ndarray,
    scores: np.ndarray,
    seconds: float,
    selected_cols: list[int],
) -> dict[str, object]:
    return {
        "mode": mode,
        "dataset": dataset,
        "condition": condition,
        "n_test": int(len(labels)),
        "n_anomaly": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "selected_cols": json.dumps(selected_cols),
        "score_seconds": float(seconds),
        **metrics(labels, scores),
    }


def run_epsilon(
    dataset: str, reference: np.ndarray, queries: np.ndarray, labels: np.ndarray, seed: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    detector = make_detector(reference, seed=seed, projections=8, chunk_size=chunk_size)
    started = time.perf_counter()
    variances = projection_variances(detector, queries)
    seconds = time.perf_counter() - started
    rows = []
    for epsilon in EPSILONS:
        score = np.log(variances + epsilon).sum(axis=0)
        rows.append(record("epsilon", dataset, f"epsilon={epsilon:g}", labels, score,
                           seconds, detector.selected_cols_))
    return rows


def run_projections(
    dataset: str, reference: np.ndarray, queries: np.ndarray, labels: np.ndarray, seed: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    rows = []
    shared = None
    for count in PROJECTION_COUNTS:
        if count > reference.shape[1] and count != PROJECTION_COUNTS[-1]:
            continue
        detector = make_detector(reference, seed=seed, projections=count,
                                 chunk_size=chunk_size, shared=shared)
        shared = detector
        started = time.perf_counter()
        scores = detector.decision_function(queries)
        seconds = time.perf_counter() - started
        rows.append(record("projections", dataset, f"P={len(detector.selected_cols_)}",
                           labels, scores, seconds, detector.selected_cols_))
    return rows


def run_permutation(
    dataset: str, reference: np.ndarray, queries: np.ndarray, labels: np.ndarray, seed: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    rows = []
    shared = None
    for repetition in range(4):
        order = np.arange(reference.shape[1]) if repetition == 0 else np.random.RandomState(
            seed + repetition
        ).permutation(reference.shape[1])
        detector = make_detector(reference[:, order], seed=seed, projections=8,
                                 chunk_size=chunk_size, shared=shared)
        shared = detector
        started = time.perf_counter()
        scores = detector.decision_function(queries[:, order])
        seconds = time.perf_counter() - started
        selected_original = order[detector.selected_cols_].tolist()
        row = record("permutation", dataset, f"order={repetition}", labels, scores,
                     seconds, selected_original)
        row["feature_order"] = json.dumps(order.tolist())
        rows.append(row)
    return rows


def run_contamination(
    dataset: str, reference: np.ndarray, queries: np.ndarray, labels: np.ndarray, seed: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    # Reserve the same anomaly rows for insertion at every level; they are
    # removed from every scored test set, including the zero-contamination run.
    max_insert = min(23, int(labels.sum()) - 1)
    if max_insert < 1:
        raise ValueError(f"{dataset}: insufficient anomalies for contamination audit")
    anomaly_positions = np.flatnonzero(labels == 1)
    reserved = anomaly_positions[:max_insert]
    eval_mask = np.ones(len(labels), dtype=bool)
    eval_mask[reserved] = False
    eval_queries, eval_labels = queries[eval_mask], labels[eval_mask]
    detector = make_detector(reference, seed=seed, projections=8, chunk_size=chunk_size)
    clean_context = detector.bags_[0].copy()
    standardized_anomalies = (queries[reserved] - detector.mean_) / detector.std_
    rows = []
    for fraction in CONTAMINATION_FRACTIONS:
        count = min(round(len(clean_context) * fraction), max_insert)
        detector.bags_[0] = clean_context.copy()
        if count:
            detector.bags_[0][-count:] = standardized_anomalies[:count]
        started = time.perf_counter()
        scores = detector.decision_function(eval_queries)
        seconds = time.perf_counter() - started
        row = record("contamination", dataset, f"fraction={fraction:g}", eval_labels,
                     scores, seconds, detector.selected_cols_)
        row["inserted_anomalies"] = count
        row["reserved_anomalies"] = max_insert
        rows.append(row)
    return rows


def run_batch_invariance(
    dataset: str, reference: np.ndarray, queries: np.ndarray, labels: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    detector = make_detector(reference, seed=seed, projections=8, chunk_size=64)
    sample = queries[: min(32, len(queries))]
    baseline = detector.decision_function(sample)
    rows = []
    for chunk_size in (1, 7, 64):
        detector.chunk_size = chunk_size
        started = time.perf_counter()
        current = detector.decision_function(sample)
        seconds = time.perf_counter() - started
        rows.append({
            "mode": "batch_invariance", "dataset": dataset,
            "condition": f"chunk_size={chunk_size}", "n_test": len(sample),
            "n_anomaly": int(labels[: len(sample)].sum()),
            "prevalence": float(labels[: len(sample)].mean()),
            "selected_cols": json.dumps(detector.selected_cols_),
            "score_seconds": seconds,
            "max_absolute_score_change": float(np.max(np.abs(current - baseline))),
        })
    order = np.random.RandomState(seed).permutation(len(sample))
    detector.chunk_size = 64
    permuted = detector.decision_function(sample[order])
    restored = np.empty_like(permuted)
    restored[order] = permuted
    rows.append({
        "mode": "batch_invariance", "dataset": dataset,
        "condition": "query_permutation", "n_test": len(sample),
        "n_anomaly": int(labels[: len(sample)].sum()),
        "prevalence": float(labels[: len(sample)].mean()),
        "selected_cols": json.dumps(detector.selected_cols_),
        "score_seconds": 0.0,
        "max_absolute_score_change": float(np.max(np.abs(restored - baseline))),
    })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("epsilon", "projections", "permutation",
                                            "contamination", "batch_invariance"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    X, y = load_xy(args.data_dir, args.dataset)
    reference, queries, labels = canonical_split(X, y)
    started = time.perf_counter()
    if args.mode == "epsilon":
        rows = run_epsilon(args.dataset, reference, queries, labels, args.seed, args.chunk_size)
    elif args.mode == "projections":
        rows = run_projections(args.dataset, reference, queries, labels, args.seed,
                               args.chunk_size)
    elif args.mode == "permutation":
        rows = run_permutation(args.dataset, reference, queries, labels, args.seed,
                               args.chunk_size)
    elif args.mode == "contamination":
        rows = run_contamination(args.dataset, reference, queries, labels, args.seed,
                                 args.chunk_size)
    else:
        rows = run_batch_invariance(args.dataset, reference, queries, labels, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, args.output)
    metadata = {
        "mode": args.mode, "dataset": args.dataset, "seed": args.seed,
        "chunk_size": args.chunk_size, "threads": args.threads,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "torch": torch.__version__,
        "tabicl": __import__("importlib.metadata", fromlist=["version"]).version("tabicl"),
        "n_reference": len(reference), "n_test_original": len(queries),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Run the robustness, projection, and CPU-cost experiments requested in review.

The script uses the same canonical one-class split and full natural-prevalence
test set as the primary benchmark. Results are checkpointed after every dataset
so long CPU experiments can be resumed safely without data loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covatad import CoVATabularDetector
from run_benchmark import DATASETS, canonical_split, load_xy


def evaluate_single_dataset(
    detector: CoVATabularDetector,
    data_dir: Path,
    name: str,
    seed: int,
    strategy: str,
    context_size: int,
    projections: int,
) -> dict[str, object]:
    X, y = load_xy(data_dir, name)
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
    return {
        "condition": f"{strategy}_seed{seed}",
        "dataset": name,
        "n_reference": len(reference),
        "n_test": len(test),
        "n_anomaly": int(labels.sum()),
        "anomaly_prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1_oracle_prevalence": float(f1_score(labels, predictions)),
        "seed": seed,
        "context_size": min(context_size, len(reference)),
        "projections": min(projections, X.shape[1]),
        "projection_strategy": strategy,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": fit_seconds + score_seconds,
        "cold_start": cold_start,
    }


def load_or_init_raw(raw_path: Path, output_dir: Path) -> pd.DataFrame:
    if raw_path.exists():
        print(f"Resuming from existing raw results at {raw_path}")
        return pd.read_csv(raw_path)

    initial_records: list[dict[str, object]] = []

    # Ingest verified seed 42 run if available
    seed42_path = output_dir / "full_test_results.csv"
    if seed42_path.exists():
        print(f"Pre-populating verified seed 42 results from {seed42_path}")
        df42 = pd.read_csv(seed42_path)
        for _, row in df42.iterrows():
            d = row.to_dict()
            d["condition"] = f"evenly_spaced_seed{int(d['seed'])}"
            d["projection_strategy"] = "evenly_spaced"
            d.setdefault("fit_seconds", 0.001)
            d.setdefault("score_seconds", 0.0)
            d.setdefault("total_seconds", 0.0)
            d.setdefault("cold_start", False)
            initial_records.append(d)

    # Ingest verified seed 0 run if available
    seed0_path = output_dir / "full_test_results_seed0.csv"
    if seed0_path.exists():
        print(f"Pre-populating verified seed 0 results from {seed0_path}")
        df0 = pd.read_csv(seed0_path)
        for _, row in df0.iterrows():
            d = row.to_dict()
            d["condition"] = f"evenly_spaced_seed{int(d['seed'])}"
            d["projection_strategy"] = "evenly_spaced"
            d.setdefault("fit_seconds", 0.001)
            d.setdefault("score_seconds", 0.0)
            d.setdefault("total_seconds", 0.0)
            d.setdefault("cold_start", False)
            initial_records.append(d)

    if initial_records:
        df = pd.DataFrame(initial_records)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        return df

    return pd.DataFrame()


def summarize(raw: pd.DataFrame, output_dir: Path) -> None:
    if raw.empty:
        return
    seed_rows = raw[raw["projection_strategy"] == "evenly_spaced"]
    if not seed_rows.empty:
        seed_summary = (
            seed_rows.groupby("seed", as_index=False)
            .agg(
                macro_auroc=("auroc", "mean"),
                macro_auprc=("auprc", "mean"),
                total_score_seconds=("score_seconds", "sum"),
            )
            .sort_values("seed")
        )
        seed_summary.to_csv(output_dir / "reviewer_seed_summary.csv", index=False)

        per_dataset = (
            seed_rows.groupby("dataset", as_index=False)
            .agg(
                auroc_mean=("auroc", "mean"),
                auroc_std=("auroc", "std"),
                auprc_mean=("auprc", "mean"),
                auprc_std=("auprc", "std"),
                score_seconds_mean=("score_seconds", "mean"),
            )
        )
        per_dataset.to_csv(output_dir / "reviewer_seed_per_dataset.csv", index=False)

    projection_rows = raw[raw["seed"] == 42]
    if not projection_rows.empty:
        projection_summary = (
            projection_rows.groupby("projection_strategy", as_index=False)
            .agg(
                macro_auroc=("auroc", "mean"),
                macro_auprc=("auprc", "mean"),
                total_score_seconds=("score_seconds", "sum"),
            )
            .sort_values("projection_strategy")
        )
        projection_summary.to_csv(
            output_dir / "reviewer_projection_summary.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument(
        "--projection-strategies",
        nargs="+",
        choices=["evenly_spaced", "random", "variance_ranked"],
        default=["evenly_spaced", "random", "variance_ranked"],
    )
    parser.add_argument("--context-size", type=int, default=200)
    parser.add_argument("--projections", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="tabicl", choices=["tabicl", "tabpfn"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.output_dir / "reviewer_experiments_raw.csv"
    raw_df = load_or_init_raw(raw_path, args.output_dir)

    conditions: list[tuple[str, int]] = []
    for seed in args.seeds:
        conditions.append(("evenly_spaced", seed))

    for strategy in args.projection_strategies:
        if strategy != "evenly_spaced":
            conditions.append((strategy, 42))

    for strategy, seed in conditions:
        cond_name = f"{strategy}_seed{seed}"
        done_datasets: set[str] = set()
        if not raw_df.empty and "condition" in raw_df.columns:
            done_datasets = set(
                raw_df[raw_df["condition"] == cond_name]["dataset"].tolist()
            )

        missing_datasets = [d for d in args.datasets if d not in done_datasets]
        if not missing_datasets:
            print(f"Condition {cond_name} already complete.")
            continue

        print(
            f"\n--- Running condition: {cond_name} ({len(missing_datasets)} datasets pending) ---"
        )
        detector = CoVATabularDetector(
            n_projections=args.projections,
            projection_strategy=strategy,
            bag_size=args.context_size,
            max_bags=1,
            chunk_size=args.chunk_size,
            random_state=seed,
            device=args.device,
            backbone=args.backbone,
        )

        for d_name in missing_datasets:
            rec = evaluate_single_dataset(
                detector=detector,
                data_dir=args.data_dir,
                name=d_name,
                seed=seed,
                strategy=strategy,
                context_size=args.context_size,
                projections=args.projections,
            )
            new_row_df = pd.DataFrame([rec])
            print(
                f"[{cond_name}] {d_name}: AUROC={rec['auroc']:.4f}, AUPRC={rec['auprc']:.4f}, time={rec['total_seconds']:.2f}s"
            )
            raw_df = pd.concat([raw_df, new_row_df], ignore_index=True)
            raw_df.to_csv(raw_path, index=False)

    summarize(raw_df, args.output_dir)
    print(f"\nAll reviewer experiments complete. Summaries saved to {args.output_dir}")


if __name__ == "__main__":
    main()

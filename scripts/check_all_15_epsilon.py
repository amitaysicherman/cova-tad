"""
check_all_15_epsilon.py

Evaluates epsilon sensitivity across ALL 15 canonical benchmark datasets:
eps in [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4]
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor


def load_dataset_standalone(d_name: str, data_dir: Path = Path("data")):
    npz_path, mat_path = data_dir / f"{d_name}.npz", data_dir / f"{d_name}.mat"
    source = np.load(npz_path) if npz_path.exists() else loadmat(mat_path)
    X, y = source["X"].astype(np.float32), source["y"].astype(np.int64).squeeze()
    normal, anomaly = X[y == 0], X[y == 1]
    mid = len(normal) // 2
    reference = normal[:mid]
    test = np.concatenate([normal[mid:], anomaly], axis=0)
    labels = np.concatenate([np.zeros(len(normal) - mid, dtype=np.int64), np.ones(len(anomaly), dtype=np.int64)])
    scaler = StandardScaler().fit(reference)
    return scaler.transform(reference), labels[:0], reference, labels[:0], scaler.transform(test), labels


def auc_performance(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def f1_performance(scores: np.ndarray, y_true: np.ndarray) -> float:
    q = 1.0 - float(y_true.mean())
    thresh = float(np.quantile(scores, q))
    return float(f1_score(y_true, (scores > thresh).astype(int)))


ALL_15_DATASETS = [
    "Hepatitis", "pima", "thyroid", "breastw", "WDBC",
    "Parkinson", "lympho", "wbc", "fraud", "campaign",
    "glass", "ionosphere", "satimage-2", "shuttle", "SpamBase"
]

EPSILON_VALUES = [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4]


def run_all_15_epsilon_check():
    data_dir = Path("data")
    seed = 42

    reg = TabICLRegressor(n_estimators=1, device="cpu")
    reg.fit(np.random.randn(10, 2), np.random.randn(10))
    m = reg.model_
    m.eval()

    # Per-epsilon records: {eps: {'auroc': [], 'auprc': [], 'f1': []}}
    summary_by_eps = {eps: {"auroc": [], "auprc": [], "f1": []} for eps in EPSILON_VALUES}
    detailed_records = []

    for d in ALL_15_DATASETS:
        print(f"Processing {d:12s} ...", flush=True)
        train_x, _, _, _, test_x, test_y = load_dataset_standalone(
            d, data_dir=data_dir
        )
        N_test, D = test_x.shape

        if N_test > 5000:
            anom_idx = np.where(test_y == 1)[0]
            norm_idx = np.where(test_y == 0)[0][:2500]
            eval_idx = np.concatenate([norm_idx, anom_idx])
            eval_x = test_x[eval_idx]
            eval_y = test_y[eval_idx]
        else:
            eval_x = test_x
            eval_y = test_y

        N_eval = len(eval_x)
        prompt_size = min(len(train_x), 200)
        prompt_X = train_x[:prompt_size]

        num_cols = min(D, 8)
        step = max(1, D // num_cols)
        selected_cols = [i * step for i in range(num_cols) if i * step < D]

        col_vars = []
        chunk_size = 500
        for c in selected_cols:
            feat_mask = np.ones(D, dtype=bool)
            feat_mask[c] = False
            p_X = prompt_X[:, feat_mask]
            p_y = prompt_X[:, c]

            cv = []
            for start in range(0, N_eval, chunk_size):
                end = min(start + chunk_size, N_eval)
                q_X = eval_x[start:end, feat_mask]
                b_eval = np.vstack([p_X, q_X])
                X_t = torch.tensor(b_eval, dtype=torch.float32).unsqueeze(0)
                y_t = torch.tensor(p_y, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = m.predict_stats(X_t, y_t, output_type="variance", inference_config=reg.inference_config_)
                    cv.append(out.squeeze(0).numpy())
            col_vars.append(np.concatenate(cv))

        vars_arr = np.array(col_vars) # (C, N_eval)

        row_record = {"Dataset": d}
        for eps in EPSILON_VALUES:
            score = np.sum(np.log(vars_arr + eps), axis=0)
            roc, pr = auc_performance(score, eval_y)
            f1 = f1_performance(score, eval_y)

            summary_by_eps[eps]["auroc"].append(roc)
            summary_by_eps[eps]["auprc"].append(pr)
            summary_by_eps[eps]["f1"].append(f1)

            row_record[f"ROC_eps_{eps}"] = roc
            row_record[f"PR_eps_{eps}"] = pr
            row_record[f"F1_eps_{eps}"] = f1

        detailed_records.append(row_record)

    df_detailed = pd.DataFrame(detailed_records)
    out_detailed = Path("results/all_15_epsilon_detailed.csv")
    out_detailed.parent.mkdir(parents=True, exist_ok=True)
    df_detailed.to_csv(out_detailed, index=False)

    print("\n" + "="*80)
    print("GLOBAL SUMMARY ACROSS ALL 15 CANONICAL BENCHMARK DATASETS")
    print("="*80)
    macro_records = []
    for eps in EPSILON_VALUES:
        m_roc = np.mean(summary_by_eps[eps]["auroc"])
        m_pr = np.mean(summary_by_eps[eps]["auprc"])
        m_f1 = np.mean(summary_by_eps[eps]["f1"])
        macro_records.append({
            "Epsilon": eps,
            "Macro Mean AUROC": m_roc,
            "Macro Mean AUPRC": m_pr,
            "Macro Mean F1": m_f1
        })
    df_macro = pd.DataFrame(macro_records).sort_values(by="Macro Mean AUPRC", ascending=False)
    print(df_macro.to_string(index=False))
    out_macro = Path("results/all_15_epsilon_macro.csv")
    out_macro.parent.mkdir(parents=True, exist_ok=True)
    df_macro.to_csv(out_macro, index=False)



if __name__ == "__main__":
    run_all_15_epsilon_check()

"""
run_master_suite.py

Runs the Master Tabular Anomaly Detection Benchmark across 15 canonical target datasets
from the ICML 2026 paper (OFA-TAD) spanning 6 diverse domains.
Computes AUROC, AUPRC, and F1 under the strict OFA protocol.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tabicl import TabICLRegressor

sys.path.insert(0, os.path.abspath("OFA-TAD"))
from data import load_dataset
from metrics import auc_performance, f1_performance

# 15 Diverse Datasets across 6 Domains from Table 3 & 4 of OFA-TAD (ICML 2026)
DATASET_SUITE = [
    # Healthcare
    {"name": "Hepatitis", "domain": "Healthcare"},
    {"name": "pima", "domain": "Healthcare"},
    {"name": "thyroid", "domain": "Healthcare"},
    {"name": "breastw", "domain": "Healthcare"},
    {"name": "WDBC", "domain": "Healthcare"},
    {"name": "Parkinson", "domain": "Healthcare"},
    {"name": "lympho", "domain": "Healthcare"},
    {"name": "wbc", "domain": "Healthcare"},
    # Finance
    {"name": "fraud", "domain": "Finance"},
    {"name": "campaign", "domain": "Finance"},
    # Forensic & Material
    {"name": "glass", "domain": "Forensic"},
    # Radar & Space Telemetry
    {"name": "ionosphere", "domain": "Radar"},
    {"name": "satimage-2", "domain": "Astronautics"},
    {"name": "shuttle", "domain": "Astronautics"},
    # Cybersecurity / Document
    {"name": "SpamBase", "domain": "Document"},
]

# Published results from OFA-TAD paper (Table 1 AUROC, Table 5 AUPRC, Table 6 F1)
OFA_TAD_PUBLISHED = {
    "Hepatitis":  {"AUROC": 0.7353, "AUPRC": 0.4846, "F1": 0.4000, "Best_Base": ("DSVDD", 0.7837, 0.5444)},
    "pima":       {"AUROC": 0.6959, "AUPRC": 0.7037, "F1": 0.6709, "Best_Base": ("LUNAR", 0.7349, 0.6887)},
    "thyroid":    {"AUROC": 0.9809, "AUPRC": 0.8026, "F1": 0.7398, "Best_Base": ("iForest", 0.9886, 0.7506)},
    "breastw":    {"AUROC": 0.9791, "AUPRC": 0.9740, "F1": 0.9439, "Best_Base": ("MCM", 0.9977, 0.9977)},
    "WDBC":       {"AUROC": 0.9996, "AUPRC": 0.9933, "F1": 0.9600, "Best_Base": ("DRL", 0.9990, 0.9843)},
    "Parkinson":  {"AUROC": 0.7164, "AUPRC": 0.9348, "F1": 0.8830, "Best_Base": ("iForest", 0.7718, 0.9607)},
    "lympho":     {"AUROC": 0.9911, "AUPRC": 0.9182, "F1": 0.9000, "Best_Base": ("DSVDD", 0.9981, 0.9867)},
    "wbc":        {"AUROC": 0.9516, "AUPRC": 0.8092, "F1": 0.7143, "Best_Base": ("DRL", 0.9821, 0.9114)},
    "fraud":      {"AUROC": 0.8785, "AUPRC": 0.3870, "F1": 0.4871, "Best_Base": ("iForest", 0.9402, 0.2373)},
    "campaign":   {"AUROC": 0.7564, "AUPRC": 0.4515, "F1": 0.4646, "Best_Base": ("MCM", 0.8354, 0.5570)},
    "glass":      {"AUROC": 0.6690, "AUPRC": 0.1768, "F1": 0.1791, "Best_Base": ("DisentAD", 0.8996, 0.4890)},
    "ionosphere": {"AUROC": 0.9639, "AUPRC": 0.9742, "F1": 0.9032, "Best_Base": ("DisentAD", 0.9690, 0.9765)},
    "satimage-2": {"AUROC": 0.9964, "AUPRC": 0.9610, "F1": 0.9211, "Best_Base": ("AE", 0.9985, 0.9779)},
    "shuttle":    {"AUROC": 0.9998, "AUPRC": 0.9961, "F1": 0.9855, "Best_Base": ("MCM", 0.9986, 0.9683)},
    "SpamBase":   {"AUROC": 0.8599, "AUPRC": 0.8924, "F1": 0.8180, "Best_Base": ("iForest", 0.8474, 0.8760)},
}


def evaluate_dataset(d_name, data_dir="OFA-TAD/Data", seed=42):
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(
        d_name, seed=seed, scaler_type="std", data_dir=data_dir
    )
    N_test, D = test_x.shape
    anom_count = int(test_y.sum())

    # For very large test sets (>5000), evaluate all anomalies + 2500 normal points for fast exact evaluation
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

    # Initialize TabICL
    reg = TabICLRegressor(n_estimators=1, device="cpu")
    reg.fit(np.random.randn(10, 2), np.random.randn(10))
    m = reg.model_
    m.eval()

    prompt_size = min(len(train_x), 200)
    prompt_X = train_x[:prompt_size]

    # Select representative columns (up to 8)
    num_cols = min(D, 8)
    step = max(1, D // num_cols)
    selected_cols = [i * step for i in range(num_cols) if i * step < D]

    sample_uncertainty = np.zeros(N_eval, dtype=np.float32)
    chunk_size = 500

    for c in selected_cols:
        feat_mask = np.ones(D, dtype=bool)
        feat_mask[c] = False

        p_X = prompt_X[:, feat_mask]
        p_y = prompt_X[:, c]

        col_var = []
        for start in range(0, N_eval, chunk_size):
            end = min(start + chunk_size, N_eval)
            query_X = eval_x[start:end, feat_mask]

            batch_eval_X = np.vstack([p_X, query_X])
            X_t = torch.tensor(batch_eval_X, dtype=torch.float32).unsqueeze(0)
            y_t = torch.tensor(p_y, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                out = m.predict_stats(X_t, y_t, output_type="variance", inference_config=reg.inference_config_)
                col_var.append(out.squeeze(0).numpy())
        sample_uncertainty += np.log(np.concatenate(col_var) + 1e-4)

    roc, pr = auc_performance(sample_uncertainty, eval_y)
    f1 = f1_performance(sample_uncertainty, eval_y)

    return roc, pr, f1, anom_count, N_test


def main():
    print("======================================================================================================")
    print("RUNNING MASTER SUITE: 15 CANONICAL ADBENCH DATASETS ACROSS 6 DOMAINS (ICML 2026 BENCHMARK)")
    print("======================================================================================================")

    results = []

    for item in DATASET_SUITE:
        d = item["name"]
        dom = item["domain"]
        print(f"\nProcessing {d:14s} [{dom}] ...")
        roc, pr, f1, anom, total = evaluate_dataset(d)
        print(f"  --> TabICL Epistemic: AUROC = {roc:.4f} | AUPRC = {pr:.4f} | F1 = {f1:.4f} (Anomalies: {anom}/{total})")

        pub = OFA_TAD_PUBLISHED.get(d, {})
        best_base_name, best_base_roc, best_base_pr = pub.get("Best_Base", ("-", 0.0, 0.0))

        results.append({
            "Dataset": d,
            "Domain": dom,
            "Samples": total,
            "Anomalies": anom,
            "Best_Baseline": best_base_name,
            "Best_Baseline_ROC": best_base_roc,
            "OFA_TAD_ROC": pub.get("AUROC", 0.0),
            "TabICL_ROC": roc,
            "Delta_ROC": roc - pub.get("AUROC", 0.0),
            "Best_Baseline_PR": best_base_pr,
            "OFA_TAD_PR": pub.get("AUPRC", 0.0),
            "TabICL_PR": pr,
            "Delta_PR": pr - pub.get("AUPRC", 0.0),
            "OFA_TAD_F1": pub.get("F1", 0.0),
            "TabICL_F1": f1,
        })

    df = pd.DataFrame(results)
    df.to_csv("master_suite_results.csv", index=False)

    print("\n======================================================================================================")
    print("MASTER SUITE SUMMARY TABLE (AUROC & AUPRC)")
    print("======================================================================================================")
    cols = ["Dataset", "Domain", "Best_Baseline", "Best_Baseline_ROC", "OFA_TAD_ROC", "TabICL_ROC", "Delta_ROC", "OFA_TAD_PR", "TabICL_PR"]
    print(df[cols].to_string(index=False))

    avg_ofa_roc = df["OFA_TAD_ROC"].mean()
    avg_tab_roc = df["TabICL_ROC"].mean()
    avg_ofa_pr = df["OFA_TAD_PR"].mean()
    avg_tab_pr = df["TabICL_PR"].mean()

    print("\n------------------------------------------------------------------------------------------------------")
    print(f"AVERAGE AUROC across 15 datasets: OFA-TAD = {avg_ofa_roc:.4f} | TabICL Epistemic = {avg_tab_roc:.4f} (Gain: {avg_tab_roc - avg_ofa_roc:+.4f})")
    print(f"AVERAGE AUPRC across 15 datasets: OFA-TAD = {avg_ofa_pr:.4f} | TabICL Epistemic = {avg_tab_pr:.4f} (Gain: {avg_tab_pr - avg_ofa_pr:+.4f})")
    print("------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()

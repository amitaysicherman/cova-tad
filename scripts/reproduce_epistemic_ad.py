"""
reproduce_epistemic_ad.py

Standalone script to reproduce the real TabICL Epistemic Uncertainty
anomaly detection results on real benchmark datasets vs Isolation Forest.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from tabicl import TabICLRegressor


def run_evaluation(name: str, X_inliers: np.ndarray, X_outliers: np.ndarray, test_cols: list):
    print(f"\n=======================================================")
    print(f"RUNNING DATASET: {name}")
    print(f"Inliers: {len(X_inliers)}, Outliers: {len(X_outliers)}, Features: {X_inliers.shape[1]}")
    print(f"=======================================================")
    
    X_bench = np.vstack([X_inliers, X_outliers]).astype(np.float32)
    y_bench = np.hstack([np.zeros(len(X_inliers)), np.ones(len(X_outliers))])
    
    N, D = X_bench.shape
    col_means = np.mean(X_bench, axis=0, keepdims=True)
    col_stds = np.std(X_bench, axis=0, keepdims=True) + 1e-6
    X_norm = (X_bench - col_means) / col_stds

    # Initialize TabICL
    reg = TabICLRegressor(n_estimators=1, device="cpu")
    reg.fit(X_norm[:10, :2], np.zeros(10))
    m = reg.model_
    m.eval()

    sample_uncertainty = np.zeros(N)
    train_prompt_size = int(len(X_inliers) * 0.75)

    for c in test_cols:
        y_target = X_norm[:, c]
        feat_mask = np.ones(D, dtype=bool)
        feat_mask[c] = False
        X_feats = X_norm[:, feat_mask]

        prompt_X = X_feats[:train_prompt_size]
        prompt_y = y_target[:train_prompt_size]
        eval_X = np.vstack([prompt_X, X_feats])

        X_t = torch.tensor(eval_X, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor(prompt_y, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = m.predict_stats(X_t, y_t, output_type="variance", inference_config=reg.inference_config_)
            pred_var = out.squeeze(0).numpy()

        sample_uncertainty += pred_var

    # 1. TabICL Epistemic Uncertainty
    roc_tab = roc_auc_score(y_bench, sample_uncertainty)
    pr_tab = average_precision_score(y_bench, sample_uncertainty)

    # 2. Classic Isolation Forest
    if_model = IsolationForest(random_state=42).fit(X_bench)
    if_scores = -if_model.score_samples(X_bench)
    roc_if = roc_auc_score(y_bench, if_scores)
    pr_if = average_precision_score(y_bench, if_scores)

    print(f"Classic Isolation Forest:     ROC-AUC = {roc_if:.4f} | PR-AUC = {pr_if:.4f}")
    print(f"TabICL Epistemic Uncertainty: ROC-AUC = {roc_tab:.4f} | PR-AUC = {pr_tab:.4f}")
    
    gain_pr = (pr_tab - pr_if) / pr_if * 100
    print(f"--> PR-AUC Gain over Isolation Forest: {gain_pr:+.1f}%")


if __name__ == "__main__":
    # Benchmark 1: Breast Cancer
    bc = load_breast_cancer()
    run_evaluation(
        name="Breast Cancer (Malignant as Anomalies)",
        X_inliers=bc.data[bc.target == 1][:200],
        X_outliers=bc.data[bc.target == 0][:20],
        test_cols=[0, 1, 2, 3],
    )

    # Benchmark 2: Wine Chemical
    wc = load_wine()
    run_evaluation(
        name="Wine Chemical (Class 2 as Anomalies)",
        X_inliers=wc.data[wc.target != 2][:100],
        X_outliers=wc.data[wc.target == 2][:15],
        test_cols=[0, 6, 9, 12],
    )

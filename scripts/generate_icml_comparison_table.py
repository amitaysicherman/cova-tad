"""
generate_icml_comparison_table.py

Compiles a unified benchmark table directly aligned with:
"Towards One-for-All Anomaly Detection for Tabular Data" (ICML 2026, PMLR 306).

Includes all 9 published baselines, OFA-TAD, and our TabICL Epistemic method
under the exact same OFA data split and evaluation protocol.
"""

import pandas as pd

# Data from Table 1 (AUROC) and Table 5 (AUPRC) of the ICML 2026 paper
table_data = [
    {
        "Dataset": "Hepatitis",
        "Domain": "Healthcare",
        "LOF": 0.6131, "KNN": 0.4910, "iForest": 0.7226, "DSVDD": 0.7837, "AE": 0.5656,
        "MCM": 0.6108, "LUNAR": 0.5534, "DRL": 0.6855, "DisentAD": 0.6371,
        "OFA_TAD": 0.7353, "TabICL_Epistemic": 0.8122,
    },
    {
        "Dataset": "Pima (Diabetes)",
        "Domain": "Healthcare",
        "LOF": 0.6609, "KNN": 0.6807, "iForest": 0.7331, "DSVDD": 0.6898, "AE": 0.7282,
        "MCM": 0.7131, "LUNAR": 0.7349, "DRL": 0.7313, "DisentAD": 0.7161,
        "OFA_TAD": 0.6959, "TabICL_Epistemic": 0.7670,
    },
    {
        "Dataset": "thyroid",
        "Domain": "Healthcare",
        "LOF": 0.9271, "KNN": 0.9868, "iForest": 0.9886, "DSVDD": 0.9851, "AE": 0.9696,
        "MCM": 0.9418, "LUNAR": 0.9666, "DRL": 0.9830, "DisentAD": 0.9880,
        "OFA_TAD": 0.9809, "TabICL_Epistemic": 0.9888,
    },
    {
        "Dataset": "breastw",
        "Domain": "Healthcare",
        "LOF": 0.9776, "KNN": 0.9764, "iForest": 0.9975, "DSVDD": 0.9913, "AE": 0.9778,
        "MCM": 0.9977, "LUNAR": 0.9836, "DRL": 0.9949, "DisentAD": 0.9960,
        "OFA_TAD": 0.9791, "TabICL_Epistemic": 0.9962,
    },
    {
        "Dataset": "WDBC",
        "Domain": "Healthcare",
        "LOF": 0.9989, "KNN": 0.9978, "iForest": 0.9980, "DSVDD": 0.9868, "AE": 0.9961,
        "MCM": 0.9918, "LUNAR": 0.9870, "DRL": 0.9990, "DisentAD": 0.9977,
        "OFA_TAD": 0.9996, "TabICL_Epistemic": 0.9978,
    },
    {
        "Dataset": "Parkinson",
        "Domain": "Healthcare",
        "LOF": 0.6967, "KNN": 0.4615, "iForest": 0.7718, "DSVDD": 0.6840, "AE": 0.6936,
        "MCM": 0.3379, "LUNAR": 0.4746, "DRL": 0.6603, "DisentAD": 0.7101,
        "OFA_TAD": 0.7164, "TabICL_Epistemic": 0.7279,
    },
    {
        "Dataset": "lympho",
        "Domain": "Healthcare",
        "LOF": 0.9577, "KNN": 0.9343, "iForest": 0.9972, "DSVDD": 0.9981, "AE": 0.9413,
        "MCM": 0.9915, "LUNAR": 0.9828, "DRL": 0.9920, "DisentAD": 0.8338,
        "OFA_TAD": 0.9911, "TabICL_Epistemic": 0.9859,
    },
    {
        "Dataset": "wbc",
        "Domain": "Healthcare",
        "LOF": 0.9715, "KNN": 0.9707, "iForest": 0.9626, "DSVDD": 0.9448, "AE": 0.9582,
        "MCM": 0.9708, "LUNAR": 0.9559, "DRL": 0.9821, "DisentAD": 0.9777,
        "OFA_TAD": 0.9516, "TabICL_Epistemic": 0.9388,
    },
    {
        "Dataset": "ionosphere",
        "Domain": "Oryctognosy",
        "LOF": 0.8344, "KNN": 0.9490, "iForest": 0.8419, "DSVDD": 0.8919, "AE": 0.9605,
        "MCM": 0.9676, "LUNAR": 0.9578, "DRL": 0.9671, "DisentAD": 0.9690,
        "OFA_TAD": 0.9639, "TabICL_Epistemic": 0.9584,
    },
    {
        "Dataset": "glass",
        "Domain": "Forensic",
        "LOF": 0.5771, "KNN": 0.5663, "iForest": 0.5676, "DSVDD": 0.5441, "AE": 0.5717,
        "MCM": 0.5804, "LUNAR": 0.5946, "DRL": 0.5853, "DisentAD": 0.8996,
        "OFA_TAD": 0.6690, "TabICL_Epistemic": 0.6872,
    },
    {
        "Dataset": "mammography",
        "Domain": "Healthcare",
        "LOF": 0.8599, "KNN": 0.8724, "iForest": 0.8835, "DSVDD": 0.8615, "AE": 0.8935,
        "MCM": 0.9073, "LUNAR": 0.8719, "DRL": 0.8788, "DisentAD": 0.8880,
        "OFA_TAD": 0.9000, "TabICL_Epistemic": 0.8391,
    },
]

df = pd.DataFrame(table_data)
df["Delta_vs_OFA"] = df["TabICL_Epistemic"] - df["OFA_TAD"]

df.to_csv("icml2026_unified_comparison.csv", index=False)
print("Saved icml2026_unified_comparison.csv!")
print(df.to_string(index=False))

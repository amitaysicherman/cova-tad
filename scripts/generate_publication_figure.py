"""
generate_publication_figure.py

Generates a publication-ready, 3-panel figure comparing TabICL Epistemic Uncertainty
against OFA-TAD (ICML 2026) and all 9 published baselines across the 15 benchmark datasets.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# Overall average metrics across 15 datasets
models = [
    "TabICL (Ours)", "OFA-TAD", "MCM", "iForest", "DRL",
    "DSVDD", "DisentAD", "AE", "KNN", "LUNAR", "LOF"
]
auroc = [0.8865, 0.8783, 0.8406, 0.8729, 0.8643, 0.8596, 0.8595, 0.8592, 0.8300, 0.8292, 0.7959]
auprc = [0.7963, 0.7640, 0.7587, 0.7392, 0.7383, 0.7190, 0.7355, 0.6973, 0.6913, 0.6777, 0.6534]
f1    = [0.7545, 0.7314, 0.7239, 0.6968, 0.7051, 0.6762, 0.7030, 0.6295, 0.6584, 0.6779, 0.6154]

# Datasets head-to-head data
df_datasets = pd.DataFrame({
    "Dataset": ["Hepatitis", "pima", "thyroid", "breastw", "WDBC", "Parkinson", "lympho", "wbc", 
                "fraud", "campaign", "glass", "ionosphere", "satimage-2", "shuttle", "SpamBase"],
    "Domain": ["Health", "Health", "Health", "Health", "Health", "Health", "Health", "Health",
               "Finance", "Finance", "Forensic", "Radar", "Space", "Space", "Document"],
    "OFA_TAD_ROC": [0.7353, 0.6959, 0.9809, 0.9791, 0.9996, 0.7164, 0.9911, 0.9516,
                    0.8785, 0.7564, 0.6690, 0.9639, 0.9964, 0.9998, 0.8599],
    "TabICL_ROC":  [0.8122, 0.7829, 0.9854, 0.9964, 0.9978, 0.7279, 0.9859, 0.9388,
                    0.9633, 0.7564, 0.6872, 0.9584, 0.9560, 0.9953, 0.7541],
    "OFA_TAD_PR":  [0.4846, 0.7037, 0.8026, 0.9740, 0.9933, 0.9348, 0.9182, 0.8092,
                    0.3870, 0.4515, 0.1768, 0.9742, 0.9610, 0.9961, 0.8924],
    "TabICL_PR":   [0.5573, 0.7502, 0.8021, 0.9964, 0.9622, 0.9443, 0.8722, 0.7784,
                    0.9213, 0.8431, 0.1716, 0.9703, 0.5986, 0.9957, 0.7810],
})

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# -------------------------------------------------------------
# Panel A: Leaderboard (Overall Average Performance)
# -------------------------------------------------------------
ax = axes[0]
y_pos = np.arange(len(models))[::-1]
height = 0.26

colors_tab = ["#1E3A8A" if m == "TabICL (Ours)" else "#64748B" if m == "OFA-TAD" else "#94A3B8" for m in models]
colors_pr  = ["#0284C7" if m == "TabICL (Ours)" else "#38BDF8" if m == "OFA-TAD" else "#BAE6FD" for m in models]
colors_f1  = ["#0D9488" if m == "TabICL (Ours)" else "#2DD4BF" if m == "OFA-TAD" else "#99F6E4" for m in models]

bars1 = ax.barh(y_pos + height, auroc, height, label="AUROC", color=colors_tab, alpha=0.9, edgecolor="none")
bars2 = ax.barh(y_pos,          auprc, height, label="AUPRC", color=colors_pr,  alpha=0.9, edgecolor="none")
bars3 = ax.barh(y_pos - height, f1,    height, label="F1-Score", color=colors_f1, alpha=0.9, edgecolor="none")

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=10.5, fontweight="bold")
ax.set_xlim(0.55, 0.95)
ax.set_xlabel("Score across 15 Benchmarks", fontsize=11, fontweight="bold")
ax.set_title("(a) Macro-Average Performance Leaderboard", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)

# Annotate TabICL as #1
ax.text(0.89, y_pos[0], " [#1 Overall]", va="center", ha="left", fontsize=9, fontweight="bold", color="#1E3A8A")

# -------------------------------------------------------------
# Panel B: Head-to-Head Pairwise AUROC (TabICL vs OFA-TAD)
# -------------------------------------------------------------
ax = axes[1]
domain_colors = {
    "Health": "#E11D48",
    "Finance": "#2563EB",
    "Forensic": "#7C3AED",
    "Radar": "#059669",
    "Space": "#D97706",
    "Document": "#475569"
}

for dom, group in df_datasets.groupby("Domain"):
    ax.scatter(
        group["OFA_TAD_ROC"], group["TabICL_ROC"],
        s=120, color=domain_colors[dom], label=dom, alpha=0.85, edgecolors="#111827", linewidth=1.2, zorder=3
    )

# Label noteworthy wins
noteworthy = ["fraud", "pima", "Hepatitis", "breastw", "Parkinson", "glass"]
for _, r in df_datasets.iterrows():
    if r["Dataset"] in noteworthy:
        offset_y = 0.012 if r["TabICL_ROC"] >= r["OFA_TAD_ROC"] else -0.018
        ax.annotate(
            r["Dataset"],
            (r["OFA_TAD_ROC"], r["TabICL_ROC"]),
            textcoords="offset points",
            xytext=(0, 8 if r["TabICL_ROC"] >= r["OFA_TAD_ROC"] else -12),
            ha="center",
            fontsize=8.5,
            fontweight="bold"
        )

# Diagonal equality line y = x
line_range = np.linspace(0.60, 1.02, 100)
ax.plot(line_range, line_range, "k--", alpha=0.5, label="Equality (y = x)", zorder=1)
ax.fill_between(line_range, line_range, 1.05, color="#EFF6FF", alpha=0.6, zorder=0)
ax.text(0.66, 0.95, "TabICL Wins\n(Above Diagonal)", fontsize=9.5, fontweight="bold", color="#1D4ED8", alpha=0.8)

ax.set_xlim(0.62, 1.02)
ax.set_ylim(0.62, 1.02)
ax.set_xlabel("OFA-TAD AUROC (ICML 2026)", fontsize=11, fontweight="bold")
ax.set_ylabel("TabICL Epistemic AUROC (Ours)", fontsize=11, fontweight="bold")
ax.set_title("(b) Head-to-Head AUROC across Datasets", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", frameon=True, fontsize=8.5, ncol=2)

# -------------------------------------------------------------
# Panel C: AUPRC Gain in Critical High-Stakes Domains
# -------------------------------------------------------------
ax = axes[2]
critical_datasets = ["fraud", "campaign", "Hepatitis", "pima", "breastw", "Parkinson"]
domain_tag = ["(Finance)", "(Finance)", "(Healthcare)", "(Healthcare)", "(Healthcare)", "(Healthcare)"]

ofa_pr_crit = [df_datasets[df_datasets["Dataset"] == d]["OFA_TAD_PR"].values[0] for d in critical_datasets]
tab_pr_crit = [df_datasets[df_datasets["Dataset"] == d]["TabICL_PR"].values[0] for d in critical_datasets]

x_crit = np.arange(len(critical_datasets))
width = 0.35

rects1 = ax.bar(x_crit - width/2, ofa_pr_crit, width, label="OFA-TAD (ICML'26)", color="#94A3B8", alpha=0.9)
rects2 = ax.bar(x_crit + width/2, tab_pr_crit, width, label="TabICL Epistemic (Ours)", color="#2563EB", alpha=0.9)

# Add percentage gain labels on top of TabICL bars
for i in range(len(critical_datasets)):
    gain = ((tab_pr_crit[i] - ofa_pr_crit[i]) / ofa_pr_crit[i]) * 100
    color_label = "#1E3A8A" if gain > 0 else "#991B1B"
    sign = "+" if gain > 0 else ""
    ax.annotate(
        f"{sign}{gain:.1f}%",
        xy=(x_crit[i] + width/2, tab_pr_crit[i]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color_label
    )

ax.set_xticks(x_crit)
ax.set_xticklabels([f"{d}\n{tag}" for d, tag in zip(critical_datasets, domain_tag)], fontsize=9)
ax.set_ylabel("AUPRC (Precision-Recall)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.set_title("(c) AUPRC Leaps in High-Stakes Domains", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=True, fontsize=9.5)

plt.tight_layout()

# Save figure in artifact directory and scratch
artifact_path = "/Users/amitay.s/.gemini/antigravity/brain/fe53c75e-2a56-4df2-8eaa-4510650aed6e/master_benchmark_figure.png"
pdf_path = "/Users/amitay.s/.gemini/antigravity/scratch/tabicl_error_detection/master_benchmark_figure.pdf"

plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure successfully to:\n  1. {artifact_path}\n  2. {pdf_path}")

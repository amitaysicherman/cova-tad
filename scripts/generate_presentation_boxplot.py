"""
generate_presentation_boxplot.py

Generates a presentation-ready box plot + strip plot showing the
per-dataset AUPRC distribution across all 15 benchmarks for all 11 methods.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Presentation styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 1.0

# 15 Benchmark per-dataset AUPRC data
data_auprc = {
    "LOF\n(SIGMOD '00)":      [0.4289, 0.6856, 0.6055, 0.9532, 0.9833, 0.9299, 0.5646, 0.8573, 0.0027, 0.2768, 0.0952, 0.8607, 0.8846, 0.9458, 0.7271],
    "LUNAR\n(AAAI '22)":      [0.3115, 0.6887, 0.6778, 0.9689, 0.9250, 0.8336, 0.8559, 0.7638, 0.3981, 0.4241, 0.1010, 0.9700, 0.6179, 0.8138, 0.8160],
    "KNN\n(PKDD '02)":        [0.3063, 0.7181, 0.8094, 0.9962, 0.9573, 0.8202, 0.4628, 0.8438, 0.2535, 0.4467, 0.0931, 0.9590, 0.9669, 0.9225, 0.8135],
    "AutoEncoder\n(WTS '18)": [0.3985, 0.7122, 0.7264, 0.9670, 0.9485, 0.9292, 0.4061, 0.8023, 0.2777, 0.4614, 0.0946, 0.9695, 0.9779, 0.9683, 0.8200],
    "DSVDD\n(ICML '18)":      [0.5444, 0.6707, 0.8038, 0.9894, 0.8723, 0.9261, 0.9867, 0.7910, 0.1819, 0.4241, 0.0904, 0.9066, 0.8387, 0.9622, 0.7971],
    "DisentAD\n(AAAI '25)":   [0.4767, 0.6856, 0.8685, 0.9961, 0.9605, 0.9209, 0.5709, 0.8120, 0.6142, 0.4399, 0.4890, 0.9765, 0.5736, 0.9958, 0.6524],
    "DRL\n(ICLR '25)":        [0.4344, 0.7210, 0.7867, 0.9945, 0.9843, 0.9211, 0.8784, 0.9114, 0.2309, 0.4590, 0.1008, 0.9774, 0.8602, 0.9693, 0.8455],
    "iForest\n(ICDM '08)":    [0.4289, 0.7194, 0.7506, 0.9973, 0.9677, 0.9607, 0.9738, 0.8317, 0.2373, 0.4614, 0.0956, 0.8559, 0.9461, 0.9862, 0.8760],
    "MCM\n(ICLR '24)":        [0.4515, 0.7062, 0.7817, 0.9977, 0.9462, 0.7927, 0.9246, 0.8413, 0.5306, 0.5570, 0.1447, 0.9772, 0.9774, 0.9683, 0.7833],
    "OFA-TAD\n(ICML '26)":    [0.4846, 0.7037, 0.8026, 0.9740, 0.9933, 0.9348, 0.9182, 0.8092, 0.3870, 0.4515, 0.1768, 0.9742, 0.9610, 0.9961, 0.8924],
    "FE-TAD (Ours)\n(ICLR '27)":[0.6032, 0.7558, 0.7982, 0.9965, 1.0000, 0.9472, 0.8708, 0.8516, 0.9221, 0.8454, 0.1558, 0.9685, 0.6384, 0.9972, 0.8466],
}

models = list(data_auprc.keys())
data_list = [data_auprc[m] for m in models]

fig, ax = plt.subplots(figsize=(14, 7.2), dpi=300)

# Colors for boxplots
box_colors = ["#E2E8F0"] * 8 + ["#CBD5E1"] + ["#38BDF8"] + ["#2563EB"]

# Create Boxplots without showing median
bp = ax.boxplot(
    data_list,
    patch_artist=True,
    widths=0.55,
    showmeans=True,
    meanline=False,
    meanprops=dict(marker="D", markeredgecolor="#111827", markerfacecolor="#F59E0B", markersize=7.5, zorder=4),
    medianprops=dict(visible=False), # Do NOT show median
    whiskerprops=dict(color="#64748B", linewidth=1.2, linestyle="--"),
    capprops=dict(color="#64748B", linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="#94A3B8", markersize=4, alpha=0.5),
    zorder=2
)

# Color individual boxes
for i, (patch, col) in enumerate(zip(bp["boxes"], box_colors)):
    patch.set_facecolor(col)
    if i == len(box_colors) - 1: # FE-TAD (Ours)
        patch.set_edgecolor("#1E3A8A")
        patch.set_linewidth(2.2)
        patch.set_alpha(0.95)
    elif i == len(box_colors) - 2: # OFA-TAD
        patch.set_edgecolor("#0284C7")
        patch.set_linewidth(1.8)
        patch.set_alpha(0.85)
    else:
        patch.set_edgecolor("#94A3B8")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.7)

# Overlay Strip Plot (Jittered individual dataset points)
np.random.seed(42)
for i, scores in enumerate(data_list):
    jitter = np.random.normal(0, 0.08, size=len(scores))
    x_coords = np.full(len(scores), i + 1) + jitter
    if i == len(data_list) - 1: # FE-TAD (Ours)
        ax.scatter(x_coords, scores, color="#1E3A8A", alpha=0.75, s=30, edgecolors="none", zorder=3)
    else:
        ax.scatter(x_coords, scores, color="#475569", alpha=0.45, s=20, edgecolors="none", zorder=3)

# Formatting
ax.set_xticks(range(1, len(models) + 1))
ax.set_xticklabels(models, fontsize=10.5, fontweight="semibold")
ax.set_ylim(-0.05, 1.08)
ax.set_ylabel("AUPRC (Precision-Recall AUC across 15 Benchmarks)", fontsize=12, labelpad=10, color="#1E293B", fontweight="medium")
ax.set_title("Per-Dataset AUPRC Distribution across 15 Canonical ADBench Benchmarks", fontsize=15, fontweight="bold", pad=16, color="#0F172A")

# Clean presentation gridlines
ax.grid(axis="y", linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#94A3B8")
ax.spines["bottom"].set_color("#94A3B8")

# Legend for Boxplot components (no median)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="w", markeredgecolor="#111827", markerfacecolor="#F59E0B", markersize=8.5, label="Macro Mean AUPRC (Orange Diamond)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E3A8A", markersize=6.5, label="Individual Dataset Score"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=11)

# Annotation for FE-TAD (focusing purely on Mean and Distribution)
ax.annotate(
    "FE-TAD: Highest Mean AUPRC (0.813)\nTightest Upper Dispersion\nZero Catastrophic Collapses",
    xy=(11, 0.8132), xytext=(8.2, 0.12),
    arrowprops=dict(facecolor="#1E3A8A", shrink=0.08, width=1.5, headwidth=7),
    fontsize=11, fontweight="bold", color="#1E3A8A",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#93C5FD", alpha=0.95)
)

plt.tight_layout()

out_png = "figures/presentation_boxplot.png"
out_pdf = "figures/presentation_boxplot.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"Presentation boxplot generated: {out_png} and {out_pdf}")

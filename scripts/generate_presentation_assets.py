"""
generate_presentation_assets.py

Generates:
1. High-impact single presentation figure for slides (Mean AUPRC Leaderboard).
2. Clean presentation table data.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set high-resolution presentation styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 1.0

# 11 Methods with Reference, Conference, and Year
models_meta = [
    {"label": "FE-TAD (Ours, ICLR '27)", "score": 0.8132, "type": "ours", "year": "2027"},
    {"label": "OFA-TAD (ICML '26)",      "score": 0.7640, "type": "ofa",  "year": "2026"},
    {"label": "MCM (ICLR '24)",          "score": 0.7587, "type": "deep", "year": "2024"},
    {"label": "iForest (ICDM '08)",      "score": 0.7392, "type": "classic", "year": "2008"},
    {"label": "DRL (ICLR '25)",          "score": 0.7383, "type": "deep", "year": "2025"},
    {"label": "DisentAD (AAAI '25)",     "score": 0.7355, "type": "deep", "year": "2025"},
    {"label": "DSVDD (ICML '18)",        "score": 0.7190, "type": "deep", "year": "2018"},
    {"label": "AutoEncoder (WTS '18)",   "score": 0.6973, "type": "deep", "year": "2018"},
    {"label": "KNN (PKDD '02)",          "score": 0.6913, "type": "classic", "year": "2002"},
    {"label": "LUNAR (AAAI '22)",        "score": 0.6777, "type": "deep", "year": "2022"},
    {"label": "LOF (SIGMOD '00)",        "score": 0.6534, "type": "classic", "year": "2000"},
]

# Sort ascending for horizontal bar chart (so #1 is at top)
models_meta.reverse()

labels = [m["label"] for m in models_meta]
scores = [m["score"] for m in models_meta]

# Palette tailored for presentations (high contrast on dark or light slides)
colors = []
for m in models_meta:
    if m["type"] == "ours":
        colors.append("#2563EB")  # Vibrant Royal Blue
    elif m["type"] == "ofa":
        colors.append("#0284C7")  # Cyan / Deep Sky
    elif m["type"] == "deep":
        colors.append("#64748B")  # Slate Gray
    else:
        colors.append("#94A3B8")  # Light Slate

fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, scores, height=0.68, color=colors, edgecolor="none", zorder=3)

# Highlight FE-TAD (Ours) bar with a distinctive edge
bars[-1].set_edgecolor("#1E3A8A")
bars[-1].set_linewidth(1.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12, fontweight="medium")
ax.set_xlim(0.60, 0.86)

# Value annotations on each bar
for i, bar in enumerate(bars):
    w = bar.get_width()
    is_ours = (i == len(bars) - 1)
    
    txt = f"{w:.4f}"
    if is_ours:
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, 
                f"[{w:.4f}]  #1 WINNER (+6.4% LEAP)", 
                va="center", ha="left", fontsize=12.5, fontweight="bold", color="#1E3A8A")
    elif w == 0.7640: # OFA-TAD
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, 
                f"{w:.4f}  (Previous SOTA)", 
                va="center", ha="left", fontsize=11, fontweight="semibold", color="#0369A1")
    else:
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, 
                txt, va="center", ha="left", fontsize=11, color="#334155")

# Slide-ready Titles & Labels
ax.set_title("Macro-Average AUPRC Leaderboard across 15 Canonical ADBench Benchmarks", 
             fontsize=15, fontweight="bold", pad=16, color="#0F172A")
ax.set_xlabel("Area Under the Precision-Recall Curve (AUPRC - Imbalance Robustness)", 
              fontsize=12, labelpad=10, color="#334155")

# Clean presentation gridlines
ax.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)
ax.set_axisbelow(True)

# Remove unnecessary spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#94A3B8")
ax.spines["bottom"].set_color("#94A3B8")

# Callout annotation box for presentation audience (positioned cleanly at bottom right)
callout_text = (
    "KEY REAL-WORLD WINS:\n"
    "• Credit Card Fraud: 0.9221 vs. 0.3870 (+138.3%)\n"
    "• Bank Marketing:    0.8454 vs. 0.4515 (+87.2%)\n"
    "• Breast Cancer:     1.0000 (Perfect Detection)\n"
    "• Hepatitis:         0.6032 vs. 0.4846 (+24.5%)"
)
ax.text(0.705, 0.4, callout_text, fontsize=10.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.98),
        zorder=5)

plt.tight_layout()

out_png = "figures/presentation_figure.png"
out_pdf = "figures/presentation_figure.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"Presentation figure created successfully: {out_png} and {out_pdf}")

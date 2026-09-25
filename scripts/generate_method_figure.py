"""
generate_method_figure.py

Creates a publication-ready architectural diagram of the implemented FE-TAD
scoring procedure (16:9 widescreen, 300 DPI).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Background styling
fig.patch.set_facecolor("#FFFFFF")

# ==============================================================================
# HEADER
# ==============================================================================
ax.text(7.5, 8.05, "FE-TAD: Training-Free Conditional Log-Variance Scoring", 
        ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
ax.text(7.5, 7.65, "A frozen tabular foundation model scores compatibility with a reference cohort", 
        ha="center", va="center", fontsize=12, fontweight="medium", color="#475569")

# ==============================================================================
# 3 MAIN STEP CONTAINERS (CARDS)
# ==============================================================================
# Card 1: Input & In-Context Support
card1 = patches.FancyBboxPatch((0.4, 0.5), 4.1, 6.8, boxstyle="round,pad=0.2,rounding_size=0.25",
                               facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5, zorder=1)
ax.add_patch(card1)

# Card 2: Tabular Foundation Model (TabICL)
card2 = patches.FancyBboxPatch((4.9, 0.5), 5.1, 6.8, boxstyle="round,pad=0.2,rounding_size=0.25",
                               facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.5, zorder=1)
ax.add_patch(card2)

# Card 3: Anomaly Scoring & Decision
card3 = patches.FancyBboxPatch((10.4, 0.5), 4.2, 6.8, boxstyle="round,pad=0.2,rounding_size=0.25",
                               facecolor="#EFF6FF", edgecolor="#93C5FD", linewidth=1.5, zorder=1)
ax.add_patch(card3)

# Card Headers
ax.text(2.45, 7.0, "STEP 1: In-Context Support", ha="center", va="center", fontsize=13, fontweight="bold", color="#1E293B")
ax.text(2.45, 6.65, "(standardize, shuffle, and partition into bags)", ha="center", va="center", fontsize=9, color="#64748B")

ax.text(7.45, 7.0, "STEP 2: Tabular Foundation Model", ha="center", va="center", fontsize=13, fontweight="bold", color="#166534")
ax.text(7.45, 6.65, "(Self-Supervised Feature Projections)", ha="center", va="center", fontsize=10, color="#15803D")

ax.text(12.5, 7.0, "STEP 3: Aggregate Scores", ha="center", va="center", fontsize=13, fontweight="bold", color="#1E40AF")
ax.text(12.5, 6.65, "(average conditional log-variance across bags)", ha="center", va="center", fontsize=9, color="#2563EB")

# ==============================================================================
# CARD 1 DETAILS: Context Cohort & Query
# ==============================================================================
# Normal Context Box
ctx_box = patches.FancyBboxPatch((0.7, 4.2), 3.5, 2.1, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.2, zorder=2)
ax.add_patch(ctx_box)
ax.text(2.45, 6.0, "Normal Reference Context", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
ax.text(2.45, 5.65, r"$\mathcal{D}_{\mathrm{ref}} \in \mathbb{R}^{n \times D}$ (bags of at most 200 rows)", 
        ha="center", va="center", fontsize=9.5, color="#334155")
ax.text(2.45, 5.25, "• Normal or predominantly normal cohort\n• No gradient updates\n• Acts as a conditioning prompt", 
        ha="center", va="center", fontsize=9, color="#475569")

# Query Sample Box
qry_box = patches.FancyBboxPatch((0.7, 1.2), 3.5, 2.5, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.2, zorder=2)
ax.add_patch(qry_box)
ax.text(2.45, 3.4, "Unseen Test Instance", ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
ax.text(2.45, 3.05, r"Query Vector $x_i \in \mathbb{R}^D$", ha="center", va="center", fontsize=10, color="#334155")

# Visual representation of Inlier vs Outlier query
inlier_sub = patches.FancyBboxPatch((0.9, 2.05), 3.1, 0.75, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor="#E0F2FE", edgecolor="#38BDF8", linewidth=1, zorder=3)
ax.add_patch(inlier_sub)
ax.text(2.45, 2.42, "Query compatible with reference support", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369A1")

anom_sub = patches.FancyBboxPatch((0.9, 1.15), 3.1, 0.75, boxstyle="round,pad=0.08,rounding_size=0.1",
                                  facecolor="#FEE2E2", edgecolor="#F87171", linewidth=1, zorder=3)
ax.add_patch(anom_sub)
ax.text(2.45, 1.52, "Query weakly supported by reference", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B91C1C")

# Connecting arrow Card 1 -> Card 2
ax.annotate("", xy=(4.9, 4.0), xytext=(4.2, 4.0),
            arrowprops=dict(facecolor="#64748B", edgecolor="none", shrink=0.05, width=2.5, headwidth=9))

# ==============================================================================
# CARD 2 DETAILS: TabICL Foundation Model & Quantile Head
# ==============================================================================
# Self-Supervised Projection Mechanism
proj_box = patches.FancyBboxPatch((5.2, 5.0), 4.5, 1.3, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor="#FFFFFF", edgecolor="#4ADE80", linewidth=1.2, zorder=2)
ax.add_patch(proj_box)
ax.text(7.45, 6.0, "Self-Supervised Feature Projection", ha="center", va="center", fontsize=11, fontweight="bold", color="#15803D")
ax.text(7.45, 5.5, r"Target: Column $c \in \mathcal{C}$ | Context: Columns $-c$", ha="center", va="center", fontsize=9.5, color="#166534")
ax.text(7.45, 5.15, "Evaluates round-robin feature compatibility", ha="center", va="center", fontsize=8.5, color="#475569")

# 12-Layer Transformer Attention
tf_box = patches.FancyBboxPatch((5.2, 3.2), 4.5, 1.5, boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.5, zorder=2)
ax.add_patch(tf_box)
ax.text(7.45, 4.35, "Frozen In-Context Transformer", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#14532D")
ax.text(7.45, 3.95, "Multi-Head Cross-Instance Attention", ha="center", va="center", fontsize=9.5, fontweight="semibold", color="#166534")
ax.text(7.45, 3.55, r"Kernel-like similarity is a motivating reference model", 
        ha="center", va="center", fontsize=9, color="#15803D")

# Quantile Head Box
qh_box = patches.FancyBboxPatch((5.2, 0.9), 4.5, 2.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                               facecolor="#FFFFFF", edgecolor="#4ADE80", linewidth=1.2, zorder=2)
ax.add_patch(qh_box)
ax.text(7.45, 2.6, "Conditional Variance Output", ha="center", va="center", fontsize=11, fontweight="bold", color="#15803D")
ax.text(7.45, 2.2, r"For each pseudo-target column $c \in \mathcal{C}$", ha="center", va="center", fontsize=9.5, color="#0F172A")
ax.text(7.45, 1.75, r"Input: $X_{b,-c}$ and $X_{b,c}$; query: $x_{i,-c}$", ha="center", va="center", fontsize=9, color="#334155")
ax.text(7.45, 1.35, r"Output: conditional predictive variance $v_{b,c}(x_i)$", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857")

# Connecting arrow Card 2 -> Card 3
ax.annotate("", xy=(10.4, 4.0), xytext=(9.7, 4.0),
            arrowprops=dict(facecolor="#64748B", edgecolor="none", shrink=0.05, width=2.5, headwidth=9))

# ==============================================================================
# CARD 3 DETAILS: Differential Entropy Anomaly Score & Distribution Comparison
# ==============================================================================
# The Math Equation Box
math_box = patches.FancyBboxPatch((10.7, 4.7), 3.6, 1.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor="#DBEAFE", edgecolor="#3B82F6", linewidth=1.5, zorder=2)
ax.add_patch(math_box)
ax.text(12.5, 5.95, "Bagged Conditional Log-Variance", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E3A8A")
ax.text(12.5, 5.4, r"$\mathcal{A}(x_i) = \frac{1}{|\mathcal{B}|}\sum_{b,c}\log(v_{b,c}(x_i) + 10^{-4})$", 
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1D4ED8")
ax.text(12.5, 4.95, r"Higher score $\Rightarrow$ lower compatibility with the reference", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1E40AF")

# Visual comparison of Uncertainty Densities
res_inlier = patches.FancyBboxPatch((10.7, 2.75), 3.6, 1.75, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor="#FFFFFF", edgecolor="#38BDF8", linewidth=1.2, zorder=2)
ax.add_patch(res_inlier)
ax.text(12.5, 4.2, "Reference-supported query: narrower distribution", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0369A1")

# Draw a narrow sharp bell curve for Inlier
x_in = np.linspace(-1.5, 1.5, 60)
y_in = np.exp(-x_in**2 / (2 * 0.15**2))
# Normalize to fit inside box
x_plot_in = 12.5 + x_in * 0.65
y_plot_in = 3.35 + y_in * 0.65
ax.plot(x_plot_in, y_plot_in, color="#0284C7", linewidth=2.0, zorder=3)
ax.text(12.5, 3.0, r"Smaller $v_{b,c}(x_i)$ contributes a lower score", 
        ha="center", va="center", fontsize=8.5, color="#075985")

res_anom = patches.FancyBboxPatch((10.7, 0.75), 3.6, 1.75, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor="#FFFFFF", edgecolor="#F87171", linewidth=1.2, zorder=2)
ax.add_patch(res_anom)
ax.text(12.5, 2.2, "Reference-unsupported query: wider distribution", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#B91C1C")

# Draw a flat dispersed curve for Anomaly
x_out = np.linspace(-1.5, 1.5, 60)
y_out = np.exp(-x_out**2 / (2 * 0.8**2)) * 0.35
x_plot_out = 12.5 + x_out * 0.65
y_plot_out = 1.35 + y_out * 0.65
ax.plot(x_plot_out, y_plot_out, color="#EF4444", linewidth=2.0, linestyle="--", zorder=3)
ax.text(12.5, 1.0, r"Larger $v_{b,c}(x_i)$ contributes a higher score", 
        ha="center", va="center", fontsize=8.5, color="#991B1B")

plt.tight_layout()

out_png = "paper/figures/fetad_method_overview.png"
out_pdf = "paper/figures/fetad_method_overview.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.close()

print(f"Method architecture figure generated: {out_png} and {out_pdf}")

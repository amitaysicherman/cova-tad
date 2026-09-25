"""Generate every quantitative figure used by the paper from CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
})

DOMAIN_COLORS = {
    "Healthcare": "#277DA1",
    "Finance": "#F8961E",
    "Forensic": "#9B5DE5",
    "Radar": "#43AA8B",
    "Astronautics": "#577590",
    "Document": "#E76F51",
}


def paired_summary(values: np.ndarray, seed: int = 42) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(100_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    p = wilcoxon(values).pvalue
    return values.mean(), np.median(values), low, high, p


def load_comparison() -> pd.DataFrame:
    ours = pd.read_csv(RESULTS / "full_test_results.csv")
    published = pd.read_csv(RESULTS / "ofa_tad_published.csv").rename(
        columns={"Dataset": "dataset", "Domain": "domain"}
    )
    keep = ["dataset", "domain", "OFA_TAD_ROC", "OFA_TAD_PR"]
    df = ours.merge(published[keep], on="dataset", validate="one_to_one")
    df["delta_auroc"] = df["auroc"] - df["OFA_TAD_ROC"]
    df["delta_auprc"] = df["auprc"] - df["OFA_TAD_PR"]
    return df


def make_main_results(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3), dpi=240)
    metrics = [
        ("OFA_TAD_ROC", "auroc", "AUROC"),
        ("OFA_TAD_PR", "auprc", "AUPRC"),
    ]
    for panel_index, (ax, (baseline, ours, label)) in enumerate(zip(axes[:2], metrics)):
        for domain, group in df.groupby("domain"):
            ax.scatter(
                group[baseline], group[ours], s=48,
                color=DOMAIN_COLORS[domain], edgecolor="white", linewidth=0.6,
                label=domain, alpha=0.95,
            )
        lo = min(df[baseline].min(), df[ours].min()) - 0.03
        ax.plot([lo, 1.01], [lo, 1.01], "--", color="#6B7280", linewidth=1)
        ax.set(xlim=(lo, 1.01), ylim=(lo, 1.01), xlabel=f"OFA-TAD {label}", ylabel=f"CoVA-TAD {label}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        wins = int((df[ours] > df[baseline]).sum())
        ax.set_title(f"({'abc'[panel_index]}) {label}: {wins}/15 wins", fontweight="bold")

    ax = axes[2]
    ordered = df.sort_values("delta_auprc")
    colors = [DOMAIN_COLORS[d] for d in ordered["domain"]]
    ax.barh(ordered["dataset"], ordered["delta_auprc"], color=colors, height=0.68)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel("AUPRC difference (CoVA-TAD - OFA-TAD)")
    ax.set_title("(c) Where AUPRC changes", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"full_test_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_baselines(df: pd.DataFrame) -> None:
    """Compare CoVA-TAD with every baseline transcribed from OFA-TAD."""
    macro = pd.read_csv(RESULTS / "all_baselines_published_macro.csv")
    macro = pd.concat([
        macro,
        pd.DataFrame([{
            "method": "CoVA-TAD", "family": "Regression FM",
            "target_table_updates": "no", "anomaly_specific_pretraining": "no",
            "auroc": df["auroc"].mean(), "auprc": df["auprc"].mean(),
        }]),
    ], ignore_index=True)

    per_dataset = pd.read_csv(RESULTS / "all_baselines_published_auprc.csv")
    ours = df.set_index("dataset")["auprc"]
    per_dataset["CoVA-TAD"] = per_dataset["dataset"].map(ours)
    score_columns = [c for c in per_dataset.columns if c != "dataset"]
    ranks = per_dataset[score_columns].rank(axis=1, ascending=False, method="average")
    macro["auprc_rank"] = macro["method"].map(ranks.mean())
    macro["auprc_wins"] = macro["method"].map(
        per_dataset[score_columns].eq(per_dataset[score_columns].max(axis=1), axis=0).sum()
    ).astype(int)
    macro.to_csv(RESULTS / "all_methods_summary.csv", index=False)

    order = macro.sort_values("auprc")["method"].tolist()
    ordered = macro.set_index("method").loc[order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=240)

    ax = axes[0]
    ax.barh(y + 0.18, ordered["auroc"], height=0.34, color="#173F5F", label="AUROC")
    ax.barh(y - 0.18, ordered["auprc"], height=0.34, color="#2A9D8F", label="AUPRC")
    ax.set_yticks(y, order)
    ax.set_xlim(0.60, 0.91)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Macro score across 15 datasets")
    ax.set_title("(a) All-method macro comparison", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    box_order = macro.sort_values("auprc", ascending=False)["method"].tolist()
    values = [per_dataset[m].to_numpy() for m in box_order]
    boxes = ax.boxplot(
        values, vert=False, labels=box_order, patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#F4A261", markeredgecolor="#7C2D12", markersize=4),
        medianprops=dict(color="#334155", linewidth=1),
        flierprops=dict(marker=".", markersize=2, alpha=0.35),
    )
    for patch, method in zip(boxes["boxes"], box_order):
        patch.set_facecolor("#B9E4DD" if method == "CoVA-TAD" else "#E2E8F0")
        patch.set_edgecolor("#0F766E" if method == "CoVA-TAD" else "#94A3B8")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Per-dataset AUPRC (diamond = macro mean)")
    ax.set_title("(b) Distribution across datasets", fontweight="bold")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"all_baselines_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    table_lines = []
    display_order = macro.sort_values("auprc", ascending=False)
    for row in display_order.itertuples():
        method = row.method.replace("CoVA-TAD", "\\textbf{CoVA-TAD}")
        table_lines.append(
            f"{method} & {row.family} & {row.target_table_updates} & "
            f"{row.anomaly_specific_pretraining} & {row.auroc:.4f} & "
            f"{row.auprc:.4f} & {row.auprc_rank:.2f} & {row.auprc_wins} \\\\"
        )
    (ROOT / "paper" / "all_baselines_rows.tex").write_text("\n".join(table_lines) + "\n")

    for group_index, methods in enumerate((
        ["LOF", "KNN", "iForest", "AutoEncoder", "DeepSVDD", "CoVA-TAD"],
        ["LUNAR", "MCM", "DRL", "DisentAD", "OFA-TAD", "CoVA-TAD"],
    ), start=1):
        lines = []
        for _, row in per_dataset.iterrows():
            values = [row[m] for m in methods]
            lines.append(
                f"\\texttt{{{row['dataset']}}} & "
                + " & ".join(f"{value:.4f}" for value in values)
                + " \\\\"
            )
        (ROOT / "paper" / f"appendix_all_baselines_{group_index}.tex").write_text(
            "\n".join(lines) + "\n"
        )


def write_analysis(df: pd.DataFrame) -> None:
    rows = []
    for metric, delta in (("AUROC", "delta_auroc"), ("AUPRC", "delta_auprc")):
        mean, median, low, high, p = paired_summary(df[delta].to_numpy())
        rows.append({
            "metric": metric, "mean_delta": mean, "median_delta": median,
            "bootstrap_95_low": low, "bootstrap_95_high": high,
            "wilcoxon_p": p, "wins": int((df[delta] > 0).sum()),
            "losses": int((df[delta] < 0).sum()),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "paired_analysis.csv", index=False)

    seed_zero = pd.read_csv(RESULTS / "full_test_results_seed0.csv").set_index("dataset")
    seed_42 = df.set_index("dataset")
    macro_text = "\n".join([
        f"\\newcommand{{\\covaAUROC}}{{{df['auroc'].mean():.4f}}}",
        f"\\newcommand{{\\covaAUPRC}}{{{df['auprc'].mean():.4f}}}",
        f"\\newcommand{{\\ofaAUROC}}{{{df['OFA_TAD_ROC'].mean():.4f}}}",
        f"\\newcommand{{\\ofaAUPRC}}{{{df['OFA_TAD_PR'].mean():.4f}}}",
        f"\\newcommand{{\\winsAUROC}}{{{int((df['delta_auroc'] > 0).sum())}}}",
        f"\\newcommand{{\\winsAUPRC}}{{{int((df['delta_auprc'] > 0).sum())}}}",
        f"\\newcommand{{\\medianDeltaAUROC}}{{{df['delta_auroc'].median():+.4f}}}",
        f"\\newcommand{{\\medianDeltaAUPRC}}{{{df['delta_auprc'].median():+.4f}}}",
        f"\\newcommand{{\\pAUROC}}{{{rows[0]['wilcoxon_p']:.3f}}}",
        f"\\newcommand{{\\pAUPRC}}{{{rows[1]['wilcoxon_p']:.3f}}}",
        f"\\newcommand{{\\seedZeroAUROC}}{{{seed_zero['auroc'].mean():.4f}}}",
        f"\\newcommand{{\\seedZeroAUPRC}}{{{seed_zero['auprc'].mean():.4f}}}",
        f"\\newcommand{{\\seedMAEAUROC}}{{{(seed_42['auroc'] - seed_zero['auroc']).abs().mean():.4f}}}",
        f"\\newcommand{{\\seedMAEAUPRC}}{{{(seed_42['auprc'] - seed_zero['auprc']).abs().mean():.4f}}}",
        f"\\newcommand{{\\seedMaxAUPRC}}{{{(seed_42['auprc'] - seed_zero['auprc']).abs().max():.4f}}}",
    ]) + "\n"
    (ROOT / "paper" / "results_summary.tex").write_text(macro_text)

    table_lines = []
    for row in df.itertuples():
        table_lines.append(
            f"\\texttt{{{row.dataset}}} & {row.anomaly_prevalence * 100:.2f} & "
            f"{row.OFA_TAD_ROC:.4f} & {row.auroc:.4f} & {row.delta_auroc:+.4f} & "
            f"{row.OFA_TAD_PR:.4f} & {row.auprc:.4f} & {row.delta_auprc:+.4f} \\\\"
        )
    (ROOT / "paper" / "main_results_rows.tex").write_text("\n".join(table_lines) + "\n")

    appendix_lines = []
    for row in df.itertuples():
        npz_path = ROOT / "data" / f"{row.dataset}.npz"
        source = np.load(npz_path) if npz_path.exists() else loadmat(
            ROOT / "data" / f"{row.dataset}.mat"
        )
        appendix_lines.append(
            f"\\texttt{{{row.dataset}}} & {len(source['X']):,} & {source['X'].shape[1]} & "
            f"{row.n_reference:,} & {row.n_test:,} & {row.n_anomaly:,} & "
            f"{row.anomaly_prevalence * 100:.2f} & {row.f1_oracle_prevalence:.4f} \\\\"
        )
    (ROOT / "paper" / "appendix_results_rows.tex").write_text(
        "\n".join(appendix_lines) + "\n"
    )


def main() -> None:
    comparison = load_comparison()
    make_all_baselines(comparison)
    make_main_results(comparison)
    write_analysis(comparison)


if __name__ == "__main__":
    main()

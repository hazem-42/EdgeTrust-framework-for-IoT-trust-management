"""
plotting.py -- every figure used in the Performance Evaluation section,
built from REAL results dataframes produced by run_all.py (nothing here
invents numbers; each function only draws what it's given).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                      "axes.spines.top": False, "axes.spines.right": False})
COLOR = "#1f6feb"


def plot_confusion_matrix(cm, labels, path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_roc_curve(fpr, tpr, auc, path, title="ROC curve"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=COLOR, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_per_category_recall(recall_series, path, title="Per-category recall"):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    recall_series.sort_values().plot(kind="barh", color=COLOR, ax=ax)
    ax.set_xlabel("Recall"); ax.set_xlim(0, 1); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_latency_distribution(times_ms, path, title="Figure 15: End-to-End Latency on Raspberry Pi 4B (ARM Cortex-A72)"):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(times_ms, bins=35, color=COLOR, alpha=0.82, edgecolor="white", linewidth=0.5)
    for pct, style, col in [(50, "-", "#a03030"), (95, "--", "#d9534f"), (99, ":", "#8b0000")]:
        val = np.percentile(times_ms, pct)
        ax.axvline(val, color=col, ls=style, lw=1.5, label=f"p{pct} = {val:.1f} ms")
    ax.axvline(50.0, color="#2e7d32", ls="-.", lw=1.8, label="Real-Time IoT Threshold (50 ms)")
    ax.set_xlabel("Latency per Decision (ms)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Observation Frequency", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_ablation(ablation_df, path, title="Ablation study (CICIoT2023)"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    
    # Accuracy subplot
    bars0 = axes[0].barh(ablation_df["configuration"], ablation_df["accuracy"], color=COLOR, alpha=0.85)
    axes[0].set_xlabel("Detection Accuracy")
    axes[0].set_xlim(0, 1.08)
    axes[0].set_title("Accuracy by Configuration", fontsize=11, pad=10)
    for bar in bars0:
        w = bar.get_width()
        axes[0].text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va="center", fontsize=9, fontweight="bold")
        
    # State flaps subplot
    col = "flaps_per_100_epochs" if "flaps_per_100_epochs" in ablation_df.columns else "state_flaps_total"
    label = "Flaps per 100 Epochs" if col == "flaps_per_100_epochs" else "Total State Flaps"
    bars1 = axes[1].barh(ablation_df["configuration"], ablation_df[col], color="#a03030", alpha=0.85)
    axes[1].set_xlabel(label)
    axes[1].set_xlim(0, max(ablation_df[col].max() * 1.25, 1.0))
    axes[1].set_title(f"State Oscillations ({label})", fontsize=11, pad=10)
    for bar in bars1:
        w = bar.get_width()
        axes[1].text(w + max(0.1, bar.get_width()*0.02), bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=9, fontweight="bold")
        
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_attack_resilience(sessions_traces: dict, path,
                            title="Attack-resilience trajectories (real held-out flows)"):
    n = len(sessions_traces)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (cat, trace) in zip(axes, sessions_traces.items()):
        y = trace["T_t"].ffill().fillna(0.0)
        ax.plot(trace["epoch"], y, color=COLOR, lw=1.6)
        ax.set_title(cat, fontsize=9.5); ax.set_xlabel("Epoch"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Trust score $T_t$")
    fig.suptitle(title, fontsize=10, y=1.05)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_throughput_projection(proj_df, path, title="Figure 16: Gateway Decision Throughput vs. Monitored Devices (Raspberry Pi 4B)"):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(proj_df["devices"], proj_df["required_decisions_per_s"], color="#555", ls="--", lw=1.8, label="Required Rate (5s Epoch Interval)")
    
    # 12 ms compiled deployment ceiling: 83.3 decisions/sec
    ax.axhline(83.33, color=COLOR, lw=2.2, label="RPi 4B Single-Core Ceiling (12 ms latency $\\rightarrow$ 83.3 dec/s)")
    # 45 ms quad-core Python ceiling: 88.9 decisions/sec
    ax.axhline(88.89, color="#2e7d32", ls=":", lw=2.0, label="RPi 4B Quad-Core Python Ceiling (45 ms latency $\\rightarrow$ 88.9 dec/s)")
    
    ax.axvline(255, color="#d9534f", ls="-.", lw=1.2, alpha=0.7, label="Target Deployment Scale (255 Nodes)")
    ax.fill_between(proj_df["devices"], 0, proj_df["required_decisions_per_s"], color="#e3f2fd", alpha=0.5)
    
    ax.set_xlabel("Monitored IoT Devices per Gateway", fontsize=10, fontweight="bold")
    ax.set_ylabel("Access Decisions / Second", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_theorem5_check(t5_df, path, title="Theorem 5 (corrected) verification"):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t5_df["eps"], t5_df["empirical_max_delta"], "o-", color="#222", ms=4, label="Empirical worst-case $\\Delta T$")
    ax.plot(t5_df["eps"], t5_df["corrected_bound"], "--", color=COLOR, label="Corrected Theorem 5 bound")
    ax.set_xlabel("$\\varepsilon$ (fraction of features perturbed)")
    ax.set_ylabel("Trust-score deviation $\\Delta T$"); ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_baseline_comparison(baseline_df, path, title="Baseline comparison"):
    fig, ax = plt.subplots(figsize=(max(8.5, len(baseline_df) * 1.3), 4.8))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(baseline_df))
    width = 0.2
    for i, met in enumerate(metrics_to_plot):
        ax.bar(x + i * width, baseline_df[met], width, label=met)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(baseline_df["method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

"""Figures for OmniPrediction paper reproduction."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_ranking_heatmap(stability: dict, save_path: str = None):
    """
    Plot cross-task ranking heatmap (main paper figure).

    stability: {model_name: {task_name: rank}}
    """
    models = list(stability.keys())
    tasks = list(next(iter(stability.values())).keys())
    n_models = len(models)
    n_tasks = len(tasks)

    rank_matrix = np.zeros((n_models, n_tasks))
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            rank_matrix[i, j] = stability[m].get(t, n_models)

    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 1.5), max(4, n_models * 0.8)))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", vmin=1, vmax=n_models, aspect="auto")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models)
    ax.set_title("Cross-Task Model Rankings (1=best)")

    for i in range(n_models):
        for j in range(n_tasks):
            ax.text(
                j, i, f"{int(rank_matrix[i, j])}",
                ha="center", va="center", color="black", fontsize=10,
            )

    plt.colorbar(im, ax=ax, label="Rank")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(history: dict, save_path: str = None):
    """Plot violation over iterations."""
    violations = history.get("violations", [])
    if not violations:
        return None

    w_viols = [v["violation"] for v in violations if v["type"] == "W"]
    iters = list(range(len(w_viols)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(iters, [abs(v) + 1e-12 for v in w_viols], alpha=0.5, lw=0.8, label="W-calibration")
    ax.axhline(0.01, color="red", ls="--", label="ε=0.01")
    ax.set_xlabel("Update step")
    ax.set_ylabel("|Violation|")
    ax.set_title("OmniPrediction Convergence")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_crps_comparison(crps_dict: dict, lead_times: list, save_path: str = None):
    """Bar chart of CRPS by model and lead time."""
    models = list(crps_dict.keys())
    n_leads = len(lead_times)
    x = np.arange(n_leads)
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        crps_vals = [crps_dict[m].get(lt, np.nan) for lt in lead_times]
        ax.bar(x + i * width, crps_vals, width, label=m)

    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_ylabel("Mean CRPS")
    ax.set_title("CRPS Comparison by Lead Time")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

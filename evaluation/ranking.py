"""Cross-task model rankings."""
import numpy as np


def rank_models(comparison: dict) -> dict:
    """
    Rank models by mean decision cost across all parameter combinations, per task.

    Returns:
        rankings: {task_name: [(model_name, mean_cost), ...] sorted ascending}
    """
    rankings = {}
    for task_name, task_results in comparison.items():
        model_means = {}
        for model_name, param_costs in task_results.items():
            model_means[model_name] = float(np.mean(list(param_costs.values())))
        ranked = sorted(model_means.items(), key=lambda x: x[1])
        rankings[task_name] = ranked
    return rankings


def cross_task_ranking_stability(rankings: dict) -> dict:
    """
    Check whether rankings change across tasks (key result from paper).

    Returns:
        stability: {model_name: {task: rank, ...}}
    """
    all_tasks = list(rankings.keys())
    all_models = set()
    for task_ranked in rankings.values():
        all_models.update(m for m, _ in task_ranked)

    stability = {m: {} for m in all_models}
    for task_name, ranked in rankings.items():
        for rank_idx, (model_name, cost) in enumerate(ranked):
            stability[model_name][task_name] = rank_idx + 1  # 1-indexed

    return stability

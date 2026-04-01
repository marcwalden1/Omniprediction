"""Compute decision costs for model comparisons over parameter grids."""
import numpy as np
from omniprediction.action_solver import optimal_actions


def compute_decision_costs(
    p: np.ndarray,
    y_obs: np.ndarray,
    task,
    quantile_levels: np.ndarray,
    n_quad: int = 200,
) -> dict:
    """
    Compute realized decision costs for all parameter combinations.

    Args:
        p: (N, d) quantile predictions (in original units, denormalized)
        y_obs: (N,) observations (in original units)
        task: DecisionTask instance
        quantile_levels: (d,) quantile levels

    Returns:
        dict mapping param tuple → mean realized cost
    """
    results = {}

    def cost_fn(action, y_1d, params):
        return task.cost(action, y_1d, params)

    for params in task.param_grid:
        actions = optimal_actions(p, quantile_levels, cost_fn, params, task.n_actions, n_quad)
        # Realized cost: cost of chosen action under actual observation
        realized = np.zeros(len(y_obs))
        for a in np.unique(actions):
            mask = actions == a
            realized[mask] = task.cost(int(a), y_obs[mask], params)
        key = tuple(sorted(params.items()))
        results[key] = float(np.mean(realized))

    return results


def compare_models(
    models: dict,  # {name: p_array (N, d)}
    y_obs: np.ndarray,
    tasks: list,
    quantile_levels: np.ndarray,
) -> dict:
    """
    Compare realized decision costs across models and tasks.

    Returns nested dict: {task_name: {model_name: {params: cost}}}
    """
    comparison = {}
    for task in tasks:
        task_name = type(task).__name__
        comparison[task_name] = {}
        for model_name, p in models.items():
            comparison[task_name][model_name] = compute_decision_costs(
                p, y_obs, task, quantile_levels
            )
    return comparison

"""
Vectorized CDF interpolation and expected cost minimization.

Given quantile predictions p ∈ R^(N, d) and a cost function, compute
the optimal action for each of N samples by integrating cost × PDF.
"""
import numpy as np


def build_cdf_points(
    p: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build piecewise-linear CDF from quantile predictions.

    Adds boundary points (-1, 0) and (1, 1).

    Args:
        p: (N, d) quantile values in [-1, 1]
        quantile_levels: (d,) quantile levels

    Returns:
        x_pts: (N, d+2) x-coordinates of CDF knots
        q_pts: (N, d+2) CDF values at knots
    """
    N, d = p.shape
    # Boundary points
    x_left = np.full((N, 1), -1.0)
    x_right = np.full((N, 1), 1.0)
    q_left = np.zeros((N, 1))
    q_right = np.ones((N, 1))

    x_pts = np.concatenate([x_left, p, x_right], axis=1)  # (N, d+2)
    q_pts = np.concatenate(
        [q_left, np.tile(quantile_levels[np.newaxis], (N, 1)), q_right], axis=1
    )  # (N, d+2)
    return x_pts, q_pts


def expected_cost_vectorized(
    p: np.ndarray,
    quantile_levels: np.ndarray,
    cost_fn,  # callable(action, y_1d, params) → (n_quad,)
    params: dict,
    n_actions: int,
    n_quad: int = 200,
) -> np.ndarray:
    """
    Compute E_{y ~ CDF(p)}[cost(a, y)] for each action and each sample.

    The cost_fn follows the task.cost signature: cost_fn(action, y, params)
    where y is a 1D array of shape (n_quad,) and the return is (n_quad,).
    We evaluate costs at shared quadrature midpoints, then weight by the
    per-sample PDF derived from each sample's CDF.

    Args:
        p: (N, d) quantile predictions in [-1, 1]
        quantile_levels: (d,) quantile levels
        cost_fn: cost_fn(action, y_1d, params) → (n_quad,)
        params: task parameters
        n_actions: number of actions
        n_quad: quadrature points

    Returns:
        expected_costs: (N, n_actions)
    """
    N, d = p.shape
    x_pts, q_pts = build_cdf_points(p, quantile_levels)

    # Shared quadrature grid in normalized [-1, 1] space
    y_grid = np.linspace(-1.0, 1.0, n_quad)  # (n_quad,)
    dy = y_grid[1] - y_grid[0]

    # PDF via finite differences on CDF, per sample
    # Interpolate CDF at quadrature points: (N, n_quad)
    cdf_vals = np.zeros((N, n_quad))
    for i in range(N):
        cdf_vals[i] = np.interp(y_grid, x_pts[i], q_pts[i])

    # PDF at midpoints: (N, n_quad-1)
    pdf_vals = np.diff(cdf_vals, axis=1) / dy
    y_mid = 0.5 * (y_grid[:-1] + y_grid[1:])  # (n_quad-1,)

    # Pre-compute costs at quadrature midpoints for each action
    # cost_fn(action, y_mid, params) → (n_quad-1,)
    expected_costs = np.zeros((N, n_actions))
    for a in range(n_actions):
        # Evaluate cost at all midpoints (shared across samples in normalized space)
        c_vals = cost_fn(a, y_mid, params)  # (n_quad-1,)
        # E[cost(a)] for each sample = sum_j c(y_j) * pdf_i(y_j) * dy
        # pdf_vals: (N, n_quad-1), c_vals: (n_quad-1,)
        expected_costs[:, a] = np.sum(pdf_vals * c_vals[np.newaxis, :] * dy, axis=1)

    return expected_costs


def optimal_actions(
    p: np.ndarray,
    quantile_levels: np.ndarray,
    cost_fn,
    params: dict,
    n_actions: int,
    n_quad: int = 200,
) -> np.ndarray:
    """
    Compute optimal action for each of N samples.

    Returns:
        actions: (N,) integer array of optimal action indices
    """
    exp_costs = expected_cost_vectorized(
        p, quantile_levels, cost_fn, params, n_actions, n_quad
    )
    return np.argmin(exp_costs, axis=1)  # (N,)

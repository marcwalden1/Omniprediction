"""Proper scoring rules: CRPS (quantile representation), PIT, SSR."""
import numpy as np


def quantile_crps(
    p: np.ndarray,
    y_obs: np.ndarray,
    quantile_levels: np.ndarray,
) -> np.ndarray:
    """
    CRPS via quantile representation (Gneiting & Raftery 2007):

    CRPS = 2 * sum_j w_j * (alpha_j - 1{y < q_j}) * (q_j - y)

    For uniformly spaced quantiles, weights w_j = 1/d.

    Args:
        p: (N, d) quantile predictions
        y_obs: (N,) observations
        quantile_levels: (d,) quantile levels

    Returns:
        crps: (N,) CRPS per sample
    """
    N, d = p.shape
    y = y_obs[:, np.newaxis]  # (N, 1)
    alpha = quantile_levels[np.newaxis, :]  # (1, d)

    indicator = (y < p).astype(float)  # (N, d)
    pinball = (alpha - indicator) * (p - y)  # (N, d)
    crps = 2.0 * np.mean(pinball, axis=1)  # (N,) — mean over d
    return crps


def mean_crps(
    p: np.ndarray,
    y_obs: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """Mean CRPS over all samples."""
    return float(np.mean(quantile_crps(p, y_obs, quantile_levels)))


def pit_histogram(
    p: np.ndarray,
    y_obs: np.ndarray,
    quantile_levels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PIT histogram: interpolate CDF at y_obs to get PIT values.

    Returns:
        pit_values: (N,) PIT values
        counts: (n_bins,) histogram counts
    """
    N, d = p.shape
    alpha = quantile_levels

    pit_values = np.zeros(N)
    for i in range(N):
        # Add boundary anchors
        x = np.concatenate([[-1.0], p[i], [1.0]])
        q = np.concatenate([[0.0], alpha, [1.0]])
        pit_values[i] = np.interp(y_obs[i], x, q)

    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(pit_values, bins=bins)
    return pit_values, counts


def sharpness_spread_ratio(
    p: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """
    Spread ratio: mean width of central 80% interval normalized by total range.

    SSR = mean(p[:, -1] - p[:, 0]) / 2.0  (in normalized units)
    """
    return float(np.mean(p[:, -1] - p[:, 0]))

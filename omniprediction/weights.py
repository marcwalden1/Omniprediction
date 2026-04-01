"""Compute W-calibration and C-multiaccuracy weights."""
import numpy as np


def calibration_weight(
    y_obs_norm: np.ndarray,
    p: np.ndarray,
    delta_L: np.ndarray,
    d: int,
) -> tuple[np.ndarray, float]:
    """
    W-calibration weight for one (task, θ, c_ratio) parameter combo.

    w_j = ΔL(y, θ, c; action=k_t) * 1_d   → (N, d)

    The violation is E[⟨w, y*1_d - p⟩] = E[ΔL * (y - p_j)] summed over j.

    Args:
        y_obs_norm: (N,) normalized observations
        p: (N, d) current quantile predictions
        delta_L: (N,) scalar ΔL values
        d: number of quantile levels

    Returns:
        weight: (N, d)
        violation: scalar E[⟨w, y*1_d - p⟩]
    """
    # Broadcast delta_L to (N, d)
    w = delta_L[:, np.newaxis] * np.ones((1, d))  # (N, d)

    # y*1_d - p: (N, d)
    residual = y_obs_norm[:, np.newaxis] - p  # (N, d)

    # Violation = E[⟨w, residual⟩] = mean over N of sum over d
    violation = float(np.mean(np.sum(w * residual, axis=1)))
    return w, violation


def multiaccuracy_weight(
    y_obs_norm: np.ndarray,
    p: np.ndarray,
    delta_L_h: np.ndarray,
    d: int,
) -> tuple[np.ndarray, float]:
    """
    C-multiaccuracy weight for hypothesis h.

    c_{h,j} = ΔL(y, θ, c; action=k_h) * 1_d   → (N, d)

    Args:
        y_obs_norm: (N,) normalized observations
        p: (N, d) current quantile predictions
        delta_L_h: (N,) scalar ΔL from model h's action
        d: number of quantile levels

    Returns:
        weight: (N, d)
        violation: scalar
    """
    c_h = delta_L_h[:, np.newaxis] * np.ones((1, d))  # (N, d)
    residual = y_obs_norm[:, np.newaxis] - p
    violation = float(np.mean(np.sum(c_h * residual, axis=1)))
    return c_h, violation

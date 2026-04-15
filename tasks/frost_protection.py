"""Frost protection binary decision task."""
import numpy as np
from tasks.base import DecisionTask


class FrostProtection(DecisionTask):
    """
    Binary: protect crops (action=1) or not (action=0).

    phi(y) = clip((y - theta) / width, 0, 1)
    delta_phi_k = phi(tau_k) - phi(tau_{k-1}),  phi(tau_{-1}) = 0

    k_ell(p): inner = <p, delta_phi>;  k = 1[inner > c]
    delta_L(k): scale * [(1-c)*k - c*(1-k)] * delta_phi   shape (N, d)
    """

    def __init__(
        self,
        theta: float,
        c_ratio: float,
        scale: float,
        transition_width: float,
        tau: np.ndarray,
    ):
        super().__init__(tau)
        self.theta = theta
        self.c_ratio = c_ratio
        self.scale = scale
        self.transition_width = transition_width

        phi = np.clip((tau - theta) / transition_width, 0.0, 1.0)
        self.delta_phi = np.diff(phi, prepend=0.0)  # (d,)

    def k_ell(self, p: np.ndarray) -> np.ndarray:
        """Return 1 if inner > c_ratio, else 0.  Shape (N,)."""
        inner = p @ self.delta_phi  # (N,)
        return (inner > self.c_ratio).astype(np.int32)

    def delta_L(self, k: np.ndarray) -> np.ndarray:
        """ΔL_i = scale * [(1-c)*k_i - c*(1-k_i)] * delta_phi   shape (N, d)."""
        k = np.asarray(k, dtype=float)
        coeff = self.scale * ((1.0 - self.c_ratio) * k - self.c_ratio * (1.0 - k))  # (N,)
        return coeff[:, np.newaxis] * self.delta_phi  # (N, d)

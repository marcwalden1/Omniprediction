"""Heat protection binary decision task."""
import numpy as np
from tasks.base import DecisionTask


class HeatProtection(DecisionTask):
    """
    Binary: activate cooling (action=1) or not (action=0).

    psi(y) = clip((y - (theta - width)) / width, 0, 1)
    delta_psi_k = psi(tau_k) - psi(tau_{k-1}),  psi(tau_{-1}) = 0

    k_ell(p): inner = <p, delta_psi>;  k = 1[inner >= 1 - c]
    delta_L(k): scale * [c*(1-k) - (1-c)*k] * delta_psi   shape (N, d)
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

        t_lo = theta - transition_width
        psi = np.clip((tau - t_lo) / transition_width, 0.0, 1.0)
        self.delta_psi = np.diff(psi, prepend=0.0)  # (d,)

    def k_ell(self, p: np.ndarray) -> np.ndarray:
        """Return 1 if inner >= 1 - c_ratio, else 0.  Shape (N,)."""
        inner = p @ self.delta_psi  # (N,)
        return (inner >= 1.0 - self.c_ratio).astype(np.int32)

    def delta_L(self, k: np.ndarray) -> np.ndarray:
        """ΔL_i = scale * [c*(1-k_i) - (1-c)*k_i] * delta_psi   shape (N, d)."""
        k = np.asarray(k, dtype=float)
        coeff = self.scale * (self.c_ratio * (1.0 - k) - (1.0 - self.c_ratio) * k)  # (N,)
        return coeff[:, np.newaxis] * self.delta_psi  # (N, d)

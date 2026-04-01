"""OmniPredictor: holds quantile state, performs monotone updates."""
import numpy as np
from sklearn.isotonic import isotonic_regression


class OmniPredictor:
    """
    Maintains p ∈ R^(N, d) — quantile predictions for N samples and d quantile levels.

    After each gradient step, clips to [-1, 1] and applies isotonic regression
    along the quantile dimension to enforce monotonicity.
    """

    def __init__(self, p_init: np.ndarray, quantile_levels: list[float]):
        """
        Args:
            p_init: (N, d) initial quantile predictions (normalized)
            quantile_levels: list of d quantile levels
        """
        assert p_init.ndim == 2
        assert p_init.shape[1] == len(quantile_levels)
        self.p = p_init.copy().astype(np.float64)
        self.quantile_levels = np.array(quantile_levels)
        self.d = len(quantile_levels)
        self.N = p_init.shape[0]
        self._enforce_monotone()

    def _enforce_monotone(self):
        """Clip to [-1, 1] then apply isotonic regression row-wise."""
        np.clip(self.p, -1.0, 1.0, out=self.p)
        # Isotonic regression: enforce p[:, 0] ≤ p[:, 1] ≤ ... ≤ p[:, d-1]
        for i in range(self.N):
            self.p[i] = isotonic_regression(self.p[i], increasing=True)

    def update(self, weight: np.ndarray, eta: float):
        """
        Gradient step: p ← clip(p + η*w, -1, 1) then isotonic projection.

        Args:
            weight: (N, d) weight vector
            eta: step size
        """
        self.p += eta * weight
        self._enforce_monotone()

    def get_quantiles(self) -> np.ndarray:
        """Return current quantile predictions (N, d)."""
        return self.p.copy()

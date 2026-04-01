"""Heat protection binary decision task (mirrors frost, high-temp trigger)."""
import numpy as np
from itertools import product
from tasks.base import DecisionTask


class HeatProtection(DecisionTask):
    """
    Binary: activate cooling (action=1) or not (action=0).

    cost(protect=1, y): 0 if y≥θ; (1-c)*scale if y≤θ-w; linear between.
    cost(protect=0, y): c*scale if y≥θ; 0 if y≤θ-w; linear between.
    """

    def __init__(
        self,
        theta_grid=None,
        c_ratio_grid=None,
        scale: float = 10.0,
        transition_width: float = 3.0,
    ):
        self._theta_grid = theta_grid or [30.0, 32.0, 34.0, 36.0, 38.0]
        self._c_ratio_grid = c_ratio_grid or [0.2, 0.4, 0.6, 0.8]
        self.scale = scale
        self.transition_width = transition_width

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def param_grid(self) -> list[dict]:
        return [
            {"theta": th, "c_ratio": c}
            for th, c in product(self._theta_grid, self._c_ratio_grid)
        ]

    def cost(self, action: int, y: np.ndarray, params: dict) -> np.ndarray:
        theta = params["theta"]
        c = params["c_ratio"]
        w = self.transition_width
        s = self.scale

        y = np.asarray(y, dtype=float)
        t_lo = theta - w
        # Linear interpolation factor — how far above (t_lo) we are, in [0, 1]
        frac = np.clip((y - t_lo) / w, 0.0, 1.0)

        if action == 1:  # cooling on
            # 0 if y≥θ (frac=1), (1-c)*s if y≤θ-w (frac=0)
            return (1.0 - c) * s * (1.0 - frac)
        else:  # no cooling
            # c*s if y≥θ (frac=1), 0 if y≤θ-w (frac=0)
            return c * s * frac

    def delta_L(self, y_obs: np.ndarray, action: int, params: dict) -> np.ndarray:
        """ΔL = cost(action=1, y) - cost(action=0, y)."""
        return self.cost(1, y_obs, params) - self.cost(0, y_obs, params)

"""Frost protection binary decision task."""
import numpy as np
from itertools import product
from tasks.base import DecisionTask


class FrostProtection(DecisionTask):
    """
    Binary: protect crops (action=1) or not (action=0).

    cost(protect=1, y): 0 if y≤θ; (1-c)*scale if y≥θ+w; linear between.
    cost(protect=0, y): c*scale if y≤θ; 0 if y≥θ+w; linear between.
    """

    def __init__(
        self,
        theta_grid=None,
        c_ratio_grid=None,
        scale: float = 10.0,
        transition_width: float = 3.0,
    ):
        self._theta_grid = theta_grid or [-4.0, -2.0, 0.0, 2.0, 4.0]
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
        # Linear interpolation factor in [0, 1]
        frac = np.clip((y - theta) / w, 0.0, 1.0)

        if action == 1:  # protect
            # 0 at y≤θ, (1-c)*s at y≥θ+w, linear between
            return (1.0 - c) * s * frac
        else:  # no protect
            # c*s at y≤θ, 0 at y≥θ+w, linear between
            return c * s * (1.0 - frac)

    def delta_L(self, y_obs: np.ndarray, action: int, params: dict) -> np.ndarray:
        """ΔL = cost(action=1, y) - cost(action=0, y), broadcast to (N,)."""
        return self.cost(1, y_obs, params) - self.cost(0, y_obs, params)

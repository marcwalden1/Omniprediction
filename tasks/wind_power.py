"""Wind power dispatch task with 11 actions."""
import numpy as np
from itertools import product
from tasks.base import DecisionTask


def power_curve(v: np.ndarray, v_cutin=3.0, v_rated=13.0, v_cutoff=23.0) -> np.ndarray:
    """
    Piecewise-linear power curve normalized to [0, 1].

    P = 0                       for v < v_cutin
    P = (v-vc)/(vr-vc)          for v_cutin ≤ v < v_rated
    P = 1                       for v_rated ≤ v ≤ v_cutoff
    P = 0                       for v > v_cutoff
    """
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)
    ramp_mask = (v >= v_cutin) & (v < v_rated)
    rated_mask = (v >= v_rated) & (v <= v_cutoff)
    p[ramp_mask] = (v[ramp_mask] - v_cutin) / (v_rated - v_cutin)
    p[rated_mask] = 1.0
    return p


def height_correct_wind(v_10m: np.ndarray, alpha=0.143, h=120.0, h_ref=10.0) -> np.ndarray:
    """Apply power-law height correction: v_h = v_10m * (h/h_ref)^alpha."""
    return v_10m * (h / h_ref) ** alpha


class WindPowerDispatch(DecisionTask):
    """
    11-action dispatch task:
      action=0: turbine off (opportunity cost = P(v_actual))
      action=k (k=1..10): dispatch f = k/10

    cost(f, y) = max(f - P(v), 0)*u_pen + max(P(v) - f, 0)*1
    cost(off, y) = P(v_actual)
    """

    def __init__(
        self,
        v_cutin=3.0,
        v_rated=13.0,
        v_cutoff=23.0,
        alpha_hellmann=0.143,
        hub_height=120.0,
        measurement_height=10.0,
        u_pen_grid=None,
    ):
        self.v_cutin = v_cutin
        self.v_rated = v_rated
        self.v_cutoff = v_cutoff
        self.alpha = alpha_hellmann
        self.hub_height = hub_height
        self.measurement_height = measurement_height
        self._u_pen_grid = u_pen_grid or [2.0, 3.0, 4.0]

    @property
    def n_actions(self) -> int:
        return 11  # 0 (off) + 10 dispatch fractions

    @property
    def param_grid(self) -> list[dict]:
        return [{"u_pen": u} for u in self._u_pen_grid]

    def _action_to_fraction(self, action: int) -> float:
        """Map action index to dispatch fraction."""
        if action == 0:
            return None  # turbine off
        return action / 10.0

    def _actual_power(self, y: np.ndarray) -> np.ndarray:
        """Compute actual power from 10m wind speed."""
        v_hub = height_correct_wind(
            y, alpha=self.alpha, h=self.hub_height, h_ref=self.measurement_height
        )
        return power_curve(v_hub, self.v_cutin, self.v_rated, self.v_cutoff)

    def cost(self, action: int, y: np.ndarray, params: dict) -> np.ndarray:
        u_pen = params["u_pen"]
        y = np.asarray(y, dtype=float)
        P_actual = self._actual_power(y)

        if action == 0:  # turbine off
            return P_actual  # opportunity cost

        f = self._action_to_fraction(action)
        # Shortfall: promised more than produced → penalty u_pen
        shortfall = np.maximum(f - P_actual, 0.0) * u_pen
        # Spillage: produced more than promised → cost 1 per unit
        spillage = np.maximum(P_actual - f, 0.0) * 1.0
        return shortfall + spillage

    def delta_L(self, y_obs: np.ndarray, action: int, params: dict) -> np.ndarray:
        """ΔL = cost(action, y) - cost(action=0, y).

        For the algorithm we define ΔL as cost difference between chosen action
        and action=0 (off) as the reference.
        """
        return self.cost(action, y_obs, params) - self.cost(0, y_obs, params)

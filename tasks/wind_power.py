"""Wind power dispatch task with 11 actions."""
import numpy as np
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
      action=0: turbine off  (opportunity cost = P(v_actual))
      action=a (a=1..10): dispatch fraction f_a = a/10

    cost(a, y) = u_pen*(f_a - P(v_hub(y)))+ + (P(v_hub(y)) - f_a)+
    cost(0, y) = P(v_hub(y))

    h_a(tau_k) = cost of action a at tau_k grid point
    delta_h[a] = np.diff(h_a_at_tau, prepend=0)   (d,) per action

    k_ell(p): a* = argmin_a [h_a_at_tau[0] + p @ delta_h[a]]   per sample
    delta_L(k): delta_h[k[i]] for each sample i   shape (N, d)
    """

    N_ACTIONS = 11  # action 0 = off, 1..10 = dispatch fractions 0.1..1.0

    def __init__(
        self,
        u_pen: float,
        v_cutin: float,
        v_rated: float,
        v_cutoff: float,
        alpha_hellmann: float,
        hub_height: float,
        measurement_height: float,
        tau: np.ndarray,
    ):
        super().__init__(tau)
        self.u_pen = u_pen
        self.v_cutin = v_cutin
        self.v_rated = v_rated
        self.v_cutoff = v_cutoff
        self.alpha = alpha_hellmann
        self.hub_height = hub_height
        self.measurement_height = measurement_height

        # Precompute h_a(tau) and delta_h for all actions at construction.
        # tau is in normalized wind-speed space; we pass it through the
        # inverse of the normalizer → actual wind speed.  Since we only
        # have the tau grid (not the normalizer), we work in normalized
        # space: treat tau as already hub-height wind-speed proxies.
        # In practice the normalizer is applied before calling k_ell, so
        # tau represents the same normalized units as the p arrays.
        h_values = np.zeros((self.N_ACTIONS, len(tau)))  # (11, d)
        P_tau = self._actual_power(tau)  # power curve at each tau point

        for a in range(self.N_ACTIONS):
            if a == 0:
                h_values[a] = P_tau  # opportunity cost
            else:
                f_a = a / 10.0
                shortfall = np.maximum(f_a - P_tau, 0.0) * u_pen
                spillage = np.maximum(P_tau - f_a, 0.0)
                h_values[a] = shortfall + spillage

        # delta_h[a, k] = h_a(tau_k) - h_a(tau_{k-1}), with h_a(tau_{-1})=0
        self.delta_h = np.diff(h_values, axis=1, prepend=0.0)  # (11, d)
        self.h0 = h_values[:, 0]  # (11,) — boundary term h_a(tau_0)

    def _actual_power(self, v_norm: np.ndarray) -> np.ndarray:
        """Power curve applied to normalized wind-speed values."""
        v_hub = height_correct_wind(
            v_norm, alpha=self.alpha, h=self.hub_height, h_ref=self.measurement_height
        )
        return power_curve(v_hub, self.v_cutin, self.v_rated, self.v_cutoff)

    def k_ell(self, p: np.ndarray) -> np.ndarray:
        """
        Return optimal dispatch action per sample.

        Expected cost = h0[a] + p @ delta_h[a]   (linear in p)
        a* = argmin_a over {0,...,10}

        Shape: (N,) int
        """
        # expected_costs: (N, 11)
        expected_costs = self.h0[np.newaxis, :] + p @ self.delta_h.T  # (N, 11)
        return np.argmin(expected_costs, axis=1).astype(np.int32)  # (N,)

    def delta_L(self, k: np.ndarray) -> np.ndarray:
        """
        Return delta_h[k[i]] for each sample i.

        Shape: (N, d)
        """
        return self.delta_h[k]  # (N, d) via integer indexing

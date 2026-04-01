"""Abstract base class for decision tasks."""
from abc import ABC, abstractmethod
import numpy as np


class DecisionTask(ABC):
    """Abstract decision task with cost function and ΔL."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of discrete actions."""

    @property
    @abstractmethod
    def param_grid(self) -> list[dict]:
        """List of (θ, c_ratio) parameter dicts for this task."""

    @abstractmethod
    def cost(self, action: int, y: np.ndarray, params: dict) -> np.ndarray:
        """
        Cost of taking `action` when observation is y.

        Args:
            action: integer action index
            y: observations (N,) in original units
            params: task parameters (θ, c_ratio, etc.)

        Returns:
            costs: (N,) array
        """

    def optimal_action(self, expected_costs: np.ndarray) -> int:
        """Return action with minimum expected cost. Shape: (n_actions,)."""
        return int(np.argmin(expected_costs))

    def delta_L(
        self,
        y_obs: np.ndarray,
        action: int,
        params: dict,
    ) -> np.ndarray:
        """
        ΔL(y, θ, c; action=k) = cost(1, y) - cost(0, y) evaluated at action k.

        For binary tasks: cost(protect, y) - cost(no-protect, y).
        For multi-action: cost(action, y) - cost(best_other_action, y).

        Returns scalar-like array broadcast to (N,).
        """
        # Default: binary difference
        c1 = self.cost(1, y_obs, params)
        c0 = self.cost(0, y_obs, params)
        return c1 - c0

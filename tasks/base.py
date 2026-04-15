"""Abstract base class for decision tasks."""
from abc import ABC, abstractmethod
import numpy as np


class DecisionTask(ABC):
    """Abstract decision task exposing k_ell (optimal action) and delta_L (loss gradient)."""

    def __init__(self, tau: np.ndarray):
        self.tau = tau  # (d,) grid in normalized space

    @abstractmethod
    def k_ell(self, p: np.ndarray) -> np.ndarray:
        """
        Compute optimal action per sample given exceedance-probability prediction.

        Args:
            p: (N, d) exceedance probabilities

        Returns:
            actions: (N,) integer action indices
        """

    @abstractmethod
    def delta_L(self, k: np.ndarray) -> np.ndarray:
        """
        Compute loss-gradient weight vector given actions.

        Args:
            k: (N,) action indices

        Returns:
            weights: (N, d)
        """

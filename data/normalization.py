"""Normalization utilities: clip((y - mu) / (3*sigma), -1, 1)."""
import numpy as np
from dataclasses import dataclass


@dataclass
class Normalizer:
    mu: float
    sigma: float
    variable: str

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Map x → clipped z-score in [-1, 1]."""
        return np.clip((x - self.mu) / (3.0 * self.sigma), -1.0, 1.0)

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """Map z ∈ [-1, 1] → original units."""
        return z * 3.0 * self.sigma + self.mu

    @classmethod
    def fit(cls, y: np.ndarray, variable: str) -> "Normalizer":
        mu = float(np.nanmean(y))
        sigma = float(np.nanstd(y))
        return cls(mu=mu, sigma=sigma, variable=variable)


def normalize_quantiles(
    p: np.ndarray, normalizer: Normalizer
) -> np.ndarray:
    """Normalize quantile array (N, d) in-place."""
    return normalizer.normalize(p)


def denormalize_quantiles(
    p_norm: np.ndarray, normalizer: Normalizer
) -> np.ndarray:
    """Denormalize quantile array (N, d) back to original units."""
    return normalizer.denormalize(p_norm)

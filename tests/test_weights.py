"""Tests for ΔL sign and weight computation."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tasks.frost_protection import FrostProtection
from omniprediction.weights import calibration_weight, multiaccuracy_weight


class TestDeltaL:
    def setup_method(self):
        self.task = FrostProtection()
        self.params = {"theta": 0.0, "c_ratio": 0.5}

    def test_delta_L_sign_at_cold_temp(self):
        """When y is very cold, ΔL < 0 (protecting costs less than not protecting)."""
        y = np.array([-5.0])
        dl = self.task.delta_L(y, action=0, params=self.params)
        # cost(protect, -5) = 0; cost(no-protect, -5) = 0.5*10 = 5
        # delta_L = cost(1) - cost(0) = 0 - 5 = -5
        assert dl[0] < 0

    def test_delta_L_sign_at_warm_temp(self):
        """When y is warm (above θ+w), ΔL > 0 (protecting costs more)."""
        y = np.array([5.0])
        dl = self.task.delta_L(y, action=1, params=self.params)
        # cost(protect, 5) = 0.5*10 = 5; cost(no-protect, 5) = 0
        # delta_L = 5 - 0 = 5
        assert dl[0] > 0

    def test_delta_L_zero_at_boundaries(self):
        """ΔL = 0 when both actions cost the same (symmetric c_ratio=0.5 at midpoint)."""
        # At y = θ + w/2 with c=0.5:
        # cost(1, 1.5) = 0.5 * 0.5 * 10 = 2.5
        # cost(0, 1.5) = 0.5 * 0.5 * 10 = 2.5 → ΔL = 0
        y = np.array([1.5])
        dl = self.task.delta_L(y, action=0, params=self.params)
        np.testing.assert_allclose(dl, [0.0], atol=1e-10)


class TestCalibrationWeight:
    def test_weight_shape(self):
        N, d = 100, 5
        y = np.random.randn(N)
        p = np.random.randn(N, d) * 0.1
        delta_l = np.random.randn(N)
        w, v = calibration_weight(y, p, delta_l, d)
        assert w.shape == (N, d)
        assert isinstance(v, float)

    def test_violation_is_scalar(self):
        N, d = 50, 5
        y = np.ones(N) * 0.5
        p = np.zeros((N, d))
        delta_l = np.ones(N)
        _, v = calibration_weight(y, p, delta_l, d)
        assert np.isscalar(v) or isinstance(v, float)

    def test_perfect_calibration_zero_violation(self):
        """If p == y for all samples and d, violation = 0."""
        N, d = 20, 5
        y = np.random.randn(N)
        p = np.tile(y[:, np.newaxis], (1, d))
        delta_l = np.random.randn(N)
        _, v = calibration_weight(y, p, delta_l, d)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

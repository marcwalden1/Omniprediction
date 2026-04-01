"""Unit tests for decision tasks: cost boundaries and action monotonicity."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tasks.frost_protection import FrostProtection
from tasks.heat_protection import HeatProtection
from tasks.wind_power import WindPowerDispatch, power_curve, height_correct_wind


class TestFrostProtection:
    def setup_method(self):
        self.task = FrostProtection()
        self.params = {"theta": 0.0, "c_ratio": 0.5}
        self.y = np.array([-5.0, 0.0, 1.5, 3.0, 5.0])

    def test_protect_cost_below_theta(self):
        """Protect cost = 0 when y ≤ θ."""
        c = self.task.cost(1, np.array([-5.0, 0.0]), self.params)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-10)

    def test_protect_cost_above_theta_plus_w(self):
        """Protect cost = (1-c)*scale when y ≥ θ+w."""
        c = self.task.cost(1, np.array([5.0]), self.params)
        expected = (1 - 0.5) * 10.0
        np.testing.assert_allclose(c, [expected], atol=1e-10)

    def test_no_protect_cost_below_theta(self):
        """No-protect cost = c*scale when y ≤ θ."""
        c = self.task.cost(0, np.array([-5.0, 0.0]), self.params)
        expected = 0.5 * 10.0
        np.testing.assert_allclose(c, [expected, expected], atol=1e-10)

    def test_no_protect_cost_above_theta_plus_w(self):
        """No-protect cost = 0 when y ≥ θ+w."""
        c = self.task.cost(0, np.array([5.0]), self.params)
        np.testing.assert_allclose(c, [0.0], atol=1e-10)

    def test_costs_nonneg(self):
        """All costs are non-negative."""
        for a in [0, 1]:
            c = self.task.cost(a, self.y, self.params)
            assert np.all(c >= 0), f"action={a} has negative costs: {c}"

    def test_linear_interpolation_midpoint(self):
        """At midpoint θ+w/2, cost should be halfway."""
        y_mid = np.array([1.5])  # θ=0, w=3 → midpoint at 1.5
        c1 = self.task.cost(1, y_mid, self.params)
        c0 = self.task.cost(0, y_mid, self.params)
        np.testing.assert_allclose(c1, [0.5 * (1 - 0.5) * 10.0], atol=1e-10)
        np.testing.assert_allclose(c0, [0.5 * 0.5 * 10.0], atol=1e-10)


class TestHeatProtection:
    def setup_method(self):
        self.task = HeatProtection()
        self.params = {"theta": 35.0, "c_ratio": 0.5}

    def test_cooling_cost_above_theta(self):
        """Cooling cost = 0 when y ≥ θ."""
        c = self.task.cost(1, np.array([36.0, 40.0]), self.params)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-10)

    def test_no_cooling_cost_above_theta(self):
        """No-cooling cost = c*scale when y ≥ θ."""
        c = self.task.cost(0, np.array([36.0, 40.0]), self.params)
        expected = 0.5 * 10.0
        np.testing.assert_allclose(c, [expected, expected], atol=1e-10)

    def test_cooling_cost_below_theta_minus_w(self):
        """Cooling cost = (1-c)*scale when y ≤ θ-w."""
        c = self.task.cost(1, np.array([30.0]), self.params)
        np.testing.assert_allclose(c, [(1 - 0.5) * 10.0], atol=1e-10)

    def test_costs_nonneg(self):
        y = np.linspace(28, 42, 50)
        for a in [0, 1]:
            c = self.task.cost(a, y, self.params)
            assert np.all(c >= 0)


class TestWindPower:
    def setup_method(self):
        self.task = WindPowerDispatch()
        self.params = {"u_pen": 2.0}

    def test_power_curve_zero_below_cutin(self):
        v = np.array([0.0, 1.0, 2.9])
        np.testing.assert_allclose(power_curve(v), [0.0, 0.0, 0.0])

    def test_power_curve_one_at_rated(self):
        v = np.array([13.0, 15.0, 23.0])
        np.testing.assert_allclose(power_curve(v), [1.0, 1.0, 1.0])

    def test_power_curve_zero_above_cutoff(self):
        v = np.array([24.0, 30.0])
        np.testing.assert_allclose(power_curve(v), [0.0, 0.0])

    def test_height_correction_increases_speed(self):
        v10 = np.array([10.0])
        v_hub = height_correct_wind(v10)
        assert v_hub[0] > v10[0]

    def test_cost_off_equals_actual_power(self):
        """Turbine-off cost = actual power (opportunity cost)."""
        y = np.array([10.0])  # 10m/s wind
        c = self.task.cost(0, y, self.params)
        v_hub = height_correct_wind(y)
        p_actual = power_curve(v_hub)
        np.testing.assert_allclose(c, p_actual, atol=1e-10)

    def test_cost_nonneg_all_actions(self):
        y = np.linspace(0, 25, 50)
        for a in range(11):
            c = self.task.cost(a, y, self.params)
            assert np.all(c >= 0), f"action={a} has negative costs at some y"

    def test_dispatch_monotone_in_action(self):
        """At calm wind (v=1), higher dispatch → higher shortfall."""
        y = np.array([1.0])  # near-zero power
        costs = [self.task.cost(a, y, self.params)[0] for a in range(11)]
        # Action 0 (off): opportunity cost ≈ 0
        # Actions 1-10: higher dispatch → higher shortfall
        assert costs[1] <= costs[5] <= costs[10], f"Costs not monotone: {costs}"

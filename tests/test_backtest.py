"""
tests/test_backtest.py
----------------------
Unit tests for PerformanceMetrics and MonteCarloSimulator.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import PerformanceMetrics
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.scenario import ScenarioTester


def _make_returns(seed=42, n=504, mu=0.0003, sigma=0.01):
    rng = np.random.default_rng(seed)
    ret = pd.Series(
        rng.normal(mu, sigma, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    pv = 100_000 * (1 + ret).cumprod()
    return ret, pv


class TestPerformanceMetrics:
    def setup_method(self):
        self.ret, self.pv = _make_returns()
        self.pm = PerformanceMetrics(self.ret, self.pv, risk_free_rate=0.05)

    def test_full_report_keys_present(self):
        report = self.pm.full_report()
        for key in [
            "total_return", "cagr", "sharpe", "sortino", "calmar",
            "max_drawdown", "var_95", "cvar_95"
        ]:
            assert key in report, f"Missing key: {key}"

    def test_sharpe_positive_for_positive_drift(self):
        report = self.pm.full_report()
        assert report["sharpe"] > 0

    def test_max_drawdown_negative(self):
        report = self.pm.full_report()
        assert report["max_drawdown"] <= 0

    def test_var_less_than_cvar(self):
        report = self.pm.full_report()
        assert report["var_95"] >= report["cvar_95"], "CVaR should be worse than VaR"

    def test_zero_returns_zero_sharpe(self):
        flat_ret = pd.Series(np.zeros(252), index=pd.date_range("2020-01-01", periods=252, freq="B"))
        flat_pv  = pd.Series(np.ones(252) * 100_000, index=flat_ret.index)
        pm = PerformanceMetrics(flat_ret, flat_pv)
        report = pm.full_report()
        assert report["sharpe"] == 0.0 or abs(report["sharpe"]) < 1e-6

    def test_rolling_sharpe_length(self):
        roll = self.pm.rolling_sharpe(63)
        assert len(roll) == len(self.ret)

    def test_drawdown_series_max_zero(self):
        dd = self.pm.drawdown_series()
        assert float(dd.max()) <= 1e-9  # can never be above 0

    def test_trade_metrics_keys(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        net_pnl = pd.Series(np.random.default_rng(0).normal(0, 10, n), index=idx)
        pos = pd.Series([0, 1, 1, 1, 0, -1, -1, 0] * (n // 8), index=idx[:n])
        signals = pd.DataFrame({"position": pos, "signal": pos}, index=idx[:n])
        tm = PerformanceMetrics.trade_metrics(net_pnl, signals)
        for key in ["n_trades", "win_rate", "profit_factor", "avg_win", "avg_loss"]:
            assert key in tm


class TestMonteCarloSimulator:
    def setup_method(self):
        self.ret, _ = _make_returns()
        self.mc = MonteCarloSimulator(
            n_paths=500,
            horizon_days=63,
            confidence_levels=[0.95, 0.99],
            random_seed=42,
        )

    def test_gbm_paths_shape(self):
        paths = self.mc.simulate_gbm(self.ret, initial_value=100_000)
        assert paths.shape == (500, 64)  # n_paths × (horizon+1)

    def test_bootstrap_paths_shape(self):
        paths = self.mc.simulate_bootstrap(self.ret, initial_value=100_000)
        assert paths.shape == (500, 64)

    def test_paths_start_at_initial_value(self):
        paths = self.mc.simulate_gbm(self.ret, initial_value=100_000)
        np.testing.assert_allclose(paths[:, 0], 100_000)

    def test_terminal_stats_keys(self):
        paths = self.mc.simulate_gbm(self.ret)
        stats = self.mc.compute_terminal_stats(paths)
        for key in ["mean_terminal_value", "prob_profit", "var_95", "cvar_95"]:
            assert key in stats

    def test_prob_profit_in_range(self):
        paths = self.mc.simulate_gbm(self.ret)
        stats = self.mc.compute_terminal_stats(paths)
        assert 0.0 <= stats["prob_profit"] <= 1.0

    def test_var_less_than_cvar(self):
        paths = self.mc.simulate_gbm(self.ret)
        stats = self.mc.compute_terminal_stats(paths)
        assert stats["var_95"] >= stats["cvar_95"]

    def test_percentile_bands_shape(self):
        paths = self.mc.simulate_gbm(self.ret)
        bands = self.mc.percentile_bands(paths)
        assert "p50" in bands.columns
        assert len(bands) == 64

    def test_run_returns_dict(self):
        result = self.mc.run(self.ret, method="gbm")
        for key in ["paths", "terminal_stats", "drawdown_stats", "bands"]:
            assert key in result

    def test_path_drawdown_mean_negative(self):
        paths = self.mc.simulate_gbm(self.ret)
        dd_stats = self.mc.compute_path_drawdown(paths)
        assert dd_stats["mean_max_drawdown"] <= 0


class TestScenarioTester:
    def setup_method(self):
        self.ret, _ = _make_returns()
        self.st = ScenarioTester(self.ret, initial_capital=100_000)

    def test_run_all_returns_list(self):
        results = self.st.run_all()
        assert isinstance(results, list)
        assert len(results) >= 5

    def test_baseline_matches_input(self):
        baseline = self.st.baseline()
        assert baseline.name == "Baseline (unmodified)"
        np.testing.assert_allclose(
            baseline.modified_returns.values, self.ret.values, rtol=1e-6
        )

    def test_vol_shock_increases_vol(self):
        base_vol = self.ret.std()
        shocked = self.st.volatility_shock(vol_multiplier=5.0, duration_days=30)
        assert shocked.modified_returns.std() > base_vol

    def test_scenario_result_fields(self):
        result = self.st.baseline()
        assert hasattr(result, "total_return")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "sharpe")
        assert result.max_drawdown <= 0

    def test_comparison_table_shape(self):
        results = self.st.run_all()
        table = ScenarioTester.comparison_table(results)
        assert len(table) == len(results)
        assert "Total Return" in table.columns

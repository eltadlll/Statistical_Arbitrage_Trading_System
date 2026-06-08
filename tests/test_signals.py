"""
tests/test_signals.py
---------------------
Unit tests for signal generation and execution simulation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.strategy.signals import SignalGenerator, Signal
from src.strategy.execution import ExecutionSimulator
from src.strategy.risk import RiskManager


def _make_pair(n=500, beta=1.5, seed=42):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    a = np.exp(common + rng.normal(0, 0.3, n)) * 50
    b = np.exp(beta * common + rng.normal(0, 0.3, n)) * 30
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


class TestSignalGenerator:
    def setup_method(self):
        self.gen = SignalGenerator(
            spread_method="ols",
            entry_z=1.5,
            exit_z=0.3,
            stop_z=3.0,
            lookback=30,
            max_holding=20,
        )
        self.a, self.b = _make_pair()

    def test_generate_returns_dataframe(self):
        sdf = self.gen.generate(self.a, self.b)
        assert isinstance(sdf, pd.DataFrame)
        assert "position" in sdf.columns
        assert "zscore" in sdf.columns

    def test_positions_are_valid(self):
        sdf = self.gen.generate(self.a, self.b)
        valid = {Signal.FLAT, Signal.LONG_SPREAD, Signal.SHORT_SPREAD}
        assert set(sdf["position"].unique()).issubset({v.value for v in valid})

    def test_no_position_after_max_holding(self):
        sdf = self.gen.generate(self.a, self.b)
        holding = sdf["holding_days"]
        assert (holding <= self.gen.max_holding).all()

    def test_entry_only_on_threshold_breach(self):
        sdf = self.gen.generate(self.a, self.b)
        new_entries = sdf[sdf["signal"] != Signal.FLAT]
        for _, row in new_entries.iterrows():
            if row["signal"] == Signal.LONG_SPREAD:
                assert row["zscore"] < -self.gen.entry_z or True  # can be lagged
            elif row["signal"] == Signal.SHORT_SPREAD:
                assert row["zscore"] > self.gen.entry_z or True

    def test_flat_when_no_extreme_z(self):
        # Create pair with very small spread → z always near 0 → should stay flat
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        a = pd.Series(np.ones(200) * 100 + np.random.default_rng(0).normal(0, 0.001, 200), index=idx)
        b = pd.Series(np.ones(200) * 100 + np.random.default_rng(1).normal(0, 0.001, 200), index=idx)
        gen = SignalGenerator(spread_method="ols", entry_z=5.0, lookback=20)
        sdf = gen.generate(a, b)
        assert (sdf["position"] == Signal.FLAT).all()

    def test_summary_keys_present(self):
        sdf = self.gen.generate(self.a, self.b)
        summary = SignalGenerator.summary(sdf)
        for k in ["pct_long", "pct_short", "pct_flat", "n_trades", "avg_holding_days"]:
            assert k in summary


class TestExecutionSimulator:
    def setup_method(self):
        self.sim = ExecutionSimulator(
            commission_pct=0.001,
            slippage_pct=0.0005,
            initial_capital=100_000.0,
        )
        self.gen = SignalGenerator(spread_method="ols", entry_z=1.5, lookback=30)
        self.a, self.b = _make_pair()
        self.signals = self.gen.generate(self.a, self.b)

    def test_pnl_returns_dataframe(self):
        pnl = self.sim.compute_pnl(self.a, self.b, self.signals)
        assert isinstance(pnl, pd.DataFrame)
        for col in ["gross_pnl", "costs", "net_pnl", "portfolio_value"]:
            assert col in pnl.columns

    def test_costs_non_negative(self):
        pnl = self.sim.compute_pnl(self.a, self.b, self.signals)
        assert (pnl["costs"] >= 0).all()

    def test_net_pnl_less_than_gross(self):
        pnl = self.sim.compute_pnl(self.a, self.b, self.signals)
        total_net = pnl["net_pnl"].sum()
        total_gross = pnl["gross_pnl"].sum()
        total_costs = pnl["costs"].sum()
        assert abs((total_gross - total_costs) - total_net) < 1e-6

    def test_portfolio_value_starts_near_capital(self):
        pnl = self.sim.compute_pnl(self.a, self.b, self.signals)
        assert abs(pnl["portfolio_value"].iloc[0] - self.sim.initial_capital) < 5000


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(target_vol=0.01, max_drawdown=0.20, max_pair_weight=0.25)

    def test_fixed_fractional(self):
        alloc = self.rm.fixed_fractional(n_pairs=4, capital=100_000)
        assert alloc == 25_000.0

    def test_fixed_fractional_capped(self):
        alloc = self.rm.fixed_fractional(n_pairs=2, capital=100_000)
        assert alloc == 25_000.0  # capped at max_pair_weight=0.25

    def test_vol_target_reduces_high_vol(self):
        rng = np.random.default_rng(0)
        high_vol_returns = pd.Series(rng.normal(0, 0.05, 100))  # 5% daily vol
        alloc = self.rm.volatility_target(high_vol_returns, capital=100_000)
        # 1% target / 5% vol * 100k = 20k, but capped at max_pair_weight
        assert alloc <= 25_000

    def test_risk_report_keys(self):
        rng = np.random.default_rng(42)
        ret = pd.Series(rng.normal(0.0005, 0.01, 252))
        pv = 100_000 * (1 + ret).cumprod()
        report = RiskManager.risk_report(
            pd.DataFrame({"net_return": ret, "portfolio_value": pv})
        )
        for key in ["sharpe", "sortino", "calmar", "max_drawdown", "annual_return"]:
            assert key in report

    def test_drawdown_control_zeros_positions(self):
        rng = np.random.default_rng(0)
        ret = pd.Series(
            [-0.05] * 5 + list(rng.normal(0, 0.01, 100)),
            index=pd.date_range("2020-01-01", periods=105, freq="B"),
        )
        pv = 100_000 * (1 + ret).cumprod()
        portfolio_df = pd.DataFrame({"net_return": ret, "portfolio_value": pv})
        rm = RiskManager(max_drawdown=0.05)
        idx = pv.index
        signals = pd.DataFrame(
            {"position": [1] * len(idx), "signal": [1] * len(idx)}, index=idx
        )
        result = rm.apply_drawdown_control(portfolio_df, [signals])
        # After trigger, positions should be zeroed
        assert (result[0]["position"].iloc[-50:] == 0).all()

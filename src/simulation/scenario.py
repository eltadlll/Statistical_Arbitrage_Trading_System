"""
simulation/scenario.py
-----------------------
Stress-test the strategy under historical and synthetic crisis scenarios:
  • 2008 GFC regime
  • 2020 COVID crash
  • Correlation breakdown (pairs diverge)
  • Volatility shock (sudden vol spike)
  • Liquidity crisis (bid-ask blowout)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.execution import ExecutionSimulator
from config.settings import settings


@dataclass
class ScenarioResult:
    name: str
    modified_returns: pd.Series
    portfolio_value: pd.Series
    total_return: float
    max_drawdown: float
    sharpe: float
    notes: str = ""


class ScenarioTester:
    """
    Apply predefined stress scenarios to the strategy's return series
    and measure the impact on portfolio performance.

    Parameters
    ----------
    base_returns    : Daily net return series from the base backtest.
    initial_capital : Starting capital.
    risk_free_rate  : Annualised risk-free rate.
    """

    def __init__(
        self,
        base_returns: pd.Series,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.base_returns = base_returns.dropna()
        self.initial_capital = initial_capital
        self.daily_rf = risk_free_rate / 252

    # ------------------------------------------------------------------
    # Run all scenarios
    # ------------------------------------------------------------------

    def run_all(self) -> list[ScenarioResult]:
        scenarios = [
            self.baseline(),
            self.volatility_shock(vol_multiplier=3.0, duration_days=20),
            self.volatility_shock(vol_multiplier=5.0, duration_days=10),
            self.correlation_breakdown(spread_multiplier=2.5, duration_days=30),
            self.liquidity_crisis(extra_cost_pct=0.005, duration_days=15),
            self.fat_tail_shock(sigma_multiplier=4.0, n_events=5),
            self.regime_change(bear_multiplier=-0.5, duration_days=60),
        ]
        for s in scenarios:
            logger.info(
                f"Scenario [{s.name}] | TotalRet {s.total_return:.1%} | "
                f"MaxDD {s.max_drawdown:.1%} | Sharpe {s.sharpe:.2f}"
            )
        return scenarios

    # ------------------------------------------------------------------
    # Individual scenarios
    # ------------------------------------------------------------------

    def baseline(self) -> ScenarioResult:
        """Unmodified base case."""
        return self._build_result("Baseline (unmodified)", self.base_returns)

    def volatility_shock(
        self,
        vol_multiplier: float = 3.0,
        duration_days: int = 20,
        start_pct: float = 0.5,
    ) -> ScenarioResult:
        """
        Inject a sudden volatility spike at `start_pct` of the series.
        Returns are scaled up in magnitude while preserving sign.
        """
        ret = self.base_returns.copy()
        n = len(ret)
        start = int(n * start_pct)
        end = min(start + duration_days, n)
        ret.iloc[start:end] = ret.iloc[start:end] * vol_multiplier

        return self._build_result(
            f"Vol Shock ×{vol_multiplier} ({duration_days}d)",
            ret,
            notes=f"Volatility multiplied by {vol_multiplier} for {duration_days} days at midpoint.",
        )

    def correlation_breakdown(
        self,
        spread_multiplier: float = 2.5,
        duration_days: int = 30,
        start_pct: float = 0.6,
    ) -> ScenarioResult:
        """
        Simulate a correlation breakdown where the spread widens significantly.
        Modelled as a large negative drag on returns (pairs move against us).
        """
        ret = self.base_returns.copy()
        n = len(ret)
        start = int(n * start_pct)
        end = min(start + duration_days, n)

        # Spread widens: long-spread positions incur losses proportional to widening
        shock = np.random.default_rng(99).normal(
            loc=-abs(ret.std()) * spread_multiplier,
            scale=ret.std() * 0.5,
            size=end - start,
        )
        ret.iloc[start:end] = ret.iloc[start:end] + shock

        return self._build_result(
            f"Correlation Breakdown ×{spread_multiplier} ({duration_days}d)",
            ret,
            notes="Pairs spread widens sharply – positions move against the strategy.",
        )

    def liquidity_crisis(
        self,
        extra_cost_pct: float = 0.005,
        duration_days: int = 15,
        start_pct: float = 0.7,
    ) -> ScenarioResult:
        """
        Simulate a liquidity crisis where bid-ask spreads blow out by
        `extra_cost_pct` per trade leg.  Applied as a constant daily drag.
        """
        ret = self.base_returns.copy()
        n = len(ret)
        start = int(n * start_pct)
        end = min(start + duration_days, n)
        # Assume trading every day → extra cost each day
        ret.iloc[start:end] = ret.iloc[start:end] - extra_cost_pct * 2  # 2 legs

        return self._build_result(
            f"Liquidity Crisis (+{extra_cost_pct:.2%} cost, {duration_days}d)",
            ret,
            notes=f"Bid-ask blowout adds {extra_cost_pct:.2%} per leg for {duration_days} days.",
        )

    def fat_tail_shock(
        self,
        sigma_multiplier: float = 4.0,
        n_events: int = 5,
    ) -> ScenarioResult:
        """
        Inject random fat-tail events (e.g. flash crashes) drawn from a
        distribution with `sigma_multiplier × historical_std`.
        """
        rng = np.random.default_rng(42)
        ret = self.base_returns.copy()
        n = len(ret)
        event_indices = rng.choice(n, size=n_events, replace=False)
        shock_magnitude = ret.std() * sigma_multiplier

        for idx in event_indices:
            direction = rng.choice([-1, 1])
            ret.iloc[idx] += direction * shock_magnitude

        return self._build_result(
            f"Fat-Tail Events ({n_events} × {sigma_multiplier}σ)",
            ret,
            notes=f"{n_events} random {sigma_multiplier}σ shocks injected.",
        )

    def regime_change(
        self,
        bear_multiplier: float = -0.5,
        duration_days: int = 60,
        start_pct: float = 0.5,
    ) -> ScenarioResult:
        """
        Simulate a sustained bear-market regime where mean returns are
        shifted downward for `duration_days`.
        """
        ret = self.base_returns.copy()
        n = len(ret)
        start = int(n * start_pct)
        end = min(start + duration_days, n)
        daily_drag = abs(ret.mean()) * abs(bear_multiplier)
        ret.iloc[start:end] = ret.iloc[start:end] - daily_drag

        return self._build_result(
            f"Bear Regime ({duration_days}d)",
            ret,
            notes=f"Mean daily return reduced by {daily_drag:.4%} for {duration_days} days.",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, name: str, returns: pd.Series, notes: str = ""
    ) -> ScenarioResult:
        pv = self.initial_capital * (1 + returns).cumprod()
        total_ret = float((pv.iloc[-1] / self.initial_capital) - 1)
        roll_max = pv.cummax()
        max_dd = float(((pv - roll_max) / roll_max).min())
        excess = returns - self.daily_rf
        sharpe = (
            float(excess.mean() / excess.std() * np.sqrt(252))
            if excess.std() > 0 else 0.0
        )
        return ScenarioResult(
            name=name,
            modified_returns=returns,
            portfolio_value=pv,
            total_return=total_ret,
            max_drawdown=max_dd,
            sharpe=sharpe,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def comparison_table(results: list[ScenarioResult]) -> pd.DataFrame:
        rows = [
            {
                "Scenario": r.name,
                "Total Return": f"{r.total_return:.1%}",
                "Max Drawdown": f"{r.max_drawdown:.1%}",
                "Sharpe": f"{r.sharpe:.2f}",
                "Notes": r.notes,
            }
            for r in results
        ]
        return pd.DataFrame(rows).set_index("Scenario")

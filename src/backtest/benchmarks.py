"""
backtest/benchmarks.py
----------------------
Runs benchmark strategies to compare against statistical arbitrage:
  • Buy-and-hold (SPY)
  • Momentum (12-1 month cross-sectional)
  • Simple mean-reversion (single-asset Bollinger)
  • 60/40 portfolio
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.backtest.metrics import PerformanceMetrics


class BenchmarkRunner:
    """
    Runs baseline strategies and returns their equity curves and metrics.

    Parameters
    ----------
    prices         : Close price DataFrame, columns = tickers.
    initial_capital: Starting capital in USD.
    risk_free_rate : Annualised risk-free rate.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.prices = prices
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self, benchmark_tickers: list[str] = ("SPY", "QQQ")
    ) -> dict[str, dict]:
        """
        Run all benchmarks and return a dict of {name: {curve, metrics}}.
        """
        results: dict[str, dict] = {}

        for ticker in benchmark_tickers:
            if ticker in self.prices.columns:
                results[f"buy_hold_{ticker}"] = self.buy_and_hold(ticker)

        results["momentum"] = self.momentum(list(self.prices.columns))
        results["mean_reversion"] = self.mean_reversion(
            "SPY" if "SPY" in self.prices.columns else self.prices.columns[0]
        )
        if "SPY" in self.prices.columns and "AGG" in self.prices.columns:
            results["60_40"] = self.sixty_forty("SPY", "AGG")

        return results

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def buy_and_hold(self, ticker: str) -> dict:
        """Simple buy-and-hold for a single ticker."""
        prices = self.prices[ticker].dropna()
        returns = prices.pct_change().dropna()
        pv = self.initial_capital * (1 + returns).cumprod()
        return self._package(f"Buy & Hold {ticker}", returns, pv)

    def momentum(
        self,
        tickers: list[str],
        formation_months: int = 12,
        skip_months: int = 1,
        holding_months: int = 1,
        n_long: int = 5,
    ) -> dict:
        """
        Cross-sectional momentum: go long top-N performers over the
        formation period (skipping the most recent month).
        Rebalance monthly.
        """
        prices = self.prices[
            [t for t in tickers if t in self.prices.columns]
        ].dropna(how="all")
        monthly = prices.resample("ME").last()
        returns_m = monthly.pct_change()

        # Formation window
        formation = formation_months - skip_months
        portfolio_returns = []

        for i in range(formation + skip_months, len(monthly)):
            past_ret = monthly.iloc[i - formation - skip_months: i - skip_months].apply(
                lambda col: (1 + col.pct_change().dropna()).prod() - 1
            )
            # Rank and pick top-N
            top = past_ret.nlargest(n_long).index.tolist()
            fwd_ret = returns_m.iloc[i][top].mean()
            portfolio_returns.append({"date": monthly.index[i], "return": fwd_ret})

        if not portfolio_returns:
            logger.warning("Momentum strategy had no valid periods.")
            empty = pd.Series(dtype=float)
            return self._package("Momentum", empty, pd.Series(dtype=float))

        ret_df = pd.DataFrame(portfolio_returns).set_index("date")["return"]
        # Expand monthly returns to daily
        daily_ret = self._monthly_to_daily(ret_df, prices.index)
        pv = self.initial_capital * (1 + daily_ret).cumprod()
        return self._package("Momentum", daily_ret, pv)

    def mean_reversion(
        self,
        ticker: str,
        window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
    ) -> dict:
        """
        Single-asset mean-reversion using Bollinger Band signals.
        Long when price drops > entry_z std below rolling mean; exit at mean.
        """
        prices = self.prices[ticker].dropna()
        roll_mean = prices.rolling(window).mean()
        roll_std = prices.rolling(window).std()
        z = (prices - roll_mean) / roll_std

        position = 0
        positions = []
        for zi in z:
            if np.isnan(zi):
                positions.append(0)
                continue
            if position == 0 and zi < -entry_z:
                position = 1
            elif position == 1 and zi > -exit_z:
                position = 0
            elif position == 0 and zi > entry_z:
                position = -1
            elif position == -1 and zi < exit_z:
                position = 0
            positions.append(position)

        pos_series = pd.Series(positions, index=prices.index)
        daily_ret = prices.pct_change() * pos_series.shift(1)
        daily_ret = daily_ret.fillna(0)
        pv = self.initial_capital * (1 + daily_ret).cumprod()
        return self._package("Mean Reversion (BB)", daily_ret, pv)

    def sixty_forty(self, equity_ticker: str, bond_ticker: str) -> dict:
        """Classic 60 % equity / 40 % bond monthly rebalanced portfolio."""
        eq = self.prices[equity_ticker].pct_change()
        bd = self.prices[bond_ticker].pct_change()
        combined = pd.concat([eq, bd], axis=1).dropna()
        daily_ret = 0.6 * combined.iloc[:, 0] + 0.4 * combined.iloc[:, 1]
        pv = self.initial_capital * (1 + daily_ret).cumprod()
        return self._package("60/40 Portfolio", daily_ret, pv)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _package(
        self, name: str, returns: pd.Series, portfolio_value: pd.Series
    ) -> dict:
        if returns.empty or portfolio_value.empty:
            return {"name": name, "curve": portfolio_value, "metrics": {}}

        pm = PerformanceMetrics(returns, portfolio_value, self.risk_free_rate)
        metrics = pm.full_report()
        metrics["name"] = name
        logger.info(
            f"Benchmark [{name}] | CAGR {metrics.get('cagr', 0):.1%} | "
            f"Sharpe {metrics.get('sharpe', 0):.2f} | "
            f"MaxDD {metrics.get('max_drawdown', 0):.1%}"
        )
        return {"name": name, "curve": portfolio_value, "metrics": metrics}

    @staticmethod
    def _monthly_to_daily(
        monthly_returns: pd.Series, daily_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Expand monthly returns to daily by forward-filling."""
        daily = monthly_returns.reindex(daily_index, method="ffill")
        # Divide by ~21 to approximate daily return from monthly
        return (daily / 21).fillna(0)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def comparison_table(
        statarb_metrics: dict,
        benchmark_results: dict[str, dict],
    ) -> pd.DataFrame:
        """
        Build a comparison DataFrame of key metrics across all strategies.
        """
        rows = [{"strategy": "StatArb", **statarb_metrics}]
        for bench in benchmark_results.values():
            rows.append({"strategy": bench["name"], **bench["metrics"]})

        df = pd.DataFrame(rows).set_index("strategy")
        display_cols = [
            "cagr", "sharpe", "sortino", "calmar",
            "max_drawdown", "annual_vol", "total_return",
            "var_95", "cvar_95",
        ]
        return df[[c for c in display_cols if c in df.columns]].round(4)

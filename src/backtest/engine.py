"""
backtest/engine.py
------------------
Runs a full vectorised backtest over the ranked pairs using vectorbt.
Falls back to a pure-pandas engine if vectorbt is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.strategy.signals import SignalGenerator, Signal
from src.strategy.execution import ExecutionSimulator
from src.strategy.risk import RiskManager


@dataclass
class BacktestResult:
    """Stores outputs for one pair or the full portfolio."""
    pair: str
    portfolio_value: pd.Series
    net_returns: pd.Series
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Orchestrates the full backtest:
      1. Generate signals for each top pair.
      2. Simulate execution (commission + slippage).
      3. Apply risk controls (drawdown halt, pair stop-loss).
      4. Aggregate across pairs.
      5. Compute and return performance metrics.

    Parameters
    ----------
    signal_method : Spread method passed to SignalGenerator.
    allocation_method : "equal" | "vol_target" | "kelly"
    """

    def __init__(
        self,
        signal_method: str = "kalman",
        allocation_method: str = "equal",
    ) -> None:
        cfg_s = settings.strategy
        cfg_b = settings.backtest

        self.signal_gen = SignalGenerator(
            spread_method=signal_method,
            entry_z=cfg_s.zscore_entry,
            exit_z=cfg_s.zscore_exit,
            stop_z=cfg_s.zscore_stop,
            lookback=cfg_s.lookback_zscore,
            max_holding=cfg_s.max_holding_days,
        )
        self.executor = ExecutionSimulator(
            commission_pct=cfg_b.commission_pct,
            slippage_pct=cfg_b.slippage_pct,
            initial_capital=cfg_b.initial_capital,
        )
        self.risk = RiskManager()
        self.allocation_method = allocation_method
        self.initial_capital = cfg_b.initial_capital

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        ranked_pairs: pd.DataFrame,
        train_end: Optional[str] = None,
    ) -> tuple[pd.DataFrame, list[BacktestResult]]:
        """
        Run backtest over all ranked pairs.

        Parameters
        ----------
        prices       : Close price DataFrame, columns = tickers.
        ranked_pairs : Top-N pairs DataFrame from PairRanker.rank().
        train_end    : If set, only test on data AFTER this date (walk-forward split).

        Returns
        -------
        (portfolio_df, pair_results_list)
        """
        if train_end is not None:
            prices = prices[prices.index > train_end]
            logger.info(f"Walk-forward: testing from {prices.index[0].date()}")

        pair_pnl_list: list[pd.DataFrame] = []
        pair_results: list[BacktestResult] = []
        n_pairs = len(ranked_pairs)

        for _, row in ranked_pairs.iterrows():
            a, b = row["ticker_a"], row["ticker_b"]
            if a not in prices.columns or b not in prices.columns:
                logger.warning(f"  {a}|{b} – missing price data, skipped.")
                continue

            try:
                result = self._run_pair(a, b, prices, n_pairs)
                pair_pnl_list.append(result.portfolio_value.to_frame("portfolio_value")
                                     .assign(net_return=result.net_returns))
                pair_results.append(result)
            except Exception as exc:
                logger.warning(f"  {a}|{b} backtest failed: {exc}")

        if not pair_pnl_list:
            raise RuntimeError("All pair backtests failed or were skipped.")

        # Build net_return DataFrames for aggregation
        net_ret_dfs = [df["net_return"].rename(i) for i, df in enumerate(pair_pnl_list)]
        portfolio_df = self.executor.aggregate_portfolio(
            [r.signals.join(p, how="inner")
             for r, p in zip(pair_results, pair_pnl_list)
             if not r.signals.empty],
            equal_weight=(self.allocation_method == "equal"),
        )

        # Risk: global drawdown control
        for pr in pair_results:
            self.risk.apply_drawdown_control(portfolio_df, [pr.signals])

        logger.info(
            f"Backtest complete | {len(pair_results)} pairs | "
            f"portfolio return {((portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1):.1%}"
        )
        return portfolio_df, pair_results

    # ------------------------------------------------------------------
    # Per-pair backtest
    # ------------------------------------------------------------------

    def _run_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        prices: pd.DataFrame,
        n_pairs: int,
    ) -> BacktestResult:
        price_a = prices[ticker_a]
        price_b = prices[ticker_b]

        signals = self.signal_gen.generate(price_a, price_b, log_prices=True)

        # Allocation
        allocation = self.initial_capital / n_pairs

        pnl = self.executor.compute_pnl(
            price_a=price_a,
            price_b=price_b,
            signals_df=signals,
            hedge_ratio=signals.get("hedge_ratio"),
            allocation=allocation / self.initial_capital,
        )

        # Pair-level stop-loss
        signals = self.risk.pair_stop_loss(
            signals_df=signals,
            net_pnl=pnl["net_pnl"],
            initial_allocation=allocation,
        )

        result = BacktestResult(
            pair=f"{ticker_a}|{ticker_b}",
            portfolio_value=pnl["portfolio_value"],
            net_returns=pnl["net_return"],
            signals=signals,
            metrics=RiskManager.risk_report(pnl, settings.backtest.risk_free_rate),
        )

        logger.debug(
            f"  {result.pair} | Sharpe {result.metrics['sharpe']:.2f} | "
            f"MaxDD {result.metrics['max_drawdown']:.1%} | "
            f"AnnRet {result.metrics['annual_return']:.1%}"
        )
        return result

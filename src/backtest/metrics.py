"""
backtest/metrics.py
-------------------
Computes detailed performance metrics from backtest results:
  • Return-based: Sharpe, Sortino, Calmar, CAGR, annual vol
  • Trade-based:  win-rate, profit factor, avg win/loss, max consecutive loss
  • Drawdown:     max DD, avg DD, recovery time
  • Rolling:      rolling Sharpe, rolling vol
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceMetrics:
    """
    Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    returns         : Daily net return series.
    portfolio_value : Portfolio equity curve.
    risk_free_rate  : Annualised risk-free rate (default 5 %).
    periods_per_year: Trading days per year (default 252).
    """

    def __init__(
        self,
        returns: pd.Series,
        portfolio_value: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> None:
        self.returns = returns.dropna()
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.periods = periods_per_year
        self.daily_rf = risk_free_rate / periods_per_year

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> dict[str, float]:
        """Return all metrics as a flat dict."""
        report = {}
        report.update(self._return_metrics())
        report.update(self._drawdown_metrics())
        report.update(self._risk_adjusted_metrics())
        return report

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def _return_metrics(self) -> dict[str, float]:
        ret = self.returns
        n = len(ret)
        total_ret = float((1 + ret).prod() - 1)
        cagr = float((1 + total_ret) ** (self.periods / n) - 1) if n > 0 else 0.0
        annual_vol = float(ret.std() * np.sqrt(self.periods))

        return {
            "total_return": total_ret,
            "cagr": cagr,
            "annual_vol": annual_vol,
            "daily_mean": float(ret.mean()),
            "daily_std": float(ret.std()),
            "skewness": float(ret.skew()),
            "kurtosis": float(ret.kurtosis()),
            "best_day": float(ret.max()),
            "worst_day": float(ret.min()),
        }

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    def _drawdown_metrics(self) -> dict[str, float]:
        pv = self.portfolio_value
        roll_max = pv.cummax()
        dd = (pv - roll_max) / roll_max

        max_dd = float(dd.min())

        # Recovery time: days from trough to new high
        trough_idx = dd.idxmin()
        post_trough = pv.loc[trough_idx:]
        recovery_mask = post_trough >= roll_max.loc[trough_idx]
        recovery_days = int(recovery_mask.idxmax() - trough_idx if recovery_mask.any() else -1)

        avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
        dd_duration = int((dd < 0).sum())

        return {
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "drawdown_duration_days": dd_duration,
            "recovery_days": recovery_days,
        }

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def _risk_adjusted_metrics(self) -> dict[str, float]:
        ret = self.returns
        excess = ret - self.daily_rf

        # Sharpe
        sharpe = (
            float(excess.mean() / excess.std() * np.sqrt(self.periods))
            if excess.std() > 0 else 0.0
        )

        # Sortino (downside deviation below risk-free)
        downside = ret[ret < self.daily_rf]
        sortino = (
            float(excess.mean() / downside.std() * np.sqrt(self.periods))
            if len(downside) > 0 and downside.std() > 0 else 0.0
        )

        # Calmar
        dd_metrics = self._drawdown_metrics()
        max_dd = dd_metrics["max_drawdown"]
        cagr = self._return_metrics()["cagr"]
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

        # Omega ratio (return / downside risk)
        gains = ret[ret > self.daily_rf] - self.daily_rf
        losses = self.daily_rf - ret[ret < self.daily_rf]
        omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")

        # VaR and CVaR
        var_95 = float(np.percentile(ret, 5))
        cvar_95 = float(ret[ret <= var_95].mean())
        var_99 = float(np.percentile(ret, 1))
        cvar_99 = float(ret[ret <= var_99].mean())

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "omega": omega,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "var_99": var_99,
            "cvar_99": cvar_99,
        }

    # ------------------------------------------------------------------
    # Trade-level metrics (requires signals DataFrame)
    # ------------------------------------------------------------------

    @staticmethod
    def trade_metrics(
        net_pnl: pd.Series,
        signals: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Compute trade-based metrics from per-bar PnL and position signals.

        Parameters
        ----------
        net_pnl : Daily net PnL series.
        signals : DataFrame with 'position' and 'signal' columns.

        Returns
        -------
        Dict with win_rate, profit_factor, avg_win, avg_loss, etc.
        """
        # Identify trade segments: group consecutive non-zero positions
        pos = signals["position"]
        trade_id = (pos != pos.shift()).cumsum()
        in_trade = pos != 0

        trade_pnl = []
        for tid, group in net_pnl[in_trade].groupby(trade_id[in_trade]):
            trade_pnl.append(float(group.sum()))

        if not trade_pnl:
            return {k: 0.0 for k in [
                "n_trades", "win_rate", "profit_factor",
                "avg_win", "avg_loss", "avg_trade_pnl",
                "max_consecutive_wins", "max_consecutive_losses",
                "expectancy",
            ]}

        pnl_arr = np.array(trade_pnl)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]

        win_rate = len(wins) / len(pnl_arr)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        profit_factor = (
            float(wins.sum() / abs(losses.sum()))
            if losses.sum() != 0 else float("inf")
        )
        expectancy = float(pnl_arr.mean())

        # Max consecutive wins / losses
        def max_consecutive(arr, condition):
            max_c = cur = 0
            for v in arr:
                cur = cur + 1 if condition(v) else 0
                max_c = max(max_c, cur)
            return max_c

        max_wins = max_consecutive(pnl_arr, lambda x: x > 0)
        max_losses = max_consecutive(pnl_arr, lambda x: x <= 0)

        return {
            "n_trades": len(pnl_arr),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade_pnl": expectancy,
            "max_consecutive_wins": float(max_wins),
            "max_consecutive_losses": float(max_losses),
            "expectancy": expectancy,
        }

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling annualised Sharpe ratio."""
        excess = self.returns - self.daily_rf
        return (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(self.periods)
        ).rename("rolling_sharpe")

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        """Rolling annualised volatility."""
        return (self.returns.rolling(window).std() * np.sqrt(self.periods)).rename(
            "rolling_vol"
        )

    def drawdown_series(self) -> pd.Series:
        """Running drawdown series."""
        pv = self.portfolio_value
        return ((pv - pv.cummax()) / pv.cummax()).rename("drawdown")

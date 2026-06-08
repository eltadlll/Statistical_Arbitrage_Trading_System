"""
strategy/risk.py
----------------
Position sizing, drawdown controls, and stop-loss logic.
Supports fixed fractional, volatility-targeting, and Kelly sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


class RiskManager:
    """
    Position sizing and risk controls for the pairs portfolio.

    Parameters
    ----------
    target_vol      : Daily portfolio volatility target (e.g. 0.01 = 1 %).
    max_drawdown    : Maximum allowed portfolio drawdown before halting.
    max_pair_weight : Maximum weight allocated to a single pair.
    kelly_fraction  : Fraction of Kelly criterion to apply (default 0.5 = half-Kelly).
    """

    def __init__(
        self,
        target_vol: float = 0.01,
        max_drawdown: float = 0.20,
        max_pair_weight: float = 0.25,
        kelly_fraction: float = 0.5,
    ) -> None:
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.max_pair_weight = max_pair_weight
        self.kelly_fraction = kelly_fraction

    # ------------------------------------------------------------------
    # Position sizing methods
    # ------------------------------------------------------------------

    def fixed_fractional(
        self, n_pairs: int, capital: float
    ) -> float:
        """Equal allocation to each pair, capped at max_pair_weight."""
        weight = min(1.0 / n_pairs, self.max_pair_weight)
        return weight * capital

    def volatility_target(
        self,
        pair_returns: pd.Series,
        capital: float,
        lookback: int = 21,
    ) -> float:
        """
        Scale position so the pair's realised vol matches target_vol.
        Returns allocation in dollars.
        """
        recent_vol = pair_returns.iloc[-lookback:].std()
        if recent_vol == 0 or np.isnan(recent_vol):
            return self.fixed_fractional(1, capital)
        weight = min(self.target_vol / recent_vol, self.max_pair_weight)
        return weight * capital

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> float:
        """
        Fractional Kelly criterion:
            f = (p/|avg_loss|) - ((1-p)/avg_win)
        Returns allocation in dollars, clipped to max_pair_weight.
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        kelly_f = (win_rate / abs(avg_loss)) - ((1 - win_rate) / avg_win)
        kelly_f = max(kelly_f, 0) * self.kelly_fraction
        weight = min(kelly_f, self.max_pair_weight)
        return weight * capital

    # ------------------------------------------------------------------
    # Portfolio-level risk controls
    # ------------------------------------------------------------------

    def apply_drawdown_control(
        self,
        portfolio_pnl: pd.DataFrame,
        signals_df_list: list[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        """
        If portfolio drawdown exceeds max_drawdown, zero out all signals
        from that date forward (trading halt).

        Parameters
        ----------
        portfolio_pnl    : Aggregated portfolio DataFrame from ExecutionSimulator.
        signals_df_list  : List of signals DataFrames (modified in-place).

        Returns
        -------
        Clipped signals DataFrames.
        """
        pv = portfolio_pnl["portfolio_value"]
        rolling_max = pv.cummax()
        drawdown = (pv - rolling_max) / rolling_max

        halt_date = drawdown[drawdown < -self.max_drawdown].first_valid_index()
        if halt_date is not None:
            logger.warning(
                f"Drawdown limit ({self.max_drawdown:.0%}) hit on {halt_date} – "
                "flattening all positions."
            )
            out = []
            for sdf in signals_df_list:
                sdf = sdf.copy()
                sdf.loc[halt_date:, "position"] = 0
                sdf.loc[halt_date:, "signal"] = 0
                out.append(sdf)
            return out

        return signals_df_list

    def compute_drawdown(self, portfolio_value: pd.Series) -> pd.Series:
        """Return running drawdown series."""
        rolling_max = portfolio_value.cummax()
        return (portfolio_value - rolling_max) / rolling_max

    def max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Maximum drawdown over the full series."""
        return float(self.compute_drawdown(portfolio_value).min())

    # ------------------------------------------------------------------
    # Stop-loss at pair level
    # ------------------------------------------------------------------

    def pair_stop_loss(
        self,
        signals_df: pd.DataFrame,
        net_pnl: pd.Series,
        pair_stop_loss_pct: float = 0.10,
        initial_allocation: float = 10_000.0,
    ) -> pd.DataFrame:
        """
        Zero out pair signals if cumulative pair PnL drops below
        `pair_stop_loss_pct` of initial allocation.
        """
        cum_pnl = net_pnl.cumsum()
        stop_level = -abs(pair_stop_loss_pct * initial_allocation)
        triggered = cum_pnl[cum_pnl < stop_level].first_valid_index()

        if triggered is not None:
            logger.warning(
                f"Pair stop-loss triggered on {triggered}. "
                f"CumPnL {cum_pnl[triggered]:.0f} < {stop_level:.0f}"
            )
            signals_df = signals_df.copy()
            signals_df.loc[triggered:, "position"] = 0
            signals_df.loc[triggered:, "signal"] = 0

        return signals_df

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    @staticmethod
    def risk_report(portfolio_pnl: pd.DataFrame, risk_free_rate: float = 0.05) -> dict:
        """High-level risk metrics for the portfolio."""
        ret = portfolio_pnl["net_return"]
        pv = portfolio_pnl["portfolio_value"]
        daily_rf = risk_free_rate / 252

        excess = ret - daily_rf
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

        downside = ret[ret < daily_rf]
        sortino = float(
            excess.mean() / downside.std() * np.sqrt(252)
        ) if len(downside) > 0 and downside.std() > 0 else 0.0

        rolling_max = pv.cummax()
        dd = (pv - rolling_max) / rolling_max
        max_dd = float(dd.min())
        calmar = float(ret.mean() * 252 / abs(max_dd)) if max_dd != 0 else 0.0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "annual_return": float(ret.mean() * 252),
            "annual_vol": float(ret.std() * np.sqrt(252)),
            "total_return": float((pv.iloc[-1] / pv.iloc[0]) - 1),
        }

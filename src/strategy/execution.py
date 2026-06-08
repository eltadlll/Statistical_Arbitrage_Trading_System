"""
strategy/execution.py
---------------------
Simulates order execution including:
  • Commission (percentage of notional)
  • Bid-ask slippage (percentage of price)
  • Pair PnL calculation (long A / short B and vice versa)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import settings


class ExecutionSimulator:
    """
    Compute realistic PnL for a pairs strategy from signal and price data.

    Parameters
    ----------
    commission_pct : Commission charged per leg as fraction (default 0.001 = 0.1%).
    slippage_pct   : Slippage per leg as fraction (default 0.0005 = 0.05%).
    initial_capital: Starting portfolio value in USD.
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Core PnL engine
    # ------------------------------------------------------------------

    def compute_pnl(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        signals_df: pd.DataFrame,
        hedge_ratio: pd.Series | float | None = None,
        allocation: float = 1.0,
    ) -> pd.DataFrame:
        """
        Simulate PnL for a single pair from signals.

        Parameters
        ----------
        price_a     : Raw price series for asset A.
        price_b     : Raw price series for asset B.
        signals_df  : Output of SignalGenerator.generate() with 'position' column.
        hedge_ratio : Per-bar hedge ratio β. If scalar or None, uses static β.
        allocation  : Fraction of initial_capital to allocate to this pair.

        Returns
        -------
        DataFrame with columns: gross_pnl, costs, net_pnl, portfolio_value,
                                 gross_return, net_return.
        """
        aligned = pd.concat(
            [price_a, price_b, signals_df["position"],
             signals_df.get("hedge_ratio", pd.Series(1.0, index=signals_df.index))],
            axis=1,
        ).dropna()
        aligned.columns = ["price_a", "price_b", "position", "beta"]

        capital = self.initial_capital * allocation

        # --- Compute per-bar gross returns ---
        # LONG spread:  long A (1 unit), short B (β units)
        # SHORT spread: short A (1 unit), long B (β units)
        ret_a = aligned["price_a"].pct_change()
        ret_b = aligned["price_b"].pct_change()

        beta = aligned["beta"]
        pos = aligned["position"]

        # Spread return (sign convention: LONG_SPREAD = +1)
        spread_return = pos * (ret_a - beta * ret_b)

        # --- Transaction costs on position changes ---
        pos_change = pos.diff().abs()
        total_notional_traded = pos_change * (
            aligned["price_a"].abs() + beta.abs() * aligned["price_b"].abs()
        )
        cost_per_bar = total_notional_traded * (self.commission_pct + self.slippage_pct)
        cost_return = cost_per_bar / capital

        gross_pnl = (spread_return * capital).fillna(0)
        costs = (cost_return * capital).fillna(0)
        net_pnl = gross_pnl - costs

        portfolio_value = capital + net_pnl.cumsum()

        return pd.DataFrame(
            {
                "gross_pnl": gross_pnl,
                "costs": costs,
                "net_pnl": net_pnl,
                "portfolio_value": portfolio_value,
                "gross_return": spread_return.fillna(0),
                "net_return": (spread_return - cost_return).fillna(0),
            },
            index=aligned.index,
        )

    # ------------------------------------------------------------------
    # Multi-pair portfolio aggregation
    # ------------------------------------------------------------------

    def aggregate_portfolio(
        self,
        pair_pnl_list: list[pd.DataFrame],
        equal_weight: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate PnL across multiple pairs into a single portfolio.

        Parameters
        ----------
        pair_pnl_list : List of DataFrames from compute_pnl().
        equal_weight  : If True, equal-weight pairs; otherwise weight by
                        inverse volatility of net_return.

        Returns
        -------
        Aggregated portfolio DataFrame with same columns as compute_pnl().
        """
        if not pair_pnl_list:
            raise ValueError("pair_pnl_list is empty.")

        # Align all on common index
        net_returns = pd.concat(
            [df["net_return"].rename(i) for i, df in enumerate(pair_pnl_list)],
            axis=1,
        ).fillna(0)

        if equal_weight:
            weights = np.ones(len(pair_pnl_list)) / len(pair_pnl_list)
        else:
            vols = net_returns.std()
            inv_vol = 1.0 / (vols + 1e-9)
            weights = (inv_vol / inv_vol.sum()).values

        portfolio_return = (net_returns * weights).sum(axis=1)
        portfolio_value = self.initial_capital * (1 + portfolio_return).cumprod()

        gross_returns = pd.concat(
            [df["gross_return"].rename(i) for i, df in enumerate(pair_pnl_list)],
            axis=1,
        ).fillna(0)
        gross_return = (gross_returns * weights).sum(axis=1)

        net_pnl = portfolio_return * self.initial_capital
        gross_pnl = gross_return * self.initial_capital

        return pd.DataFrame(
            {
                "gross_pnl": gross_pnl,
                "costs": gross_pnl - net_pnl,
                "net_pnl": net_pnl,
                "portfolio_value": portfolio_value,
                "gross_return": gross_return,
                "net_return": portfolio_return,
            },
            index=net_returns.index,
        )

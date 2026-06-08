"""
strategy/signals.py
-------------------
Generates long/short signals for a pair based on spread z-score.
Supports OLS-static, rolling-OLS, and Kalman-filter spread methods.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.analysis.spread import SpreadBuilder


class Signal(IntEnum):
    FLAT = 0
    LONG_SPREAD = 1    # long A, short B (spread below entry threshold)
    SHORT_SPREAD = -1  # short A, long B (spread above entry threshold)


class SignalGenerator:
    """
    Converts a price pair into a time-series of trading signals.

    Parameters
    ----------
    spread_method   : "ols" | "rolling_ols" | "kalman"
    entry_z         : |z-score| threshold to open a position.
    exit_z          : |z-score| threshold to close a position.
    stop_z          : |z-score| threshold for hard stop-loss.
    lookback        : Rolling window for z-score and rolling OLS.
    max_holding     : Max bars a position can be held before forced exit.
    """

    def __init__(
        self,
        spread_method: str = "kalman",
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        lookback: int = 60,
        max_holding: int = 30,
    ) -> None:
        self.spread_method = spread_method
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback = lookback
        self.max_holding = max_holding
        self._sb = SpreadBuilder(
            zscore_window=lookback,
            kalman_trans_cov=settings.strategy.kalman_transition_cov,
            kalman_obs_cov=settings.strategy.kalman_observation_cov,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        log_prices: bool = True,
    ) -> pd.DataFrame:
        """
        Generate signals for a pair.

        Returns a DataFrame with columns:
            spread, zscore, signal, position, holding_days
        """
        if log_prices:
            lp_a = np.log(price_a)
            lp_b = np.log(price_b)
        else:
            lp_a, lp_b = price_a, price_b

        spread, hedge_ratio = self._compute_spread(lp_a, lp_b)
        zscore = self._sb.zscore(spread, window=self.lookback)

        signals = self._apply_rules(zscore)
        signals["spread"] = spread
        signals["zscore"] = zscore
        signals["hedge_ratio"] = (
            hedge_ratio if isinstance(hedge_ratio, pd.Series)
            else pd.Series(hedge_ratio, index=spread.index)
        )

        logger.debug(
            f"Signals generated | method={self.spread_method} | "
            f"entries={( signals['signal'] != Signal.FLAT).sum()}"
        )
        return signals.dropna(subset=["zscore"])

    # ------------------------------------------------------------------
    # Spread construction
    # ------------------------------------------------------------------

    def _compute_spread(
        self, lp_a: pd.Series, lp_b: pd.Series
    ) -> tuple[pd.Series, pd.Series | float]:
        if self.spread_method == "kalman":
            spread, beta, _ = self._sb.kalman_spread(lp_a, lp_b)
            return spread, beta
        elif self.spread_method == "rolling_ols":
            return self._sb.rolling_ols_spread(lp_a, lp_b, window=self.lookback)
        else:  # ols
            spread, beta = self._sb.ols_spread(lp_a, lp_b)
            return spread, float(beta)

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    def _apply_rules(self, zscore: pd.Series) -> pd.DataFrame:
        """
        State machine over z-score:
          • FLAT → LONG_SPREAD  when z < -entry_z
          • FLAT → SHORT_SPREAD when z >  entry_z
          • LONG_SPREAD  → FLAT when z > -exit_z OR z > stop_z OR max_holding
          • SHORT_SPREAD → FLAT when z <  exit_z OR z < -stop_z OR max_holding
        """
        n = len(zscore)
        signal = np.zeros(n, dtype=int)
        position = np.zeros(n, dtype=int)
        holding = np.zeros(n, dtype=int)

        current_pos = Signal.FLAT
        days_held = 0

        for i, z in enumerate(zscore):
            if np.isnan(z):
                position[i] = Signal.FLAT
                continue

            if current_pos == Signal.FLAT:
                if z < -self.entry_z:
                    current_pos = Signal.LONG_SPREAD
                    signal[i] = Signal.LONG_SPREAD
                    days_held = 1
                elif z > self.entry_z:
                    current_pos = Signal.SHORT_SPREAD
                    signal[i] = Signal.SHORT_SPREAD
                    days_held = 1

            elif current_pos == Signal.LONG_SPREAD:
                days_held += 1
                exit_cond = (
                    z > -self.exit_z
                    or z > self.stop_z
                    or days_held >= self.max_holding
                )
                if exit_cond:
                    signal[i] = Signal.FLAT
                    current_pos = Signal.FLAT
                    days_held = 0

            elif current_pos == Signal.SHORT_SPREAD:
                days_held += 1
                exit_cond = (
                    z < self.exit_z
                    or z < -self.stop_z
                    or days_held >= self.max_holding
                )
                if exit_cond:
                    signal[i] = Signal.FLAT
                    current_pos = Signal.FLAT
                    days_held = 0

            position[i] = current_pos
            holding[i] = days_held

        return pd.DataFrame(
            {"signal": signal, "position": position, "holding_days": holding},
            index=zscore.index,
        )

    # ------------------------------------------------------------------
    # Convenience: signal summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(signals_df: pd.DataFrame) -> dict[str, float]:
        """High-level statistics on a signals DataFrame."""
        pos = signals_df["position"]
        n = len(pos)
        n_long = (pos == Signal.LONG_SPREAD).sum()
        n_short = (pos == Signal.SHORT_SPREAD).sum()
        n_flat = (pos == Signal.FLAT).sum()
        return {
            "pct_long": n_long / n,
            "pct_short": n_short / n,
            "pct_flat": n_flat / n,
            "n_trades": int((signals_df["signal"] != Signal.FLAT).sum()),
            "avg_holding_days": float(
                signals_df.loc[pos != Signal.FLAT, "holding_days"].mean()
            ) if (pos != Signal.FLAT).any() else 0.0,
        }

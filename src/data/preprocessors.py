"""
preprocessors.py
----------------
Transforms raw OHLCV price data into clean, analysis-ready DataFrames:
  • Forward-fill gaps, drop all-NaN columns
  • Log-prices and arithmetic returns
  • Winsorisation of extreme returns
  • Normalised / standardised price series
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class Preprocessor:
    """
    Clean and transform a raw Close price DataFrame.

    Parameters
    ----------
    min_history : int
        Minimum number of non-NaN observations required to keep a ticker.
    winsorise_pct : float
        Winsorise daily returns at this percentile (e.g. 0.01 = 1 % tails).
    fill_method : str
        How to fill internal NaNs: "ffill", "bfill", or "interpolate".
    """

    def __init__(
        self,
        min_history: int = 252,
        winsorise_pct: float = 0.01,
        fill_method: str = "ffill",
    ) -> None:
        self.min_history = min_history
        self.winsorise_pct = winsorise_pct
        self.fill_method = fill_method

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Full preprocessing pipeline.

        Returns a dict with keys:
            "prices"       – cleaned Close prices
            "log_prices"   – natural log of prices
            "returns"      – arithmetic daily returns
            "log_returns"  – log returns
        """
        prices = self._clean(prices)
        log_prices = np.log(prices)
        returns = prices.pct_change().iloc[1:]
        returns = self._winsorise(returns)
        log_returns = log_prices.diff().iloc[1:]

        logger.info(
            f"Preprocessing complete | "
            f"{prices.shape[1]} tickers × {prices.shape[0]} days"
        )
        return {
            "prices": prices,
            "log_prices": log_prices,
            "returns": returns,
            "log_returns": log_returns,
        }

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    def _clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop thin tickers, forward-fill gaps, remove remaining NaNs."""
        prices = prices.copy()

        # 1. Drop tickers with insufficient history
        n_valid = prices.notna().sum()
        thin = n_valid[n_valid < self.min_history].index.tolist()
        if thin:
            logger.warning(f"Dropping {len(thin)} tickers (< {self.min_history} obs): {thin}")
        prices = prices.drop(columns=thin)

        # 2. Fill internal gaps
        if self.fill_method == "ffill":
            prices = prices.ffill()
        elif self.fill_method == "bfill":
            prices = prices.bfill()
        elif self.fill_method == "interpolate":
            prices = prices.interpolate(method="time")

        # 3. Drop rows where everything is NaN (weekends in mixed datasets)
        prices = prices.dropna(how="all")

        # 4. Ensure positive prices
        if (prices <= 0).any().any():
            logger.warning("Non-positive prices found – replaced with NaN then ffilled.")
            prices = prices.where(prices > 0).ffill()

        return prices

    def _winsorise(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clip each column's returns at the given percentile tails."""
        lo = returns.quantile(self.winsorise_pct)
        hi = returns.quantile(1 - self.winsorise_pct)
        return returns.clip(lower=lo, upper=hi, axis=1)

    # ------------------------------------------------------------------
    # Derived series helpers (called externally)
    # ------------------------------------------------------------------

    @staticmethod
    def normalise(series: pd.Series) -> pd.Series:
        """Min-max normalise a price series to [0, 1]."""
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn)

    @staticmethod
    def standardise(series: pd.Series) -> pd.Series:
        """Z-score standardise a price series (mean=0, std=1)."""
        return (series - series.mean()) / series.std()

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score of a series over `window` periods."""
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return (series - mu) / sigma

    @staticmethod
    def rebase(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
        """Rebase all price series so they start at `base`."""
        return prices.divide(prices.iloc[0]) * base

    @staticmethod
    def compute_volatility(
        returns: pd.DataFrame, window: int = 21, annualise: bool = True
    ) -> pd.DataFrame:
        """
        Rolling realised volatility.
        Annualised by √252 if `annualise=True`.
        """
        vol = returns.rolling(window).std()
        return vol * np.sqrt(252) if annualise else vol

    @staticmethod
    def align_series(*series: pd.Series) -> tuple[pd.Series, ...]:
        """
        Align multiple price/return series to their common date index.
        Drops any dates where any series has NaN.
        """
        combined = pd.concat(series, axis=1).dropna()
        return tuple(combined.iloc[:, i] for i in range(combined.shape[1]))

    @staticmethod
    def compute_beta(asset: pd.Series, market: pd.Series) -> float:
        """OLS beta of asset returns on market returns."""
        asset, market = Preprocessor.align_series(asset, market)
        slope, _, _, _, _ = stats.linregress(market.values, asset.values)
        return float(slope)

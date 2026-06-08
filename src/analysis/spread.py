"""
spread.py
---------
Constructs and characterises the spread between a cointegrated pair:
  • OLS static spread
  • Kalman filter dynamic hedge ratio
  • Rolling z-score
  • Hurst exponent
  • Ornstein-Uhlenbeck parameter estimation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from pykalman import KalmanFilter
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS


class SpreadBuilder:
    """
    Construct and characterise the spread for a pair (A, B).

    Parameters
    ----------
    zscore_window : Rolling window for z-score standardisation.
    kalman_trans_cov  : Kalman transition covariance (controls how fast β can change).
    kalman_obs_cov    : Kalman observation covariance.
    """

    def __init__(
        self,
        zscore_window: int = 60,
        kalman_trans_cov: float = 1e-4,
        kalman_obs_cov: float = 1e-2,
    ) -> None:
        self.zscore_window = zscore_window
        self.kalman_trans_cov = kalman_trans_cov
        self.kalman_obs_cov = kalman_obs_cov

    # ------------------------------------------------------------------
    # OLS spread
    # ------------------------------------------------------------------

    def ols_spread(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> tuple[pd.Series, float]:
        """
        Compute OLS spread: spread = A - β·B.

        Returns (spread series, static hedge ratio β).
        """
        aligned = pd.concat([price_a, price_b], axis=1).dropna()
        a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
        X = add_constant(b)
        model = OLS(a, X).fit()
        beta = float(model.params.iloc[1])
        spread = a - beta * b
        return spread, beta

    def rolling_ols_spread(
        self, price_a: pd.Series, price_b: pd.Series, window: int = 60
    ) -> tuple[pd.Series, pd.Series]:
        """
        Rolling OLS hedge ratio and spread.

        Returns (spread series, rolling beta series).
        """
        aligned = pd.concat([price_a, price_b], axis=1).dropna()
        a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

        betas = np.full(len(a), np.nan)
        for i in range(window, len(a) + 1):
            a_win = a.iloc[i - window:i]
            b_win = b.iloc[i - window:i]
            X = add_constant(b_win)
            m = OLS(a_win, X).fit()
            betas[i - 1] = m.params.iloc[1]

        beta_series = pd.Series(betas, index=a.index, name="rolling_beta")
        spread = a - beta_series * b
        return spread, beta_series

    # ------------------------------------------------------------------
    # Kalman filter hedge ratio
    # ------------------------------------------------------------------

    def kalman_spread(
        self, price_a: pd.Series, price_b: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Dynamic hedge ratio via Kalman filter.
        State vector: [β, intercept]

        Returns (spread, kalman_beta, kalman_intercept) as Series.
        """
        aligned = pd.concat([price_a, price_b], axis=1).dropna()
        a = aligned.iloc[:, 0].values
        b = aligned.iloc[:, 1].values
        n = len(a)

        # Observation matrix: each row is [b_t, 1]
        obs_mat = np.stack([b, np.ones(n)], axis=1)[:, np.newaxis, :]

        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.eye(2),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=self.kalman_obs_cov * np.eye(1),
            transition_covariance=self.kalman_trans_cov * np.eye(2),
        )

        state_means, _ = kf.filter(a)
        beta_k = pd.Series(state_means[:, 0], index=aligned.index, name="kalman_beta")
        intercept_k = pd.Series(state_means[:, 1], index=aligned.index, name="kalman_intercept")
        spread = pd.Series(
            a - state_means[:, 0] * b - state_means[:, 1],
            index=aligned.index,
            name="kalman_spread",
        )
        return spread, beta_k, intercept_k

    # ------------------------------------------------------------------
    # Z-score
    # ------------------------------------------------------------------

    def zscore(
        self, spread: pd.Series, window: int | None = None
    ) -> pd.Series:
        """Rolling z-score of the spread series."""
        w = window or self.zscore_window
        mu = spread.rolling(w).mean()
        sigma = spread.rolling(w).std()
        return ((spread - mu) / sigma).rename("zscore")

    # ------------------------------------------------------------------
    # Hurst exponent
    # ------------------------------------------------------------------

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
        """
        Estimate the Hurst exponent using R/S analysis.
          H < 0.5  → mean-reverting (antipersistent)
          H = 0.5  → random walk
          H > 0.5  → trending (persistent)
        """
        series = series.dropna().values
        lags = range(2, min(max_lag, len(series) // 2))
        tau = []
        for lag in lags:
            chunks = [series[i:i + lag] for i in range(0, len(series) - lag, lag)]
            if not chunks:
                continue
            rs_values = []
            for chunk in chunks:
                if len(chunk) < 2 or np.std(chunk) == 0:
                    continue
                mean = np.mean(chunk)
                deviate = np.cumsum(chunk - mean)
                rs = (np.max(deviate) - np.min(deviate)) / np.std(chunk)
                rs_values.append(rs)
            if rs_values:
                tau.append((lag, np.mean(rs_values)))

        if len(tau) < 2:
            return 0.5  # fallback

        lags_arr = np.log([t[0] for t in tau])
        rs_arr = np.log([t[1] for t in tau])
        hurst = float(np.polyfit(lags_arr, rs_arr, 1)[0])
        return hurst

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck parameters
    # ------------------------------------------------------------------

    @staticmethod
    def ou_parameters(spread: pd.Series) -> dict[str, float]:
        """
        Fit a discrete OU process: Δy_t = κ·(μ - y_{t-1})·Δt + σ·ε_t

        Returns dict with keys: kappa (speed), mu (long-run mean), sigma, half_life.
        """
        y = spread.dropna()
        y_lag = y.shift(1)
        dy = y - y_lag
        combined = pd.concat([dy, y_lag], axis=1).dropna()
        X = add_constant(combined.iloc[:, 1])
        model = OLS(combined.iloc[:, 0], X).fit()

        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
        kappa = -beta
        mu = alpha / kappa if kappa != 0 else float("nan")
        sigma = float(model.resid.std())
        half_life = float(np.log(2) / kappa) if kappa > 0 else float("inf")

        return {
            "kappa": float(kappa),
            "mu": float(mu),
            "sigma": sigma,
            "half_life": half_life,
        }

    # ------------------------------------------------------------------
    # Full characterisation
    # ------------------------------------------------------------------

    def characterise(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        method: str = "ols",
    ) -> dict:
        """
        Return a full summary dict for a pair using OLS or Kalman spread.

        method : "ols" | "kalman"
        """
        if method == "kalman":
            spread, beta, _ = self.kalman_spread(price_a, price_b)
        else:
            spread, beta_val = self.ols_spread(price_a, price_b)
            beta = beta_val  # scalar

        z = self.zscore(spread)
        ou = self.ou_parameters(spread)
        hurst = self.hurst_exponent(spread)

        return {
            "spread": spread,
            "zscore": z,
            "hedge_ratio": beta,
            "hurst": hurst,
            **ou,
        }

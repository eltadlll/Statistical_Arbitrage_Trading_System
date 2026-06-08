"""
simulation/monte_carlo.py
--------------------------
Monte Carlo simulation engine:
  • Geometric Brownian Motion (GBM) path generation
  • Historical bootstrap (block bootstrap preserving autocorrelation)
  • VaR / CVaR at configurable confidence levels
  • Distributional summary (percentile bands, fan chart data)
  • Numba-accelerated inner loops for 10k+ paths
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):          # no-op decorator fallback
        return func

from config.settings import settings


# ------------------------------------------------------------------
# Numba-accelerated path simulation (falls back to numpy if unavailable)
# ------------------------------------------------------------------

@njit
def _simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    GBM simulation: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    Returns shape (n_paths, n_steps + 1).
    """
    np.random.seed(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    log_drift = (mu - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * np.sqrt(dt)
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(log_drift + vol_dt * z)
    return paths


class MonteCarloSimulator:
    """
    Simulate portfolio equity paths and compute risk statistics.

    Parameters
    ----------
    n_paths         : Number of simulation paths.
    horizon_days    : Simulation horizon in trading days.
    confidence_levels: List of confidence levels for VaR/CVaR.
    random_seed     : RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_paths: int = 10_000,
        horizon_days: int = 252,
        confidence_levels: list[float] = (0.95, 0.99),
        random_seed: int = 42,
    ) -> None:
        self.n_paths = n_paths
        self.horizon_days = horizon_days
        self.confidence_levels = list(confidence_levels)
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # GBM simulation
    # ------------------------------------------------------------------

    def simulate_gbm(
        self,
        returns: pd.Series,
        initial_value: float = 100_000.0,
    ) -> np.ndarray:
        """
        Fit GBM parameters from historical returns and simulate paths.

        Returns
        -------
        np.ndarray of shape (n_paths, horizon_days + 1).
        """
        mu = float(returns.mean() * 252)
        sigma = float(returns.std() * np.sqrt(252))
        dt = 1 / 252

        logger.info(
            f"GBM simulation | mu={mu:.2%} sigma={sigma:.2%} "
            f"paths={self.n_paths} horizon={self.horizon_days}d"
        )

        paths = _simulate_gbm_paths(
            S0=initial_value,
            mu=mu,
            sigma=sigma,
            dt=dt,
            n_steps=self.horizon_days,
            n_paths=self.n_paths,
            seed=self.random_seed,
        )
        return paths

    # ------------------------------------------------------------------
    # Block bootstrap simulation
    # ------------------------------------------------------------------

    def simulate_bootstrap(
        self,
        returns: pd.Series,
        initial_value: float = 100_000.0,
        block_size: int = 21,
    ) -> np.ndarray:
        """
        Block bootstrap: resample overlapping blocks of historical returns
        to preserve short-term autocorrelation structure.

        Returns
        -------
        np.ndarray of shape (n_paths, horizon_days + 1).
        """
        rng = np.random.default_rng(self.random_seed)
        ret_arr = returns.dropna().values
        n = len(ret_arr)

        paths = np.empty((self.n_paths, self.horizon_days + 1))
        paths[:, 0] = initial_value

        blocks_needed = int(np.ceil(self.horizon_days / block_size))

        logger.info(
            f"Block bootstrap | block_size={block_size} "
            f"paths={self.n_paths} horizon={self.horizon_days}d"
        )

        for p in range(self.n_paths):
            sampled = []
            for _ in range(blocks_needed):
                start = rng.integers(0, n - block_size)
                sampled.extend(ret_arr[start: start + block_size])
            sampled = np.array(sampled[: self.horizon_days])

            pv = initial_value
            for t, r in enumerate(sampled, 1):
                pv = pv * (1 + r)
                paths[p, t] = pv

        return paths

    # ------------------------------------------------------------------
    # Risk statistics
    # ------------------------------------------------------------------

    def compute_terminal_stats(
        self, paths: np.ndarray, initial_value: float = 100_000.0
    ) -> dict:
        """
        Compute summary statistics from terminal values (last column of paths).
        """
        terminal = paths[:, -1]
        terminal_returns = (terminal / initial_value) - 1

        stats: dict = {
            "mean_terminal_value": float(terminal.mean()),
            "median_terminal_value": float(np.median(terminal)),
            "std_terminal_value": float(terminal.std()),
            "mean_return": float(terminal_returns.mean()),
            "median_return": float(np.median(terminal_returns)),
            "prob_profit": float((terminal > initial_value).mean()),
            "prob_loss_10pct": float((terminal_returns < -0.10).mean()),
            "prob_loss_20pct": float((terminal_returns < -0.20).mean()),
            "prob_loss_50pct": float((terminal_returns < -0.50).mean()),
        }

        for cl in self.confidence_levels:
            alpha = 1 - cl
            var = float(np.percentile(terminal_returns, alpha * 100))
            cvar = float(terminal_returns[terminal_returns <= var].mean())
            stats[f"var_{int(cl*100)}"] = var
            stats[f"cvar_{int(cl*100)}"] = cvar
            stats[f"pct_{int(cl*100)}_terminal"] = float(np.percentile(terminal, cl * 100))
            stats[f"pct_{int((1-cl)*100)}_terminal"] = float(np.percentile(terminal, (1-cl) * 100))

        return stats

    def compute_path_drawdown(self, paths: np.ndarray) -> dict:
        """
        Compute per-path maximum drawdown and return summary statistics.
        """
        max_dds = []
        for path in paths:
            roll_max = np.maximum.accumulate(path)
            dd = (path - roll_max) / roll_max
            max_dds.append(float(dd.min()))

        arr = np.array(max_dds)
        result = {
            "mean_max_drawdown": float(arr.mean()),
            "median_max_drawdown": float(np.median(arr)),
            "worst_max_drawdown": float(arr.min()),
        }
        for cl in self.confidence_levels:
            result[f"max_dd_var_{int(cl*100)}"] = float(np.percentile(arr, (1 - cl) * 100))
        return result

    # ------------------------------------------------------------------
    # Percentile fan-chart data
    # ------------------------------------------------------------------

    def percentile_bands(
        self,
        paths: np.ndarray,
        percentiles: list[int] = (5, 10, 25, 50, 75, 90, 95),
    ) -> pd.DataFrame:
        """
        Return a DataFrame of percentile bands over time.
        Columns = percentile labels, index = timestep (0 … horizon_days).
        """
        data = {f"p{p}": np.percentile(paths, p, axis=0) for p in percentiles}
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Full simulation report
    # ------------------------------------------------------------------

    def run(
        self,
        returns: pd.Series,
        initial_value: float = 100_000.0,
        method: str = "gbm",
    ) -> dict:
        """
        Run full Monte Carlo simulation and return a complete results dict.

        Parameters
        ----------
        returns       : Historical daily returns series.
        initial_value : Starting portfolio value.
        method        : "gbm" | "bootstrap"

        Returns
        -------
        Dict with keys: paths, terminal_stats, drawdown_stats, bands.
        """
        if method == "bootstrap":
            paths = self.simulate_bootstrap(returns, initial_value)
        else:
            paths = self.simulate_gbm(returns, initial_value)

        terminal_stats = self.compute_terminal_stats(paths, initial_value)
        drawdown_stats = self.compute_path_drawdown(paths)
        bands = self.percentile_bands(paths)

        logger.info(
            f"MC [{method}] | E[terminal] ${terminal_stats['mean_terminal_value']:,.0f} | "
            f"P(profit) {terminal_stats['prob_profit']:.1%} | "
            f"VaR95 {terminal_stats.get('var_95', 0):.1%}"
        )

        return {
            "paths": paths,
            "terminal_stats": terminal_stats,
            "drawdown_stats": drawdown_stats,
            "bands": bands,
            "method": method,
        }

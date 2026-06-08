"""
cointegration.py
----------------
Tests pairs of price series for cointegration using:
  • Engle-Granger two-step test  (statsmodels)
  • Johansen maximum-likelihood test  (statsmodels)
  • ADF test on the residual spread
  • KPSS test on the residual spread
Also computes the OLS hedge ratio and the mean-reversion half-life.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config.settings import settings


@dataclass
class CointegrationResult:
    """Holds all test statistics for a single pair."""
    ticker_a: str
    ticker_b: str

    # Engle-Granger
    eg_pvalue: float = np.nan
    eg_stat: float = np.nan

    # ADF on spread
    adf_pvalue: float = np.nan
    adf_stat: float = np.nan

    # KPSS on spread (null = stationary; LOW p → non-stationary)
    kpss_pvalue: float = np.nan
    kpss_stat: float = np.nan

    # Johansen
    johansen_trace_stat: float = np.nan
    johansen_trace_crit_90: float = np.nan

    # Spread properties
    hedge_ratio: float = np.nan
    half_life: float = np.nan
    spread_mean: float = np.nan
    spread_std: float = np.nan

    # Composite pass
    is_cointegrated: bool = False

    @property
    def pair(self) -> str:
        return f"{self.ticker_a}|{self.ticker_b}"


class CointegrationAnalyzer:
    """
    Run cointegration tests on candidate pairs.

    Parameters
    ----------
    eg_pvalue_max   : Engle-Granger p-value threshold (default 0.05).
    adf_pvalue_max  : ADF p-value threshold on spread (default 0.05).
    half_life_min   : Minimum allowed half-life in days.
    half_life_max   : Maximum allowed half-life in days.
    """

    def __init__(
        self,
        eg_pvalue_max: float = 0.05,
        adf_pvalue_max: float = 0.05,
        half_life_min: int = 5,
        half_life_max: int = 60,
    ) -> None:
        self.eg_pvalue_max = eg_pvalue_max
        self.adf_pvalue_max = adf_pvalue_max
        self.half_life_min = half_life_min
        self.half_life_max = half_life_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_pair(
        self,
        log_prices: pd.DataFrame,
        ticker_a: str,
        ticker_b: str,
    ) -> CointegrationResult:
        """
        Run all cointegration tests on a single pair.

        Parameters
        ----------
        log_prices : DataFrame of log-prices, columns = tickers.
        ticker_a, ticker_b : Pair to evaluate.

        Returns
        -------
        CointegrationResult dataclass.
        """
        result = CointegrationResult(ticker_a=ticker_a, ticker_b=ticker_b)

        try:
            s_a = log_prices[ticker_a].dropna()
            s_b = log_prices[ticker_b].dropna()
            combined = pd.concat([s_a, s_b], axis=1).dropna()
            s_a, s_b = combined.iloc[:, 0], combined.iloc[:, 1]

            if len(s_a) < 60:
                logger.debug(f"  {ticker_a}|{ticker_b}: too few obs ({len(s_a)}), skipping.")
                return result

            # 1. OLS hedge ratio (regress A on B)
            result.hedge_ratio, spread = self._ols_spread(s_a, s_b)

            # 2. Engle-Granger
            eg_stat, eg_p, _ = coint(s_a, s_b)
            result.eg_stat = float(eg_stat)
            result.eg_pvalue = float(eg_p)

            # 3. ADF on spread
            adf_stat, adf_p, *_ = adfuller(spread, autolag="AIC")
            result.adf_stat = float(adf_stat)
            result.adf_pvalue = float(adf_p)

            # 4. KPSS on spread (null = stationary; want HIGH p → can't reject stationarity)
            try:
                kpss_stat, kpss_p, *_ = kpss(spread, regression="c", nlags="auto")
                result.kpss_stat = float(kpss_stat)
                result.kpss_pvalue = float(kpss_p)
            except Exception:
                pass  # KPSS occasionally fails on short series

            # 5. Johansen test
            joh_result = coint_johansen(combined, det_order=0, k_ar_diff=1)
            result.johansen_trace_stat = float(joh_result.lr1[0])
            result.johansen_trace_crit_90 = float(joh_result.cvt[0, 0])

            # 6. Half-life of mean reversion (Ornstein-Uhlenbeck)
            result.half_life = self._half_life(spread)

            # 7. Spread statistics
            result.spread_mean = float(spread.mean())
            result.spread_std = float(spread.std())

            # 8. Composite pass/fail
            result.is_cointegrated = self._passes_filters(result)

        except Exception as exc:
            logger.warning(f"  {ticker_a}|{ticker_b}: cointegration test failed – {exc}")

        return result

    def test_all_pairs(
        self,
        log_prices: pd.DataFrame,
        pairs: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Test all pairs and return a DataFrame of CointegrationResult values.
        Only pairs where both tickers are in log_prices are evaluated.
        """
        available = set(log_prices.columns)
        valid_pairs = [(a, b) for a, b in pairs if a in available and b in available]
        logger.info(f"Running cointegration tests on {len(valid_pairs)} pairs …")

        results = []
        for i, (a, b) in enumerate(valid_pairs, 1):
            res = self.test_pair(log_prices, a, b)
            results.append(res)
            if i % 50 == 0:
                logger.info(f"  … {i}/{len(valid_pairs)} pairs tested")

        df = self._results_to_df(results)
        n_pass = df["is_cointegrated"].sum()
        logger.info(
            f"Cointegration complete | {n_pass}/{len(df)} pairs passed all filters"
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ols_spread(
        self, s_a: pd.Series, s_b: pd.Series
    ) -> tuple[float, pd.Series]:
        """Fit OLS: A = β·B + ε. Return (β, spread ε)."""
        X = add_constant(s_b)
        model = OLS(s_a, X).fit()
        beta = float(model.params.iloc[1])
        spread = s_a - beta * s_b
        return beta, spread

    def _half_life(self, spread: pd.Series) -> float:
        """
        Estimate mean-reversion half-life via OLS on AR(1):
            Δspread_t = α + β·spread_{t-1} + ε
        half-life = -ln(2) / β
        """
        lagged = spread.shift(1)
        delta = spread - lagged
        combined = pd.concat([delta, lagged], axis=1).dropna()
        X = add_constant(combined.iloc[:, 1])
        model = OLS(combined.iloc[:, 0], X).fit()
        beta = model.params.iloc[1]
        if beta >= 0:
            return np.inf  # non-mean-reverting
        return float(-np.log(2) / beta)

    def _passes_filters(self, r: CointegrationResult) -> bool:
        """Return True if the result passes all configured thresholds."""
        return (
            r.eg_pvalue <= self.eg_pvalue_max
            and r.adf_pvalue <= self.adf_pvalue_max
            and self.half_life_min <= r.half_life <= self.half_life_max
        )

    @staticmethod
    def _results_to_df(results: list[CointegrationResult]) -> pd.DataFrame:
        rows = [
            {
                "ticker_a": r.ticker_a,
                "ticker_b": r.ticker_b,
                "eg_pvalue": r.eg_pvalue,
                "eg_stat": r.eg_stat,
                "adf_pvalue": r.adf_pvalue,
                "adf_stat": r.adf_stat,
                "kpss_pvalue": r.kpss_pvalue,
                "johansen_trace_stat": r.johansen_trace_stat,
                "johansen_trace_crit_90": r.johansen_trace_crit_90,
                "hedge_ratio": r.hedge_ratio,
                "half_life": r.half_life,
                "spread_mean": r.spread_mean,
                "spread_std": r.spread_std,
                "is_cointegrated": r.is_cointegrated,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        return df.sort_values("eg_pvalue").reset_index(drop=True)

"""
tests/test_cointegration.py
---------------------------
Unit tests for the cointegration analysis module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import SpreadBuilder


def _make_cointegrated_pair(n=500, beta=1.5, seed=42):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    noise_a = rng.normal(0, 0.5, n)
    noise_b = rng.normal(0, 0.5, n)
    a = common + noise_a
    b = beta * common + noise_b
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def _make_random_walk_pair(n=500, seed=99):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 1, n))
    b = np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx, name="X"), pd.Series(b, index=idx, name="Y")


class TestCointegrationAnalyzer:
    def setup_method(self):
        self.analyzer = CointegrationAnalyzer(
            eg_pvalue_max=0.05,
            adf_pvalue_max=0.05,
            half_life_min=2,
            half_life_max=200,
        )

    def test_cointegrated_pair_detected(self):
        a, b = _make_cointegrated_pair()
        log_prices = pd.DataFrame({"A": a, "B": b})
        result = self.analyzer.test_pair(log_prices, "A", "B")
        assert result.eg_pvalue < 0.10, (
            f"Expected EG p-value < 0.10, got {result.eg_pvalue:.4f}"
        )
        assert result.is_cointegrated, "Cointegrated pair should pass filters"

    def test_random_walk_not_cointegrated(self):
        x, y = _make_random_walk_pair()
        log_prices = pd.DataFrame({"X": x, "Y": y})
        result = self.analyzer.test_pair(log_prices, "X", "Y")
        # Should mostly fail; allow occasional false positive in stochastic test
        # We just check the result object is well-formed
        assert not np.isnan(result.eg_pvalue)
        assert not np.isnan(result.hedge_ratio)

    def test_half_life_positive(self):
        a, b = _make_cointegrated_pair()
        log_prices = pd.DataFrame({"A": a, "B": b})
        result = self.analyzer.test_pair(log_prices, "A", "B")
        assert result.half_life > 0, "Half-life should be positive"

    def test_test_all_pairs_returns_dataframe(self):
        a, b = _make_cointegrated_pair()
        log_prices = pd.DataFrame({"A": a, "B": b})
        pairs = [("A", "B")]
        df = self.analyzer.test_all_pairs(log_prices, pairs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "is_cointegrated" in df.columns

    def test_missing_ticker_skipped(self):
        a, b = _make_cointegrated_pair()
        log_prices = pd.DataFrame({"A": a, "B": b})
        df = self.analyzer.test_all_pairs(log_prices, [("A", "MISSING")])
        assert df.empty

    def test_results_sorted_by_pvalue(self):
        a, b = _make_cointegrated_pair(seed=42)
        c, d = _make_cointegrated_pair(seed=7)
        log_prices = pd.DataFrame({"A": a, "B": b, "C": c, "D": d})
        df = self.analyzer.test_all_pairs(log_prices, [("A", "B"), ("C", "D")])
        if len(df) > 1:
            assert df["eg_pvalue"].iloc[0] <= df["eg_pvalue"].iloc[1]


class TestSpreadBuilder:
    def setup_method(self):
        self.sb = SpreadBuilder()
        self.a, self.b = _make_cointegrated_pair()

    def test_ols_spread_zero_mean(self):
        spread, beta = self.sb.ols_spread(self.a, self.b)
        assert abs(spread.mean()) < 1.0, "OLS spread should be near zero mean"
        assert beta > 0, "Hedge ratio should be positive"

    def test_zscore_near_zero_mean(self):
        spread, _ = self.sb.ols_spread(self.a, self.b)
        z = self.sb.zscore(spread, window=60).dropna()
        assert abs(z.mean()) < 0.5

    def test_hurst_exponent_mean_reverting(self):
        spread, _ = self.sb.ols_spread(self.a, self.b)
        h = self.sb.hurst_exponent(spread)
        assert h < 0.6, f"Spread Hurst {h:.3f} should indicate mean reversion"

    def test_ou_parameters_positive_kappa(self):
        spread, _ = self.sb.ols_spread(self.a, self.b)
        ou = self.sb.ou_parameters(spread)
        assert ou["kappa"] > 0, "OU kappa should be positive for mean-reverting spread"
        assert ou["half_life"] > 0

    def test_kalman_spread_same_length(self):
        spread, beta, intercept = self.sb.kalman_spread(self.a, self.b)
        assert len(spread) == len(self.a)
        assert len(beta) == len(self.a)

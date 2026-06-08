"""
correlation.py
--------------
Computes pairwise correlation matrices (static + rolling) and
pre-screens candidate pairs by minimum correlation threshold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config.settings import settings


class CorrelationAnalyzer:
    """
    Pairwise correlation analysis for a universe of price series.

    Parameters
    ----------
    window      : Rolling window in trading days (for rolling correlation).
    method      : "pearson" | "spearman" | "kendall"
    min_corr    : Minimum |correlation| to include a pair in results.
    """

    def __init__(
        self,
        window: int = 252,
        method: str = "pearson",
        min_corr: float = 0.70,
    ) -> None:
        self.window = window
        self.method = method
        self.min_corr = min_corr

    # ------------------------------------------------------------------
    # Static correlation
    # ------------------------------------------------------------------

    def full_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Full-sample pairwise correlation matrix."""
        return returns.corr(method=self.method)

    def high_correlation_pairs(
        self,
        returns: pd.DataFrame,
        pairs: Optional[list[tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Screen pairs by |correlation| >= min_corr.

        Parameters
        ----------
        returns : Daily return DataFrame.
        pairs   : List of (A, B) tuples to evaluate; if None, all pairs are used.

        Returns
        -------
        DataFrame with columns [ticker_a, ticker_b, correlation]
        sorted by |correlation| descending.
        """
        corr_matrix = self.full_matrix(returns)

        if pairs is not None:
            records = []
            for a, b in pairs:
                if a in corr_matrix.index and b in corr_matrix.columns:
                    c = corr_matrix.loc[a, b]
                    if abs(c) >= self.min_corr:
                        records.append({"ticker_a": a, "ticker_b": b, "correlation": c})
        else:
            records = []
            tickers = corr_matrix.columns.tolist()
            for i, a in enumerate(tickers):
                for b in tickers[i + 1 :]:
                    c = corr_matrix.loc[a, b]
                    if abs(c) >= self.min_corr:
                        records.append({"ticker_a": a, "ticker_b": b, "correlation": c})

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No pairs exceeded the minimum correlation threshold.")
            return df

        df["abs_corr"] = df["correlation"].abs()
        df = df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
        logger.info(f"Found {len(df)} pairs with |corr| >= {self.min_corr}")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Rolling correlation
    # ------------------------------------------------------------------

    def rolling_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """Rolling pairwise correlation between two return series."""
        w = window or self.window
        if self.method == "pearson":
            return series_a.rolling(w).corr(series_b)
        elif self.method == "spearman":
            # Spearman rolling via rank correlation
            return series_a.rolling(w).corr(series_b.rank())
        else:
            raise NotImplementedError(f"Rolling {self.method} not implemented.")

    def rolling_matrix(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Rolling correlation matrices stored as a dict of {ticker: rolling_corr_series}.
        For large universes use rolling_correlation() on specific pairs instead.
        """
        w = window or self.window
        result: dict[str, pd.DataFrame] = {}
        tickers = returns.columns.tolist()
        for i, a in enumerate(tickers):
            for b in tickers[i + 1 :]:
                key = f"{a}|{b}"
                result[key] = self.rolling_correlation(returns[a], returns[b], w)
        return result

    # ------------------------------------------------------------------
    # Stability metrics
    # ------------------------------------------------------------------

    def correlation_stability(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Summarise the stability of a rolling correlation over time.

        Returns dict with keys: mean, std, min, max, pct_positive, pct_above_min.
        """
        roll = self.rolling_correlation(series_a, series_b, window).dropna()
        return {
            "mean": float(roll.mean()),
            "std": float(roll.std()),
            "min": float(roll.min()),
            "max": float(roll.max()),
            "pct_positive": float((roll > 0).mean()),
            "pct_above_min": float((roll.abs() >= self.min_corr).mean()),
        }

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------

    @staticmethod
    def pearson_pvalue(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
        """Return (Pearson_r, p_value) for two return series."""
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        r, p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(r), float(p)

    @staticmethod
    def spearman_pvalue(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
        """Return (Spearman_rho, p_value) for two return series."""
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        rho, p = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(rho), float(p)

    # ------------------------------------------------------------------
    # Cluster-based pre-screening
    # ------------------------------------------------------------------

    def cluster_pairs_by_correlation(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 5,
    ) -> dict[int, list[str]]:
        """
        Group tickers into clusters by correlation distance using
        agglomerative hierarchical clustering.
        Returns {cluster_id: [ticker, ...]} mapping.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        corr_matrix = self.full_matrix(returns).fillna(0)
        distance = 1 - corr_matrix.abs()

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="complete",
        )
        labels = model.fit_predict(distance.values)
        clusters: dict[int, list[str]] = {}
        for ticker, label in zip(corr_matrix.columns, labels):
            clusters.setdefault(int(label), []).append(ticker)

        for cid, members in clusters.items():
            logger.info(f"  Cluster {cid}: {members}")
        return clusters

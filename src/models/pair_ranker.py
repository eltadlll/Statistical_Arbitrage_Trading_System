"""
models/pair_ranker.py
---------------------
Combines ML scores with statistical filter scores into a
composite ranking and selects the top-N pairs for strategy execution.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


class PairRanker:
    """
    Produce a ranked shortlist of pairs from the combined output of the
    CointegrationAnalyzer and MLPairSelector.

    Parameters
    ----------
    top_n          : Number of pairs to select.
    ml_weight      : Weight of ML score in composite (0-1).
    stat_weight    : Weight of statistical score in composite (0-1).
    enforce_hurst  : If True, drop pairs with Hurst >= 0.5.
    enforce_hl     : If True, enforce half-life bounds from settings.
    """

    def __init__(
        self,
        top_n: int = 20,
        ml_weight: float = 0.5,
        stat_weight: float = 0.5,
        enforce_hurst: bool = True,
        enforce_hl: bool = True,
    ) -> None:
        self.top_n = top_n
        self.ml_weight = ml_weight
        self.stat_weight = stat_weight
        self.enforce_hurst = enforce_hurst
        self.enforce_hl = enforce_hl

    def rank(
        self,
        coint_df: pd.DataFrame,
        ml_scores: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank pairs and return top-N.

        Parameters
        ----------
        coint_df  : DataFrame from CointegrationAnalyzer (has eg_pvalue, half_life, etc.)
        ml_scores : Optional DataFrame from MLPairSelector.score_pairs() with 'ml_score'.

        Returns
        -------
        DataFrame of top-N pairs with composite_score column, sorted descending.
        """
        df = coint_df.copy()

        # ------ hard filters ------
        df = df[df["is_cointegrated"]]

        if self.enforce_hurst and "hurst" in df.columns:
            df = df[df["hurst"] < 0.5]

        if self.enforce_hl:
            df = df[
                (df["half_life"] >= settings.analysis.half_life_min)
                & (df["half_life"] <= settings.analysis.half_life_max)
            ]

        if df.empty:
            logger.warning("No pairs passed hard filters.")
            return df

        # ------ statistical score (lower eg_pvalue = better) ------
        # Combine eg_pvalue, adf_pvalue, half_life centrality
        df["_stat_score"] = self._stat_score(df)

        # ------ merge ML scores ------
        if ml_scores is not None:
            df = df.merge(ml_scores[["ticker_a", "ticker_b", "ml_score"]],
                          on=["ticker_a", "ticker_b"], how="left")
            df["ml_score"] = df["ml_score"].fillna(0.5)
        else:
            df["ml_score"] = 0.5  # neutral when no ML

        # ------ composite score ------
        df["composite_score"] = (
            self.stat_weight * df["_stat_score"]
            + self.ml_weight * df["ml_score"]
        )

        df = df.drop(columns=["_stat_score"])
        result = (
            df.sort_values("composite_score", ascending=False)
            .head(self.top_n)
            .reset_index(drop=True)
        )

        logger.info(
            f"PairRanker: {len(coint_df)} candidates → "
            f"{len(df)} passed filters → top {len(result)} selected"
        )
        self._log_top_pairs(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stat_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Statistical score in [0, 1].
        Combines EG p-value (lower = better), ADF p-value (lower = better),
        and half-life centrality (closer to midpoint of [min, max] = better).
        """
        cfg = settings.analysis

        # Invert and normalise eg_pvalue
        eg_norm = 1.0 - self._minmax(df["eg_pvalue"])

        # Invert and normalise adf_pvalue
        adf_norm = 1.0 - self._minmax(df["adf_pvalue"])

        # Half-life centrality: best at midpoint of acceptable range
        hl_mid = (cfg.half_life_min + cfg.half_life_max) / 2
        hl_norm = 1.0 - (df["half_life"] - hl_mid).abs() / hl_mid
        hl_norm = hl_norm.clip(0, 1)

        return (0.4 * eg_norm + 0.3 * adf_norm + 0.3 * hl_norm).fillna(0)

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    @staticmethod
    def _log_top_pairs(df: pd.DataFrame) -> None:
        for _, row in df.head(5).iterrows():
            logger.info(
                f"  #{row.name+1:02d}  {row['ticker_a']}|{row['ticker_b']}  "
                f"composite={row['composite_score']:.3f}  "
                f"half_life={row.get('half_life', float('nan')):.1f}d  "
                f"eg_p={row.get('eg_pvalue', float('nan')):.4f}"
            )

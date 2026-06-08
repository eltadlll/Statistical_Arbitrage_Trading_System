"""
features/builder.py
-------------------
Assembles the feature matrix that feeds the ML pair-selection models.
One row per candidate pair; columns are statistical + price-based features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.spread import SpreadBuilder
from src.data.preprocessors import Preprocessor


_spread_builder = SpreadBuilder()


class FeatureBuilder:
    """
    Build the feature matrix for ML pair selection.

    Parameters
    ----------
    log_prices  : Log-price DataFrame (columns = tickers).
    returns     : Daily returns DataFrame (columns = tickers).
    coint_df    : Output of CointegrationAnalyzer.test_all_pairs().
    """

    FEATURE_COLS = [
        "eg_pvalue",
        "adf_pvalue",
        "kpss_pvalue",
        "johansen_trace_ratio",
        "hedge_ratio",
        "half_life",
        "half_life_norm",
        "spread_std",
        "hurst",
        "ou_kappa",
        "ou_sigma",
        "rolling_corr_mean",
        "rolling_corr_std",
        "rolling_corr_pct_above_07",
        "vol_ratio",
        "price_ratio_mean",
        "price_ratio_std",
        "zscore_std",
        "n_crossings",
        "days_of_data",
    ]

    def __init__(
        self,
        log_prices: pd.DataFrame,
        returns: pd.DataFrame,
        coint_df: pd.DataFrame,
    ) -> None:
        self.log_prices = log_prices
        self.returns = returns
        self.coint_df = coint_df

    def build(
        self, pairs: Optional[list[tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Compute all features for each pair in `coint_df` (or a subset `pairs`).

        Returns a DataFrame with one row per pair + a 'label' column
        (1 if cointegrated, 0 otherwise) for supervised training.
        """
        df = self.coint_df.copy()
        if pairs is not None:
            pair_set = {(a, b) for a, b in pairs}
            mask = df.apply(
                lambda r: (r["ticker_a"], r["ticker_b"]) in pair_set, axis=1
            )
            df = df[mask].reset_index(drop=True)

        logger.info(f"Building features for {len(df)} pairs …")
        extra_rows = []
        for _, row in df.iterrows():
            a, b = row["ticker_a"], row["ticker_b"]
            feats = self._pair_features(a, b, row)
            extra_rows.append(feats)

        extras = pd.DataFrame(extra_rows, index=df.index)
        result = pd.concat([df, extras], axis=1)

        # Normalise half-life to [0, 1] range
        hl = result["half_life"].replace([np.inf, -np.inf], np.nan)
        result["half_life_norm"] = (hl - hl.min()) / (hl.max() - hl.min() + 1e-9)

        # Label: 1 if cointegrated
        result["label"] = result["is_cointegrated"].astype(int)

        logger.info("Feature matrix built.")
        return result

    # ------------------------------------------------------------------
    # Per-pair feature computation
    # ------------------------------------------------------------------

    def _pair_features(
        self, ticker_a: str, ticker_b: str, coint_row: pd.Series
    ) -> dict:
        feats: dict = {}

        try:
            lp_a = self.log_prices[ticker_a].dropna()
            lp_b = self.log_prices[ticker_b].dropna()
            ret_a = self.returns[ticker_a].dropna()
            ret_b = self.returns[ticker_b].dropna()

            # Align series
            log_aligned = pd.concat([lp_a, lp_b], axis=1).dropna()
            ret_aligned = pd.concat([ret_a, ret_b], axis=1).dropna()
            lp_a, lp_b = log_aligned.iloc[:, 0], log_aligned.iloc[:, 1]
            ret_a, ret_b = ret_aligned.iloc[:, 0], ret_aligned.iloc[:, 1]

            n = len(lp_a)
            feats["days_of_data"] = n

            # --- Johansen trace ratio ---
            joh_stat = coint_row.get("johansen_trace_stat", np.nan)
            joh_crit = coint_row.get("johansen_trace_crit_90", np.nan)
            feats["johansen_trace_ratio"] = (
                joh_stat / joh_crit if not np.isnan(joh_crit) and joh_crit != 0 else np.nan
            )

            # --- Spread features ---
            spread, _ = _spread_builder.ols_spread(lp_a, lp_b)
            ou = _spread_builder.ou_parameters(spread)
            feats["hurst"] = _spread_builder.hurst_exponent(spread)
            feats["ou_kappa"] = ou["kappa"]
            feats["ou_sigma"] = ou["sigma"]
            feats["zscore_std"] = _spread_builder.zscore(spread).std()

            # Number of times z-score crosses zero (mean-reversion proxy)
            z = _spread_builder.zscore(spread).dropna()
            feats["n_crossings"] = int(((z.shift(1) * z) < 0).sum())

            # --- Rolling correlation ---
            roll_corr = ret_a.rolling(60).corr(ret_b).dropna()
            feats["rolling_corr_mean"] = float(roll_corr.mean())
            feats["rolling_corr_std"] = float(roll_corr.std())
            feats["rolling_corr_pct_above_07"] = float((roll_corr.abs() >= 0.7).mean())

            # --- Volatility ratio ---
            vol_a = float(ret_a.std())
            vol_b = float(ret_b.std())
            feats["vol_ratio"] = vol_a / vol_b if vol_b != 0 else np.nan

            # --- Price ratio stats ---
            price_ratio = np.exp(lp_a) / np.exp(lp_b)
            feats["price_ratio_mean"] = float(price_ratio.mean())
            feats["price_ratio_std"] = float(price_ratio.std())

        except Exception as exc:
            logger.warning(f"Feature error for {ticker_a}|{ticker_b}: {exc}")
            for col in [
                "days_of_data", "johansen_trace_ratio", "hurst",
                "ou_kappa", "ou_sigma", "zscore_std", "n_crossings",
                "rolling_corr_mean", "rolling_corr_std",
                "rolling_corr_pct_above_07", "vol_ratio",
                "price_ratio_mean", "price_ratio_std",
            ]:
                feats.setdefault(col, np.nan)

        return feats

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def feature_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract (X, y) from the full feature DataFrame for ML training.
        Drops rows with NaN in any feature column.
        """
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available].replace([np.inf, -np.inf], np.nan).dropna()
        y = df.loc[X.index, "label"]
        return X, y

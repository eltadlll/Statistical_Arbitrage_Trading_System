"""
universe.py
-----------
Loads tickers from config/universe.yaml, applies liquidity filters,
and generates candidate pairs within the same sector group.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]


class UniverseBuilder:
    """
    Build and filter the trading universe from universe.yaml.

    Parameters
    ----------
    universe_path : Path | None
        Override path to universe.yaml.
    min_avg_volume : float
        Minimum average daily dollar volume to pass the liquidity filter.
    min_price : float
        Minimum average price (avoids penny stocks).
    """

    def __init__(
        self,
        universe_path: Optional[Path] = None,
        min_avg_volume: float = 1e6,
        min_price: float = 5.0,
    ) -> None:
        self.universe_path = universe_path or ROOT / "config" / "universe.yaml"
        self.min_avg_volume = min_avg_volume
        self.min_price = min_price
        self._universe: dict = {}

    # ------------------------------------------------------------------
    # Universe loading
    # ------------------------------------------------------------------

    def load(self) -> dict[str, list[str]]:
        """
        Parse universe.yaml and return a flat {group_name: [tickers]} dict.
        All groups are flattened (equities.technology → "technology").
        """
        with open(self.universe_path) as fh:
            raw = yaml.safe_load(fh)

        groups: dict[str, list[str]] = {}

        # Equities – nested by sector
        for sector, tickers in raw.get("equities", {}).items():
            groups[sector] = tickers or []

        # ETFs – nested by sub-group
        for subgroup, tickers in raw.get("etfs", {}).items():
            groups[f"etf_{subgroup}"] = tickers or []

        # Commodities – flat list
        if "commodities" in raw:
            groups["commodities"] = raw["commodities"]

        self._universe = groups
        all_tickers = [t for tickers in groups.values() for t in tickers]
        logger.info(
            f"Universe loaded: {len(groups)} groups, {len(all_tickers)} tickers total"
        )
        return groups

    def all_tickers(self) -> list[str]:
        """Return a deduplicated flat list of all tickers in the universe."""
        if not self._universe:
            self.load()
        return sorted({t for tickers in self._universe.values() for t in tickers})

    # ------------------------------------------------------------------
    # Liquidity filter
    # ------------------------------------------------------------------

    def apply_liquidity_filter(
        self,
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
    ) -> list[str]:
        """
        Keep tickers that pass minimum price and (optionally) volume thresholds.

        Parameters
        ----------
        prices  : DataFrame of Close prices, columns = tickers
        volume  : DataFrame of daily volume (shares), same shape as prices.
                  If None, only the price filter is applied.

        Returns
        -------
        List of tickers that pass all filters.
        """
        passed: list[str] = []
        for ticker in prices.columns:
            avg_price = prices[ticker].mean()
            if avg_price < self.min_price:
                logger.debug(f"  {ticker} dropped – avg price ${avg_price:.2f} < ${self.min_price}")
                continue

            if volume is not None and ticker in volume.columns:
                # Dollar volume proxy: price × share volume
                dollar_vol = (prices[ticker] * volume[ticker]).mean()
                if dollar_vol < self.min_avg_volume:
                    logger.debug(
                        f"  {ticker} dropped – avg $ vol ${dollar_vol:,.0f} < ${self.min_avg_volume:,.0f}"
                    )
                    continue

            passed.append(ticker)

        logger.info(
            f"Liquidity filter: {len(passed)}/{len(prices.columns)} tickers passed"
        )
        return passed

    # ------------------------------------------------------------------
    # Pair generation
    # ------------------------------------------------------------------

    def generate_pairs(
        self,
        tickers: Optional[list[str]] = None,
        cross_sector: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Generate candidate pairs from the universe.

        Parameters
        ----------
        tickers     : Restrict to this list (e.g. post-liquidity-filter tickers).
        cross_sector: If True, generate all cross-sector pairs too.

        Returns
        -------
        List of (ticker_A, ticker_B) tuples (no duplicates, A < B alphabetically).
        """
        if not self._universe:
            self.load()

        eligible = set(tickers) if tickers else None
        pairs: list[tuple[str, str]] = []

        # Within-sector pairs (always generated)
        for group, group_tickers in self._universe.items():
            filtered = [t for t in group_tickers if eligible is None or t in eligible]
            if len(filtered) < 2:
                continue
            for a, b in itertools.combinations(sorted(filtered), 2):
                pairs.append((a, b))

        # Cross-sector pairs (optional)
        if cross_sector:
            all_filtered = sorted(
                {t for g in self._universe.values() for t in g}
                & (eligible or {t for g in self._universe.values() for t in g})
            )
            for a, b in itertools.combinations(all_filtered, 2):
                if (a, b) not in pairs:
                    pairs.append((a, b))

        # Deduplicate
        pairs = list(dict.fromkeys(pairs))
        logger.info(f"Generated {len(pairs)} candidate pairs")
        return pairs

    def group_of(self, ticker: str) -> Optional[str]:
        """Return the sector/group name for a given ticker."""
        if not self._universe:
            self.load()
        for group, tickers in self._universe.items():
            if ticker in tickers:
                return group
        return None

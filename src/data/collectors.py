"""
collectors.py
-------------
Downloads OHLCV data from yfinance (primary) and Alpha Vantage (fallback).
All results are disk-cached via joblib to avoid redundant API calls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from joblib import Memory
from loguru import logger

from config.settings import settings

_memory = Memory(location=str(settings.data.cache_dir), verbose=0)


class DataCollector:
    """
    Download and cache OHLCV price data for a list of tickers.

    Parameters
    ----------
    start : str  e.g. "2015-01-01"
    end   : str  e.g. "2024-12-31"
    interval : str  yfinance interval string, default "1d"
    """

    def __init__(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> None:
        self.start = start or settings.data.start_date
        self.end = end or settings.data.end_date
        self.interval = interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, tickers: list[str], column: str = "Close") -> pd.DataFrame:
        """
        Return a DataFrame of `column` prices, one column per ticker.
        Missing tickers are dropped with a warning.
        """
        logger.info(f"Fetching {len(tickers)} tickers | {self.start} → {self.end}")
        raw = self._batch_download(tickers)
        if raw.empty:
            raise ValueError("Download returned an empty DataFrame.")

        # yfinance multi-ticker download returns MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                prices = raw[column]
            except KeyError:
                prices = raw.xs(column, axis=1, level=0)
        else:
            prices = raw[[column]] if column in raw.columns else raw

        dropped = [t for t in tickers if t not in prices.columns]
        if dropped:
            logger.warning(f"Tickers not returned by yfinance: {dropped}")

        return prices.dropna(how="all")

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Return full OHLCV DataFrame for a single ticker."""
        logger.info(f"Fetching OHLCV for {ticker}")
        return self._single_download(ticker)

    def fetch_all_ohlcv(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Return {ticker: OHLCV DataFrame} for every ticker."""
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                result[ticker] = self._single_download(ticker)
            except Exception as exc:
                logger.warning(f"Failed to fetch {ticker}: {exc}")
            time.sleep(0.1)  # polite rate-limiting
        return result

    def save_raw(self, prices: pd.DataFrame, name: str = "prices") -> Path:
        """Persist a price DataFrame to data/raw/."""
        path = settings.data.raw_dir / f"{name}.parquet"
        prices.to_parquet(path)
        logger.info(f"Saved raw data → {path}")
        return path

    def load_raw(self, name: str = "prices") -> pd.DataFrame:
        """Load a previously saved raw price DataFrame."""
        path = settings.data.raw_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No raw file at {path}. Run fetch() first.")
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Internal helpers (cached)
    # ------------------------------------------------------------------

    @_memory.cache
    def _batch_download(self, tickers: list[str]) -> pd.DataFrame:
        return yf.download(
            tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    @_memory.cache
    def _single_download(self, ticker: str) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        df = tkr.history(
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]


# ------------------------------------------------------------------
# Alpha Vantage fallback (used when yfinance is unavailable)
# ------------------------------------------------------------------

class AlphaVantageCollector:
    """
    Thin wrapper around the Alpha Vantage REST API.
    Requires AV_API_KEY environment variable.
    Free tier: 25 requests / day, 500 / month.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.data.alpha_vantage_key
        if self.api_key == "demo":
            logger.warning("Using Alpha Vantage demo key – limited to MSFT only.")

    def fetch_daily(self, ticker: str, outputsize: str = "full") -> pd.DataFrame:
        import requests

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        key = "Time Series (Daily)"
        if key not in data:
            raise ValueError(f"AV response missing '{key}': {data.get('Note', data)}")

        df = pd.DataFrame(data[key]).T
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(float)
        return df.sort_index()

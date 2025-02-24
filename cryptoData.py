"""Functions for fetching and processing cryptocurrency data using CCXT."""

import pandas as pd
import ccxt
import numpy as np
from typing import Tuple, List, Optional


class CryptoDataFetcher:
    """Fetches and processes OHLCV data for cryptocurrencies via CCXT."""

    def __init__(self, exchangeName: str = 'binance'):
        """Initialize with a CCXT exchange (default: Binance)."""
        self.exchange = getattr(ccxt, exchangeName)({
            'enableRateLimit': True,
        })

    def fetchOhlcvData(self, symbol: str, timeframe: str = '1d', limit: int = 30) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a given symbol and timeframe."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.NetworkError as e:
            print(f"Network error fetching {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            print(f"Exchange error fetching {symbol}: {e}")
            return None

    def extractFeatures(self, symbol: str) -> Tuple[List[float], pd.Series]:
        """Extract quantitative features and 7-day price trend from OHLCV data."""
        df = self.fetchOhlcvData(symbol)
        if df is None or df.empty:
            return [np.nan] * 5, pd.Series()

        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(14).std().iloc[-1] * (252 ** 0.5)  # Annualized
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100 if len(df) >= 7 else 0
        volumeSkew = df['volume'].tail(3).mean() / df['volume'].mean() if len(df) >= 3 else 1
        autocorrelation = returns.autocorr(lag=1)
        sma = df['close'].rolling(20).mean()
        meanReversionStrength = ((df['close'].iloc[-1] - sma.iloc[-1]) / sma.iloc[-1]) / volatility if volatility else 0

        features = [volatility, momentum, volumeSkew, autocorrelation, meanReversionStrength]
        priceTrend = df['close'].tail(7)

        return features, priceTrend


# Global instance for convenience
dataFetcher = CryptoDataFetcher()
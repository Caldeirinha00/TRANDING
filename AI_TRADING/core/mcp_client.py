import MetaTrader5 as mt5
import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from core.config import CONFIG
from core.mt5_executor import mt5_call, mt5_exec, get_mt5_executor

logger = logging.getLogger(__name__)

class MT5Client:
    def __init__(self) -> None:
        self.connected: bool = False

    def connect(self) -> bool:
        """Initialize and authenticate with MetaTrader 5 terminal."""
        if self.connected:
            return True

        def _init():
            return mt5.initialize(
                path=CONFIG.MT5_PATH,
                login=CONFIG.MT5_LOGIN,
                password=CONFIG.MT5_PASSWORD,
                server=CONFIG.MT5_SERVER,
            )

        future = get_mt5_executor().submit(_init)
        if not future.result():
            logger.error(f"initialize() failed, error code = {mt5.last_error()}")
            return False

        self.connected = True
        logger.info(f"Connected to MT5 account #{CONFIG.MT5_LOGIN}")
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self.connected:
            future = get_mt5_executor().submit(mt5.shutdown)
            future.result()
            self.connected = False
            logger.info("Disconnected from MT5")

    def reconnect(self) -> bool:
        """Disconnect then reconnect to MT5."""
        self.disconnect()
        return self.connect()

    async def get_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get the latest bid/ask price for a symbol."""
        if not self.connected and not self.connect():
            return None

        # Fetch price — run in executor to avoid blocking the event loop
        def _get_tick():
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
            return mt5.symbol_info_tick(symbol)
        
        tick = await mt5_exec(_get_tick)

        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time,
            "time_msc": tick.time_msc,
        }

    async def get_candles_latest(
        self,
        symbol: str,
        timeframe: int = mt5.TIMEFRAME_M5,
        count: int = 100,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get the latest candles for a symbol."""
        if not self.connected and not self.connect():
            return None

        rates = await mt5_call(mt5.copy_rates_from_pos, symbol, timeframe, 1, count)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get candles for {symbol}")
            return None

        # Convert to a list of dicts
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df.to_dict('records')

# Global instance
mcp_client = MT5Client()

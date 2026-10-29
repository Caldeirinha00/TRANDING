from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import json
import os
from typing import List

class Settings(BaseSettings):
    # Networking
    LAN_IP: str = Field(default="192.168.1.90")
    QUESTDB_HOST: str = Field(default="192.168.1.90")
    QUESTDB_PORT: int = Field(default=9009)
    QUESTDB_AUTH_TOKEN: str = Field(default="")
    DRAGONFLY_HOST: str = Field(default="192.168.1.90")
    DRAGONFLY_PORT: int = Field(default=6379)
    QDRANT_HOST: str = Field(default="192.168.1.90")
    QDRANT_PORT: int = Field(default=6333)

    # API Keys
    FMP_API_KEY: str
    ALPHA_VANTAGE_API_KEY: str = Field(default="")
    MISTRAL_API_KEYS: str = Field(default="")

    # LLM Configuration
    LLM_MODEL: str = Field(default="ministral-8b-2512")
    LLM_FALLBACK_MODEL: str = Field(default="ministral-3b-2512")

    # Trading Configuration
    DEFAULT_SYMBOLS: str = Field(default='["EURUSD"]')
    MAX_DAILY_DRAWDOWN_PCT: float = Field(default=3.0)
    MAX_TOTAL_DRAWDOWN_PCT: float = Field(default=20.0)
    MAX_SINGLE_TRADE_LOSS_PCT: float = Field(default=1.0)
    MAX_CONCURRENT_POSITIONS: int = Field(default=3)
    MAX_LOT_SIZE: float = Field(default=0.5)
    MAX_TRADES_PER_DAY: int = Field(default=15)

    # MT5 Connection
    MT5_PATH: str = Field(default='C:\\Program Files\\MetaTrader 5\\terminal64.exe')
    MT5_LOGIN: int = Field(default=0)
    MT5_PASSWORD: str = Field(default="")
    MT5_SERVER: str = Field(default="PepperstoneUK-Demo")

    # Analysis
    ANALYSIS_INTERVAL_SECONDS: int = Field(default=300)

    # Twin Mode (MQL5 EA executes on chart, Python acts as AI Copilot / Risk Guardian)
    TWIN_MODE: bool = Field(default=True)
    TWIN_SYMBOLS: str = Field(default='["EURUSD"]')

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def symbols_list(self) -> List[str]:
        try:
            return json.loads(self.DEFAULT_SYMBOLS)
        except Exception:
            return ["EURUSD"]

    @property
    def twin_symbols_list(self) -> List[str]:
        try:
            return [s.upper() for s in json.loads(self.TWIN_SYMBOLS)]
        except Exception:
            return ["EURUSD"]

    @property
    def redis_url(self) -> str:
        return f'redis://{self.DRAGONFLY_HOST}:{self.DRAGONFLY_PORT}/0'

# Global configuration instance
CONFIG = Settings()

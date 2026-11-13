from pathlib import Path
from typing import Dict, List, Any
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

_DOTENV_CONTROL_ESCAPE_MAP = str.maketrans(
    {
        "\a": r"\a",
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
        "\v": r"\v",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MetaTrader 5
    mt5_terminal_path: str = Field(default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    mt5_data_path: str = Field(
        default=r"C:\Users\mob70\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
    )
    mt5_login: int = Field(default=0)
    mt5_password: str = Field(default="")
    mt5_server: str = Field(default="PepperstoneUK-Demo")

    # Docker LAN Infrastructure
    dragonfly_host: str = Field(default="192.168.1.90")
    dragonfly_port: int = Field(default=6379)
    questdb_host: str = Field(default="192.168.1.90")
    questdb_rest_port: int = Field(default=9000)
    questdb_pg_port: int = Field(default=8812)
    questdb_ilp_port: int = Field(default=9009)
    questdb_pg_user: str = Field(default="admin")
    questdb_pg_password: str = Field(default="quest")
    questdb_pg_dbname: str = Field(default="qdb")

    # Orçamento de Risco ESMA (1.000 EUR)
    initial_capital_eur: float = Field(default=1000.0)
    commission_per_lot_roundtrip_eur: float = Field(default=0.0)  # 0.0 Pepperstone Standard; ~6.50 Razor
    max_risk_per_trade_pct: float = Field(default=0.0100)      # 1.00% (<= 10.00 EUR)
    max_portfolio_heat_pct: float = Field(default=0.0210)      # 2.10% (<= 21.00 EUR)
    max_active_risk_positions: int = Field(default=3)          # Máx 3 posições fora de BE
    daily_dd_kill_switch_pct: float = Field(default=0.0285)    # 2.85% (Preventivo de 3.0%)
    total_dd_kill_switch_pct: float = Field(default=0.1950)    # 19.5% (Preventivo de 20.0%)
    max_swap_to_1r_ratio: float = Field(default=0.1500)        # Veto se swap 4d > 15% de 1R

    # Meta-Labeling & Option A Pass-Through Controls
    use_meta_labeling: bool = Field(default=False)  # Option A: deterministic Layer 1 Pass-Through at 0.01 lot
    fixed_pass_through_lots: float = Field(default=0.01)
    meta_min_probability: float = Field(default=0.55)
    meta_high_conviction_probability: float = Field(default=0.68)
    meta_min_expected_net_r: float = Field(default=0.15)

    # Live Execution & Safety Guards (Fase 3/4)
    allow_live_real_money: bool = Field(default=False)
    mt5_magic_number: int = Field(default=20261112)
    order_deviation_points: int = Field(default=20)
    h4_close_offset_seconds: float = Field(default=2.0)
    max_live_spread_multiplier: float = Field(default=3.0)  # Reject live entry if spread > 3x active benchmark

    # Telegram
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    @field_validator("mt5_terminal_path", "mt5_data_path", mode="before")
    @classmethod
    def _restore_dotenv_windows_escapes(cls, value: Any) -> Any:
        # python-dotenv interprets \t, \n, etc. inside double-quoted Windows paths as control chars
        if isinstance(value, str):
            return value.translate(_DOTENV_CONTROL_ESCAPE_MAP)
        return value

    def load_universe(self) -> Dict[str, Any]:
        universe_path = BASE_DIR / "config" / "universe.yaml"
        with open(universe_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def approved_symbols(self) -> List[str]:
        return list(self.load_universe()["approved_symbols"].keys())

    @property
    def benchmark_spread_points(self) -> Dict[str, int]:
        universe = self.load_universe().get("approved_symbols", {})
        return {
            sym: int(meta["typical_spread_points"])
            for sym, meta in universe.items()
            if isinstance(meta, dict) and "typical_spread_points" in meta
        }


settings = Settings()

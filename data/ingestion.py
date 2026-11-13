from datetime import datetime, timezone
from typing import Dict, List, Tuple
import MetaTrader5 as mt5
import pandas as pd
import structlog
from config.settings import settings
from core.mt5_connector import MT5Connector, SymbolContractSpec
from core.storage import StorageClient, SymbolCostSnapshot
from data.cost_model import CostModel
from strategy.features import compute_atr

logger = structlog.get_logger(__name__)


class HistoricalDataIngestor:
    """Downloads closed H4/D1 MT5 histories and real contract specs into QuestDB."""

    def __init__(self, connector: MT5Connector, storage: StorageClient) -> None:
        self.connector = connector
        self.storage = storage
        self.cost_model = CostModel()

    @staticmethod
    def fetch_closed_range(symbol: str, timeframe_str: str, start_utc: datetime) -> pd.DataFrame:
        """Fetch historical MT5 bars strictly trimming out any unclosed current bar [0]."""
        tf_map = {"H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
        tf = tf_map[timeframe_str]
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select {symbol}: {mt5.last_error()}")

        latest_closed = mt5.copy_rates_from_pos(symbol, tf, 1, 1)
        if latest_closed is None or len(latest_closed) == 0:
            raise RuntimeError(f"No closed bar reference for {symbol} {timeframe_str}: {mt5.last_error()}")
        max_closed_ts = int(latest_closed[0]["time"])

        now_upper = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year + 1)
        rates = mt5.copy_rates_range(symbol, tf, start_utc, now_upper)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No historical range returned for {symbol} {timeframe_str}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df = df[df["time"] <= max_closed_ts].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index(ascending=True)
        return df

    def sync_all_approved_symbols(
        self,
        start_utc: datetime = datetime(2015, 1, 1, tzinfo=timezone.utc),
        symbols: List[str] | None = None,
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, SymbolContractSpec]]:
        """Sync H4, D1 closed histories and live contract specs for the given symbols (default approved 8)."""
        self.storage.ensure_ohlcv_tables()
        counts: Dict[str, Dict[str, int]] = {}
        specs: Dict[str, SymbolContractSpec] = {}
        snapshots: List[SymbolCostSnapshot] = []
        now_utc = datetime.now(timezone.utc)

        target_symbols = symbols if symbols is not None else settings.approved_symbols
        for sym in target_symbols:
            spec = self.connector.get_symbol_spec(sym)
            if spec is None:
                raise RuntimeError(f"Symbol spec unavailable for {sym}")
            specs[sym] = spec

            df_h4 = self.fetch_closed_range(sym, "H4", start_utc=start_utc)
            df_d1 = self.fetch_closed_range(sym, "D1", start_utc=start_utc)

            ok_h4, msg_h4 = self.storage.write_ohlcv_dataframe("ohlcv_h4", sym, df_h4)
            if not ok_h4:
                raise RuntimeError(msg_h4)

            ok_d1, msg_d1 = self.storage.write_ohlcv_dataframe("ohlcv_d1", sym, df_d1)
            if not ok_d1:
                raise RuntimeError(msg_d1)

            atr_price = float(compute_atr(df_h4.tail(150), period=14).iloc[-1])
            sl_dist = 2.0 * atr_price
            friction_long = self.cost_model.evaluate_trade_friction(
                spec=spec, side=+1, lots=0.01, sl_distance_price=sl_dist, entry_time_utc=now_utc, holding_bars_h4=24
            )
            friction_short = self.cost_model.evaluate_trade_friction(
                spec=spec, side=-1, lots=0.01, sl_distance_price=sl_dist, entry_time_utc=now_utc, holding_bars_h4=24
            )
            status = "APROVADO (<=1%)" if friction_long.gross_1r_risk_eur <= 10.0 else "BLOQUEADO (>1%)"
            snapshots.append(
                SymbolCostSnapshot(
                    symbol=sym,
                    status=status,
                    margin_001_eur=spec.margin_initial_001_eur,
                    atr14_h4_pips=round(atr_price / spec.pip_size, 2),
                    sl_pips=round(sl_dist / spec.pip_size, 2),
                    risk_1r_eur=friction_long.gross_1r_risk_eur,
                    spread_points=spec.spread_points,
                    friction_pct=round(friction_long.friction_to_risk_ratio * 100.0, 2),
                    swap_mode=spec.swap_mode,
                    swap_long_raw=spec.swap_long,
                    swap_short_raw=spec.swap_short,
                    swap_4d_long_eur=friction_long.expected_swap_eur,
                    swap_4d_short_eur=friction_short.expected_swap_eur,
                    captured_at=now_utc,
                )
            )
            counts[sym] = {"H4": len(df_h4), "D1": len(df_d1)}

        self.storage.write_symbol_cost_snapshots(snapshots)
        return counts, specs

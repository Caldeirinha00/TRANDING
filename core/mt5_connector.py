from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import asyncio
import time
from typing import Optional, Dict, Any, List, Sequence, Tuple
import MetaTrader5 as mt5
import pandas as pd
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SymbolContractSpec:
    """Especificação contratual extraída em tempo real do servidor MT5."""
    symbol: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    digits: int
    point: float
    pip_size: float
    trade_contract_size: float
    trade_tick_value: float
    trade_tick_value_profit: float
    trade_tick_value_loss: float
    trade_tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread_points: int
    swap_mode: int
    swap_long: float
    swap_short: float
    swap_rollover3days: int
    stops_level_points: int
    margin_initial_001_eur: float
    bid: float
    ask: float


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    login: int
    server: str
    currency: str
    balance_eur: float
    equity_eur: float
    margin_eur: float
    margin_free_eur: float
    leverage: int
    is_demo: bool


@dataclass(frozen=True)
class BrokerOpenPosition:
    ticket: int
    symbol: str
    side: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    swap_eur: float
    profit_eur: float
    time_open_server: datetime
    magic: int
    comment: str


@dataclass(frozen=True)
class OrderExecutionResult:
    success: bool
    action: str
    symbol: str
    retcode: int
    comment: str
    ticket: int = 0
    order_id: int = 0
    price: float = 0.0
    volume: float = 0.0


class MT5Connector:
    """
    Wrapper Thread-Safe de Fonte Única para a API Python do MetaTrader 5.
    Elimina a necessidade de EAs em MQL5 ou Named Pipes binários.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5_single_thread")
        self._connected = False

    def connect(self) -> bool:
        init_kwargs: Dict[str, Any] = {"path": settings.mt5_terminal_path}
        if settings.mt5_login > 0 and settings.mt5_password:
            init_kwargs["login"] = settings.mt5_login
            init_kwargs["password"] = settings.mt5_password
            init_kwargs["server"] = settings.mt5_server

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            logger.error("mt5_init_failed", error=err, terminal=settings.mt5_terminal_path)
            self._connected = False
            return False

        account_info = mt5.account_info()
        if account_info is None:
            logger.error("mt5_account_info_unavailable", error=mt5.last_error())
            self._connected = False
            return False

        self._connected = True
        if account_info.balance >= 50000.0:
            logger.warning(
                "mt5_account_mismatch_expected_1000_eur",
                server=account_info.server,
                currency=account_info.currency,
                leverage=account_info.leverage,
            )
        else:
            logger.info(
                "mt5_connected",
                login=account_info.login,
                server=account_info.server,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                leverage=account_info.leverage,
            )
        return True

    def shutdown(self) -> None:
        mt5.shutdown()
        self._connected = False

    def ensure_connected(self) -> bool:
        """Verify live MT5 terminal link and reconnect if dropped."""
        if self._connected and mt5.terminal_info() is not None and mt5.account_info() is not None:
            return True
        return self.reconnect()

    def reconnect(self) -> bool:
        """Reset MT5 IPC session and re-initialize connection."""
        logger.warning("mt5_reconnecting")
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        return self.connect()

    async def call_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        if kwargs and args:
            return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
        if kwargs:
            return await loop.run_in_executor(self._executor, lambda: func(**kwargs))
        return await loop.run_in_executor(self._executor, lambda: func(*args))

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolContractSpec]:
        if not mt5.symbol_select(symbol, True):
            logger.warning("symbol_select_failed", symbol=symbol, error=mt5.last_error())
            return None

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None:
            return None

        # Newly subscribed Market Watch symbols can briefly report bid==0 or spread==0 until first tick arrives
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and (
            tick is None
            or tick.bid <= 0.0
            or tick.ask <= 0.0
            or (info.spread == 0 and tick.ask == tick.bid)
        ):
            time.sleep(0.05)
            info = mt5.symbol_info(symbol) or info
            tick = mt5.symbol_info_tick(symbol)

        if tick is None or tick.bid <= 0.0:
            return None

        tick_spread_points = int(round((tick.ask - tick.bid) / info.point)) if info.point > 0 else 0
        effective_spread_points = max(int(info.spread), max(0, tick_spread_points))

        # Determinação do tamanho de 1 Pip padrão (4ª casa ou 2ª casa em pares JPY)
        pip_size = info.point * 10.0 if info.digits in (3, 5) else info.point

        # Cálculo exato da margem exigida pelo servidor MT5 para 0,01 lote
        margin_001 = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 0.01, tick.ask)
        if margin_001 is None:
            # Fallback ESMA conservador caso o servidor não devolva em mercado fechado
            esma_lev = 20.0 if "JPY" in symbol and not symbol.startswith("USD") else 30.0
            margin_001 = (0.01 * info.trade_contract_size) / esma_lev

        return SymbolContractSpec(
            symbol=symbol,
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            currency_margin=info.currency_margin,
            digits=info.digits,
            point=info.point,
            pip_size=pip_size,
            trade_contract_size=info.trade_contract_size,
            trade_tick_value=info.trade_tick_value,
            trade_tick_value_profit=getattr(info, "trade_tick_value_profit", info.trade_tick_value),
            trade_tick_value_loss=getattr(info, "trade_tick_value_loss", info.trade_tick_value),
            trade_tick_size=info.trade_tick_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            spread_points=effective_spread_points,
            swap_mode=info.swap_mode,
            swap_long=info.swap_long,
            swap_short=info.swap_short,
            swap_rollover3days=info.swap_rollover3days,
            stops_level_points=info.trade_stops_level,
            margin_initial_001_eur=float(margin_001),
            bid=tick.bid,
            ask=tick.ask,
        )

    def get_historical_benchmark_spec(
        self,
        symbol: str,
        benchmark_spread_points: Optional[int] = None,
    ) -> Optional[SymbolContractSpec]:
        """
        Return live broker contract spec with deterministic active-market spread override
        for reproducible offline backtesting and Meta-Labeling training (avoiding midnight
        rollover or weekend quote artifacts across historical bars).
        """
        spec = self.get_symbol_spec(symbol)
        if spec is None or benchmark_spread_points is None:
            return spec
        if spec.spread_points != int(benchmark_spread_points):
            logger.info(
                "benchmark_spread_applied",
                symbol=symbol,
                live_spread_points=spec.spread_points,
                benchmark_spread_points=int(benchmark_spread_points),
            )
        return replace(spec, spread_points=int(benchmark_spread_points))

    def fetch_closed_bars(
        self,
        symbol: str,
        timeframe_str: str,
        count: int = 500,
        server_now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Extrai EXCLUSIVAMENTE barras já fechadas indexadas pelo timestamp do servidor.
        Quando `server_now` é fornecido, filtra estritamente barras cujo fecho
        (`bar_open + tf_duration`) <= `server_now`, prevenindo falhas quando um símbolo
        ainda não recebeu o primeiro tick da nova barra após o fecho.
        """
        tf_map = {
            "H4": (mt5.TIMEFRAME_H4, timedelta(hours=4)),
            "D1": (mt5.TIMEFRAME_D1, timedelta(days=1)),
            "H1": (mt5.TIMEFRAME_H1, timedelta(hours=1)),
        }
        tf, tf_delta = tf_map[timeframe_str]
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Falha ao selecionar {symbol} no Market Watch: {mt5.last_error()}")

        if server_now is None:
            rates = mt5.copy_rates_from_pos(symbol, tf, 1, count)
            if rates is None or len(rates) == 0:
                time.sleep(0.25)
                rates = mt5.copy_rates_from_pos(symbol, tf, 1, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count + 2)
            if rates is None or len(rates) == 0:
                time.sleep(0.25)
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count + 2)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Falha ao obter barras {timeframe_str} para {symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(ascending=True, inplace=True)

        if server_now is not None:
            server_ts = pd.Timestamp(server_now)
            if server_ts.tzinfo is None:
                server_ts = server_ts.tz_localize("UTC")
            else:
                server_ts = server_ts.tz_convert("UTC")
            closed_mask = (df.index + pd.Timedelta(tf_delta)) <= server_ts
            df = df.loc[closed_mask]
            if len(df) > count:
                df = df.iloc[-count:]
        return df

    def get_account_snapshot(self) -> Optional[BrokerAccountSnapshot]:
        if not self.ensure_connected():
            return None
        info = mt5.account_info()
        if info is None:
            return None
        is_demo = "demo" in str(info.server).lower() or getattr(info, "trade_mode", 0) == getattr(
            mt5, "ACCOUNT_TRADE_MODE_DEMO", 0
        )
        return BrokerAccountSnapshot(
            login=int(info.login),
            server=str(info.server),
            currency=str(info.currency),
            balance_eur=float(info.balance),
            equity_eur=float(info.equity),
            margin_eur=float(info.margin),
            margin_free_eur=float(info.margin_free),
            leverage=int(info.leverage),
            is_demo=is_demo,
        )

    def get_server_time_utc(self, symbols: Optional[Sequence[str]] = None) -> Optional[datetime]:
        """Return latest MT5 server clock timestamp (encoded as UTC-naive-equivalent server epoch)."""
        if not self.ensure_connected():
            return None
        sym_list = list(symbols) if symbols else settings.approved_symbols
        latest_ts = 0
        for sym in sym_list:
            mt5.symbol_select(sym, True)
            tick = mt5.symbol_info_tick(sym)
            if tick is not None and int(getattr(tick, "time", 0)) > latest_ts:
                latest_ts = int(tick.time)
        if latest_ts <= 0:
            return None
        return datetime.fromtimestamp(latest_ts, tz=timezone.utc)

    def resolve_filling_mode(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        filling_mask = int(getattr(info, "filling_mode", 0))
        if filling_mask & 2:
            return mt5.ORDER_FILLING_IOC
        if filling_mask & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def get_open_positions(self, magic: Optional[int] = None) -> List[BrokerOpenPosition]:
        if not self.ensure_connected():
            return []
        raw_positions = mt5.positions_get()
        if raw_positions is None:
            return []

        result: List[BrokerOpenPosition] = []
        buy_type = getattr(mt5, "POSITION_TYPE_BUY", 0)
        for pos in raw_positions:
            if magic is not None and int(pos.magic) != int(magic):
                continue
            side = 1 if int(pos.type) == buy_type else -1
            result.append(
                BrokerOpenPosition(
                    ticket=int(pos.ticket),
                    symbol=str(pos.symbol),
                    side=side,
                    volume=float(pos.volume),
                    price_open=float(pos.price_open),
                    sl=float(pos.sl),
                    tp=float(pos.tp),
                    price_current=float(pos.price_current),
                    swap_eur=float(pos.swap),
                    profit_eur=float(pos.profit),
                    time_open_server=datetime.fromtimestamp(int(pos.time), tz=timezone.utc),
                    magic=int(pos.magic),
                    comment=str(pos.comment),
                )
            )
        return result

    def _validate_trading_account_safety(self, action: str, symbol: str) -> Optional[OrderExecutionResult]:
        snap = self.get_account_snapshot()
        if snap is None:
            return OrderExecutionResult(False, action, symbol, -1, "MT5_NOT_CONNECTED")
        if not snap.is_demo and not settings.allow_live_real_money:
            logger.critical("blocked_live_real_money_order", server=snap.server, login=snap.login)
            return OrderExecutionResult(False, action, symbol, -2, "BLOCKED_LIVE_REAL_MONEY_DISABLED")
        if snap.balance_eur >= 50000.0:
            logger.critical("blocked_wrong_account_balance", login=snap.login, server=snap.server)
            return OrderExecutionResult(False, action, symbol, -3, "BLOCKED_WRONG_MT5_ACCOUNT")
        return None

    def _build_market_order_request(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_OP_A",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        spec = self.get_symbol_spec(symbol)
        if spec is None:
            return None, "SYMBOL_SPEC_UNAVAILABLE"
        if sl_price <= 0.0 or tp_price <= 0.0:
            return None, "NAKED_ORDER_FORBIDDEN"

        order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
        price = spec.ask if side > 0 else spec.bid
        dev = settings.order_deviation_points if deviation_points is None else int(deviation_points)
        mgc = settings.mt5_magic_number if magic is None else int(magic)

        request: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(float(volume), 2),
            "type": order_type,
            "price": round(float(price), spec.digits),
            "sl": round(float(sl_price), spec.digits),
            "tp": round(float(tp_price), spec.digits),
            "deviation": dev,
            "magic": mgc,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.resolve_filling_mode(symbol),
        }
        return request, None

    def check_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_CHECK",
    ) -> OrderExecutionResult:
        """Dry-run server validation via `mt5.order_check` without submitting any live trade."""
        guard_err = self._validate_trading_account_safety("ORDER_CHECK", symbol)
        if guard_err is not None:
            return guard_err
        request, build_err = self._build_market_order_request(
            symbol, side, volume, sl_price, tp_price, deviation_points, magic, comment
        )
        if request is None:
            return OrderExecutionResult(False, "ORDER_CHECK", symbol, -4, str(build_err))
        chk = mt5.order_check(request)
        if chk is None:
            return OrderExecutionResult(False, "ORDER_CHECK", symbol, -5, f"MT5_ORDER_CHECK_NONE:{mt5.last_error()}")
        ret = int(getattr(chk, "retcode", -1))
        ok = ret == 0 or ret == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        return OrderExecutionResult(
            success=ok,
            action="ORDER_CHECK",
            symbol=symbol,
            retcode=ret,
            comment=str(getattr(chk, "comment", "")),
            price=float(request["price"]),
            volume=float(request["volume"]),
        )

    def send_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_OP_A",
    ) -> OrderExecutionResult:
        guard_err = self._validate_trading_account_safety("ORDER_OPEN", symbol)
        if guard_err is not None:
            return guard_err
        request, build_err = self._build_market_order_request(
            symbol, side, volume, sl_price, tp_price, deviation_points, magic, comment
        )
        if request is None:
            return OrderExecutionResult(False, "ORDER_OPEN", symbol, -4, str(build_err))

        res = mt5.order_send(request)
        if res is None:
            return OrderExecutionResult(False, "ORDER_OPEN", symbol, -5, f"MT5_ORDER_SEND_NONE:{mt5.last_error()}")

        ret = int(res.retcode)
        done_codes = {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        if ret not in done_codes:
            logger.error("mt5_order_send_failed", symbol=symbol, retcode=ret, comment=res.comment)
            return OrderExecutionResult(False, "ORDER_OPEN", symbol, ret, str(res.comment))

        # Resolve resulting position ticket from deal/order
        pos_ticket = 0
        for p in self.get_open_positions(magic=request["magic"]):
            if p.symbol == symbol:
                pos_ticket = p.ticket
                break
        if pos_ticket == 0:
            pos_ticket = int(getattr(res, "order", 0))

        return OrderExecutionResult(
            success=True,
            action="ORDER_OPEN",
            symbol=symbol,
            retcode=ret,
            comment=str(res.comment),
            ticket=pos_ticket,
            order_id=int(getattr(res, "order", 0)),
            price=float(getattr(res, "price", request["price"]) or request["price"]),
            volume=float(getattr(res, "volume", request["volume"]) or request["volume"]),
        )

    def modify_position_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: float,
        magic: Optional[int] = None,
    ) -> OrderExecutionResult:
        guard_err = self._validate_trading_account_safety("MODIFY_SLTP", symbol)
        if guard_err is not None:
            return guard_err
        spec = self.get_symbol_spec(symbol)
        if spec is None:
            return OrderExecutionResult(False, "MODIFY_SLTP", symbol, -4, "SYMBOL_SPEC_UNAVAILABLE", ticket=ticket)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(ticket),
            "sl": round(float(new_sl), spec.digits),
            "tp": round(float(new_tp), spec.digits),
            "magic": settings.mt5_magic_number if magic is None else int(magic),
        }
        res = mt5.order_send(request)
        if res is None:
            return OrderExecutionResult(
                False, "MODIFY_SLTP", symbol, -5, f"MT5_ORDER_SEND_NONE:{mt5.last_error()}", ticket=ticket
            )
        ret = int(res.retcode)
        ok = ret == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        if not ok:
            logger.warning("mt5_modify_sltp_rejected", ticket=ticket, symbol=symbol, retcode=ret, comment=res.comment)
        return OrderExecutionResult(
            success=ok,
            action="MODIFY_SLTP",
            symbol=symbol,
            retcode=ret,
            comment=str(res.comment),
            ticket=ticket,
            order_id=int(getattr(res, "order", 0)),
        )

    def close_position(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        side: int,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_TIME_STOP",
    ) -> OrderExecutionResult:
        guard_err = self._validate_trading_account_safety("CLOSE_POSITION", symbol)
        if guard_err is not None:
            return guard_err
        spec = self.get_symbol_spec(symbol)
        if spec is None:
            return OrderExecutionResult(False, "CLOSE_POSITION", symbol, -4, "SYMBOL_SPEC_UNAVAILABLE", ticket=ticket)

        close_type = mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY
        close_px = spec.bid if side > 0 else spec.ask
        dev = settings.order_deviation_points if deviation_points is None else int(deviation_points)
        mgc = settings.mt5_magic_number if magic is None else int(magic)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(float(volume), 2),
            "type": close_type,
            "position": int(ticket),
            "price": round(float(close_px), spec.digits),
            "deviation": dev,
            "magic": mgc,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.resolve_filling_mode(symbol),
        }
        res = mt5.order_send(request)
        if res is None:
            return OrderExecutionResult(
                False, "CLOSE_POSITION", symbol, -5, f"MT5_ORDER_SEND_NONE:{mt5.last_error()}", ticket=ticket
            )
        ret = int(res.retcode)
        ok = ret in {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        return OrderExecutionResult(
            success=ok,
            action="CLOSE_POSITION",
            symbol=symbol,
            retcode=ret,
            comment=str(res.comment),
            ticket=ticket,
            order_id=int(getattr(res, "order", 0)),
            price=float(getattr(res, "price", close_px) or close_px),
            volume=float(getattr(res, "volume", volume) or volume),
        )

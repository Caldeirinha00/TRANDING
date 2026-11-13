from dataclasses import dataclass
from datetime import datetime, timezone
import json
import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import psycopg
from questdb import Sender, TimestampNanos
import redis
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

VALID_OHLCV_TABLES = {"ohlcv_h4", "ohlcv_d1"}


@dataclass(frozen=True)
class SymbolCostSnapshot:
    """Broker contract & H4 cost snapshot persisted to QuestDB `symbol_costs_snapshot`."""

    symbol: str
    status: str
    margin_001_eur: float
    atr14_h4_pips: float
    sl_pips: float
    risk_1r_eur: float
    spread_points: int
    friction_pct: float
    swap_mode: int
    swap_long_raw: float
    swap_short_raw: float
    swap_4d_long_eur: float
    swap_4d_short_eur: float
    captured_at: datetime


@dataclass(frozen=True)
class FeatureAuditRecord:
    """Live/simulated H4-close stationary feature & decision record persisted to `feature_audit_log`."""

    ts: datetime
    symbol: str
    side: int
    trigger_type: str
    mode: str
    atr14_h4: float
    sl_distance_price: float
    tp_distance_price: float
    cs_spread: float
    ker_d1: float
    atr_regime_ratio: float
    cs_momentum_spread: float
    breakout_extension_atr: float
    bar_conviction_ratio: float
    d1_trend_age_norm: float
    net_carry_to_risk_ratio: float
    liquidity_session_sin: float
    liquidity_session_cos: float
    meta_probability: float
    expected_net_r: float
    approved: bool
    allocated_lots: float
    rejection_reason: str
    ticket: int = 0


@dataclass(frozen=True)
class TradeExecutionRecord:
    """Lifecycle audit log entry persisted to `trade_executions`."""

    ts: datetime
    symbol: str
    action: str  # OPEN, MODIFY_BE, MODIFY_TRAIL, CLOSE_TIME_STOP, CLOSE_EMERGENCY, ORDER_CHECK
    ticket: int
    side: int
    lots: float
    entry_price: float
    sl_price: float
    tp_price: float
    initial_sl_dist: float
    exit_price: float = 0.0
    realized_pnl_eur: float = 0.0
    swap_eur: float = 0.0
    reason: str = ""
    entry_bar_open: Optional[datetime] = None


class StorageClient:
    """Bounded-timeout storage client for QuestDB (ILP + Postgres wire) and DragonflyDB (Redis)."""

    def __init__(
        self,
        questdb_host: Optional[str] = None,
        questdb_rest_port: Optional[int] = None,
        questdb_ilp_port: Optional[int] = None,
        questdb_pg_port: Optional[int] = None,
        dragonfly_host: Optional[str] = None,
        dragonfly_port: Optional[int] = None,
    ) -> None:
        self.questdb_host = questdb_host or settings.questdb_host
        self.questdb_rest_port = questdb_rest_port or settings.questdb_rest_port
        self.questdb_ilp_port = questdb_ilp_port or settings.questdb_ilp_port
        self.questdb_pg_port = questdb_pg_port or settings.questdb_pg_port
        self.dragonfly_host = dragonfly_host or settings.dragonfly_host
        self.dragonfly_port = dragonfly_port or settings.dragonfly_port

    @staticmethod
    def is_tcp_reachable(host: str, port: int, timeout_sec: float = 1.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout_sec):
                return True
        except OSError:
            return False

    def _pg_connect(self, connect_timeout: int = 5) -> psycopg.Connection:
        return psycopg.connect(
            host=self.questdb_host,
            port=self.questdb_pg_port,
            user=settings.questdb_pg_user,
            password=settings.questdb_pg_password,
            dbname=settings.questdb_pg_dbname,
            connect_timeout=connect_timeout,
            autocommit=True,
        )

    def _resolve_ilp_conf(self, timeout_sec: float = 1.5, request_timeout_ms: int = 5000) -> Optional[str]:
        if self.is_tcp_reachable(self.questdb_host, self.questdb_rest_port, timeout_sec=timeout_sec):
            return (
                f"http::addr={self.questdb_host}:{self.questdb_rest_port};"
                f"retry_timeout=2000;request_timeout={request_timeout_ms};"
            )
        if self.is_tcp_reachable(self.questdb_host, self.questdb_ilp_port, timeout_sec=timeout_sec):
            return f"tcp::addr={self.questdb_host}:{self.questdb_ilp_port};"
        return None

    def ping_dragonfly(self, timeout_sec: float = 1.5) -> bool:
        try:
            client = redis.Redis(
                host=self.dragonfly_host,
                port=self.dragonfly_port,
                socket_connect_timeout=timeout_sec,
                socket_timeout=timeout_sec,
            )
            return bool(client.ping())
        except Exception as exc:
            logger.warning("dragonfly_unreachable", host=self.dragonfly_host, port=self.dragonfly_port, error=str(exc))
            return False

    def ensure_ohlcv_tables(self) -> None:
        """Create deduplicated OHLCV tables in QuestDB if they do not exist."""
        with self._pg_connect() as conn:
            for table_name in sorted(VALID_OHLCV_TABLES):
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        ts TIMESTAMP,
                        symbol SYMBOL CAPACITY 64 CACHE,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        tick_volume LONG,
                        spread INT,
                        real_volume LONG
                    ) TIMESTAMP(ts) PARTITION BY YEAR WAL
                    DEDUP UPSERT KEYS(ts, symbol);
                    """
                )

    def write_ohlcv_dataframe(
        self,
        table_name: str,
        symbol: str,
        df: pd.DataFrame,
        chunk_size: int = 5000,
    ) -> Tuple[bool, str]:
        """Persist a closed-bars OHLCV DataFrame (indexed by bar open time) to QuestDB."""
        if table_name not in VALID_OHLCV_TABLES:
            raise ValueError(f"Invalid OHLCV table name: {table_name}")
        if df.empty:
            return True, f"NO_ROWS ({symbol} {table_name})"

        conf_str = self._resolve_ilp_conf(timeout_sec=1.5, request_timeout_ms=30000)
        if conf_str is None:
            msg = f"QuestDB offline/unreachable at {self.questdb_host}:{self.questdb_rest_port}"
            return False, msg

        timestamps_ns = pd.to_datetime(df.index, utc=True).as_unit("ns").astype("int64").to_numpy()
        opens = df["open"].astype(float).to_numpy()
        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        closes = df["close"].astype(float).to_numpy()
        tick_vols = df["tick_volume"].astype("int64").to_numpy()
        spreads = df["spread"].astype("int32").to_numpy()
        real_vols = df["real_volume"].astype("int64").to_numpy()

        total_rows = len(df)
        try:
            with Sender.from_conf(conf_str) as sender:
                for idx in range(total_rows):
                    sender.row(
                        table_name,
                        symbols={"symbol": symbol},
                        columns={
                            "open": float(opens[idx]),
                            "high": float(highs[idx]),
                            "low": float(lows[idx]),
                            "close": float(closes[idx]),
                            "tick_volume": int(tick_vols[idx]),
                            "spread": int(spreads[idx]),
                            "real_volume": int(real_vols[idx]),
                        },
                        at=TimestampNanos(int(timestamps_ns[idx])),
                    )
                    if (idx + 1) % chunk_size == 0:
                        sender.flush()
                sender.flush()
            logger.info("questdb_ohlcv_written", table=table_name, symbol=symbol, rows=total_rows)
            return True, f"Ingested {total_rows} rows into {table_name} for {symbol}"
        except Exception as exc:
            logger.error("questdb_ohlcv_write_failed", table=table_name, symbol=symbol, error=str(exc))
            return False, f"QuestDB OHLCV write error ({symbol} {table_name}): {exc}"

    def load_ohlcv_dataframe(self, table_name: str, symbol: str) -> pd.DataFrame:
        """Load deduplicated OHLCV bars for `symbol` ordered chronologically."""
        if table_name not in VALID_OHLCV_TABLES:
            raise ValueError(f"Invalid OHLCV table name: {table_name}")

        query = f"""
            SELECT ts, open, high, low, close, tick_volume, spread, real_volume
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY ts ASC;
        """
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (symbol,))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume", "spread", "real_volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="time"),
            )

        df = pd.DataFrame(
            rows,
            columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"],
        )
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset=["time"], keep="last").set_index("time").sort_index(ascending=True)
        return df

    def count_ohlcv_rows(self, table_name: str) -> Dict[str, Tuple[int, str, str]]:
        """Return {symbol: (row_count, min_ts, max_ts)} from QuestDB."""
        if table_name not in VALID_OHLCV_TABLES:
            raise ValueError(f"Invalid OHLCV table name: {table_name}")

        query = f"""
            SELECT symbol, count(*), min(ts), max(ts)
            FROM {table_name}
            GROUP BY symbol
            ORDER BY symbol ASC;
        """
        stats: Dict[str, Tuple[int, str, str]] = {}
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                for sym, cnt, min_ts, max_ts in cur.fetchall():
                    stats[str(sym)] = (int(cnt), str(min_ts), str(max_ts))
        return stats

    def write_symbol_cost_snapshots(
        self,
        snapshots: Sequence[SymbolCostSnapshot],
        timeout_sec: float = 1.5,
    ) -> Tuple[bool, str]:
        """Persist broker profiler snapshots to QuestDB `symbol_costs_snapshot`."""
        if not snapshots:
            return True, "NO_ROWS"

        conf_str = self._resolve_ilp_conf(timeout_sec=timeout_sec, request_timeout_ms=3000)
        if conf_str is None:
            msg = (
                f"QuestDB offline/unreachable at {self.questdb_host}:"
                f"{self.questdb_rest_port}/{self.questdb_ilp_port}"
            )
            logger.warning("questdb_unreachable", message=msg)
            return False, msg

        try:
            with Sender.from_conf(conf_str) as sender:
                for snap in snapshots:
                    ts = snap.captured_at.astimezone(timezone.utc)
                    ts_ns = TimestampNanos(int(ts.timestamp() * 1_000_000_000))
                    sender.row(
                        "symbol_costs_snapshot",
                        symbols={"symbol": snap.symbol, "status": snap.status},
                        columns={
                            "margin_001_eur": float(snap.margin_001_eur),
                            "atr14_h4_pips": float(snap.atr14_h4_pips),
                            "sl_pips": float(snap.sl_pips),
                            "risk_1r_eur": float(snap.risk_1r_eur),
                            "spread_points": int(snap.spread_points),
                            "friction_pct": float(snap.friction_pct),
                            "swap_mode": int(snap.swap_mode),
                            "swap_long_raw": float(snap.swap_long_raw),
                            "swap_short_raw": float(snap.swap_short_raw),
                            "swap_4d_long_eur": float(snap.swap_4d_long_eur),
                            "swap_4d_short_eur": float(snap.swap_4d_short_eur),
                        },
                        at=ts_ns,
                    )
                sender.flush()
            logger.info("questdb_symbol_snapshots_written", count=len(snapshots), conf=conf_str.split("::")[0])
            return True, f"Wrote {len(snapshots)} snapshots to QuestDB ({conf_str.split('::')[0].upper()})"
        except Exception as exc:
            logger.error("questdb_write_failed", error=str(exc))
            return False, f"QuestDB write error: {exc}"

    def _redis_client(self, timeout_sec: float = 1.5) -> redis.Redis:
        return redis.Redis(
            host=self.dragonfly_host,
            port=self.dragonfly_port,
            socket_connect_timeout=timeout_sec,
            socket_timeout=timeout_sec,
            decode_responses=True,
        )

    def set_runtime_state(self, key: str, payload: Dict[str, Any], timeout_sec: float = 1.5) -> bool:
        """Persist JSON operational state (e.g. HWM, day start equity, open ticket metadata) in DragonflyDB."""
        try:
            self._redis_client(timeout_sec=timeout_sec).set(key, json.dumps(payload))
            return True
        except Exception as exc:
            logger.warning("dragonfly_set_failed", key=key, error=str(exc))
            return False

    def get_runtime_state(self, key: str, timeout_sec: float = 1.5) -> Optional[Dict[str, Any]]:
        try:
            raw = self._redis_client(timeout_sec=timeout_sec).get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("dragonfly_get_failed", key=key, error=str(exc))
            return None

    def ensure_feature_audit_table(self) -> None:
        """Create deduplicated QuestDB `feature_audit_log` table for live-vs-backtest parity checks."""
        with self._pg_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_audit_log (
                    ts TIMESTAMP,
                    symbol SYMBOL CAPACITY 64 CACHE,
                    side INT,
                    trigger_type SYMBOL CAPACITY 64 CACHE,
                    mode SYMBOL CAPACITY 32 CACHE,
                    atr14_h4 DOUBLE,
                    sl_distance_price DOUBLE,
                    tp_distance_price DOUBLE,
                    cs_spread DOUBLE,
                    ker_d1 DOUBLE,
                    atr_regime_ratio DOUBLE,
                    cs_momentum_spread DOUBLE,
                    breakout_extension_atr DOUBLE,
                    bar_conviction_ratio DOUBLE,
                    d1_trend_age_norm DOUBLE,
                    net_carry_to_risk_ratio DOUBLE,
                    liquidity_session_sin DOUBLE,
                    liquidity_session_cos DOUBLE,
                    meta_probability DOUBLE,
                    expected_net_r DOUBLE,
                    approved BOOLEAN,
                    allocated_lots DOUBLE,
                    rejection_reason STRING,
                    ticket LONG
                ) TIMESTAMP(ts) PARTITION BY YEAR WAL
                DEDUP UPSERT KEYS(ts, symbol);
                """
            )

    def write_feature_audit_records(
        self,
        records: Sequence[FeatureAuditRecord],
        timeout_sec: float = 1.5,
    ) -> Tuple[bool, str]:
        """Write closed-bar Layer 1/2 feature vectors and decisions to QuestDB `feature_audit_log`."""
        if not records:
            return True, "NO_ROWS"
        conf_str = self._resolve_ilp_conf(timeout_sec=timeout_sec, request_timeout_ms=5000)
        if conf_str is None:
            return False, f"QuestDB offline/unreachable at {self.questdb_host}:{self.questdb_rest_port}"

        try:
            self.ensure_feature_audit_table()
            with Sender.from_conf(conf_str) as sender:
                for rec in records:
                    ts = pd.Timestamp(rec.ts)
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                    sender.row(
                        "feature_audit_log",
                        symbols={
                            "symbol": rec.symbol,
                            "trigger_type": rec.trigger_type,
                            "mode": rec.mode,
                        },
                        columns={
                            "side": int(rec.side),
                            "atr14_h4": float(rec.atr14_h4),
                            "sl_distance_price": float(rec.sl_distance_price),
                            "tp_distance_price": float(rec.tp_distance_price),
                            "cs_spread": float(rec.cs_spread),
                            "ker_d1": float(rec.ker_d1),
                            "atr_regime_ratio": float(rec.atr_regime_ratio),
                            "cs_momentum_spread": float(rec.cs_momentum_spread),
                            "breakout_extension_atr": float(rec.breakout_extension_atr),
                            "bar_conviction_ratio": float(rec.bar_conviction_ratio),
                            "d1_trend_age_norm": float(rec.d1_trend_age_norm),
                            "net_carry_to_risk_ratio": float(rec.net_carry_to_risk_ratio),
                            "liquidity_session_sin": float(rec.liquidity_session_sin),
                            "liquidity_session_cos": float(rec.liquidity_session_cos),
                            "meta_probability": float(rec.meta_probability),
                            "expected_net_r": float(rec.expected_net_r),
                            "approved": bool(rec.approved),
                            "allocated_lots": float(rec.allocated_lots),
                            "rejection_reason": str(rec.rejection_reason),
                            "ticket": int(rec.ticket),
                        },
                        at=TimestampNanos(int(ts.value)),
                    )
                sender.flush()
            return True, f"Wrote {len(records)} rows to feature_audit_log"
        except Exception as exc:
            logger.error("questdb_feature_audit_write_failed", error=str(exc))
            return False, f"QuestDB feature_audit_log write error: {exc}"

    def load_feature_audit_records(
        self,
        symbol: Optional[str] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Load audited live feature vectors from QuestDB `feature_audit_log` chronologically."""
        self.ensure_feature_audit_table()
        clauses: List[str] = []
        params: List[Any] = []
        if symbol is not None:
            clauses.append("symbol = %s")
            params.append(symbol)
        if start_ts is not None:
            clauses.append("ts >= %s")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts <= %s")
            params.append(end_ts)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT
                ts, symbol, side, trigger_type, mode,
                atr14_h4, sl_distance_price, tp_distance_price, cs_spread,
                ker_d1, atr_regime_ratio, cs_momentum_spread, breakout_extension_atr,
                bar_conviction_ratio, d1_trend_age_norm, net_carry_to_risk_ratio,
                liquidity_session_sin, liquidity_session_cos,
                meta_probability, expected_net_r, approved, allocated_lots,
                rejection_reason, ticket
            FROM feature_audit_log
            {where_sql}
            ORDER BY ts ASC, symbol ASC
            LIMIT {int(limit)};
        """
        with self._pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

        cols = [
            "ts", "symbol", "side", "trigger_type", "mode",
            "atr14_h4", "sl_distance_price", "tp_distance_price", "cs_spread",
            "ker_d1", "atr_regime_ratio", "cs_momentum_spread", "breakout_extension_atr",
            "bar_conviction_ratio", "d1_trend_age_norm", "net_carry_to_risk_ratio",
            "liquidity_session_sin", "liquidity_session_cos",
            "meta_probability", "expected_net_r", "approved", "allocated_lots",
            "rejection_reason", "ticket",
        ]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows, columns=cols)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def ensure_trade_executions_table(self) -> None:
        """Create QuestDB `trade_executions` table tracking order open/BE/trail/close actions."""
        with self._pg_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_executions (
                    ts TIMESTAMP,
                    symbol SYMBOL CAPACITY 64 CACHE,
                    action SYMBOL CAPACITY 32 CACHE,
                    ticket LONG,
                    side INT,
                    lots DOUBLE,
                    entry_price DOUBLE,
                    sl_price DOUBLE,
                    tp_price DOUBLE,
                    initial_sl_dist DOUBLE,
                    exit_price DOUBLE,
                    realized_pnl_eur DOUBLE,
                    swap_eur DOUBLE,
                    reason STRING,
                    entry_bar_open TIMESTAMP
                ) TIMESTAMP(ts) PARTITION BY YEAR WAL;
                """
            )

    def write_trade_execution_record(
        self,
        rec: TradeExecutionRecord,
        timeout_sec: float = 1.5,
    ) -> Tuple[bool, str]:
        conf_str = self._resolve_ilp_conf(timeout_sec=timeout_sec, request_timeout_ms=5000)
        if conf_str is None:
            return False, f"QuestDB offline/unreachable at {self.questdb_host}:{self.questdb_rest_port}"
        try:
            self.ensure_trade_executions_table()
            ts = pd.Timestamp(rec.ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            cols: Dict[str, Any] = {
                "ticket": int(rec.ticket),
                "side": int(rec.side),
                "lots": float(rec.lots),
                "entry_price": float(rec.entry_price),
                "sl_price": float(rec.sl_price),
                "tp_price": float(rec.tp_price),
                "initial_sl_dist": float(rec.initial_sl_dist),
                "exit_price": float(rec.exit_price),
                "realized_pnl_eur": float(rec.realized_pnl_eur),
                "swap_eur": float(rec.swap_eur),
                "reason": str(rec.reason),
            }
            if rec.entry_bar_open is not None:
                eb = pd.Timestamp(rec.entry_bar_open)
                eb = eb.tz_localize("UTC") if eb.tzinfo is None else eb.tz_convert("UTC")
                cols["entry_bar_open"] = eb.to_pydatetime()

            with Sender.from_conf(conf_str) as sender:
                sender.row(
                    "trade_executions",
                    symbols={"symbol": rec.symbol, "action": rec.action},
                    columns=cols,
                    at=TimestampNanos(int(ts.value)),
                )
                sender.flush()
            return True, f"Recorded {rec.action} for {rec.symbol} ticket={rec.ticket}"
        except Exception as exc:
            logger.error("questdb_trade_execution_write_failed", error=str(exc))
            return False, f"QuestDB trade_executions write error: {exc}"

    def load_open_trade_metadata(self, tickets: Sequence[int]) -> Dict[int, Tuple[float, Optional[datetime]]]:
        """Recover `(initial_sl_dist, entry_bar_open)` for open tickets from QuestDB `trade_executions`."""
        if not tickets:
            return {}
        try:
            self.ensure_trade_executions_table()
            placeholders = ",".join(["%s"] * len(tickets))
            query = f"""
                SELECT ticket, initial_sl_dist, entry_bar_open
                FROM trade_executions
                WHERE action = 'OPEN' AND ticket IN ({placeholders})
                ORDER BY ts DESC;
            """
            out: Dict[int, Tuple[float, Optional[datetime]]] = {}
            with self._pg_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, tuple(int(t) for t in tickets))
                    for ticket, sl_dist, entry_bar_open in cur.fetchall():
                        tk = int(ticket)
                        if tk not in out:
                            eb_dt = pd.to_datetime(entry_bar_open, utc=True).to_pydatetime() if entry_bar_open else None
                            out[tk] = (float(sl_dist), eb_dt)
            return out
        except Exception as exc:
            logger.warning("questdb_load_open_trade_metadata_failed", error=str(exc))
            return {}

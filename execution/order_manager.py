"""
H4 Deterministic Order Manager & Position Lifecycle Engine (Fase 3).

Executes on every broker H4 bar close (T + 2s server time) with 100% logic
parity against `PrimaryTrendStrategy` (`H2_BB_TREND_DIP_8P`) and
`strategy.triple_barrier.step_barrier_state_bar`:
  - Option A Pass-Through mode (`use_meta_labeling=False`): fixed `0.01` lot
    allocation ranked cross-sectionally by `cs_spread`, enforcing all Layer 0/1/3
    safety & currency-heat vetoes.
  - Existing Position Lifecycle:
      * Break-Even lock (+0.10R) once MFE >= +1.10R
      * Chandelier Trailing (2.0 * ATR14_H4) once MFE >= +1.50R
      * Vertical Time Stop exit at 36 closed H4 trading bars
  - Continuous QuestDB Audit Logging (`feature_audit_log`, `trade_executions`)
    plus DragonflyDB trade metadata backup (`pos_meta:{ticket}`).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from core.clock import H4ServerClock, ensure_utc
from core.mt5_connector import (
    BrokerOpenPosition,
    OrderExecutionResult,
    SymbolContractSpec,
)
from core.storage import FeatureAuditRecord, StorageClient, TradeExecutionRecord
from data.cost_model import CostModel
from execution.mt5_adapter import BrokerExecutionAdapter, build_fallback_contract_spec
from execution.telegram_notifier import TelegramNotifier
from risk.currency_guard import CurrencyAndHeatGuard, OpenPositionRiskState
from risk.portfolio_sizer import CandidateSignal, PortfolioRiskSizer, SizingDecision
from strategy.features import FEATURE_COLUMNS, compute_atr
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy
from strategy.triple_barrier import build_symbol_feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class H4CycleResult:
    """Structured summary of one H4 bar-close evaluation and execution cycle."""

    cycle_time_server: datetime
    latest_closed_bar_open: datetime
    latest_closed_d1_bar_open: datetime
    mode: str
    equity_eur: float
    open_positions_count_start: int
    open_positions_count_end: int
    positions_closed_time_stop: List[int] = field(default_factory=list)
    positions_sl_updated: List[int] = field(default_factory=list)
    signals_detected: int = 0
    orders_approved: int = 0
    orders_executed: List[OrderExecutionResult] = field(default_factory=list)
    audit_records: List[FeatureAuditRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class H4OrderManager:
    """
    Production & simulation-compatible H4 bar-close execution coordinator.
    """

    def __init__(
        self,
        adapter: BrokerExecutionAdapter,
        storage: Optional[StorageClient] = None,
        clock: Optional[H4ServerClock] = None,
        strategy: Optional[PrimaryTrendStrategy] = None,
        cost_model: Optional[CostModel] = None,
        currency_guard: Optional[CurrencyAndHeatGuard] = None,
        notifier: Optional[TelegramNotifier] = None,
        symbols: Optional[List[str]] = None,
        use_meta_labeling: Optional[bool] = None,
        fixed_lots: Optional[float] = None,
        meta_model: Optional[Any] = None,
        max_live_spread_multiplier: Optional[float] = None,
        min_live_spread_cap_points: int = 45,
        h4_history_bars: int = 1200,
        d1_history_bars: int = 800,
        sync_to_questdb: bool = True,
        enforce_bar_alignment_wait: bool = True,
    ) -> None:
        self.adapter = adapter
        self.storage = storage
        self.clock = clock or H4ServerClock(offset_seconds=settings.h4_close_offset_seconds)
        self.strategy = strategy or PrimaryTrendStrategy(PrimaryStrategyConfig())
        self.cost_model = cost_model or CostModel()
        self.currency_guard = currency_guard or CurrencyAndHeatGuard()
        self.notifier = notifier

        self.symbols = list(symbols or settings.approved_symbols)
        self.use_meta_labeling = (
            settings.use_meta_labeling if use_meta_labeling is None else bool(use_meta_labeling)
        )
        self.fixed_lots = (
            settings.fixed_pass_through_lots if fixed_lots is None else float(fixed_lots)
        )
        self.meta_model = meta_model
        self.max_live_spread_multiplier = (
            settings.max_live_spread_multiplier
            if max_live_spread_multiplier is None
            else float(max_live_spread_multiplier)
        )
        self.min_live_spread_cap_points = int(min_live_spread_cap_points)
        self.h4_history_bars = max(400, int(h4_history_bars))
        self.d1_history_bars = max(250, int(d1_history_bars))
        self.sync_to_questdb = bool(sync_to_questdb)
        self.enforce_bar_alignment_wait = bool(enforce_bar_alignment_wait)

        self.portfolio_sizer = PortfolioRiskSizer(
            cost_model=self.cost_model,
            guard=self.currency_guard,
            use_meta_labeling=self.use_meta_labeling,
            strategy_cfg=self.strategy.cfg,
        )
        self._in_memory_pos_meta: Dict[int, Tuple[float, datetime]] = {}

    @property
    def mode_label(self) -> str:
        return "META_LABELING" if self.use_meta_labeling else "PASS_THROUGH_OP_A"

    @staticmethod
    def _merge_and_trim_ohlcv(
        df_db: pd.DataFrame,
        df_live: pd.DataFrame,
        cutoff_inclusive: datetime,
    ) -> pd.DataFrame:
        frames = [f for f in (df_db, df_live) if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume", "spread", "real_volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="time"),
            )
        merged = pd.concat(frames, axis=0)
        if merged.index.tz is None:
            merged.index = merged.index.tz_localize("UTC")
        else:
            merged.index = merged.index.tz_convert("UTC")
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        cutoff_ts = pd.Timestamp(ensure_utc(cutoff_inclusive))
        return merged[merged.index <= cutoff_ts]

    def _fetch_and_sync_symbol_bars(
        self,
        symbol: str,
        query_srv_now: datetime,
        target_closed_h4: datetime,
        target_closed_d1: datetime,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_mt5_h4 = self.adapter.fetch_closed_bars(
            symbol=symbol,
            timeframe_str="H4",
            count=self.h4_history_bars,
            server_now=query_srv_now,
        )
        if (
            self.enforce_bar_alignment_wait
            and H4ServerClock.is_trading_h4_bar_open(target_closed_h4)
            and (df_mt5_h4.empty or df_mt5_h4.index[-1].to_pydatetime() < target_closed_h4)
        ):
            for _ in range(6):
                time.sleep(0.5)
                df_mt5_h4 = self.adapter.fetch_closed_bars(
                    symbol=symbol,
                    timeframe_str="H4",
                    count=self.h4_history_bars,
                    server_now=query_srv_now,
                )
                if not df_mt5_h4.empty and df_mt5_h4.index[-1].to_pydatetime() >= target_closed_h4:
                    break

        df_mt5_d1 = self.adapter.fetch_closed_bars(
            symbol=symbol,
            timeframe_str="D1",
            count=self.d1_history_bars,
            server_now=query_srv_now,
        )

        df_db_h4 = pd.DataFrame()
        df_db_d1 = pd.DataFrame()
        if self.storage is not None:
            try:
                df_db_h4 = self.storage.load_ohlcv_dataframe("ohlcv_h4", symbol)
            except Exception as exc:
                logger.warning("Failed loading QuestDB H4 history for %s: %s", symbol, exc)
            try:
                df_db_d1 = self.storage.load_ohlcv_dataframe("ohlcv_d1", symbol)
            except Exception as exc:
                logger.warning("Failed loading QuestDB D1 history for %s: %s", symbol, exc)

            if self.sync_to_questdb:
                for tbl, df_live, df_existing in (
                    ("ohlcv_h4", df_mt5_h4, df_db_h4),
                    ("ohlcv_d1", df_mt5_d1, df_db_d1),
                ):
                    if df_live.empty:
                        continue
                    to_write = (
                        df_live
                        if df_existing.empty
                        else df_live[df_live.index >= df_existing.index[-1]]
                    )
                    if not to_write.empty:
                        ok, msg = self.storage.write_ohlcv_dataframe(tbl, symbol, to_write)
                        if not ok:
                            logger.warning("QuestDB sync warning (%s %s): %s", symbol, tbl, msg)

        merged_h4 = self._merge_and_trim_ohlcv(df_db_h4, df_mt5_h4, target_closed_h4)
        merged_d1 = self._merge_and_trim_ohlcv(df_db_d1, df_mt5_d1, target_closed_d1)
        return merged_h4, merged_d1

    def _save_position_metadata(
        self,
        ticket: int,
        symbol: str,
        initial_sl_dist: float,
        entry_bar_open: datetime,
    ) -> None:
        entry_utc = ensure_utc(entry_bar_open)
        self._in_memory_pos_meta[ticket] = (float(initial_sl_dist), entry_utc)
        if self.storage is not None:
            self.storage.set_runtime_state(
                f"pos_meta:{ticket}",
                {
                    "symbol": symbol,
                    "initial_sl_dist": float(initial_sl_dist),
                    "entry_bar_open": entry_utc.isoformat(),
                },
            )

    def _load_positions_metadata(
        self,
        positions: List[BrokerOpenPosition],
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[int, Tuple[float, datetime]]:
        tickets = [p.ticket for p in positions]
        meta: Dict[int, Tuple[float, datetime]] = {}

        if self.storage is not None and tickets:
            try:
                db_meta = self.storage.load_open_trade_metadata(tickets)
                for tk, (sl_d, entry_b) in db_meta.items():
                    if sl_d > 0.0 and entry_b is not None:
                        meta[tk] = (float(sl_d), ensure_utc(entry_b))
            except Exception as exc:
                logger.warning("Could not load QuestDB trade metadata: %s", exc)

        for pos in positions:
            if pos.ticket not in meta and pos.ticket in self._in_memory_pos_meta:
                meta[pos.ticket] = self._in_memory_pos_meta[pos.ticket]

            if pos.ticket not in meta and self.storage is not None:
                kv = self.storage.get_runtime_state(f"pos_meta:{pos.ticket}")
                if isinstance(kv, dict) and kv.get("initial_sl_dist") and kv.get("entry_bar_open"):
                    meta[pos.ticket] = (
                        float(kv["initial_sl_dist"]),
                        ensure_utc(datetime.fromisoformat(str(kv["entry_bar_open"]))),
                    )

            if pos.ticket not in meta:
                sl_dist = abs(pos.price_open - pos.sl) if pos.sl > 0.0 else 0.0
                if sl_dist <= 0.0 and pos.symbol in h4_bars_by_symbol and not h4_bars_by_symbol[pos.symbol].empty:
                    atr = compute_atr(
                        h4_bars_by_symbol[pos.symbol], period=self.strategy.cfg.atr_h4_period
                    ).iloc[-1]
                    if pd.notna(atr) and atr > 0.0:
                        sl_dist = float(atr) * self.strategy.cfg.sl_atr_mult
                entry_bar = H4ServerClock.current_h4_bar_open(pos.time_open_server)
                meta[pos.ticket] = (float(sl_dist), entry_bar)
                if sl_dist > 0.0:
                    self._in_memory_pos_meta[pos.ticket] = meta[pos.ticket]

        return meta

    def _compute_bars_held(
        self,
        entry_bar_open: datetime,
        target_closed_h4: datetime,
        df_h4: Optional[pd.DataFrame],
    ) -> int:
        entry_ts = ensure_utc(entry_bar_open)
        close_ts = ensure_utc(target_closed_h4)
        if close_ts < entry_ts:
            return 0
        if df_h4 is not None and not df_h4.empty:
            idx = df_h4.index
            first_bar = idx[0].to_pydatetime()
            if first_bar <= entry_ts:
                return int(((idx >= pd.Timestamp(entry_ts)) & (idx <= pd.Timestamp(close_ts))).sum())
        return H4ServerClock.count_trading_h4_bars_between(
            entry_ts, close_ts, inclusive_entry_bar=True
        )

    def _compute_target_sl(
        self,
        pos: BrokerOpenPosition,
        initial_sl_dist: float,
        entry_bar_open: datetime,
        target_closed_h4: datetime,
        df_h4: pd.DataFrame,
        spec: SymbolContractSpec,
    ) -> Tuple[float, str]:
        """
        Deterministically replays closed H4 bars since `entry_bar_open` using the
        exact MFE, Break-Even (+0.10R at +1.10R), and Chandelier Trail (+1.50R,
        2.0 * ATR14) math from `step_barrier_state_bar`.
        """
        cfg = self.strategy.cfg
        if df_h4.empty or initial_sl_dist <= 0.0:
            return pos.sl, "NONE"

        atr_series = compute_atr(df_h4, period=cfg.atr_h4_period).fillna(0.0)
        bars_since = df_h4[
            (df_h4.index >= pd.Timestamp(ensure_utc(entry_bar_open)))
            & (df_h4.index <= pd.Timestamp(ensure_utc(target_closed_h4)))
        ]
        if bars_since.empty:
            return pos.sl, "NONE"

        desired_sl = float(pos.sl)
        favorable_peak = float(pos.price_open)
        stage = "NONE"

        for ts, row in bars_since.iterrows():
            curr_atr = float(atr_series.loc[ts])
            eff_atr = (
                curr_atr
                if curr_atr > 0.0
                else (initial_sl_dist / max(cfg.sl_atr_mult, 1e-9))
            )

            if pos.side > 0:
                favorable_peak = max(favorable_peak, float(row["high"]))
                excursion_r = (favorable_peak - pos.price_open) / max(initial_sl_dist, 1e-9)
                if excursion_r >= cfg.breakeven_trigger_r:
                    be_lock = pos.price_open + (cfg.breakeven_lock_r * initial_sl_dist)
                    if be_lock > desired_sl:
                        desired_sl = be_lock
                        stage = "MODIFY_SL_BE"
                if cfg.trailing_activation_r > 0 and excursion_r >= cfg.trailing_activation_r and eff_atr > 0.0:
                    trail_sl = favorable_peak - (cfg.trailing_atr_mult * eff_atr)
                    if trail_sl > desired_sl:
                        desired_sl = trail_sl
                        stage = "MODIFY_SL_TRAIL"
            else:
                bar_sp_pts = max(0, int(row.get("spread", spec.spread_points)))
                ask_low = float(row["low"]) + (bar_sp_pts * spec.point)
                favorable_peak = min(favorable_peak, ask_low)
                excursion_r = (pos.price_open - favorable_peak) / max(initial_sl_dist, 1e-9)
                if excursion_r >= cfg.breakeven_trigger_r:
                    be_lock = pos.price_open - (cfg.breakeven_lock_r * initial_sl_dist)
                    if desired_sl <= 0.0 or be_lock < desired_sl:
                        desired_sl = be_lock
                        stage = "MODIFY_SL_BE"
                if cfg.trailing_activation_r > 0 and excursion_r >= cfg.trailing_activation_r and eff_atr > 0.0:
                    trail_sl = favorable_peak + (cfg.trailing_atr_mult * eff_atr)
                    if desired_sl <= 0.0 or trail_sl < desired_sl:
                        desired_sl = trail_sl
                        stage = "MODIFY_SL_TRAIL"

        return round(desired_sl, spec.digits), stage

    def _manage_open_positions(
        self,
        open_positions: List[BrokerOpenPosition],
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
        specs_by_symbol: Dict[str, SymbolContractSpec],
        target_closed_h4: datetime,
        result: H4CycleResult,
    ) -> List[BrokerOpenPosition]:
        if not open_positions:
            return []

        pos_meta = self._load_positions_metadata(open_positions, h4_bars_by_symbol)
        remaining: List[BrokerOpenPosition] = []
        cfg = self.strategy.cfg

        for pos in open_positions:
            spec = specs_by_symbol.get(pos.symbol) or self.adapter.get_symbol_spec(pos.symbol)
            if spec is None:
                remaining.append(pos)
                result.errors.append(f"Missing live spec managing ticket={pos.ticket} ({pos.symbol})")
                continue

            initial_sl_dist, entry_bar_open = pos_meta[pos.ticket]
            df_h4 = h4_bars_by_symbol.get(pos.symbol, pd.DataFrame())
            bars_held = self._compute_bars_held(entry_bar_open, target_closed_h4, df_h4)

            if bars_held >= cfg.vertical_barrier_bars_h4:
                close_res = self.adapter.close_position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    volume=pos.volume,
                    side=pos.side,
                    magic=pos.magic,
                    comment="AI_TRADING_TIME_STOP",
                )
                result.orders_executed.append(close_res)
                if close_res.success:
                    result.positions_closed_time_stop.append(pos.ticket)
                    if self.storage is not None:
                        self.storage.write_trade_execution_record(
                            TradeExecutionRecord(
                                ts=datetime.now(timezone.utc),
                                symbol=pos.symbol,
                                action="TIME_STOP_CLOSE",
                                ticket=pos.ticket,
                                side=pos.side,
                                lots=pos.volume,
                                entry_price=pos.price_open,
                                sl_price=pos.sl,
                                tp_price=pos.tp,
                                initial_sl_dist=initial_sl_dist,
                                exit_price=close_res.price,
                                realized_pnl_eur=pos.profit_eur,
                                swap_eur=pos.swap_eur,
                                reason="TIME_STOP_36H4",
                                entry_bar_open=entry_bar_open,
                            )
                        )
                    if self.notifier is not None:
                        self.notifier.notify_position_closed(
                            pos.symbol, pos.ticket, "TIME_STOP_36H4", pos.profit_eur
                        )
                    continue
                result.errors.append(
                    f"TIME_STOP close failed ticket={pos.ticket} ({pos.symbol}): {close_res.comment}"
                )
                remaining.append(pos)
                continue

            target_sl, stage = self._compute_target_sl(
                pos=pos,
                initial_sl_dist=initial_sl_dist,
                entry_bar_open=entry_bar_open,
                target_closed_h4=target_closed_h4,
                df_h4=df_h4,
                spec=spec,
            )

            tolerance = spec.point * 0.5
            tightened = (
                (pos.side > 0 and target_sl > pos.sl + tolerance and target_sl < (spec.bid - 2.0 * spec.point))
                or (
                    pos.side < 0
                    and (pos.sl <= 0.0 or target_sl < pos.sl - tolerance)
                    and target_sl > (spec.ask + 2.0 * spec.point)
                )
            )
            if tightened and stage != "NONE":
                mod_res = self.adapter.modify_position_sl_tp(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    new_sl=target_sl,
                    new_tp=pos.tp,
                    magic=pos.magic,
                )
                result.orders_executed.append(mod_res)
                if mod_res.success:
                    result.positions_sl_updated.append(pos.ticket)
                    pos = BrokerOpenPosition(
                        ticket=pos.ticket,
                        symbol=pos.symbol,
                        side=pos.side,
                        volume=pos.volume,
                        price_open=pos.price_open,
                        sl=target_sl,
                        tp=pos.tp,
                        price_current=pos.price_current,
                        swap_eur=pos.swap_eur,
                        profit_eur=pos.profit_eur,
                        time_open_server=pos.time_open_server,
                        magic=pos.magic,
                        comment=pos.comment,
                    )
                    if self.storage is not None:
                        self.storage.write_trade_execution_record(
                            TradeExecutionRecord(
                                ts=datetime.now(timezone.utc),
                                symbol=pos.symbol,
                                action=stage,
                                ticket=pos.ticket,
                                side=pos.side,
                                lots=pos.volume,
                                entry_price=pos.price_open,
                                sl_price=target_sl,
                                tp_price=pos.tp,
                                initial_sl_dist=initial_sl_dist,
                                reason=stage,
                                entry_bar_open=entry_bar_open,
                            )
                        )
                    if self.notifier is not None:
                        ref_px = spec.bid if pos.side > 0 else spec.ask
                        self.notifier.notify_position_updated(
                            pos.symbol, pos.ticket, stage, target_sl, ref_px
                        )
                else:
                    result.errors.append(
                        f"SL modify failed ticket={pos.ticket} ({pos.symbol}): {mod_res.comment}"
                    )

            remaining.append(pos)

        return remaining

    def _max_allowed_live_spread_points(self, symbol: str) -> int:
        bench = settings.benchmark_spread_points.get(symbol, 15)
        return max(
            int(round(bench * self.max_live_spread_multiplier)),
            self.min_live_spread_cap_points,
        )

    def _build_open_risk_states(
        self,
        positions: List[BrokerOpenPosition],
        specs_by_symbol: Dict[str, SymbolContractSpec],
    ) -> Tuple[List[OpenPositionRiskState], float]:
        risk_states: List[OpenPositionRiskState] = []
        used_margin = 0.0
        for p in positions:
            spec = specs_by_symbol.get(p.symbol) or self.adapter.get_symbol_spec(p.symbol)
            open_risk_eur = 10.0
            if spec is not None:
                used_margin += spec.margin_initial_001_eur * (p.volume / 0.01)
                if p.sl > 0.0:
                    sl_dist = abs(p.price_open - p.sl)
                    open_risk_eur = self.cost_model.calculate_sl_risk_eur(
                        spec=spec, sl_distance_price=sl_dist, lots=p.volume
                    )
            risk_states.append(
                OpenPositionRiskState(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    side=p.side,
                    lots=p.volume,
                    entry_price=p.price_open,
                    current_sl=p.sl,
                    open_risk_eur=open_risk_eur,
                )
            )
        return risk_states, used_margin

    def run_h4_cycle(
        self,
        server_now: Optional[datetime] = None,
        force_closed_bar_open: Optional[datetime] = None,
    ) -> H4CycleResult:
        """
        Executes a single full H4 bar-close management, inference, sizing, execution,
        and QuestDB audit cycle. Safe in both live PepperstoneUK-Demo and tests.
        """
        srv_now = ensure_utc(server_now) if server_now is not None else self.clock.estimate_server_now(self.adapter)
        target_closed_h4 = (
            ensure_utc(force_closed_bar_open)
            if force_closed_bar_open is not None
            else self.clock.latest_closed_h4_bar_open(srv_now)
        )
        # Ensure D1 cutoff respects T_d1 + 24h <= T_h4 + 4h even during forced historical cycles.
        effective_h4_close = target_closed_h4 + timedelta(hours=4)
        target_closed_d1 = H4ServerClock.latest_closed_d1_bar_open(effective_h4_close)
        query_srv_now = (
            max(srv_now, effective_h4_close) if force_closed_bar_open is not None else srv_now
        )

        result = H4CycleResult(
            cycle_time_server=srv_now,
            latest_closed_bar_open=target_closed_h4,
            latest_closed_d1_bar_open=target_closed_d1,
            mode=self.mode_label,
            equity_eur=0.0,
            open_positions_count_start=0,
            open_positions_count_end=0,
        )

        if self.storage is not None:
            try:
                self.storage.ensure_feature_audit_table()
                self.storage.ensure_trade_executions_table()
            except Exception as exc:
                logger.warning("QuestDB DDL warning: %s", exc)

        account = self.adapter.get_account_snapshot()
        if account is None:
            result.errors.append("Broker account snapshot unavailable (MT5 disconnected or wrong account).")
            return result
        if (not account.is_demo and not settings.allow_live_real_money) or account.balance_eur >= 50000.0:
            result.errors.append(
                f"Account safety block triggered (login={account.login}, balance={account.balance_eur:.2f}, is_demo={account.is_demo})."
            )
            return result

        result.equity_eur = float(account.equity_eur)

        h4_bars_by_symbol: Dict[str, pd.DataFrame] = {}
        d1_bars_by_symbol: Dict[str, pd.DataFrame] = {}
        specs_by_symbol: Dict[str, SymbolContractSpec] = {}

        for sym in self.symbols:
            spec = self.adapter.get_symbol_spec(sym)
            if spec is not None:
                specs_by_symbol[sym] = spec
                if self.storage is not None:
                    self.storage.set_runtime_state(f"parity_spec:{sym}", asdict(spec))

            df_h4, df_d1 = self._fetch_and_sync_symbol_bars(
                symbol=sym,
                query_srv_now=query_srv_now,
                target_closed_h4=target_closed_h4,
                target_closed_d1=target_closed_d1,
            )
            if not df_h4.empty and not df_d1.empty:
                h4_bars_by_symbol[sym] = df_h4
                d1_bars_by_symbol[sym] = df_d1

        open_positions_start = self.adapter.get_open_positions()
        result.open_positions_count_start = len(open_positions_start)

        # Step 1: Manage existing positions (Time Stop / BE / Chandelier Trailing)
        remaining_positions = self._manage_open_positions(
            open_positions=open_positions_start,
            h4_bars_by_symbol=h4_bars_by_symbol,
            specs_by_symbol=specs_by_symbol,
            target_closed_h4=target_closed_h4,
            result=result,
        )

        # Step 2: Generate Basket Layer 1 Signals & Exact Layer 2 Stationary Feature Vectors
        if not h4_bars_by_symbol or not d1_bars_by_symbol:
            result.open_positions_count_end = len(remaining_positions)
            result.errors.append("No synchronized H4/D1 bars available for signal evaluation.")
            return result

        signals_by_symbol, _ = self.strategy.generate_signals_for_basket(
            h4_bars_by_symbol=h4_bars_by_symbol,
            d1_bars_by_symbol=d1_bars_by_symbol,
        )

        target_ts = pd.Timestamp(target_closed_h4)
        open_symbols = {p.symbol for p in remaining_positions}
        open_risk_states, used_margin = self._build_open_risk_states(
            remaining_positions, specs_by_symbol
        )

        features_at_t: Dict[str, pd.Series] = {}
        sig_rows_at_t: Dict[str, pd.Series] = {}
        pre_veto_reasons: Dict[str, str] = {}
        eligible_candidates: List[CandidateSignal] = []

        for sym in self.symbols:
            if sym not in signals_by_symbol or target_ts not in signals_by_symbol[sym].index:
                continue
            df_h4 = h4_bars_by_symbol[sym]
            df_d1 = d1_bars_by_symbol[sym]
            sig_df = signals_by_symbol[sym]
            spec = specs_by_symbol.get(sym) or build_fallback_contract_spec(sym)

            feat_df = build_symbol_feature_matrix(
                df_h4=df_h4,
                df_d1=df_d1,
                sig_df=sig_df,
                spec=spec,
                cost_model=self.cost_model,
                cfg=self.strategy.cfg,
                reference_lots=self.fixed_lots,
            )
            if target_ts not in feat_df.index:
                continue

            sig_row = sig_df.loc[target_ts]
            feat_row = feat_df.loc[target_ts]
            sig_rows_at_t[sym] = sig_row
            features_at_t[sym] = feat_row

            side = int(sig_row["side"])
            if side == 0:
                continue

            result.signals_detected += 1
            live_spec = specs_by_symbol.get(sym)
            if live_spec is None:
                pre_veto_reasons[sym] = "VETO_MISSING_LIVE_SPEC"
                continue
            if sym in open_symbols:
                pre_veto_reasons[sym] = "VETO_SYMBOL_ALREADY_OPEN"
                continue

            max_sp = self._max_allowed_live_spread_points(sym)
            if live_spec.spread_points > max_sp:
                pre_veto_reasons[sym] = (
                    f"VETO_LIVE_ROLLOVER_SPREAD:{live_spec.spread_points}pts>{max_sp}pts"
                )
                continue

            entry_px = live_spec.ask if side > 0 else live_spec.bid
            sl_dist = float(sig_row["sl_distance_price"])
            tp_dist = float(sig_row["tp_distance_price"])
            if entry_px <= 0.0 or sl_dist <= 0.0 or tp_dist <= 0.0:
                pre_veto_reasons[sym] = "VETO_INVALID_BARRIER_GEOMETRY"
                continue

            sl_px = round(entry_px - (side * sl_dist), live_spec.digits)
            tp_px = round(entry_px + (side * tp_dist), live_spec.digits)

            meta_prob = 0.50
            if self.use_meta_labeling and self.meta_model is not None:
                X_one = pd.DataFrame([feat_row[FEATURE_COLUMNS].to_dict()])
                meta_prob = float(self.meta_model.predict_proba(X_one)[0])

            eligible_candidates.append(
                CandidateSignal(
                    symbol=sym,
                    side=side,
                    entry_price=entry_px,
                    sl_price=sl_px,
                    tp_price=tp_px,
                    meta_probability=meta_prob,
                    signal_time_utc=target_closed_h4,
                    ker_d1=float(sig_row["ker_d1"]),
                    priority_score=float(sig_row["cs_spread"]),
                    holding_bars_h4=self.strategy.cfg.vertical_barrier_bars_h4,
                )
            )

        decisions_by_sym: Dict[str, SizingDecision] = {}
        candidates_by_sym = {c.symbol: c for c in eligible_candidates}

        if eligible_candidates:
            ranked_decisions = self.portfolio_sizer.rank_and_allocate_batch(
                candidates=eligible_candidates,
                specs=specs_by_symbol,
                equity_eur=result.equity_eur,
                open_positions=open_risk_states,
            )
            exec_bar_open = max(
                H4ServerClock.current_h4_bar_open(srv_now),
                target_closed_h4 + timedelta(hours=4),
            )
            for dec in ranked_decisions:
                cand = candidates_by_sym[dec.symbol]
                live_spec = specs_by_symbol[dec.symbol]
                if not dec.approved:
                    decisions_by_sym[dec.symbol] = dec
                    continue

                if (used_margin + dec.margin_required_eur) > (result.equity_eur * 0.85):
                    decisions_by_sym[dec.symbol] = SizingDecision(
                        symbol=dec.symbol,
                        side=dec.side,
                        approved=False,
                        allocated_lots=0.0,
                        risk_eur=dec.risk_eur,
                        risk_pct_equity=dec.risk_pct_equity,
                        expected_net_r=dec.expected_net_r,
                        margin_required_eur=0.0,
                        rejection_reason="VETO_ESMA_MARGIN_BUFFER_85PCT",
                    )
                    continue

                sig_row = sig_rows_at_t[dec.symbol]
                sl_dist = float(sig_row["sl_distance_price"])
                tp_dist = float(sig_row["tp_distance_price"])
                # Refresh SL/TP around live quote at order moment
                entry_px = live_spec.ask if cand.side > 0 else live_spec.bid
                sl_px = round(entry_px - (cand.side * sl_dist), live_spec.digits)
                tp_px = round(entry_px + (cand.side * tp_dist), live_spec.digits)

                order_res = self.adapter.send_market_order(
                    symbol=dec.symbol,
                    side=cand.side,
                    volume=dec.allocated_lots,
                    sl_price=sl_px,
                    tp_price=tp_px,
                    comment=f"H2_{str(sig_row['trigger_type'])[:12]}",
                )
                result.orders_executed.append(order_res)

                if order_res.success:
                    result.orders_approved += 1
                    used_margin += dec.margin_required_eur
                    open_symbols.add(dec.symbol)
                    decisions_by_sym[dec.symbol] = dec
                    self._save_position_metadata(
                        ticket=order_res.ticket,
                        symbol=dec.symbol,
                        initial_sl_dist=sl_dist,
                        entry_bar_open=exec_bar_open,
                    )
                    if self.storage is not None:
                        self.storage.write_trade_execution_record(
                            TradeExecutionRecord(
                                ts=datetime.now(timezone.utc),
                                symbol=dec.symbol,
                                action=order_res.action,
                                ticket=order_res.ticket,
                                side=cand.side,
                                lots=dec.allocated_lots,
                                entry_price=order_res.price or entry_px,
                                sl_price=sl_px,
                                tp_price=tp_px,
                                initial_sl_dist=sl_dist,
                                reason=str(sig_row["trigger_type"]),
                                entry_bar_open=exec_bar_open,
                            )
                        )
                    if self.notifier is not None:
                        self.notifier.notify_order_opened(
                            symbol=dec.symbol,
                            side=cand.side,
                            lots=dec.allocated_lots,
                            price=order_res.price or entry_px,
                            sl=sl_px,
                            tp=tp_px,
                            ticket=order_res.ticket,
                            mode=self.mode_label,
                        )
                else:
                    result.errors.append(
                        f"Order open failed ({dec.symbol}): {order_res.comment}"
                    )
                    decisions_by_sym[dec.symbol] = SizingDecision(
                        symbol=dec.symbol,
                        side=dec.side,
                        approved=False,
                        allocated_lots=0.0,
                        risk_eur=dec.risk_eur,
                        risk_pct_equity=dec.risk_pct_equity,
                        expected_net_r=dec.expected_net_r,
                        margin_required_eur=0.0,
                        rejection_reason=f"BROKER_EXECUTION_FAILED:{order_res.comment}",
                    )

        # Step 3: Emit Continuous Feature Audit Records for EVERY symbol at bar T
        for sym in self.symbols:
            if sym not in features_at_t or sym not in sig_rows_at_t:
                continue
            sig_row = sig_rows_at_t[sym]
            feat_row = features_at_t[sym]
            side = int(sig_row["side"])
            cand = candidates_by_sym.get(sym)
            dec = decisions_by_sym.get(sym)

            if side == 0:
                approved = False
                lots = 0.0
                reason = "NO_PRIMARY_SIGNAL"
                exp_r = 0.0
                meta_prob = 0.0
                ticket = 0
            elif sym in pre_veto_reasons:
                approved = False
                lots = 0.0
                reason = pre_veto_reasons[sym]
                exp_r = 0.0
                meta_prob = 0.50 if not self.use_meta_labeling else 0.0
                ticket = 0
            elif dec is not None:
                approved = dec.approved
                lots = dec.allocated_lots if dec.approved else 0.0
                reason = dec.rejection_reason
                exp_r = dec.expected_net_r
                meta_prob = cand.meta_probability if cand is not None else 0.50
                matching_order = next(
                    (
                        o
                        for o in result.orders_executed
                        if o.symbol == sym and o.success and o.action in {"ORDER_OPEN", "DRY_RUN_OPEN", "SIM_OPEN"}
                    ),
                    None,
                )
                ticket = matching_order.ticket if matching_order is not None else 0
            else:
                approved = False
                lots = 0.0
                reason = "UNPROCESSED"
                exp_r = 0.0
                meta_prob = 0.0
                ticket = 0

            result.audit_records.append(
                FeatureAuditRecord(
                    ts=target_closed_h4,
                    symbol=sym,
                    side=side,
                    trigger_type=str(sig_row["trigger_type"]),
                    mode=self.mode_label,
                    atr14_h4=float(sig_row["atr14_h4"]),
                    sl_distance_price=float(sig_row["sl_distance_price"]),
                    tp_distance_price=float(sig_row["tp_distance_price"]),
                    cs_spread=float(sig_row["cs_spread"]),
                    ker_d1=float(feat_row["ker_d1"]),
                    atr_regime_ratio=float(feat_row["atr_regime_ratio"]),
                    cs_momentum_spread=float(feat_row["cs_momentum_spread"]),
                    breakout_extension_atr=float(feat_row["breakout_extension_atr"]),
                    bar_conviction_ratio=float(feat_row["bar_conviction_ratio"]),
                    d1_trend_age_norm=float(feat_row["d1_trend_age_norm"]),
                    net_carry_to_risk_ratio=float(feat_row["net_carry_to_risk_ratio"]),
                    liquidity_session_sin=float(feat_row["liquidity_session_sin"]),
                    liquidity_session_cos=float(feat_row["liquidity_session_cos"]),
                    meta_probability=float(meta_prob),
                    expected_net_r=float(exp_r),
                    approved=bool(approved),
                    allocated_lots=float(lots),
                    rejection_reason=str(reason),
                    ticket=int(ticket),
                )
            )

        if self.storage is not None and result.audit_records:
            ok, msg = self.storage.write_feature_audit_records(result.audit_records)
            if not ok:
                result.errors.append(f"Feature audit QuestDB write failed: {msg}")

        result.open_positions_count_end = len(self.adapter.get_open_positions())
        return result

    async def run_h4_cycle_async(
        self,
        server_now: Optional[datetime] = None,
        force_closed_bar_open: Optional[datetime] = None,
    ) -> H4CycleResult:
        return await asyncio.to_thread(
            self.run_h4_cycle,
            server_now=server_now,
            force_closed_bar_open=force_closed_bar_open,
        )

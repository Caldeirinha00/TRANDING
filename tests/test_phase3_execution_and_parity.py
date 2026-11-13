"""
Phase 3 Verification Suite: Option A Pass-Through, H4 Server Clock,
OrderManager Lifecycle (BE +1.10R / Chandelier Trail +1.50R / 36-bar TimeStop),
Account Safety Guards, and Live-vs-Batch Feature Parity (<= 1e-9).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest.parity_checker import ParityChecker
from config.settings import settings
from core.clock import H4ServerClock
from core.mt5_connector import (
    BrokerAccountSnapshot,
    BrokerOpenPosition,
    MT5Connector,
    OrderExecutionResult,
    SymbolContractSpec,
)
from core.storage import FeatureAuditRecord
from data.cost_model import CostModel
from execution.mt5_adapter import BrokerExecutionAdapter, build_fallback_contract_spec
from execution.order_manager import H4OrderManager
from risk.currency_guard import CurrencyAndHeatGuard
from risk.portfolio_sizer import CandidateSignal, PortfolioRiskSizer
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy


# ---------------------------------------------------------------------------
# Synthetic In-Memory Broker & Market Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_fx_basket(
    symbols: List[str],
    n_h4_bars: int = 1600,
    start_dt: datetime = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Generates deterministic weekday-only H4 bars (00,04,08,12,16,20) and matching
    D1 bars with realistic trends + dips for testing Option A Pass-Through & Parity.
    """
    rng = np.random.default_rng(20261112)
    h4_dates: List[datetime] = []
    cur = start_dt
    while len(h4_dates) < n_h4_bars:
        if cur.weekday() < 5:
            h4_dates.append(cur)
        cur += timedelta(hours=4)
    h4_index = pd.DatetimeIndex(h4_dates, tz="UTC", name="time")

    h4_map: Dict[str, pd.DataFrame] = {}
    d1_map: Dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols):
        base_px = 152.0 if "JPY" in sym else (1.08 + i * 0.05)
        pip = 0.01 if "JPY" in sym else 0.0001
        drift = (0.6 if sym == "EURUSD" else (-0.4 if i % 2 else 0.25)) * pip
        noise = rng.normal(0.0, 4.0 * pip, size=n_h4_bars)
        closes = base_px + np.cumsum(drift + noise)
        opens = np.concatenate([[base_px], closes[:-1]])
        spread_pts = settings.benchmark_spread_points.get(sym, 12)
        bar_range = np.abs(rng.normal(10.0 * pip, 3.0 * pip, size=n_h4_bars)) + 4.0 * pip
        highs = np.maximum(opens, closes) + bar_range * 0.5
        lows = np.minimum(opens, closes) - bar_range * 0.5

        df_h4 = pd.DataFrame(
            {
                "open": np.round(opens, 5),
                "high": np.round(highs, 5),
                "low": np.round(lows, 5),
                "close": np.round(closes, 5),
                "tick_volume": rng.integers(500, 3000, size=n_h4_bars),
                "spread": np.full(n_h4_bars, spread_pts, dtype=int),
                "real_volume": np.zeros(n_h4_bars, dtype=int),
            },
            index=h4_index,
        )
        # Aggregate D1 strictly on weekday server bars
        df_d1 = (
            df_h4.resample("1D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "median",
                    "real_volume": "sum",
                }
            )
            .dropna()
        )
        df_d1["spread"] = df_d1["spread"].round().astype(int)
        h4_map[sym] = df_h4
        d1_map[sym] = df_d1

    return h4_map, d1_map


def _find_signal_bar(
    symbols: List[str],
    h4_map: Dict[str, pd.DataFrame],
    d1_map: Dict[str, pd.DataFrame],
    strategy: PrimaryTrendStrategy,
) -> tuple[str, datetime, int, float, float]:
    sigs, _ = strategy.generate_signals_for_basket(h4_map, d1_map)
    for sym in symbols:
        df_s = sigs[sym]
        # Skip final 60 bars so we have room to step 36 bars forward in lifecycle test
        eligible = df_s.iloc[900:-60]
        nz = eligible[eligible["side"] != 0]
        if not nz.empty:
            ts = nz.index[0].to_pydatetime()
            row = nz.iloc[0]
            return (
                sym,
                ts,
                int(row["side"]),
                float(row["sl_distance_price"]),
                float(row["tp_distance_price"]),
            )
    raise AssertionError("Synthetic basket did not produce any Layer 1 signal in test window.")


class InMemoryBrokerAdapter(BrokerExecutionAdapter):
    """Deterministic BrokerExecutionAdapter for unit & lifecycle testing."""

    def __init__(
        self,
        h4_map: Dict[str, pd.DataFrame],
        d1_map: Dict[str, pd.DataFrame],
        account: BrokerAccountSnapshot,
        specs: Optional[Dict[str, SymbolContractSpec]] = None,
    ) -> None:
        self.h4_map = h4_map
        self.d1_map = d1_map
        self.account = account
        self.specs: Dict[str, SymbolContractSpec] = specs or {
            sym: build_fallback_contract_spec(sym) for sym in h4_map
        }
        self.server_now = h4_map[next(iter(h4_map))].index[-1].to_pydatetime() + timedelta(hours=4, seconds=2)
        self.positions: List[BrokerOpenPosition] = []
        self._ticket_seq = 1000

    def set_server_now(self, srv_now: datetime) -> None:
        self.server_now = srv_now
        # Refresh live bid/ask on specs from latest closed bar close
        closed_cutoff = H4ServerClock.latest_closed_h4_bar_open(srv_now)
        for sym, df in self.h4_map.items():
            sub = df[df.index <= pd.Timestamp(closed_cutoff)]
            if not sub.empty:
                bid = float(sub["close"].iloc[-1])
                sp = self.specs[sym]
                ask = round(bid + sp.spread_points * sp.point, sp.digits)
                self.specs[sym] = replace(sp, bid=bid, ask=ask)

    def get_server_time_utc(self) -> Optional[datetime]:
        return self.server_now

    def get_account_snapshot(self) -> Optional[BrokerAccountSnapshot]:
        return self.account

    def get_open_positions(self, magic: Optional[int] = None) -> List[BrokerOpenPosition]:
        if magic is None:
            return list(self.positions)
        return [p for p in self.positions if p.magic == magic]

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolContractSpec]:
        return self.specs.get(symbol)

    def fetch_closed_bars(
        self,
        symbol: str,
        timeframe_str: str,
        count: int = 500,
        server_now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        ref_now = server_now or self.server_now
        tf_delta = timedelta(hours=4) if timeframe_str.upper() == "H4" else timedelta(days=1)
        df = self.h4_map[symbol] if timeframe_str.upper() == "H4" else self.d1_map[symbol]
        cutoff = pd.Timestamp(ref_now - tf_delta)
        return df[df.index <= cutoff].tail(count).copy()

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
        sp = self.specs[symbol]
        px = sp.ask if side > 0 else sp.bid
        return OrderExecutionResult(True, "ORDER_CHECK", symbol, 0, "OK", price=px, volume=volume)

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
        sp = self.specs[symbol]
        px = sp.ask if side > 0 else sp.bid
        self._ticket_seq += 1
        tk = self._ticket_seq
        self.positions.append(
            BrokerOpenPosition(
                ticket=tk,
                symbol=symbol,
                side=side,
                volume=volume,
                price_open=px,
                sl=round(sl_price, sp.digits),
                tp=round(tp_price, sp.digits),
                price_current=px,
                swap_eur=0.0,
                profit_eur=0.0,
                time_open_server=self.server_now,
                magic=settings.mt5_magic_number if magic is None else magic,
                comment=comment,
            )
        )
        return OrderExecutionResult(
            True, "ORDER_OPEN", symbol, 10009, "DONE", ticket=tk, order_id=tk, price=px, volume=volume
        )

    def modify_position_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: float,
        magic: Optional[int] = None,
    ) -> OrderExecutionResult:
        sp = self.specs[symbol]
        for idx, p in enumerate(self.positions):
            if p.ticket == ticket:
                self.positions[idx] = replace(
                    p, sl=round(new_sl, sp.digits), tp=round(new_tp, sp.digits)
                )
                return OrderExecutionResult(
                    True, "MODIFY_SLTP", symbol, 10009, "DONE", ticket=ticket
                )
        return OrderExecutionResult(False, "MODIFY_SLTP", symbol, -1, "NOT_FOUND", ticket=ticket)

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
        for idx, p in enumerate(self.positions):
            if p.ticket == ticket:
                self.positions.pop(idx)
                sp = self.specs[symbol]
                px = sp.bid if side > 0 else sp.ask
                return OrderExecutionResult(
                    True, "CLOSE_POSITION", symbol, 10009, "DONE", ticket=ticket, price=px, volume=volume
                )
        return OrderExecutionResult(False, "CLOSE_POSITION", symbol, -1, "NOT_FOUND", ticket=ticket)


# ---------------------------------------------------------------------------
# Unit & Integration Tests
# ---------------------------------------------------------------------------


def test_option_a_passthrough_sizer_rules_and_priority_ranking() -> None:
    cm = CostModel()
    specs = {sym: build_fallback_contract_spec(sym) for sym in ["EURUSD", "AUDUSD", "AUDJPY"]}
    sizer_op_a = PortfolioRiskSizer(
        cost_model=cm,
        guard=CurrencyAndHeatGuard(),
        use_meta_labeling=False,
        strategy_cfg=PrimaryStrategyConfig(),
    )

    t0 = datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc)
    # 1. Rank priority by `cs_spread` (EURUSD wins SHORT-USD bucket over AUDUSD despite lower meta_prob)
    c_eur = CandidateSignal(
        symbol="EURUSD",
        side=1,
        entry_price=1.08000,
        sl_price=1.07400,  # 60 pips -> ~5.52 EUR at 0.01 lot (< 10 EUR)
        tp_price=1.09050,
        meta_probability=0.35,
        signal_time_utc=t0,
        ker_d1=0.35,
        priority_score=0.90,
        holding_bars_h4=36,
    )
    c_aud = CandidateSignal(
        symbol="AUDUSD",
        side=1,
        entry_price=0.66000,
        sl_price=0.65400,
        tp_price=0.67050,
        meta_probability=0.80,  # Higher fake prob, LOWER cs_spread priority & same SHORT USD bucket
        signal_time_utc=t0,
        ker_d1=0.35,
        priority_score=0.62,
        holding_bars_h4=36,
    )
    # 2. Orthogonal pair AUDJPY (+AUD, -JPY) is approved, strictly capped at 0.01 lot
    c_audjpy = CandidateSignal(
        symbol="AUDJPY",
        side=1,
        entry_price=100.000,
        sl_price=99.200,  # 80 pips -> ~4.96 EUR at 0.01 lot (< 10 EUR)
        tp_price=101.400,
        meta_probability=0.80,
        signal_time_utc=t0,
        ker_d1=0.35,
        priority_score=0.70,
        holding_bars_h4=36,
    )

    decisions = sizer_op_a.rank_and_allocate_batch(
        candidates=[c_aud, c_audjpy, c_eur],
        specs=specs,
        equity_eur=1000.0,
        open_positions=[],
    )
    assert [d.symbol for d in decisions] == ["EURUSD", "AUDJPY", "AUDUSD"]
    assert decisions[0].approved is True
    assert decisions[0].allocated_lots == pytest.approx(0.01)
    assert decisions[1].approved is True
    # Pass-Through strictly caps at 0.01 lot even when meta_probability=0.80
    assert decisions[1].allocated_lots == pytest.approx(0.01)
    assert decisions[2].approved is False
    assert "VETO_CURRENCY_BUCKET" in decisions[2].rejection_reason

    # Confirm default ML sizer rejects meta_probability=0.35 and keeps Fase 0/2 compatibility
    sizer_ml = PortfolioRiskSizer(cost_model=cm, guard=CurrencyAndHeatGuard(), use_meta_labeling=True)
    ml_dec = sizer_ml.evaluate_single_candidate(c_eur, specs["EURUSD"], 1000.0, [])
    assert ml_dec.approved is False
    assert "VETO_META_PROBABILITY" in ml_dec.rejection_reason


def test_h4_server_clock_boundaries_and_weekend_skip() -> None:
    clock = H4ServerClock(offset_seconds=2.0)

    # Sync winter EET (UTC+2): real UTC 09:59:58 => server 11:59:58
    delta = clock.sync_offset_from_server_time(
        server_now=datetime(2026, 11, 11, 11, 59, 58, tzinfo=timezone.utc),
        real_utc_now=datetime(2026, 11, 11, 9, 59, 58, tzinfo=timezone.utc),
    )
    assert delta == pytest.approx(7200.0)
    srv_now = datetime(2026, 11, 11, 11, 59, 58, tzinfo=timezone.utc)
    assert clock.current_h4_bar_open(srv_now) == datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
    assert clock.latest_closed_h4_bar_open(srv_now) == datetime(2026, 11, 11, 4, 0, tzinfo=timezone.utc)
    assert clock.next_h4_close_wakeup_server(srv_now) == datetime(2026, 11, 11, 12, 0, 2, tzinfo=timezone.utc)
    assert clock.seconds_until_next_h4_wakeup(srv_now) == pytest.approx(4.0)

    # 1 second before wakeup vs exact wakeup at 12:00:02
    just_opened = datetime(2026, 11, 11, 12, 0, 1, tzinfo=timezone.utc)
    assert clock.next_h4_close_wakeup_server(just_opened) == datetime(2026, 11, 11, 12, 0, 2, tzinfo=timezone.utc)
    at_wakeup = datetime(2026, 11, 11, 12, 0, 2, tzinfo=timezone.utc)
    assert clock.latest_closed_h4_bar_open(at_wakeup) == datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
    assert clock.next_h4_close_wakeup_server(at_wakeup) == datetime(2026, 11, 11, 16, 0, 2, tzinfo=timezone.utc)

    # Weekend closure: Friday 20:00 closes at Saturday 00:00:02; next wakeup is Monday 04:00:02
    fri_night = datetime(2026, 11, 13, 23, 59, 0, tzinfo=timezone.utc)  # Friday
    sat_close = clock.next_h4_close_wakeup_server(fri_night, skip_weekend=True)
    assert sat_close == datetime(2026, 11, 14, 0, 0, 2, tzinfo=timezone.utc)
    mon_close = clock.next_h4_close_wakeup_server(sat_close, skip_weekend=True)
    assert mon_close == datetime(2026, 11, 16, 4, 0, 2, tzinfo=timezone.utc)

    # Friday 00:00..20:00 = 6 bars + Monday 00:00..20:00 = 6 bars => 12 trading bars over Mon+Fri span
    bars = H4ServerClock.count_trading_h4_bars_between(
        datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc),
        inclusive_entry_bar=True,
    )
    assert bars == 12


def test_account_safety_blocks_1m_and_real_accounts() -> None:
    symbols = ["EURUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "GBPUSD", "AUDJPY"]
    h4_map, d1_map = _make_synthetic_fx_basket(symbols, n_h4_bars=500)

    # 1. H4OrderManager blocks 1,000,000 EUR wrong account even if is_demo=True
    wrong_acc = BrokerAccountSnapshot(
        login=99999999,
        server="PepperstoneUK-Demo",
        currency="EUR",
        balance_eur=1_000_000.0,
        equity_eur=1_000_000.0,
        margin_eur=0.0,
        margin_free_eur=1_000_000.0,
        leverage=30,
        is_demo=True,
    )
    adapter = InMemoryBrokerAdapter(h4_map, d1_map, wrong_acc)
    mgr = H4OrderManager(adapter=adapter, symbols=symbols, use_meta_labeling=False, sync_to_questdb=False)
    res_1m = mgr.run_h4_cycle()
    assert res_1m.errors and "Account safety block" in res_1m.errors[0]
    assert res_1m.orders_executed == []

    # 2. Live real-money account blocked by default when allow_live_real_money=False
    real_acc = BrokerAccountSnapshot(
        login=1234567,
        server="PepperstoneUK-Live",
        currency="EUR",
        balance_eur=1000.0,
        equity_eur=1000.0,
        margin_eur=0.0,
        margin_free_eur=1000.0,
        leverage=30,
        is_demo=False,
    )
    adapter_real = InMemoryBrokerAdapter(h4_map, d1_map, real_acc)
    mgr_real = H4OrderManager(adapter=adapter_real, symbols=symbols, use_meta_labeling=False, sync_to_questdb=False)
    res_real = mgr_real.run_h4_cycle()
    assert res_real.errors and "Account safety block" in res_real.errors[0]

    # 3. Direct MT5Connector._validate_trading_account_safety check
    conn = MT5Connector()
    conn.get_account_snapshot = lambda: wrong_acc  # type: ignore[method-assign]
    chk = conn._validate_trading_account_safety("ORDER_OPEN", "EURUSD")
    assert chk is not None and not chk.success and "BLOCKED_WRONG_MT5_ACCOUNT" in chk.comment


def test_order_manager_rollover_veto_be_trail_timestop_and_parity() -> None:
    symbols = ["EURUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "GBPUSD", "AUDJPY"]
    h4_map, d1_map = _make_synthetic_fx_basket(symbols, n_h4_bars=1600)
    strategy = PrimaryTrendStrategy(PrimaryStrategyConfig())
    sig_sym, sig_ts, sig_side, sl_dist, tp_dist = _find_signal_bar(
        symbols, h4_map, d1_map, strategy
    )

    demo_acc = BrokerAccountSnapshot(
        login=62143460,
        server="PepperstoneUK-Demo",
        currency="EUR",
        balance_eur=1000.0,
        equity_eur=1000.0,
        margin_eur=0.0,
        margin_free_eur=1000.0,
        leverage=30,
        is_demo=True,
    )

    # A. Rollover spread spike veto (e.g. 95 pts across rollover) blocks execution while keeping audit rows
    adapter_spike = InMemoryBrokerAdapter(h4_map, d1_map, demo_acc)
    for sym_k in adapter_spike.specs:
        adapter_spike.specs[sym_k] = replace(adapter_spike.specs[sym_k], spread_points=95)
    adapter_spike.set_server_now(sig_ts + timedelta(hours=4, seconds=2))

    mgr_spike = H4OrderManager(
        adapter=adapter_spike,
        symbols=symbols,
        strategy=strategy,
        use_meta_labeling=False,
        sync_to_questdb=False,
        enforce_bar_alignment_wait=False,
    )
    res_spike = mgr_spike.run_h4_cycle(server_now=sig_ts + timedelta(hours=4, seconds=2))
    audit_sig = next(r for r in res_spike.audit_records if r.symbol == sig_sym)
    assert audit_sig.side == sig_side
    assert "VETO_LIVE_ROLLOVER_SPREAD" in audit_sig.rejection_reason
    assert audit_sig.approved is False
    assert len(adapter_spike.positions) == 0

    # B. Normal active spread -> Opens 0.01 lot -> steps BE -> Trail -> 36-bar TIME_STOP
    adapter = InMemoryBrokerAdapter(h4_map, d1_map, demo_acc)
    mgr = H4OrderManager(
        adapter=adapter,
        symbols=symbols,
        strategy=strategy,
        use_meta_labeling=False,
        sync_to_questdb=False,
        enforce_bar_alignment_wait=False,
    )

    open_srv_now = sig_ts + timedelta(hours=4, seconds=2)
    adapter.set_server_now(open_srv_now)
    res_open = mgr.run_h4_cycle(server_now=open_srv_now)
    open_rec = next(r for r in res_open.audit_records if r.symbol == sig_sym)
    assert open_rec.approved is True
    assert open_rec.allocated_lots == pytest.approx(0.01)

    pos = next(p for p in adapter.positions if p.symbol == sig_sym)
    ticket = pos.ticket
    entry_bar_open = H4ServerClock.current_h4_bar_open(open_srv_now)

    # Construct favorable path over subsequent H4 bars for `sig_sym` to test BE & Trail deterministically
    h4_df = adapter.h4_map[sig_sym]
    entry_idx_loc = int(h4_df.index.get_indexer([pd.Timestamp(entry_bar_open)])[0])
    assert entry_idx_loc > 0
    # Prevent subsequent signals or wild prices from interfering; gently advance sig_sym favorably
    # Bar 0 (entry_bar_open) reaches +1.20R (triggers BE@+1.10R locking +0.10R)
    # Bar 1 reaches +1.70R (triggers Chandelier Trail@+1.50R)
    # Bars 2..35 stay inside (BE+0.5R .. TP-0.2R) so position survives until exact bar 36 TIME_STOP!
    orig_close = pos.price_open
    sp = adapter.specs[sig_sym]
    sp_px = sp.spread_points * sp.point
    mid_safe = orig_close + sig_side * (0.8 * sl_dist)
    for step in range(36):
        b_idx = entry_idx_loc + step
        ts_idx = h4_df.index[b_idx]
        if step == 0:
            fav_px = (orig_close + 1.20 * sl_dist) if sig_side > 0 else (orig_close - 1.20 * sl_dist - sp_px)
            adv_px = (orig_close - 0.10 * sl_dist) if sig_side > 0 else (orig_close + 0.10 * sl_dist)
        elif step == 1:
            fav_px = (orig_close + 1.70 * sl_dist) if sig_side > 0 else (orig_close - 1.70 * sl_dist - sp_px)
            adv_px = mid_safe
        else:
            fav_px = orig_close + sig_side * (1.20 * sl_dist)
            adv_px = mid_safe

        bar_h = max(fav_px, adv_px) if sig_side > 0 else max(adv_px, fav_px)
        bar_l = min(fav_px, adv_px)
        h4_df.loc[ts_idx, "open"] = round(mid_safe, sp.digits)
        h4_df.loc[ts_idx, "high"] = round(bar_h, sp.digits)
        h4_df.loc[ts_idx, "low"] = round(bar_l, sp.digits)
        h4_df.loc[ts_idx, "close"] = round((bar_h + bar_l) * 0.5, sp.digits)

    # Bar #1 closes at entry_bar_open + 4h -> triggers BE lock (+0.10R)
    t_bar1_close = h4_df.index[entry_idx_loc].to_pydatetime() + timedelta(hours=4, seconds=2)
    adapter.set_server_now(t_bar1_close)
    # Adjust live bid/ask comfortably away from BE SL for stop-level guard
    adapter.specs[sig_sym] = replace(
        adapter.specs[sig_sym],
        bid=round(orig_close + sig_side * 1.15 * sl_dist, sp.digits),
        ask=round(orig_close + sig_side * 1.15 * sl_dist + sp_px, sp.digits),
    )
    res_be = mgr.run_h4_cycle(server_now=t_bar1_close)
    assert ticket in res_be.positions_sl_updated
    pos_after_be = next(p for p in adapter.positions if p.ticket == ticket)
    expected_be_sl = round(orig_close + sig_side * (0.10 * sl_dist), sp.digits)
    assert pos_after_be.sl == pytest.approx(expected_be_sl, abs=2 * sp.point)

    # Check risk guard now sees position as BE-protected (0.0 open risk!)
    risk_states, _ = mgr._build_open_risk_states([pos_after_be], adapter.specs)
    assert risk_states[0].is_at_breakeven is True
    assert risk_states[0].effective_active_risk_eur == 0.0

    # Bar #2 closes -> excursion reached +1.70R (>= 1.50R), triggers Chandelier Trail
    t_bar2_close = h4_df.index[entry_idx_loc + 1].to_pydatetime() + timedelta(hours=4, seconds=2)
    adapter.set_server_now(t_bar2_close)
    adapter.specs[sig_sym] = replace(
        adapter.specs[sig_sym],
        bid=round(orig_close + sig_side * 1.60 * sl_dist, sp.digits),
        ask=round(orig_close + sig_side * 1.60 * sl_dist + sp_px, sp.digits),
    )
    res_trail = mgr.run_h4_cycle(server_now=t_bar2_close)
    pos_after_trail = next(p for p in adapter.positions if p.ticket == ticket)
    assert (pos_after_trail.sl > pos_after_be.sl) if sig_side > 0 else (pos_after_trail.sl < pos_after_be.sl)

    # Bar #35 still keeps position alive; Bar #36 triggers TIME_STOP close
    t_bar35_close = h4_df.index[entry_idx_loc + 34].to_pydatetime() + timedelta(hours=4, seconds=2)
    adapter.set_server_now(t_bar35_close)
    res_bar35 = mgr.run_h4_cycle(server_now=t_bar35_close)
    assert ticket not in res_bar35.positions_closed_time_stop
    assert any(p.ticket == ticket for p in adapter.positions)

    t_bar36_close = h4_df.index[entry_idx_loc + 35].to_pydatetime() + timedelta(hours=4, seconds=2)
    adapter.set_server_now(t_bar36_close)
    res_bar36 = mgr.run_h4_cycle(server_now=t_bar36_close)
    assert ticket in res_bar36.positions_closed_time_stop
    assert not any(p.ticket == ticket for p in adapter.positions)

    # C. Parity Audit against full-batch computation on unmodified historical slice up to open_srv_now
    # Collect multi-bar audit trail on a fresh clean adapter for strict <= 1e-9 parity proof
    h4_clean, d1_clean = _make_synthetic_fx_basket(symbols, n_h4_bars=1400)
    adapter_clean = InMemoryBrokerAdapter(h4_clean, d1_clean, demo_acc)
    mgr_clean = H4OrderManager(
        adapter=adapter_clean,
        symbols=symbols,
        strategy=strategy,
        use_meta_labeling=False,
        sync_to_questdb=False,
        enforce_bar_alignment_wait=False,
    )
    all_audit: List[FeatureAuditRecord] = []
    # Run 5 consecutive closed H4 bars ending near sample tail
    sample_bars = h4_clean[symbols[0]].index[-10:-5]
    for bar_open in sample_bars:
        bar_dt = bar_open.to_pydatetime()
        srv_t = bar_dt + timedelta(hours=4, seconds=2)
        adapter_clean.set_server_now(srv_t)
        cyc = mgr_clean.run_h4_cycle(server_now=srv_t)
        all_audit.extend(cyc.audit_records)

    checker = ParityChecker(strategy=strategy, cost_model=CostModel(), tolerance=1e-9)
    parity_report = checker.check_records(
        audit_records=all_audit,
        h4_bars_by_symbol=h4_clean,
        d1_bars_by_symbol=d1_clean,
        specs_by_symbol=adapter_clean.specs,
        reference_lots=0.01,
    )
    assert parity_report.passed is True, parity_report.summary
    assert parity_report.records_checked == len(symbols) * len(sample_bars)
    assert parity_report.max_abs_diff <= 1e-9
    assert parity_report.max_signal_diff <= 1e-9
    assert parity_report.signal_mismatches == 0

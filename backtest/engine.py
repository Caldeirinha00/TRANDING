from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import settings
from core.mt5_connector import SymbolContractSpec
from data.cost_model import CostModel
from risk.currency_guard import CurrencyAndHeatGuard, OpenPositionRiskState
from strategy.features import compute_atr
from strategy.primary_trend import PrimaryStrategyConfig
from strategy.triple_barrier import effective_spread_price, step_barrier_state_bar


@dataclass
class _OpenBacktestPosition:
    trade_id: int
    symbol: str
    side: int
    lots: float
    entry_time: datetime
    entry_price: float
    initial_sl_dist: float
    current_sl: float
    tp_price: float
    gross_1r_eur: float
    spread_cost_eur: float
    commission_eur: float
    margin_eur: float
    last_bar_ts: datetime
    favorable_peak: float
    bars_held: int = 0
    swap_eur: float = 0.0
    is_be: bool = False
    is_trailing: bool = False
    trigger_type: str = "NONE"
    meta_probability: float = 1.0


@dataclass(frozen=True)
class BacktestTrade:
    trade_id: int
    symbol: str
    side: int
    lots: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    bars_held: int
    exit_reason: str
    trigger_type: str
    gross_1r_eur: float
    spread_cost_eur: float
    commission_eur: float
    swap_eur: float
    net_pnl_eur: float
    realized_r: float
    meta_probability: float = 1.0


@dataclass
class BacktestReport:
    initial_capital_eur: float
    final_equity_eur: float
    net_profit_eur: float
    net_return_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_r: float
    expectancy_eur: float
    max_drawdown_pct: float
    sharpe_annualized: float
    calmar_ratio: float
    total_gross_1r_eur: float
    total_spread_plus_commission_eur: float
    friction_to_gross_risk_pct: float
    total_swap_eur: float
    kill_switch_daily_hits: int
    kill_switch_total_hit: bool
    rejected_counts: Dict[str, int] = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class PortfolioBacktestEngine:
    """
    Event-driven bar-by-bar multi-asset backtester aligned with MT5 Bid/Ask execution,
    ESMA margin & portfolio heat constraints, BE@1.20R (+0.10R lock), Chandelier ATR
    Trailing@1.80R, Vertical Barrier (30 H4 bars), and Wednesday triple rollover swaps.
    """

    def __init__(
        self,
        specs: Dict[str, SymbolContractSpec],
        cost_model: Optional[CostModel] = None,
        trend_config: Optional[PrimaryStrategyConfig] = None,
        initial_capital_eur: Optional[float] = None,
    ) -> None:
        self.specs = specs
        self.cost_model = cost_model or CostModel()
        self.trend_config = trend_config or PrimaryStrategyConfig()
        self.initial_capital_eur = (
            settings.initial_capital_eur if initial_capital_eur is None else initial_capital_eur
        )
        self.guard = CurrencyAndHeatGuard()

    def _effective_spread_price(self, spec: SymbolContractSpec, bar_spread_pts: int) -> float:
        return effective_spread_price(spec, bar_spread_pts)

    def _calc_unrealized_eur(
        self,
        pos: _OpenBacktestPosition,
        bid_close: float,
        spread_price: float,
    ) -> float:
        spec = self.specs[pos.symbol]
        mark_price = bid_close if pos.side > 0 else (bid_close + spread_price)
        price_diff = (mark_price - pos.entry_price) * pos.side
        val_per_unit = self.cost_model.value_per_price_unit_for_lots(
            spec, pos.lots, for_loss=(price_diff < 0.0)
        )
        return price_diff * val_per_unit

    def _close_position(
        self,
        pos: _OpenBacktestPosition,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        meta_prob: Optional[float] = None,
    ) -> BacktestTrade:
        spec = self.specs[pos.symbol]
        resolved_meta_prob = pos.meta_probability if meta_prob is None else meta_prob
        price_diff = (exit_price - pos.entry_price) * pos.side
        val_per_unit = self.cost_model.value_per_price_unit_for_lots(
            spec, pos.lots, for_loss=(price_diff < 0.0)
        )
        pnl_after_spread = price_diff * val_per_unit
        net_pnl_eur = pnl_after_spread - pos.commission_eur + pos.swap_eur
        realized_r = net_pnl_eur / pos.gross_1r_eur if pos.gross_1r_eur > 0 else 0.0

        return BacktestTrade(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            lots=pos.lots,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            bars_held=pos.bars_held,
            exit_reason=exit_reason,
            trigger_type=pos.trigger_type,
            gross_1r_eur=pos.gross_1r_eur,
            spread_cost_eur=pos.spread_cost_eur,
            commission_eur=pos.commission_eur,
            swap_eur=round(pos.swap_eur, 4),
            net_pnl_eur=round(net_pnl_eur, 4),
            realized_r=round(realized_r, 4),
            meta_probability=resolved_meta_prob,
        )

    def _step_open_position_bar(
        self,
        pos: _OpenBacktestPosition,
        bar_ts: datetime,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        curr_atr: float,
        spread_price: float,
        charge_swap: bool,
    ) -> Optional[Tuple[float, str]]:
        return step_barrier_state_bar(
            pos=pos,  # type: ignore[arg-type]
            spec=self.specs[pos.symbol],
            cost_model=self.cost_model,
            cfg=self.trend_config,
            bar_ts=bar_ts,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
            curr_atr=curr_atr,
            spread_price=spread_price,
            charge_swap=charge_swap,
        )

    def run(
        self,
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
        fixed_lots: Optional[float] = 0.01,
        meta_decisions_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> BacktestReport:
        """
        Simulate multi-asset H4 portfolio execution strictly using next-bar open fills
        after closed-bar signals.
        """
        symbols = sorted(h4_bars_by_symbol.keys())
        # Tuple: (open, high, low, close, spread_pts, atr14_h4)
        bar_lookup: Dict[str, Dict[datetime, Tuple[float, float, float, float, int, float]]] = {}
        # Tuple: (side, sl_dist, tp_dist, cs_spread, ker_d1, trigger_type)
        sig_lookup: Dict[str, Dict[datetime, Tuple[int, float, float, float, float, str]]] = {}
        meta_lookup: Dict[str, Dict[datetime, Tuple[float, float]]] = {}
        all_timestamps_set: set[datetime] = set()

        for sym in symbols:
            df_full = h4_bars_by_symbol[sym]
            atr_full = compute_atr(df_full, period=self.trend_config.atr_h4_period).fillna(0.0)
            df_b = df_full
            if start_time is not None:
                df_b = df_b[df_b.index >= start_time]
            if end_time is not None:
                df_b = df_b[df_b.index <= end_time]
            atr_slice = atr_full.reindex(df_b.index).fillna(0.0)

            df_s = signals_by_symbol[sym].reindex(df_b.index).fillna(
                {
                    "side": 0,
                    "sl_distance_price": 0.0,
                    "tp_distance_price": 0.0,
                    "cs_spread": 0.0,
                    "ker_d1": 0.0,
                    "trigger_type": "NONE",
                }
            )

            b_dict: Dict[datetime, Tuple[float, float, float, float, int, float]] = {}
            s_dict: Dict[datetime, Tuple[int, float, float, float, float, str]] = {}
            for ts, row in df_b.iterrows():
                ts_dt = ts.to_pydatetime()
                all_timestamps_set.add(ts_dt)
                b_dict[ts_dt] = (
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row.get("spread", 0)),
                    float(atr_slice.loc[ts]),
                )
            for ts, row in df_s.iterrows():
                side_val = int(row["side"])
                if side_val != 0:
                    s_dict[ts.to_pydatetime()] = (
                        side_val,
                        float(row["sl_distance_price"]),
                        float(row["tp_distance_price"]),
                        float(row["cs_spread"]),
                        float(row.get("ker_d1", 0.0)),
                        str(row["trigger_type"]),
                    )
            bar_lookup[sym] = b_dict
            sig_lookup[sym] = s_dict

            if meta_decisions_by_symbol and sym in meta_decisions_by_symbol:
                m_df = meta_decisions_by_symbol[sym]
                meta_lookup[sym] = {
                    ts.to_pydatetime(): (float(r["meta_prob"]), float(r["expected_net_r"]))
                    for ts, r in m_df.iterrows()
                }
            else:
                meta_lookup[sym] = {}

        all_timestamps = sorted(all_timestamps_set)
        if not all_timestamps:
            raise ValueError("No H4 bars available for backtest.")

        balance_eur = float(self.initial_capital_eur)
        hwm_equity_eur = float(self.initial_capital_eur)
        day_start_equity_eur = float(self.initial_capital_eur)
        current_day: Optional[date] = None
        daily_locked = False
        halted_total_dd = False

        open_positions: Dict[str, _OpenBacktestPosition] = {}
        # Pending tuple: (side, sl_dist, tp_dist, prio, meta_prob, lots_hint, ker_d1, trig)
        pending_signals: Dict[str, Tuple[int, float, float, float, float, float, float, str]] = {}
        closed_trades: List[BacktestTrade] = []
        rejected_counts: Dict[str, int] = {}
        equity_rows: List[Tuple[datetime, float, float, int, float, float]] = []
        last_known_bid: Dict[str, float] = {}
        last_known_spread: Dict[str, float] = {}

        kill_switch_daily_hits = 0
        kill_switch_total_hit = False
        next_trade_id = 1

        def _reject(reason_key: str) -> None:
            rejected_counts[reason_key] = rejected_counts.get(reason_key, 0) + 1

        def _compute_mtm_equity() -> float:
            unreal = 0.0
            for pos in open_positions.values():
                bid_px = last_known_bid[pos.symbol]
                sp_px = last_known_spread[pos.symbol]
                unreal += self._calc_unrealized_eur(pos, bid_px, sp_px) + pos.swap_eur
            return balance_eur + unreal

        for ts in all_timestamps:
            if halted_total_dd:
                break

            bar_day = ts.date()
            if current_day is None or bar_day > current_day:
                current_day = bar_day
                daily_locked = False
                day_start_equity_eur = _compute_mtm_equity() if open_positions else balance_eur

            for sym in symbols:
                bar = bar_lookup[sym].get(ts)
                if bar is not None:
                    last_known_bid[sym] = bar[3]
                    last_known_spread[sym] = self._effective_spread_price(self.specs[sym], bar[4])

            # 1. Step already-open positions through bar `ts`
            to_close: List[Tuple[str, float, str]] = []
            for sym, pos in open_positions.items():
                bar = bar_lookup[sym].get(ts)
                if bar is None:
                    continue
                _, b_high, b_low, b_close, b_sp, b_atr = bar
                sp_px = self._effective_spread_price(self.specs[sym], b_sp)
                res = self._step_open_position_bar(
                    pos=pos,
                    bar_ts=ts,
                    bar_high=b_high,
                    bar_low=b_low,
                    bar_close=b_close,
                    curr_atr=b_atr,
                    spread_price=sp_px,
                    charge_swap=True,
                )
                if res is not None:
                    to_close.append((sym, res[0], res[1]))

            for sym, exit_px, reason in to_close:
                pos = open_positions.pop(sym)
                trade = self._close_position(pos, ts, exit_px, reason)
                balance_eur += trade.net_pnl_eur
                closed_trades.append(trade)

            # 2. Execute pending signals from previous closed bar at `ts` Open
            if not daily_locked and not halted_total_dd and pending_signals:
                ready_candidates = []
                remaining_pending = {}
                for sym, payload in pending_signals.items():
                    if sym in bar_lookup and ts in bar_lookup[sym]:
                        ready_candidates.append((sym, payload))
                    else:
                        remaining_pending[sym] = payload
                pending_signals = remaining_pending

                ready_candidates.sort(key=lambda item: item[1][3], reverse=True)

                for sym, (side, sl_dist, tp_dist, _prio, meta_prob, lots_hint, ker_d1, trig) in ready_candidates:
                    spec = self.specs[sym]
                    bar = bar_lookup[sym][ts]
                    b_open, b_high, b_low, b_close, b_sp, b_atr = bar
                    sp_px = self._effective_spread_price(spec, b_sp)

                    if sym in open_positions:
                        _reject("VETO_SYMBOL_ALREADY_OPEN")
                        continue

                    lots_candidates = [fixed_lots] if fixed_lots is not None else ([0.02, 0.01] if lots_hint >= 0.02 else [0.01])
                    mtm_equity = _compute_mtm_equity()
                    max_unit_risk = mtm_equity * settings.max_risk_per_trade_pct
                    used_margin = sum(p.margin_eur for p in open_positions.values())
                    risk_states = [
                        OpenPositionRiskState(
                            ticket=p.trade_id,
                            symbol=p.symbol,
                            side=p.side,
                            lots=p.lots,
                            entry_price=p.entry_price,
                            current_sl=p.current_sl,
                            open_risk_eur=p.gross_1r_eur,
                        )
                        for p in open_positions.values()
                    ]
                    swap_cap = (
                        self.trend_config.max_swap_strong_trend_ratio
                        if ker_d1 >= self.trend_config.ker_strong_trend_threshold
                        else self.trend_config.max_swap_normal_ratio
                    )

                    selected_lots = 0.0
                    selected_friction = None
                    selected_margin = 0.0
                    last_reject_key = "VETO_ZERO_LOTS"

                    for cand_lots in lots_candidates:
                        if cand_lots < 0.01:
                            last_reject_key = "VETO_ZERO_LOTS"
                            continue
                        friction = self.cost_model.evaluate_trade_friction(
                            spec=spec,
                            side=side,
                            lots=cand_lots,
                            sl_distance_price=sl_dist,
                            entry_time_utc=ts,
                            holding_bars_h4=self.trend_config.vertical_barrier_bars_h4,
                            max_swap_ratio_override=swap_cap,
                        )
                        if friction.is_hostile_swap_vetoed:
                            last_reject_key = "VETO_HOSTILE_SWAP"
                            break
                        if friction.gross_1r_risk_eur > max_unit_risk:
                            last_reject_key = "VETO_UNIT_RISK_EXCEEDED"
                            continue
                        margin_req = spec.margin_initial_001_eur * (cand_lots / 0.01)
                        if (used_margin + margin_req) > (mtm_equity * 0.85):
                            last_reject_key = "VETO_ESMA_MARGIN"
                            continue
                        guard_ok, guard_reason = self.guard.can_accept_new_risk(
                            candidate_symbol=sym,
                            candidate_side=side,
                            candidate_risk_eur=friction.gross_1r_risk_eur,
                            equity_eur=mtm_equity,
                            open_positions=risk_states,
                        )
                        if not guard_ok:
                            last_reject_key = guard_reason.split(":", 1)[0]
                            continue

                        selected_lots = cand_lots
                        selected_friction = friction
                        selected_margin = margin_req
                        break

                    if selected_lots < 0.01 or selected_friction is None:
                        _reject(last_reject_key)
                        continue

                    lots = selected_lots
                    friction = selected_friction
                    margin_req = selected_margin

                    entry_px = (b_open + sp_px) if side > 0 else b_open
                    sl_px = entry_px - (side * sl_dist)
                    tp_px = entry_px + (side * tp_dist)

                    pos = _OpenBacktestPosition(
                        trade_id=next_trade_id,
                        symbol=sym,
                        side=side,
                        lots=lots,
                        entry_time=ts,
                        entry_price=entry_px,
                        initial_sl_dist=sl_dist,
                        current_sl=sl_px,
                        tp_price=tp_px,
                        gross_1r_eur=friction.gross_1r_risk_eur,
                        spread_cost_eur=friction.spread_cost_eur,
                        commission_eur=friction.commission_roundtrip_eur,
                        margin_eur=margin_req,
                        last_bar_ts=ts,
                        favorable_peak=entry_px,
                        trigger_type=trig,
                        meta_probability=meta_prob,
                    )
                    next_trade_id += 1

                    entry_exit = self._step_open_position_bar(
                        pos=pos,
                        bar_ts=ts,
                        bar_high=b_high,
                        bar_low=b_low,
                        bar_close=b_close,
                        curr_atr=b_atr,
                        spread_price=sp_px,
                        charge_swap=False,
                    )
                    if entry_exit is not None:
                        trade = self._close_position(pos, ts, entry_exit[0], entry_exit[1], meta_prob=meta_prob)
                        balance_eur += trade.net_pnl_eur
                        closed_trades.append(trade)
                    else:
                        open_positions[sym] = pos

            # 3. Check Mark-to-Market Equity & Kill Switches at bar close
            mtm_equity = _compute_mtm_equity()
            hwm_equity_eur = max(hwm_equity_eur, mtm_equity)
            daily_dd = max(0.0, (day_start_equity_eur - mtm_equity) / max(1e-6, day_start_equity_eur))
            total_dd = max(0.0, (hwm_equity_eur - mtm_equity) / max(1e-6, hwm_equity_eur))

            if total_dd >= settings.total_dd_kill_switch_pct:
                kill_switch_total_hit = True
                halted_total_dd = True
                for sym, pos in list(open_positions.items()):
                    exit_px = (
                        last_known_bid[sym] if pos.side > 0 else (last_known_bid[sym] + last_known_spread[sym])
                    )
                    trade = self._close_position(pos, ts, exit_px, "KILL_SWITCH_TOTAL")
                    balance_eur += trade.net_pnl_eur
                    closed_trades.append(trade)
                open_positions.clear()
                mtm_equity = balance_eur
            elif daily_dd >= settings.daily_dd_kill_switch_pct and not daily_locked:
                daily_locked = True
                kill_switch_daily_hits += 1
                pending_signals.clear()
                for sym, pos in list(open_positions.items()):
                    exit_px = (
                        last_known_bid[sym] if pos.side > 0 else (last_known_bid[sym] + last_known_spread[sym])
                    )
                    trade = self._close_position(pos, ts, exit_px, "KILL_SWITCH_DAILY")
                    balance_eur += trade.net_pnl_eur
                    closed_trades.append(trade)
                open_positions.clear()
                mtm_equity = balance_eur

            used_margin_now = sum(p.margin_eur for p in open_positions.values())
            dd_pct = max(0.0, (hwm_equity_eur - mtm_equity) / max(1e-6, hwm_equity_eur)) * 100.0
            equity_rows.append(
                (ts, round(mtm_equity, 4), round(balance_eur, 4), len(open_positions), round(used_margin_now, 2), round(dd_pct, 4))
            )

            # 4. Queue new closed-bar signals from `ts` for execution at next bar Open
            if not daily_locked and not halted_total_dd:
                for sym in symbols:
                    sig = sig_lookup[sym].get(ts)
                    if sig is None:
                        continue
                    side, sl_dist, tp_dist, cs_spread, ker_d1, trig = sig
                    meta_prob = 1.0
                    prio = cs_spread
                    lots_hint = 0.01

                    if meta_decisions_by_symbol is not None:
                        m_val = meta_lookup[sym].get(ts)
                        if m_val is None:
                            _reject("VETO_MISSING_META")
                            continue
                        meta_prob, expected_net_r = m_val
                        if meta_prob < settings.meta_min_probability:
                            _reject("VETO_META_PROBABILITY")
                            continue
                        if expected_net_r < settings.meta_min_expected_net_r:
                            _reject("VETO_LOW_EXPECTED_R")
                            continue
                        prio = expected_net_r
                        lots_hint = 0.02 if meta_prob >= settings.meta_high_conviction_probability else 0.01

                    pending_signals[sym] = (side, sl_dist, tp_dist, prio, meta_prob, lots_hint, ker_d1, trig)

        final_ts = all_timestamps[-1]
        for sym, pos in list(open_positions.items()):
            exit_px = last_known_bid[sym] if pos.side > 0 else (last_known_bid[sym] + last_known_spread[sym])
            trade = self._close_position(pos, final_ts, exit_px, "END_OF_SAMPLE")
            balance_eur += trade.net_pnl_eur
            closed_trades.append(trade)
        open_positions.clear()

        eq_df = pd.DataFrame(
            equity_rows,
            columns=["time", "equity_eur", "balance_eur", "open_positions", "used_margin_eur", "drawdown_pct"],
        ).set_index("time")

        return self._summarize(
            closed_trades=closed_trades,
            eq_df=eq_df,
            rejected_counts=rejected_counts,
            kill_switch_daily_hits=kill_switch_daily_hits,
            kill_switch_total_hit=kill_switch_total_hit,
        )

    def _summarize(
        self,
        closed_trades: List[BacktestTrade],
        eq_df: pd.DataFrame,
        rejected_counts: Dict[str, int],
        kill_switch_daily_hits: int,
        kill_switch_total_hit: bool,
    ) -> BacktestReport:
        total_trades = len(closed_trades)
        final_equity = float(eq_df["equity_eur"].iloc[-1]) if not eq_df.empty else self.initial_capital_eur
        net_profit = final_equity - self.initial_capital_eur
        net_ret_pct = (net_profit / self.initial_capital_eur) * 100.0
        max_dd_pct = float(eq_df["drawdown_pct"].max()) if not eq_df.empty else 0.0

        if total_trades == 0:
            return BacktestReport(
                initial_capital_eur=self.initial_capital_eur,
                final_equity_eur=round(final_equity, 2),
                net_profit_eur=round(net_profit, 2),
                net_return_pct=round(net_ret_pct, 2),
                total_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_r=0.0,
                expectancy_eur=0.0,
                max_drawdown_pct=round(max_dd_pct, 2),
                sharpe_annualized=0.0,
                calmar_ratio=0.0,
                total_gross_1r_eur=0.0,
                total_spread_plus_commission_eur=0.0,
                friction_to_gross_risk_pct=0.0,
                total_swap_eur=0.0,
                kill_switch_daily_hits=kill_switch_daily_hits,
                kill_switch_total_hit=kill_switch_total_hit,
                rejected_counts=rejected_counts,
                trades=closed_trades,
                equity_curve=eq_df,
            )

        net_pnls = np.array([t.net_pnl_eur for t in closed_trades], dtype=float)
        r_vals = np.array([t.realized_r for t in closed_trades], dtype=float)
        wins = net_pnls[net_pnls > 0.0]
        losses = net_pnls[net_pnls < 0.0]

        win_rate = (len(wins) / total_trades) * 100.0
        gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        exp_r = float(np.mean(r_vals))
        exp_eur = float(np.mean(net_pnls))
        total_gross_1r = float(sum(t.gross_1r_eur for t in closed_trades))
        total_friction = float(sum(t.spread_cost_eur + t.commission_eur for t in closed_trades))
        friction_pct = (total_friction / total_gross_1r * 100.0) if total_gross_1r > 0 else 0.0
        total_swap = float(sum(t.swap_eur for t in closed_trades))

        eq_daily = eq_df["equity_eur"].resample("1D").last().dropna()
        daily_rets = eq_daily.pct_change().dropna()
        if len(daily_rets) > 10 and daily_rets.std(ddof=1) > 0:
            sharpe = float((daily_rets.mean() / daily_rets.std(ddof=1)) * np.sqrt(252.0))
        else:
            sharpe = 0.0

        years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 1.0 / 12.0)
        cagr_pct = ((max(final_equity, 1.0) / self.initial_capital_eur) ** (1.0 / years) - 1.0) * 100.0
        calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        return BacktestReport(
            initial_capital_eur=self.initial_capital_eur,
            final_equity_eur=round(final_equity, 2),
            net_profit_eur=round(net_profit, 2),
            net_return_pct=round(net_ret_pct, 2),
            total_trades=total_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 3),
            expectancy_r=round(exp_r, 4),
            expectancy_eur=round(exp_eur, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_annualized=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            total_gross_1r_eur=round(total_gross_1r, 2),
            total_spread_plus_commission_eur=round(total_friction, 2),
            friction_to_gross_risk_pct=round(friction_pct, 2),
            total_swap_eur=round(total_swap, 2),
            kill_switch_daily_hits=kill_switch_daily_hits,
            kill_switch_total_hit=kill_switch_total_hit,
            rejected_counts=rejected_counts,
            trades=closed_trades,
            equity_curve=eq_df,
        )

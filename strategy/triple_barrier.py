from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import settings
from core.mt5_connector import SymbolContractSpec
from data.cost_model import CostModel
from strategy.features import FEATURE_COLUMNS, build_stationary_feature_frame, compute_atr
from strategy.primary_trend import PrimaryStrategyConfig


@dataclass
class BarrierState:
    """Mutable barrier execution state shared between Triple-Barrier labeling and backtesting."""

    side: int
    lots: float
    entry_price: float
    initial_sl_dist: float
    current_sl: float
    tp_price: float
    last_bar_ts: datetime
    favorable_peak: float
    bars_held: int = 0
    swap_eur: float = 0.0
    is_be: bool = False
    is_trailing: bool = False


def effective_spread_price(spec: SymbolContractSpec, bar_spread_pts: int) -> float:
    """Return effective Ask-Bid spread in price units from max(spec.spread_points, bar_spread)."""
    pts = max(int(spec.spread_points), max(0, int(bar_spread_pts)))
    return pts * spec.point


def step_barrier_state_bar(
    pos: BarrierState,
    spec: SymbolContractSpec,
    cost_model: CostModel,
    cfg: PrimaryStrategyConfig,
    bar_ts: datetime,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    curr_atr: float,
    spread_price: float,
    charge_swap: bool,
) -> Optional[Tuple[float, str]]:
    """
    Advance a single open position across one H4 bar with strict conservative precedence:
      1. Swap rollover accounting (if already held from previous bar)
      2. SL / BE / Trailing Stop check
      3. Take-Profit check
      4. Vertical Time-Stop Barrier check
      5. At bar close (only if surviving): update MFE, Break-Even lock, and Chandelier Trailing SL
    """
    if charge_swap:
        swap_delta = cost_model.compute_rollover_swap_between(
            spec=spec,
            side=pos.side,
            lots=pos.lots,
            prev_time=pos.last_bar_ts,
            curr_time=bar_ts,
        )
        pos.swap_eur += swap_delta
    pos.last_bar_ts = bar_ts
    pos.bars_held += 1

    sl_reason = "TRAIL" if pos.is_trailing else ("BE" if pos.is_be else "SL")

    if pos.side > 0:
        if bar_low <= pos.current_sl:
            return pos.current_sl, sl_reason
        if bar_high >= pos.tp_price:
            return pos.tp_price, "TP"
        if pos.bars_held >= cfg.vertical_barrier_bars_h4:
            return bar_close, "TIME_STOP"

        pos.favorable_peak = max(pos.favorable_peak, bar_high)
        excursion_r = (pos.favorable_peak - pos.entry_price) / max(pos.initial_sl_dist, 1e-9)

        if excursion_r >= cfg.breakeven_trigger_r:
            be_lock_sl = pos.entry_price + (cfg.breakeven_lock_r * pos.initial_sl_dist)
            if be_lock_sl > pos.current_sl:
                pos.current_sl = be_lock_sl
            pos.is_be = True

        eff_atr = curr_atr if curr_atr > 0 else (pos.initial_sl_dist / max(cfg.sl_atr_mult, 1e-9))
        if cfg.trailing_activation_r > 0 and excursion_r >= cfg.trailing_activation_r and eff_atr > 0:
            trail_sl = pos.favorable_peak - (cfg.trailing_atr_mult * eff_atr)
            if trail_sl > pos.current_sl:
                pos.current_sl = trail_sl
                pos.is_trailing = True
                pos.is_be = True
    else:
        ask_high = bar_high + spread_price
        ask_low = bar_low + spread_price
        if ask_high >= pos.current_sl:
            return pos.current_sl, sl_reason
        if ask_low <= pos.tp_price:
            return pos.tp_price, "TP"
        if pos.bars_held >= cfg.vertical_barrier_bars_h4:
            return (bar_close + spread_price), "TIME_STOP"

        pos.favorable_peak = min(pos.favorable_peak, ask_low)
        excursion_r = (pos.entry_price - pos.favorable_peak) / max(pos.initial_sl_dist, 1e-9)

        if excursion_r >= cfg.breakeven_trigger_r:
            be_lock_sl = pos.entry_price - (cfg.breakeven_lock_r * pos.initial_sl_dist)
            if be_lock_sl < pos.current_sl:
                pos.current_sl = be_lock_sl
            pos.is_be = True

        eff_atr = curr_atr if curr_atr > 0 else (pos.initial_sl_dist / max(cfg.sl_atr_mult, 1e-9))
        if cfg.trailing_activation_r > 0 and excursion_r >= cfg.trailing_activation_r and eff_atr > 0:
            trail_sl = pos.favorable_peak + (cfg.trailing_atr_mult * eff_atr)
            if trail_sl < pos.current_sl:
                pos.current_sl = trail_sl
                pos.is_trailing = True
                pos.is_be = True

    return None


def compute_net_carry_to_risk_series(
    df_h4_index: pd.DatetimeIndex,
    sig_df: pd.DataFrame,
    spec: SymbolContractSpec,
    cost_model: CostModel,
    cfg: PrimaryStrategyConfig,
    reference_lots: float = 0.01,
) -> pd.Series:
    """
    Compute projected net carry over `cfg.vertical_barrier_bars_h4` divided by gross 1R risk
    strictly at signal close time (for bars with non-zero primary signal).
    """
    carry_ratios = pd.Series(0.0, index=df_h4_index, dtype=float)
    active_mask = sig_df["side"] != 0
    if not active_mask.any():
        return carry_ratios

    for ts in sig_df.index[active_mask]:
        row = sig_df.loc[ts]
        side_val = int(row["side"])
        sl_dist = float(row["sl_distance_price"])
        if sl_dist <= 0.0 or ts not in carry_ratios.index:
            continue
        gross_1r = cost_model.calculate_sl_risk_eur(spec, reference_lots, sl_dist)
        if gross_1r <= 0.0:
            continue
        est_entry_utc = ts.to_pydatetime() + pd.Timedelta(hours=4)
        proj_swap = cost_model.project_holding_swap_eur(
            spec=spec,
            side=side_val,
            lots=reference_lots,
            entry_time_utc=est_entry_utc,
            holding_bars_h4=cfg.vertical_barrier_bars_h4,
        )
        carry_ratios.loc[ts] = proj_swap / gross_1r

    return carry_ratios


def build_symbol_feature_matrix(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    sig_df: pd.DataFrame,
    spec: SymbolContractSpec,
    cost_model: CostModel,
    cfg: PrimaryStrategyConfig,
    reference_lots: float = 0.01,
) -> pd.DataFrame:
    """
    Single-source-of-truth Layer 2 feature builder shared between Triple-Barrier training,
    OOS backtesting, and live H4 close inference.
    """
    if "cs_long_spread" in sig_df.columns:
        cs_series = sig_df["cs_long_spread"]
    else:
        cs_series = sig_df["cs_spread"] * sig_df["side"]

    carry_series = compute_net_carry_to_risk_series(
        df_h4_index=df_h4.index,
        sig_df=sig_df,
        spec=spec,
        cost_model=cost_model,
        cfg=cfg,
        reference_lots=reference_lots,
    )
    return build_stationary_feature_frame(
        df_h4=df_h4,
        df_d1=df_d1,
        cs_momentum_spread_series=cs_series,
        net_carry_to_risk_series=carry_series,
        side_series=sig_df["side"],
    )


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for path-dependent Triple-Barrier Meta-Label generation."""

    reference_capital_eur: float = settings.initial_capital_eur
    reference_lots: float = 0.01
    filter_risk_and_swap_ineligible: bool = True
    drop_unresolved_end_of_sample: bool = True


class TripleBarrierLabeler:
    """
    Generates Meta-Labeling supervised datasets aligned bit-for-bit with PortfolioBacktestEngine:
      - Candidate event triggered on H4 closed bar t
      - Enters strictly at bar t + 1 Open (Ask for Long, Bid for Short)
      - Tracks exact Bid/Ask barriers, Wednesday 3x swap rollovers, BE@1.10R, Trailing@1.50R, TimeStop@36
      - Binary label y = 1 iff net realized_r > 0 (net of spread, commission, and accumulated swap)
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        strategy_cfg: Optional[PrimaryStrategyConfig] = None,
        barrier_cfg: Optional[TripleBarrierConfig] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.strategy_cfg = strategy_cfg or PrimaryStrategyConfig()
        self.barrier_cfg = barrier_cfg or TripleBarrierConfig()

    def label_symbol_events(
        self,
        symbol: str,
        df_h4: pd.DataFrame,
        df_d1: pd.DataFrame,
        sig_df: pd.DataFrame,
        spec: SymbolContractSpec,
    ) -> pd.DataFrame:
        """Label every eligible Layer 1 candidate signal for one symbol."""
        feat_df = build_symbol_feature_matrix(
            df_h4=df_h4,
            df_d1=df_d1,
            sig_df=sig_df,
            spec=spec,
            cost_model=self.cost_model,
            cfg=self.strategy_cfg,
            reference_lots=self.barrier_cfg.reference_lots,
        )

        atr_arr = compute_atr(df_h4, period=self.strategy_cfg.atr_h4_period).fillna(0.0).to_numpy()
        times = df_h4.index.to_pydatetime()
        opens = df_h4["open"].to_numpy(dtype=float)
        highs = df_h4["high"].to_numpy(dtype=float)
        lows = df_h4["low"].to_numpy(dtype=float)
        closes = df_h4["close"].to_numpy(dtype=float)
        spreads = (
            df_h4["spread"].to_numpy(dtype=int)
            if "spread" in df_h4.columns
            else np.zeros(len(df_h4), dtype=int)
        )

        sides = sig_df["side"].reindex(df_h4.index).fillna(0).to_numpy(dtype=int)
        sl_dists = sig_df["sl_distance_price"].reindex(df_h4.index).fillna(0.0).to_numpy(dtype=float)
        tp_dists = sig_df["tp_distance_price"].reindex(df_h4.index).fillna(0.0).to_numpy(dtype=float)
        kers = (
            sig_df["ker_d1"].reindex(df_h4.index).fillna(0.0).to_numpy(dtype=float)
            if "ker_d1" in sig_df.columns
            else np.zeros(len(df_h4), dtype=float)
        )
        trigs = (
            sig_df["trigger_type"].reindex(df_h4.index).fillna("NONE").astype(str).to_numpy()
            if "trigger_type" in sig_df.columns
            else np.array(["NONE"] * len(df_h4), dtype=object)
        )
        feat_matrix = feat_df[FEATURE_COLUMNS].to_numpy(dtype=float)

        max_unit_risk = self.barrier_cfg.reference_capital_eur * settings.max_risk_per_trade_pct
        lots = self.barrier_cfg.reference_lots
        n_bars = len(df_h4)
        rows: List[Dict[str, object]] = []

        for idx in range(n_bars - 1):
            side = int(sides[idx])
            if side == 0:
                continue

            sl_dist = float(sl_dists[idx])
            tp_dist = float(tp_dists[idx])
            if sl_dist <= 0.0 or tp_dist <= 0.0:
                continue

            entry_idx = idx + 1
            entry_time = times[entry_idx]
            swap_cap = (
                self.strategy_cfg.max_swap_strong_trend_ratio
                if kers[idx] >= self.strategy_cfg.ker_strong_trend_threshold
                else self.strategy_cfg.max_swap_normal_ratio
            )
            friction = self.cost_model.evaluate_trade_friction(
                spec=spec,
                side=side,
                lots=lots,
                sl_distance_price=sl_dist,
                entry_time_utc=entry_time,
                holding_bars_h4=self.strategy_cfg.vertical_barrier_bars_h4,
                max_swap_ratio_override=swap_cap,
            )
            is_risk_swap_eligible = (
                not friction.is_hostile_swap_vetoed and friction.gross_1r_risk_eur <= max_unit_risk
            )
            if self.barrier_cfg.filter_risk_and_swap_ineligible and not is_risk_swap_eligible:
                continue

            sp_entry = effective_spread_price(spec, int(spreads[entry_idx]))
            entry_px = (opens[entry_idx] + sp_entry) if side > 0 else opens[entry_idx]
            sl_px = entry_px - (side * sl_dist)
            tp_px = entry_px + (side * tp_dist)

            pos = BarrierState(
                side=side,
                lots=lots,
                entry_price=entry_px,
                initial_sl_dist=sl_dist,
                current_sl=sl_px,
                tp_price=tp_px,
                last_bar_ts=entry_time,
                favorable_peak=entry_px,
            )

            exit_time: Optional[datetime] = None
            exit_px: float = entry_px
            exit_reason: str = "END_OF_SAMPLE"

            for step_idx in range(entry_idx, n_bars):
                sp_step = effective_spread_price(spec, int(spreads[step_idx]))
                res = step_barrier_state_bar(
                    pos=pos,
                    spec=spec,
                    cost_model=self.cost_model,
                    cfg=self.strategy_cfg,
                    bar_ts=times[step_idx],
                    bar_high=float(highs[step_idx]),
                    bar_low=float(lows[step_idx]),
                    bar_close=float(closes[step_idx]),
                    curr_atr=float(atr_arr[step_idx]),
                    spread_price=sp_step,
                    charge_swap=(step_idx > entry_idx),
                )
                if res is not None:
                    exit_px, exit_reason = res
                    exit_time = times[step_idx]
                    break

            if exit_time is None:
                if self.barrier_cfg.drop_unresolved_end_of_sample:
                    continue
                last_idx = n_bars - 1
                exit_time = times[last_idx]
                sp_last = effective_spread_price(spec, int(spreads[last_idx]))
                exit_px = closes[last_idx] if side > 0 else (closes[last_idx] + sp_last)

            price_diff = (exit_px - pos.entry_price) * side
            val_per_unit = self.cost_model.value_per_price_unit_for_lots(
                spec=spec,
                lots=lots,
                for_loss=(price_diff < 0.0),
            )
            pnl_after_spread = price_diff * val_per_unit
            net_pnl_eur = pnl_after_spread - friction.commission_roundtrip_eur + pos.swap_eur
            realized_r = net_pnl_eur / friction.gross_1r_risk_eur if friction.gross_1r_risk_eur > 0 else 0.0

            record: Dict[str, object] = {
                "symbol": symbol,
                "signal_time": pd.Timestamp(times[idx]).tz_convert("UTC"),
                "entry_time": pd.Timestamp(entry_time).tz_convert("UTC"),
                "exit_time": pd.Timestamp(exit_time).tz_convert("UTC"),
                "side": side,
                "trigger_type": str(trigs[idx]),
                "entry_price": float(pos.entry_price),
                "exit_price": float(exit_px),
                "bars_held": int(pos.bars_held),
                "exit_reason": exit_reason,
                "gross_1r_eur": round(friction.gross_1r_risk_eur, 4),
                "spread_cost_eur": round(friction.spread_cost_eur, 4),
                "commission_eur": round(friction.commission_roundtrip_eur, 4),
                "swap_eur": round(pos.swap_eur, 4),
                "net_pnl_eur": round(net_pnl_eur, 4),
                "realized_r": round(realized_r, 4),
                "meta_label": int(realized_r > 0.0),
                "is_risk_swap_eligible": bool(is_risk_swap_eligible),
                "is_resolved_label": bool(exit_reason != "END_OF_SAMPLE"),
            }
            f_vals = feat_matrix[idx]
            for col_idx, col_name in enumerate(FEATURE_COLUMNS):
                val = float(f_vals[col_idx])
                record[col_name] = 0.0 if np.isnan(val) else val
            rows.append(record)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def label_basket(
        self,
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
        d1_bars_by_symbol: Dict[str, pd.DataFrame],
        signals_by_symbol: Dict[str, pd.DataFrame],
        specs: Dict[str, SymbolContractSpec],
    ) -> pd.DataFrame:
        """Generate unified multi-symbol Triple-Barrier event table sorted strictly chronologically."""
        frames: List[pd.DataFrame] = []
        for sym in sorted(h4_bars_by_symbol.keys()):
            sym_df = self.label_symbol_events(
                symbol=sym,
                df_h4=h4_bars_by_symbol[sym],
                df_d1=d1_bars_by_symbol[sym],
                sig_df=signals_by_symbol[sym],
                spec=specs[sym],
            )
            if not sym_df.empty:
                frames.append(sym_df)

        if not frames:
            return pd.DataFrame()

        events_df = pd.concat(frames, axis=0, ignore_index=True)
        events_df = events_df.sort_values(["signal_time", "symbol"], ascending=[True, True]).reset_index(drop=True)
        return events_df

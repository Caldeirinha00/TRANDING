from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from strategy.features import (
    align_d1_to_h4_strict_causal,
    compute_atr,
    compute_kaufman_efficiency_ratio,
)


@dataclass(frozen=True)
class PrimaryStrategyConfig:
    """
    Canonical invariant configuration for Layer 1 deterministic white-box signals.
    Supports modes:
      - "DONCHIAN_DUAL_MOM" : H4 Donchian(20) Breakout + D1 Dual Momentum
      - "BB_RANGE_REV"      : H4 Bollinger(20, 2.2s) Reversion in Sideways D1 Regime
      - "BB_TREND_DIP"      : H4 Bollinger(20, 2.0s) Pullback Rejection with D1 Trend
      - "D1_TREND_CARRY"    : D1 Donchian(20) + Dual Momentum + Carry (multi-week hold)
      - "DUAL_REGIME"       : Orthogonal combination (D1_TREND_CARRY | BB_TREND_DIP | BB_RANGE_REV)
    """

    mode: str = "BB_TREND_DIP"
    ema_fast_d1: int = 20
    ema_slow_d1: int = 50
    ker_d1_period: int = 10
    min_ker_d1: float = 0.25
    max_range_ker_d1: float = 0.28
    cs_momentum_lookback_d1: int = 20
    min_cs_currency_spread: float = 0.55
    donchian_h4_period: int = 20
    donchian_d1_period: int = 20
    bb_h4_period: int = 20
    bb_std_range: float = 2.2
    bb_std_dip: float = 2.0
    ema_h4_period: int = 20
    atr_h4_period: int = 14
    max_extension_atr: float = 2.8
    min_bar_close_location: float = 0.60
    sl_atr_mult: float = 2.4
    tp_atr_mult: float = 4.2
    breakeven_trigger_r: float = 1.10
    breakeven_lock_r: float = 0.10
    trailing_activation_r: float = 1.50
    trailing_atr_mult: float = 2.0
    vertical_barrier_bars_h4: int = 36
    # Ablation & swap regime parameters
    use_legacy_ema_pullback: bool = False
    use_legacy_d1_filter: bool = False
    require_close_location: bool = True
    max_swap_normal_ratio: float = 0.04
    max_swap_strong_trend_ratio: float = 0.10
    ker_strong_trend_threshold: float = 0.45


def compute_cross_sectional_currency_strength(
    d1_bars_by_symbol: Dict[str, pd.DataFrame],
    lookback_days: int = 20,
) -> Dict[str, pd.Series]:
    """
    Compute causal cross-sectional Currency Relative Strength on closed D1 bars.
    Pair momentum = 20d price change normalized by ATR14_D1 * sqrt(20), decomposed
    into individual currency scores and standardized cross-sectionally (Z-score).
    """
    pair_norm_mom: Dict[str, pd.Series] = {}
    currencies: set[str] = set()

    for sym, df_d1 in d1_bars_by_symbol.items():
        if len(sym) < 6 or df_d1.empty:
            continue
        base, quote = sym[:3].upper(), sym[3:6].upper()
        currencies.update([base, quote])

        atr_d1 = compute_atr(df_d1, period=14).replace(0.0, np.nan)
        ret_lookback = df_d1["close"].astype(float) - df_d1["close"].astype(float).shift(lookback_days)
        pair_norm_mom[sym] = (ret_lookback / (atr_d1 * np.sqrt(lookback_days))).fillna(0.0)

    if not pair_norm_mom:
        return {}

    common_index = sorted(set().union(*(s.index for s in pair_norm_mom.values())))
    ccy_scores: Dict[str, pd.Series] = {
        ccy: pd.Series(0.0, index=common_index, dtype=float) for ccy in currencies
    }
    ccy_counts: Dict[str, int] = {ccy: 0 for ccy in currencies}

    for sym, series in pair_norm_mom.items():
        base, quote = sym[:3].upper(), sym[3:6].upper()
        aligned = series.reindex(common_index).ffill().fillna(0.0)
        ccy_scores[base] = ccy_scores[base] + aligned
        ccy_counts[base] += 1
        ccy_scores[quote] = ccy_scores[quote] - aligned
        ccy_counts[quote] += 1

    for ccy in currencies:
        if ccy_counts[ccy] > 0:
            ccy_scores[ccy] = ccy_scores[ccy] / float(ccy_counts[ccy])

    ccy_df = pd.DataFrame(ccy_scores, index=common_index)
    row_mean = ccy_df.mean(axis=1)
    row_std = ccy_df.std(axis=1).replace(0.0, 1.0)
    ccy_z = ccy_df.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)

    cs_spread_by_symbol: Dict[str, pd.Series] = {}
    for sym, df_d1 in d1_bars_by_symbol.items():
        base, quote = sym[:3].upper(), sym[3:6].upper()
        spread_d1 = (ccy_z[base] - ccy_z[quote]).reindex(df_d1.index).ffill().fillna(0.0)
        cs_spread_by_symbol[sym] = spread_d1

    return cs_spread_by_symbol


class PrimaryTrendStrategy:
    """
    Layer 1 Deterministic White-Box Model:
    Evaluates Trend Following (D1/H4) and Bollinger Band Mean Reversion / Pullback theses
    strictly causally on closed bars.
    """

    def __init__(self, config: Optional[PrimaryStrategyConfig] = None) -> None:
        self.cfg = config or PrimaryStrategyConfig()

    def generate_signals_for_symbol(
        self,
        symbol: str,
        df_h4: pd.DataFrame,
        df_d1: pd.DataFrame,
        cs_spread_d1: pd.Series,
    ) -> pd.DataFrame:
        # --- 1. Structural D1 Indicators (strictly on closed D1 bars) ---
        d1 = pd.DataFrame(index=df_d1.index)
        d1_close = df_d1["close"].astype(float)
        d1_high = df_d1["high"].astype(float)
        d1_low = df_d1["low"].astype(float)

        d1["d1_close"] = d1_close
        d1["ema20_d1"] = d1_close.ewm(span=self.cfg.ema_fast_d1, min_periods=self.cfg.ema_fast_d1, adjust=False).mean()
        d1["ema50_d1"] = d1_close.ewm(span=self.cfg.ema_slow_d1, min_periods=self.cfg.ema_slow_d1, adjust=False).mean()
        d1["ema50_slope_d1"] = d1["ema50_d1"].diff(1 if self.cfg.use_legacy_d1_filter else 3)
        d1["ts_mom_20_d1"] = d1_close - d1_close.shift(self.cfg.cs_momentum_lookback_d1)
        d1["ker_d1"] = compute_kaufman_efficiency_ratio(d1_close, window=self.cfg.ker_d1_period)
        d1["cs_spread_d1"] = cs_spread_d1.reindex(d1.index).ffill().fillna(0.0)
        d1["d1_bar_id"] = np.arange(len(d1), dtype=np.int64)

        d1_donchian_high_prev = d1_high.shift(1).rolling(
            window=self.cfg.donchian_d1_period, min_periods=self.cfg.donchian_d1_period
        ).max()
        d1_donchian_low_prev = d1_low.shift(1).rolling(
            window=self.cfg.donchian_d1_period, min_periods=self.cfg.donchian_d1_period
        ).min()
        d1_prev_close = d1_close.shift(1)
        d1["d1_bo_long"] = (d1_close > d1_donchian_high_prev) & (d1_prev_close <= d1_donchian_high_prev.shift(1))
        d1["d1_bo_short"] = (d1_close < d1_donchian_low_prev) & (d1_prev_close >= d1_donchian_low_prev.shift(1))

        d1_ready = d1["ema50_d1"].notna() & d1["ema50_slope_d1"].notna()
        d1["macro_trend_up"] = d1_ready & (d1_close > d1["ema50_d1"]) & (d1["ema50_slope_d1"] > 0.0)
        d1["macro_trend_down"] = d1_ready & (d1_close < d1["ema50_d1"]) & (d1["ema50_slope_d1"] < 0.0)

        if self.cfg.use_legacy_d1_filter:
            d1["macro_strong_up"] = d1["macro_trend_up"] & (d1["cs_spread_d1"] > 0.0)
            d1["macro_strong_down"] = d1["macro_trend_down"] & (d1["cs_spread_d1"] < 0.0)
        else:
            d1["macro_strong_up"] = (
                d1["macro_trend_up"]
                & (d1_close > d1["ema20_d1"])
                & (d1["ema20_d1"] > d1["ema50_d1"])
                & (d1["ts_mom_20_d1"] > 0.0)
                & (d1["ker_d1"] >= self.cfg.min_ker_d1)
                & (d1["cs_spread_d1"] >= self.cfg.min_cs_currency_spread)
            )
            d1["macro_strong_down"] = (
                d1["macro_trend_down"]
                & (d1_close < d1["ema20_d1"])
                & (d1["ema20_d1"] < d1["ema50_d1"])
                & (d1["ts_mom_20_d1"] < 0.0)
                & (d1["ker_d1"] >= self.cfg.min_ker_d1)
                & (d1["cs_spread_d1"] <= -self.cfg.min_cs_currency_spread)
            )

        d1["macro_range_regime"] = d1_ready & (d1["ker_d1"] <= self.cfg.max_range_ker_d1) & (
            d1["cs_spread_d1"].abs() <= 1.15
        )

        # --- 2. Strict Causal D1 -> H4 Close-Time Alignment ---
        d1_cols = [
            "d1_bar_id",
            "macro_trend_up",
            "macro_trend_down",
            "macro_strong_up",
            "macro_strong_down",
            "macro_range_regime",
            "d1_bo_long",
            "d1_bo_short",
            "ker_d1",
            "cs_spread_d1",
        ]
        aligned_d1 = align_d1_to_h4_strict_causal(df_h4.index, d1, d1_cols)
        h4 = df_h4.copy()
        for bcol in [
            "macro_trend_up",
            "macro_trend_down",
            "macro_strong_up",
            "macro_strong_down",
            "macro_range_regime",
            "d1_bo_long",
            "d1_bo_short",
        ]:
            h4[bcol] = aligned_d1[bcol].fillna(False).astype(bool)
        h4["ker_d1"] = aligned_d1["ker_d1"].fillna(0.0).astype(float)
        h4["cs_spread_d1"] = aligned_d1["cs_spread_d1"].fillna(0.0).astype(float)
        d1_bar_id = aligned_d1["d1_bar_id"].fillna(-1).astype("int64")
        is_d1_close_h4 = (d1_bar_id > 0) & (d1_bar_id != d1_bar_id.shift(1))

        # --- 3. Operational H4 Indicators ---
        h4_open = h4["open"].astype(float)
        h4_high = h4["high"].astype(float)
        h4_low = h4["low"].astype(float)
        h4_close = h4["close"].astype(float)

        atr14 = compute_atr(h4, period=self.cfg.atr_h4_period)
        ema20_h4 = h4_close.ewm(span=self.cfg.ema_h4_period, min_periods=self.cfg.ema_h4_period, adjust=False).mean()
        donchian_high_prev = h4_high.shift(1).rolling(
            window=self.cfg.donchian_h4_period, min_periods=self.cfg.donchian_h4_period
        ).max()
        donchian_low_prev = h4_low.shift(1).rolling(
            window=self.cfg.donchian_h4_period, min_periods=self.cfg.donchian_h4_period
        ).min()

        bb_mid = h4_close.rolling(window=self.cfg.bb_h4_period, min_periods=self.cfg.bb_h4_period).mean()
        bb_std = h4_close.rolling(window=self.cfg.bb_h4_period, min_periods=self.cfg.bb_h4_period).std(ddof=0)
        bb_upper_range = bb_mid + self.cfg.bb_std_range * bb_std
        bb_lower_range = bb_mid - self.cfg.bb_std_range * bb_std
        bb_upper_dip = bb_mid + self.cfg.bb_std_dip * bb_std
        bb_lower_dip = bb_mid - self.cfg.bb_std_dip * bb_std

        bar_range = (h4_high - h4_low).replace(0.0, np.nan)
        close_loc = ((h4_close - h4_low) / bar_range).fillna(0.5)
        extension_atr = ((h4_close - ema20_h4).abs() / atr14.replace(0.0, np.nan)).fillna(99.0)
        prev_close = h4_close.shift(1)

        # --- Trigger A: Donchian H4 Dual Momentum (+ optional legacy pullback for Config A ablation) ---
        fresh_bo_long = (h4_close > donchian_high_prev) & (prev_close <= donchian_high_prev.shift(1))
        fresh_bo_short = (h4_close < donchian_low_prev) & (prev_close >= donchian_low_prev.shift(1))
        if self.cfg.require_close_location:
            bo_l = fresh_bo_long & (close_loc >= self.cfg.min_bar_close_location) & (extension_atr <= self.cfg.max_extension_atr)
            bo_s = fresh_bo_short & (close_loc <= (1.0 - self.cfg.min_bar_close_location)) & (extension_atr <= self.cfg.max_extension_atr)
        else:
            bo_l, bo_s = fresh_bo_long, fresh_bo_short

        if self.cfg.use_legacy_ema_pullback:
            touched_l = (h4_low <= ema20_h4) | (h4_low.shift(1) <= ema20_h4.shift(1))
            touched_s = (h4_high >= ema20_h4) | (h4_high.shift(1) >= ema20_h4.shift(1))
            pb_l = touched_l & (h4_close > ema20_h4) & (h4_close > h4_open) & (h4_close > h4_high.shift(1)) & ((h4_close - ema20_h4) <= 1.5 * atr14)
            pb_s = touched_s & (h4_close < ema20_h4) & (h4_close < h4_open) & (h4_close < h4_low.shift(1)) & ((ema20_h4 - h4_close) <= 1.5 * atr14)
            donchian_long = h4["macro_strong_up"] & (bo_l | pb_l) & atr14.gt(0.0)
            donchian_short = h4["macro_strong_down"] & (bo_s | pb_s) & atr14.gt(0.0)
        else:
            donchian_long = h4["macro_strong_up"] & bo_l & atr14.gt(0.0)
            donchian_short = h4["macro_strong_down"] & bo_s & atr14.gt(0.0)

        # --- Trigger B: H1_BB_RANGE_REV (Bollinger 20, 2.2s mean reversion in sideways D1 regime) ---
        bb_range_long = (
            h4["macro_range_regime"]
            & (prev_close < bb_lower_range.shift(1))
            & (h4_close > bb_lower_range)
            & (h4_close > h4_open)
            & (close_loc >= 0.55)
            & atr14.gt(0.0)
        )
        bb_range_short = (
            h4["macro_range_regime"]
            & (prev_close > bb_upper_range.shift(1))
            & (h4_close < bb_upper_range)
            & (h4_close < h4_open)
            & (close_loc <= 0.45)
            & atr14.gt(0.0)
        )

        # --- Trigger C: H2_BB_TREND_DIP (Bollinger 20, 2.0s pullback rejection in D1 trend) ---
        bb_dip_long = (
            h4["macro_trend_up"]
            & (h4["cs_spread_d1"] >= 0.15)
            & ((h4_low.shift(1) <= bb_lower_dip.shift(1)) | (prev_close <= bb_lower_dip.shift(1)))
            & (h4_close > bb_lower_dip)
            & (h4_close > h4_open)
            & (close_loc >= 0.60)
            & atr14.gt(0.0)
        )
        bb_dip_short = (
            h4["macro_trend_down"]
            & (h4["cs_spread_d1"] <= -0.15)
            & ((h4_high.shift(1) >= bb_upper_dip.shift(1)) | (prev_close >= bb_upper_dip.shift(1)))
            & (h4_close < bb_upper_dip)
            & (h4_close < h4_open)
            & (close_loc <= 0.40)
            & atr14.gt(0.0)
        )

        # --- Trigger D: H3_D1_TREND_CARRY (Evaluated once exactly when D1 bar closes) ---
        d1_trend_long = is_d1_close_h4 & h4["macro_strong_up"] & h4["d1_bo_long"] & atr14.gt(0.0)
        d1_trend_short = is_d1_close_h4 & h4["macro_strong_down"] & h4["d1_bo_short"] & atr14.gt(0.0)

        mode = self.cfg.mode.upper()
        if mode == "DONCHIAN_DUAL_MOM":
            sig_long, sig_short = donchian_long, donchian_short
            trig_label = "DONCHIAN_DUAL_MOM"
        elif mode == "BB_RANGE_REV":
            sig_long, sig_short = bb_range_long, bb_range_short
            trig_label = "BB_RANGE_REV"
        elif mode == "BB_TREND_DIP":
            sig_long, sig_short = bb_dip_long, bb_dip_short
            trig_label = "BB_TREND_DIP"
        elif mode == "D1_TREND_CARRY":
            sig_long, sig_short = d1_trend_long, d1_trend_short
            trig_label = "D1_TREND_CARRY"
        elif mode == "DUAL_REGIME":
            sig_long = d1_trend_long | bb_dip_long | bb_range_long
            sig_short = d1_trend_short | bb_dip_short | bb_range_short
            trig_label = "DUAL_REGIME"
        else:
            raise ValueError(f"Unknown strategy mode: {self.cfg.mode}")

        side = np.where(sig_long, 1, np.where(sig_short, -1, 0)).astype(int)
        if mode == "DUAL_REGIME":
            trigger_type = np.where(
                d1_trend_long | d1_trend_short,
                "D1_TREND_CARRY",
                np.where(
                    bb_dip_long | bb_dip_short,
                    "BB_TREND_DIP",
                    np.where(bb_range_long | bb_range_short, "BB_RANGE_REV", "NONE"),
                ),
            )
        else:
            trigger_type = np.where(side != 0, trig_label, "NONE")

        cs_directional = np.where(side == -1, -h4["cs_spread_d1"], h4["cs_spread_d1"])

        return pd.DataFrame(
            {
                "side": side,
                "atr14_h4": atr14.fillna(0.0).astype(float),
                "sl_distance_price": (self.cfg.sl_atr_mult * atr14).fillna(0.0).astype(float),
                "tp_distance_price": (self.cfg.tp_atr_mult * atr14).fillna(0.0).astype(float),
                "cs_spread": cs_directional.astype(float),
                "cs_long_spread": h4["cs_spread_d1"].astype(float),
                "ker_d1": h4["ker_d1"].astype(float),
                "trigger_type": trigger_type,
            },
            index=df_h4.index,
        )

    def generate_signals_for_basket(
        self,
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
        d1_bars_by_symbol: Dict[str, pd.DataFrame],
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """Generate Layer 1 signals and causal cross-sectional currency spreads for the full basket."""
        cs_d1_map = compute_cross_sectional_currency_strength(
            d1_bars_by_symbol=d1_bars_by_symbol,
            lookback_days=self.cfg.cs_momentum_lookback_d1,
        )

        signals_by_symbol: Dict[str, pd.DataFrame] = {}
        cs_h4_by_symbol: Dict[str, pd.Series] = {}

        for sym, df_h4 in h4_bars_by_symbol.items():
            sig_df = self.generate_signals_for_symbol(
                symbol=sym,
                df_h4=df_h4,
                df_d1=d1_bars_by_symbol[sym],
                cs_spread_d1=cs_d1_map[sym],
            )
            signals_by_symbol[sym] = sig_df
            cs_h4_by_symbol[sym] = sig_df["cs_long_spread"]

        return signals_by_symbol, cs_h4_by_symbol

"""
Live vs. Batch Feature & Signal Parity Auditor (Fase 3).

Verifies zero-discrepancy mathematical parity (`<= 1e-9`) between online H4
bar-close vectors emitted into QuestDB (`feature_audit_log`) by `H4OrderManager`
and offline batch recalculation via `PrimaryTrendStrategy` +
`strategy.triple_barrier.build_symbol_feature_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from config.settings import settings
from core.mt5_connector import SymbolContractSpec
from core.storage import FeatureAuditRecord, StorageClient
from data.cost_model import CostModel
from execution.mt5_adapter import build_fallback_contract_spec
from strategy.features import FEATURE_COLUMNS
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy
from strategy.triple_barrier import build_symbol_feature_matrix

SIGNAL_NUMERIC_COLUMNS = (
    "atr14_h4",
    "sl_distance_price",
    "tp_distance_price",
    "cs_spread",
)


@dataclass(frozen=True)
class ParityMismatch:
    ts: datetime
    symbol: str
    field: str
    live_value: str
    batch_value: str
    abs_diff: float


@dataclass
class ParityReport:
    passed: bool
    tolerance: float
    records_checked: int
    symbols_checked: int
    missing_bars: int
    signal_mismatches: int
    max_abs_diff: float
    max_signal_diff: float
    feature_max_diffs: Dict[str, float] = field(default_factory=dict)
    mismatches: List[ParityMismatch] = field(default_factory=list)
    summary: str = ""


class ParityChecker:
    """
    Compares persisted live H4 audit records against full-batch recomputation.
    """

    def __init__(
        self,
        strategy: Optional[PrimaryTrendStrategy] = None,
        cost_model: Optional[CostModel] = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.strategy = strategy or PrimaryTrendStrategy(PrimaryStrategyConfig())
        self.cost_model = cost_model or CostModel()
        self.tolerance = float(tolerance)

    @staticmethod
    def _records_to_dataframe(
        audit_records: Union[Sequence[FeatureAuditRecord], pd.DataFrame],
    ) -> pd.DataFrame:
        if isinstance(audit_records, pd.DataFrame):
            df = audit_records.copy()
        else:
            if not audit_records:
                return pd.DataFrame()
            df = pd.DataFrame([r.__dict__ for r in audit_records])

        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    @staticmethod
    def _hydrate_specs_from_storage(
        storage: StorageClient,
        symbols: Sequence[str],
        cost_model: CostModel,
    ) -> Dict[str, SymbolContractSpec]:
        valid_keys = {f.name for f in fields(SymbolContractSpec)}
        specs: Dict[str, SymbolContractSpec] = {}
        for sym in symbols:
            kv = storage.get_runtime_state(f"parity_spec:{sym}")
            if isinstance(kv, dict) and valid_keys.issubset(kv.keys()):
                filtered = {k: kv[k] for k in valid_keys}
                specs[sym] = SymbolContractSpec(**filtered)
            else:
                specs[sym] = build_fallback_contract_spec(sym)
        return specs

    def check_records(
        self,
        audit_records: Union[Sequence[FeatureAuditRecord], pd.DataFrame],
        h4_bars_by_symbol: Dict[str, pd.DataFrame],
        d1_bars_by_symbol: Dict[str, pd.DataFrame],
        specs_by_symbol: Optional[Dict[str, SymbolContractSpec]] = None,
        reference_lots: float = settings.fixed_pass_through_lots,
    ) -> ParityReport:
        audit_df = self._records_to_dataframe(audit_records)
        if audit_df.empty:
            return ParityReport(
                passed=False,
                tolerance=self.tolerance,
                records_checked=0,
                symbols_checked=0,
                missing_bars=0,
                signal_mismatches=0,
                max_abs_diff=0.0,
                max_signal_diff=0.0,
                summary="PARITY_NO_AUDIT_RECORDS",
            )

        specs: Dict[str, SymbolContractSpec] = {}
        for sym in h4_bars_by_symbol:
            if specs_by_symbol and sym in specs_by_symbol:
                specs[sym] = specs_by_symbol[sym]
            else:
                specs[sym] = build_fallback_contract_spec(sym)

        signals_by_symbol, _ = self.strategy.generate_signals_for_basket(
            h4_bars_by_symbol=h4_bars_by_symbol,
            d1_bars_by_symbol=d1_bars_by_symbol,
        )

        features_by_symbol: Dict[str, pd.DataFrame] = {}
        for sym, df_h4 in h4_bars_by_symbol.items():
            if sym not in d1_bars_by_symbol or sym not in signals_by_symbol:
                continue
            features_by_symbol[sym] = build_symbol_feature_matrix(
                df_h4=df_h4,
                df_d1=d1_bars_by_symbol[sym],
                sig_df=signals_by_symbol[sym],
                spec=specs[sym],
                cost_model=self.cost_model,
                cfg=self.strategy.cfg,
                reference_lots=reference_lots,
            )

        feature_max_diffs: Dict[str, float] = {col: 0.0 for col in FEATURE_COLUMNS}
        max_abs_diff = 0.0
        max_signal_diff = 0.0
        missing_bars = 0
        signal_mismatches = 0
        mismatches: List[ParityMismatch] = []
        symbols_seen: set[str] = set()

        for _, row in audit_df.iterrows():
            sym = str(row["symbol"])
            ts = pd.Timestamp(row["ts"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            symbols_seen.add(sym)

            if (
                sym not in signals_by_symbol
                or sym not in features_by_symbol
                or ts not in signals_by_symbol[sym].index
                or ts not in features_by_symbol[sym].index
            ):
                missing_bars += 1
                mismatches.append(
                    ParityMismatch(
                        ts=ts.to_pydatetime(),
                        symbol=sym,
                        field="OHLCV_BAR",
                        live_value="PRESENT_IN_AUDIT",
                        batch_value="MISSING_IN_OHLCV",
                        abs_diff=float("inf"),
                    )
                )
                continue

            batch_sig = signals_by_symbol[sym].loc[ts]
            batch_feat = features_by_symbol[sym].loc[ts]

            live_side = int(row["side"])
            batch_side = int(batch_sig["side"])
            live_trig = str(row["trigger_type"])
            batch_trig = str(batch_sig["trigger_type"])

            if live_side != batch_side or live_trig != batch_trig:
                signal_mismatches += 1
                mismatches.append(
                    ParityMismatch(
                        ts=ts.to_pydatetime(),
                        symbol=sym,
                        field="side/trigger_type",
                        live_value=f"{live_side}:{live_trig}",
                        batch_value=f"{batch_side}:{batch_trig}",
                        abs_diff=float(abs(live_side - batch_side)),
                    )
                )

            for col in SIGNAL_NUMERIC_COLUMNS:
                diff = abs(float(row[col]) - float(batch_sig[col]))
                if diff > max_signal_diff:
                    max_signal_diff = diff
                if diff > self.tolerance:
                    signal_mismatches += 1
                    mismatches.append(
                        ParityMismatch(
                            ts=ts.to_pydatetime(),
                            symbol=sym,
                            field=col,
                            live_value=f"{float(row[col]):.12g}",
                            batch_value=f"{float(batch_sig[col]):.12g}",
                            abs_diff=diff,
                        )
                    )

            for col in FEATURE_COLUMNS:
                diff = abs(float(row[col]) - float(batch_feat[col]))
                if diff > feature_max_diffs[col]:
                    feature_max_diffs[col] = diff
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if diff > self.tolerance:
                    mismatches.append(
                        ParityMismatch(
                            ts=ts.to_pydatetime(),
                            symbol=sym,
                            field=col,
                            live_value=f"{float(row[col]):.12g}",
                            batch_value=f"{float(batch_feat[col]):.12g}",
                            abs_diff=diff,
                        )
                    )

        records_checked = len(audit_df)
        passed = (
            records_checked > 0
            and missing_bars == 0
            and signal_mismatches == 0
            and max_abs_diff <= self.tolerance
            and max_signal_diff <= self.tolerance
        )
        summary = (
            f"PARITY_{'PASS' if passed else 'FAIL'}: "
            f"records={records_checked}, symbols={len(symbols_seen)}, "
            f"max_feature_diff={max_abs_diff:.3e}, max_signal_diff={max_signal_diff:.3e}, "
            f"signal_mismatches={signal_mismatches}, missing_bars={missing_bars}"
        )
        return ParityReport(
            passed=passed,
            tolerance=self.tolerance,
            records_checked=records_checked,
            symbols_checked=len(symbols_seen),
            missing_bars=missing_bars,
            signal_mismatches=signal_mismatches,
            max_abs_diff=max_abs_diff,
            max_signal_diff=max_signal_diff,
            feature_max_diffs=feature_max_diffs,
            mismatches=mismatches,
            summary=summary,
        )

    def verify_from_storage(
        self,
        storage: StorageClient,
        symbols: Optional[Sequence[str]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        specs_by_symbol: Optional[Dict[str, SymbolContractSpec]] = None,
        limit: int = 5000,
        reference_lots: float = settings.fixed_pass_through_lots,
    ) -> ParityReport:
        target_symbols = list(symbols or settings.approved_symbols)
        audit_df = storage.load_feature_audit_records(
            symbol=None, start_ts=start_ts, end_ts=end_ts, limit=limit
        )
        if not audit_df.empty and symbols is not None:
            audit_df = audit_df[audit_df["symbol"].isin(target_symbols)]

        h4_bars: Dict[str, pd.DataFrame] = {}
        d1_bars: Dict[str, pd.DataFrame] = {}
        for sym in target_symbols:
            df_h4 = storage.load_ohlcv_dataframe("ohlcv_h4", sym)
            df_d1 = storage.load_ohlcv_dataframe("ohlcv_d1", sym)
            if end_ts is not None:
                cutoff_ts = pd.Timestamp(end_ts)
                if cutoff_ts.tzinfo is None:
                    cutoff_ts = cutoff_ts.tz_localize("UTC")
                if not df_h4.empty:
                    df_h4 = df_h4[df_h4.index <= cutoff_ts]
                # Keep D1 bars aligned with H4 close boundary
                d1_cutoff = cutoff_ts + pd.Timedelta(hours=4) - pd.Timedelta(days=1)
                if not df_d1.empty:
                    df_d1 = df_d1[df_d1.index <= d1_cutoff]
            if not df_h4.empty and not df_d1.empty:
                h4_bars[sym] = df_h4
                d1_bars[sym] = df_d1

        resolved_specs = specs_by_symbol or self._hydrate_specs_from_storage(
            storage, target_symbols, self.cost_model
        )
        return self.check_records(
            audit_records=audit_df,
            h4_bars_by_symbol=h4_bars,
            d1_bars_by_symbol=d1_bars,
            specs_by_symbol=resolved_specs,
            reference_lots=reference_lots,
        )

from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient, SymbolCostSnapshot
from data.cost_model import CostModel
from strategy.features import compute_atr


def main() -> None:
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Não foi possível conectar ao terminal MetaTrader 5. Verifica se está aberto.")
        sys.exit(1)

    cost_model = CostModel()
    storage = StorageClient()
    symbols_to_audit = settings.approved_symbols + ["XAUUSD"]
    now_utc = datetime.now(timezone.utc)
    snapshots: List[SymbolCostSnapshot] = []

    print("\n" + "=" * 126)
    print(
        f" AUDITORIA CONTRATUAL PEPPERSTONE MT5 | CONTA BASE: {settings.initial_capital_eur:.2f} EUR "
        f"| TETO 1R: 10.00 EUR (1.0%)"
    )
    print("=" * 126)
    header = (
        f"{'SÍMBOLO':<8} | {'MARGEM 0.01':>11} | {'ATR14(H4)':>10} | {'SL (2xATR)':>10} | "
        f"{'RISCO 0.01':>10} | {'FRICÇÃO %':>9} | {'SWAP_M':>6} | {'SWAP 4D (L/S)':>16} | {'STATUS 1.000€':<14}"
    )
    print(header)
    print("-" * 126)

    for sym in symbols_to_audit:
        spec = connector.get_symbol_spec(sym)
        if spec is None:
            print(f"{sym:<8} | Símbolo indisponível no Market Watch da corretora.")
            continue

        df_h4 = connector.fetch_closed_bars(sym, "H4", count=150)
        atr_price = float(compute_atr(df_h4, period=14).iloc[-1])
        sl_price_dist = 2.0 * atr_price

        atr_pips = atr_price / spec.pip_size
        sl_pips = sl_price_dist / spec.pip_size

        friction_long = cost_model.evaluate_trade_friction(
            spec=spec,
            side=+1,
            lots=0.01,
            sl_distance_price=sl_price_dist,
            entry_time_utc=now_utc,
            holding_bars_h4=24,
        )
        friction_short = cost_model.evaluate_trade_friction(
            spec=spec,
            side=-1,
            lots=0.01,
            sl_distance_price=sl_price_dist,
            entry_time_utc=now_utc,
            holding_bars_h4=24,
        )

        risk_eur = friction_long.gross_1r_risk_eur
        status = "APROVADO (<=1%)" if (risk_eur <= 10.0 and sym != "XAUUSD") else "BLOQUEADO (>1%)"

        print(
            f"{sym:<8} | {spec.margin_initial_001_eur:>9.2f} € | {atr_pips:>7.1f} p | {sl_pips:>7.1f} p | "
            f"{risk_eur:>8.2f} € | {friction_long.friction_to_risk_ratio*100:>8.2f}% | "
            f"{spec.swap_mode:>6d} | {friction_long.expected_swap_eur:>+6.2f}€/{friction_short.expected_swap_eur:>+6.2f}€ | "
            f"{status:<14}"
        )

        snapshots.append(
            SymbolCostSnapshot(
                symbol=sym,
                status=status,
                margin_001_eur=spec.margin_initial_001_eur,
                atr14_h4_pips=round(atr_pips, 2),
                sl_pips=round(sl_pips, 2),
                risk_1r_eur=risk_eur,
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

    print("=" * 126)

    qdb_ok, qdb_msg = storage.write_symbol_cost_snapshots(snapshots)
    df_ok = storage.ping_dragonfly()
    print(f"[QuestDB] {qdb_msg}")
    print(f"[DragonflyDB] {'ONLINE' if df_ok else 'OFFLINE/UNREACHABLE'} ({storage.dragonfly_host}:{storage.dragonfly_port})\n")

    connector.shutdown()


if __name__ == "__main__":
    main()

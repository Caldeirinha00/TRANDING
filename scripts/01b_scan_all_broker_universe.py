from datetime import datetime, timezone
import sys
from pathlib import Path
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mt5_connector import MT5Connector
from data.cost_model import CostModel
from strategy.features import compute_atr

CANDIDATE_UNIVERSE = [
    # Current 8 pairs
    "EURUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "GBPUSD", "AUDJPY",
    # Liquid G10 crosses (macro dispersion & carry)
    "EURJPY", "EURGBP", "CADJPY", "CHFJPY", "NZDJPY", "EURCAD", "EURCHF", "AUDCAD", "AUDNZD", "GBPJPY",
    # Indices, Energy & Metals (check if any micro contract fits <= 8.50 EUR risk)
    "US500", "NAS100", "GER40", "UK100", "JPN225", "XTIUSD", "XBRUSD", "XAGUSD", "XAUUSD",
]


def main() -> None:
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Falha ao conectar ao MT5.")
        sys.exit(1)

    cost_model = CostModel(commission_per_lot_roundtrip_eur=0.0)
    now_utc = datetime.now(timezone.utc)

    print("\n" + "=" * 134)
    print(" SCANNER DE UNIVERSO PEPPERSTONE (1.000 EUR) | CRITÉRIOS: RISCO(MIN_LOTE) <= 8.50€ | FRICÇÃO <= 2.2% | MARGEM <= 45€")
    print("=" * 134)
    print(
        f"{'SÍMBOLO':<10} | {'CLASSE':<9} | {'MIN_LOT':>7} | {'MARGEM MIN':>10} | {'RISCO 2xATR':>11} | "
        f"{'SPREAD €':>9} | {'FRICÇÃO %':>9} | {'SWAP 4D (L/S)':>16} | {'VEREDITO'}"
    )
    print("-" * 134)

    eligible_symbols = []

    for sym in CANDIDATE_UNIVERSE:
        if not mt5.symbol_select(sym, True):
            continue
        info = mt5.symbol_info(sym)
        if info is None:
            continue

        spec = connector.get_symbol_spec(sym)
        if spec is None:
            continue

        try:
            df_h4 = connector.fetch_closed_bars(sym, "H4", count=200)
        except Exception:
            continue

        if len(df_h4) < 50:
            continue

        min_lot = max(0.01, float(spec.volume_min))
        tick = mt5.symbol_info_tick(sym)
        margin_min = (
            mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, sym, min_lot, tick.ask)
            if tick is not None
            else None
        )
        margin_eur = float(margin_min) if margin_min is not None else spec.margin_initial_001_eur * (min_lot / 0.01)

        atr_price = float(compute_atr(df_h4, period=14).iloc[-1])
        sl_dist = 2.0 * atr_price

        f_long = cost_model.evaluate_trade_friction(spec, +1, min_lot, sl_dist, now_utc, holding_bars_h4=24)
        f_short = cost_model.evaluate_trade_friction(spec, -1, min_lot, sl_dist, now_utc, holding_bars_h4=24)

        risk_eur = f_long.gross_1r_risk_eur
        fric_pct = f_long.friction_to_risk_ratio * 100.0

        is_viable = (
            (min_lot <= 0.01 + 1e-9)
            and (0.50 <= risk_eur <= 8.50)
            and (fric_pct <= 2.20)
            and (margin_eur <= 45.0)
        )
        verdict = "ELEGÍVEL" if is_viable else "REJEITADO"
        if is_viable:
            eligible_symbols.append((sym, risk_eur, fric_pct, margin_eur))

        asset_class = (
            "FOREX"
            if len(sym) == 6 and sym.isalpha() and sym not in ("XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD")
            else "CFD/OTHER"
        )
        print(
            f"{sym:<10} | {asset_class:<9} | {min_lot:>7.2f} | {margin_eur:>8.2f} € | {risk_eur:>9.2f} € | "
            f"{f_long.spread_cost_eur:>7.3f} € | {fric_pct:>8.2f}% | "
            f"{f_long.expected_swap_eur:>+6.2f}€/{f_short.expected_swap_eur:>+6.2f}€ | {verdict}"
        )

    print("=" * 134)
    print(f"Total de símbolos matematicamente compatíveis com 1.000 EUR: {len(eligible_symbols)}")
    print("Lista Elegível:", [s[0] for s in eligible_symbols])
    connector.shutdown()


if __name__ == "__main__":
    main()

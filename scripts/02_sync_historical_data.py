from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from data.ingestion import HistoricalDataIngestor


def main() -> None:
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Falha ao ligar ao terminal MetaTrader 5.")
        sys.exit(1)

    storage = StorageClient()
    ingestor = HistoricalDataIngestor(connector=connector, storage=storage)

    print("A sincronizar historico fechado H4 e D1 (2015-presente) dos 8 pares para a QuestDB...")
    counts, _ = ingestor.sync_all_approved_symbols(
        start_utc=datetime(2015, 1, 1, tzinfo=timezone.utc)
    )

    h4_stats = storage.count_ohlcv_rows("ohlcv_h4")
    d1_stats = storage.count_ohlcv_rows("ohlcv_d1")

    print("\n=== SINCRONIZAÇÃO QUESTDB CONCLUÍDA ===")
    for sym, c in counts.items():
        h4_cnt, h4_min, h4_max = h4_stats.get(sym, (0, "-", "-"))
        d1_cnt, d1_min, d1_max = d1_stats.get(sym, (0, "-", "-"))
        print(
            f"{sym:<7} | MT5 sync H4={c['H4']:>5} D1={c['D1']:>4} | "
            f"QDB ohlcv_h4={h4_cnt:>5} [{h4_min[:10]}..{h4_max[:10]}] | "
            f"ohlcv_d1={d1_cnt:>4} [{d1_min[:10]}..{d1_max[:10]}]"
        )

    connector.shutdown()


if __name__ == "__main__":
    main()

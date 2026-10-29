import asyncio
import logging
import time
from questdb.ingress import Sender, TimestampNanos
from core.config import CONFIG
from core.mcp_client import mcp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_candles_loop():
    symbols = CONFIG.symbols_list
    logger.info(f"Starting candle collector for symbols: {symbols}")
    
    # We poll every 1 minute for new M1 candles
    # Actually, we can poll every 10 seconds to ensure we don't miss the close
    poll_interval = 10
    last_processed_time = {sym: 0 for sym in symbols}
    
    questdb_conf = f'tcp::addr={CONFIG.QUESTDB_HOST}:{CONFIG.QUESTDB_PORT};'
    
    try:
        while True:
            for symbol in symbols:
                # Fetch last 2 candles to get the recently closed one
                candles = await mcp_client.get_candles_latest(symbol, count=2)
                
                if candles and len(candles) > 0:
                    # The last candle might still be open, so we process the previous ones
                    # or process all and QuestDB handles updates (if configured with deduplication)
                    # For simplicity, we just send all to QuestDB. ILP appends by default.
                    
                    try:
                        def _send_to_questdb():
                            with Sender.from_conf(questdb_conf) as sender:
                                sent_any = False
                                for candle in candles:
                                    ts_ns = int(candle['time'].timestamp() * 1_000_000_000)
                                    if ts_ns <= last_processed_time[symbol]:
                                        continue
                                    sender.row(
                                        'candles',
                                        symbols={'symbol': symbol},
                                        columns={
                                            'open': float(candle['open']),
                                            'high': float(candle['high']),
                                            'low': float(candle['low']),
                                            'close': float(candle['close']),
                                            'tick_volume': int(candle['tick_volume']),
                                            'spread': int(candle['spread']),
                                            'real_volume': int(candle['real_volume'])
                                        },
                                        at=TimestampNanos(ts_ns)
                                    )
                                    last_processed_time[symbol] = ts_ns
                                    sent_any = True
                                if sent_any:
                                    sender.flush()
                            return last_processed_time[symbol]
                        
                        last_processed_time[symbol] = await asyncio.to_thread(_send_to_questdb)
                        logger.debug(f"Flushed candles for {symbol} to QuestDB")
                    except Exception as e:
                        logger.error(f"Failed to send to QuestDB: {e}")
                        
            await asyncio.sleep(poll_interval)
            
    except asyncio.CancelledError:
        logger.info("Candle collector loop cancelled")
    except Exception as e:
        logger.error(f"Candle collector error: {e}")
    finally:
        mcp_client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(collect_candles_loop())
    except KeyboardInterrupt:
        logger.info("Candle collector stopped by user")

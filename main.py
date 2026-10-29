"""
AI Trading System — Main Entry Point
Starts all subsystems: MT5 connection, risk daemon, and trading loop.
Usage: python main.py
"""
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime

from core.config import CONFIG
from core.mcp_client import mcp_client
from agents.orchestrator import orchestrator
from risk.risk_daemon import risk_daemon_loop
from risk.trade_monitor import trade_monitor
from agents.ipc_watcher import IPCWatcher
from redis.asyncio import Redis
import time

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Configure logging ONCE here at the entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("main")


_lock_file = None

def acquire_single_instance_lock() -> bool:
    global _lock_file
    from pathlib import Path
    lock_path = Path(__file__).resolve().parent / "logs" / "ai_trading.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _lock_file = open(lock_path, "a+")
        if sys.platform == "win32":
            import msvcrt
            _lock_file.seek(0)
            msvcrt.locking(_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        _lock_file.truncate(0)
        _lock_file.write(f"{os.getpid()}\n")
        _lock_file.flush()
        return True
    except (IOError, OSError):
        return False

async def trading_loop():
    """Main trading cycle for non-Twin symbols. Twin symbols are executed on chart by MQL5 EA."""
    interval = CONFIG.ANALYSIS_INTERVAL_SECONDS
    symbols = [s for s in CONFIG.symbols_list if not (getattr(CONFIG, "TWIN_MODE", True) and s.upper() in getattr(CONFIG, "twin_symbols_list", ["EURUSD"]))]
    
    if not symbols:
        logger.info(f"Twin Mode Active: All configured symbols ({CONFIG.symbols_list}) are handled by on-chart MQL5 Twin EA. Python direct order loop idle.")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        return

    logger.info(f"Trading loop started for Python-managed symbols: {symbols}. Interval: {interval}s")
    
    try:
        while True:
            cycle_start = datetime.now()
            logger.info(f"=== NEW CYCLE at {cycle_start.strftime('%H:%M:%S')} ===")
            
            # Active Position Management (Phase 1)
            try:
                await orchestrator.manage_open_positions()
            except Exception as e:
                logger.error(f"Error in active management: {e}", exc_info=True)
                elapsed = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval - elapsed)
                logger.info(f"Skipping trade evaluation due to management error. Sleeping {sleep_time:.0f}s...")
                await asyncio.sleep(sleep_time)
                continue
                
            # New Trade Evaluation (Phase 2) - Em Paralelo para rapidez máxima
            async def evaluate_safely(sym):
                try:
                    res = await orchestrator.evaluate_and_trade(sym)
                    if res:
                        logger.info(f"TRADE PLACED for {sym}: {res}")
                    else:
                        logger.info(f"No trade for {sym} this cycle.")
                except Exception as e:
                    logger.error(f"Error evaluating {sym}: {e}", exc_info=True)
            
            await asyncio.gather(*(evaluate_safely(s) for s in symbols))
            
            elapsed = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, interval - elapsed)
            logger.info(f"Cycle complete in {elapsed:.1f}s. Sleeping {sleep_time:.0f}s...")
            await asyncio.sleep(sleep_time)
            
    except asyncio.CancelledError:
        logger.info("Trading loop cancelled.")

async def heartbeat_loop():
    """Emits a heartbeat to Redis every 5 seconds for the Watchdog."""
    redis = Redis.from_url(CONFIG.redis_url, decode_responses=True)
    try:
        while True:
            try:
                await redis.set("system:heartbeat", str(time.time()))
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    finally:
        await redis.aclose()


async def main():
    """Initialize all subsystems and run."""
    # Ensure only ONE instance of the AI Trading core is active
    if not acquire_single_instance_lock():
        logger.critical("ANOTHER INSTANCE OF AI_TRADING IS ALREADY RUNNING! Aborting to prevent race conditions.")
        return

    logger.info("=" * 60)
    logger.info("  AI TRADING SYSTEM v0.2.0")
    logger.info(f"  Server: {CONFIG.MT5_SERVER}")
    logger.info(f"  Account: {CONFIG.MT5_LOGIN}")
    logger.info(f"  Symbols: {CONFIG.symbols_list}")
    logger.info(f"  Twin Mode: {getattr(CONFIG, 'TWIN_MODE', True)} (Twin: {getattr(CONFIG, 'twin_symbols_list', ['EURUSD'])})")
    logger.info(f"  Cycle Interval: {CONFIG.ANALYSIS_INTERVAL_SECONDS}s")
    logger.info("=" * 60)
    
    # 0. Check System Lock
    redis = Redis.from_url(CONFIG.redis_url, decode_responses=True)
    try:
        if await redis.get("system:lock") == "WATCHDOG_ACTIVE":
            logger.critical("SYSTEM LOCKED: Watchdog is active. Please clear 'system:lock' in Redis to resume.")
            return
    except Exception as e:
        logger.error(f"Failed to check lock in Redis: {e}")
    finally:
        await redis.aclose()

    # 1. Connect to MT5
    if not mcp_client.connect():
        logger.critical("Failed to connect to MT5. Exiting.")
        return
    
    import MetaTrader5 as mt5
    
    acc = mt5.account_info()
    if acc and acc.balance < 50000:
        logger.info(f"Target Account Balance: {acc.balance} {acc.currency} | Equity: {acc.equity}")
    
    # 2. Start subsystems
    pipe_task = asyncio.create_task(orchestrator.pipe_server.handle_client_loop())
    ipc_watcher = IPCWatcher()
    ipc_task = asyncio.create_task(ipc_watcher.run())
    monitor_task = asyncio.create_task(trade_monitor.run())
    trade_task = asyncio.create_task(trading_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    # Check if RiskDaemon runs embedded or in separate standalone process
    run_embedded_risk = os.getenv("RUN_EMBEDDED_RISK", "1").strip().lower() not in ("0", "false", "no")
    tasks = [pipe_task, ipc_task, monitor_task, trade_task, heartbeat_task]
    risk_task = None
    
    if run_embedded_risk:
        risk_task = asyncio.create_task(risk_daemon_loop())
        tasks.append(risk_task)
        logger.info("Embedded Risk Daemon ACTIVE (1 Hz).")
    else:
        logger.info("Embedded Risk Daemon DISABLED via RUN_EMBEDDED_RISK=0 (Standalone Risk Daemon mode).")
    
    logger.info("All systems GO (Named Pipe IPC + File Fallback active). Press Ctrl+C to stop.")
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        pipe_task.cancel()
        ipc_task.cancel()
        if risk_task:
            risk_task.cancel()
        monitor_task.cancel()
        trade_task.cancel()
        heartbeat_task.cancel()
        mcp_client.disconnect()
        logger.info("System stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")

import psutil
import logging
import asyncio
from typing import Dict, Any

from core.dragonfly_bus import DragonflyBus
from core.mcp_client import mcp_client
from core.mt5_executor import mt5_call

logger = logging.getLogger(__name__)

async def trigger_kill_switch(reason: str, dragonfly_bus: DragonflyBus = None):
    """
    EMERGENCY: Triggers the system-level kill switch.
    1. Broadcasts a fatal alert to DragonflyDB.
    2. Forcefully kills the local MCP server (metatrader-mcp-server).
    3. Shuts down MT5 client connections.
    """
    logger.critical(f"KILL SWITCH TRIGGERED! Reason: {reason}")
    
    # 1. Broadcast fatal alert
    if dragonfly_bus:
        try:
            alert = {
                "level": "FATAL",
                "reason": reason,
                "action": "HALT_TRADING"
            }
            await dragonfly_bus.publish("system:risk", alert)
            logger.info("Fatal alert broadcasted to system:risk channel.")
        except Exception as e:
            logger.error(f"Failed to broadcast fatal alert: {e}")
            
    # 1.5. Close all open positions
    try:
        from core.mt5_connection import mt5
        from agents.trade_executor import trade_executor
        positions = await mt5_call(mt5.positions_get)
        if positions:
            logger.critical(f"Found {len(positions)} open positions. Force closing ALL in parallel...")
            
            close_tasks = [trade_executor.close_position(pos.ticket) for pos in positions]
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            
            failed_closes = []
            for pos, result in zip(positions, results):
                if isinstance(result, Exception):
                    logger.error(f"EXCEPTION while closing {pos.symbol} (Ticket {pos.ticket}): {result}", exc_info=result)
                    failed_closes.append(pos)
                elif result is False:
                    logger.error(f"FAILED to close {pos.symbol} (Ticket {pos.ticket}) - trade_executor returned False")
                    failed_closes.append(pos)
                    
            if failed_closes:
                logger.critical(f"CRITICAL: Failed to close {len(failed_closes)} positions! Manual intervention required.")
                if dragonfly_bus:
                    try:
                        await dragonfly_bus.publish("system:risk", {
                            "level": "FATAL",
                            "reason": f"Failed to close {len(failed_closes)} positions during kill switch",
                            "action": "MANUAL_INTERVENTION_REQUIRED"
                        })
                    except Exception:
                        pass
            else:
                logger.critical("All positions closed successfully.")
        else:
            logger.info("No open positions to close.")
    except Exception as e:
        logger.error(f"Failed to fetch or process positions during kill switch: {e}", exc_info=e)
            
    # 2. Kill the MCP Server process
    def _kill_mcp_sync():
        killed = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            # Look for the metatrader-mcp-server or generic python processes running the server
            cmd = proc.info.get('cmdline')
            if cmd and 'metatrader-mcp-server' in ' '.join(cmd):
                proc.kill()
                logger.critical(f"Killed MCP Server process (PID: {proc.info['pid']})")
                killed = True
        return killed
    
    try:
        mcp_killed = await asyncio.to_thread(_kill_mcp_sync)
        if not mcp_killed:
            logger.warning("MCP Server process not found, it might already be dead.")
    except Exception as e:
        logger.error(f"Failed to kill MCP Server: {e}")
        
    # 3. Shutdown local MT5 direct connection if it exists
    try:
        mcp_client.disconnect()
        logger.critical("Disconnected local MT5 client.")
    except Exception as e:
        logger.error(f"Failed to disconnect MT5: {e}")
        
    logger.critical("SYSTEM SECURED. Trading is halted.")

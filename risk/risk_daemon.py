import asyncio
import logging
from core.mt5_connection import mt5
from core.mt5_executor import mt5_call
from core.config import CONFIG
from core.dragonfly_bus import DragonflyBus
from risk.rules import RiskManager
from risk.kill_switch import trigger_kill_switch

logger = logging.getLogger(__name__)


async def risk_daemon_loop():
    """Continuous risk monitoring loop. Checks account health every second."""
    logger.info("Initializing Risk Daemon...")
    bus = DragonflyBus()
    risk_manager = RiskManager()
    
    # NOTE: We do NOT call mt5.initialize() here.
    # The main.py entry point already initializes MT5 once for the entire process.
    # Calling it again can cause random disconnections and mt5.shutdown() in finally
    # would kill MT5 for the entire trading system.

    try:
        await bus.connect()
    except Exception as e:
        logger.warning(f"Risk Daemon could not connect to DragonflyDB: {e}. Running without pub/sub.")
        bus = None

    logger.info("Risk Daemon actively monitoring...")

    try:
        while True:
            # 1. Fetch current account info
            account_info = await mt5_call(mt5.account_info)
            account_balance = 0.0
            
            if not account_info:
                logger.error("Failed to retrieve MT5 account info. Activating Circuit Breaker alert.")
                if bus:
                    await bus.publish("system:risk", {"type": "CIRCUIT_BREAKER", "reason": "MT5 connection lost"})
                    
                # Try fallback to Redis for position loss calculation
                start_balance_bytes = await risk_manager.redis.get("risk:start_of_day_balance") if risk_manager.redis else None
                if start_balance_bytes:
                    account_balance = float(start_balance_bytes)
                    logger.warning(f"Using Redis start_of_day_balance ({account_balance}) as fallback for loss calculation.")
                else:
                    logger.critical("CIRCUIT BREAKER: No MT5 connection AND no Redis fallback balance. Risk daemon blind.")
            else:
                account_dict = account_info._asdict()
                account_balance = account_dict.get("balance", 0.0)
                
                # 2. Evaluate Account Drawdown Risk
                if not await risk_manager.evaluate_account_risk(account_dict):
                    await trigger_kill_switch("Account Drawdown Limit Breached!", bus)
                    break
            
            # 3. Fetch and evaluate open positions
            positions = await mt5_call(mt5.positions_get)
            if positions is not None:
                pos_list = [p._asdict() for p in positions]
                num_pos = len(pos_list)
                if num_pos >= risk_manager.max_positions:
                    # Position limit is NOT an emergency — orchestrator already blocks new trades.
                    # Only log a warning, do NOT trigger kill switch.
                    logger.warning(f"Position limit reached ({num_pos}/{risk_manager.max_positions}). Orchestrator will block new trades.")
                
                # Enforce Single Trade Loss Limit
                if account_balance > 0:
                    for pos in pos_list:
                        net_profit = pos.get("profit", 0.0) + pos.get("swap", 0.0) + pos.get("commission", 0.0)
                        if net_profit < 0:
                            loss_pct = (abs(net_profit) / account_balance) * 100
                            if loss_pct >= risk_manager.max_single_loss:
                                ticket = pos.get("ticket")
                                logger.critical(f"SINGLE TRADE LOSS BREACH! Position {ticket} loss {loss_pct:.2f}% >= Limit {risk_manager.max_single_loss:.2f}%. FORCE CLOSING.")
                                try:
                                    from agents.trade_executor import trade_executor
                                    await trade_executor.close_position(ticket)
                                except Exception as e:
                                    logger.error(f"Failed to force close position {ticket}: {e}")
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Risk Daemon cancelled.")
    except Exception as e:
        logger.error(f"Risk Daemon crashed: {e}")
        try:
            await trigger_kill_switch(f"Risk Daemon Crashed: {e}", bus)
        except Exception:
            pass
    finally:
        if bus:
            await bus.disconnect()
        # Do NOT call mt5.shutdown() here — main.py owns the MT5 lifecycle


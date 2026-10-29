import asyncio
import logging
import sys
from pathlib import Path
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.mt5_executor import mt5_call
from core.config import CONFIG
from core.dragonfly_bus import DragonflyBus
from risk.rules import RiskManager
from risk.kill_switch import trigger_kill_switch

logger = logging.getLogger(__name__)


async def risk_daemon_loop(standalone: bool = False):
    """Continuous risk monitoring loop. Checks account health every second."""
    logger.info("Initializing Risk Daemon...")
    bus = DragonflyBus()
    risk_manager = RiskManager()

    try:
        await bus.connect()
    except Exception as e:
        logger.warning(f"Risk Daemon could not connect to DragonflyDB: {e}. Running without pub/sub.")
        bus = None

    logger.info("Risk Daemon actively monitoring (1 Hz)...")
    drawdown_breach_count = 0

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
                start_balance_bytes = await risk_manager.redis.get("risk:start_of_day_balance") if hasattr(risk_manager, "redis") and risk_manager.redis else None
                if start_balance_bytes:
                    account_balance = float(start_balance_bytes)
                    logger.warning(f"Using Redis start_of_day_balance ({account_balance}) as fallback for loss calculation.")
                else:
                    logger.critical("CIRCUIT BREAKER: No MT5 connection AND no Redis fallback balance. Risk daemon blind.")
            else:
                account_dict = account_info._asdict()
                account_balance = account_dict.get("balance", 0.0)

                # Regra de Ouro: Ignorar completamente qualquer conta de 1.000.000 EUR
                if account_balance > 50000:
                    await asyncio.sleep(1)
                    continue

                # 2. Evaluate Account Drawdown Risk (requires 3 consecutive readings to filter rollover spread widening)
                is_safe = await risk_manager.evaluate_account_risk(account_dict)
                if not is_safe:
                    drawdown_breach_count += 1
                    logger.warning(f"Account Drawdown breach detected! Reading {drawdown_breach_count}/3 before KillSwitch...")
                    if drawdown_breach_count >= 3:
                        await trigger_kill_switch("Account Drawdown Limit Breached (Confirmed 3 consecutive seconds)!", bus)
                        break
                else:
                    drawdown_breach_count = 0

            # 3. Fetch and evaluate open positions
            positions = await mt5_call(mt5.positions_get)
            if positions is not None:
                pos_list = [p._asdict() for p in positions]
                num_pos = len(pos_list)
                if num_pos >= risk_manager.max_positions:
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
        if standalone:
            mt5.shutdown()


if __name__ == "__main__":
    from core.mcp_client import mcp_client

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logger.info("Starting Risk Daemon in standalone process...")
    if not mcp_client.connect():
        logger.critical("Failed to connect to MT5 in standalone Risk Daemon. Exiting.")
        sys.exit(1)

    try:
        asyncio.run(risk_daemon_loop(standalone=True))
    except KeyboardInterrupt:
        logger.info("Standalone Risk Daemon stopped by user.")
    finally:
        mcp_client.disconnect()

import MetaTrader5 as mt5
import logging
from typing import Dict, Any, Optional
import asyncio
import aiohttp
from core.config import CONFIG
from core.mcp_client import mcp_client
from core.redis_broadcaster import broadcaster
from core.mt5_executor import mt5_call

logger = logging.getLogger(__name__)


def _get_filling_mode(symbol_info) -> int:
    """Determine the correct filling mode using the symbol's filling_mode bitmask."""
    fm = symbol_info.filling_mode
    # filling_mode is a bitmask: bit0 = FOK supported, bit1 = IOC supported
    if fm & 1:  # FOK supported
        return mt5.ORDER_FILLING_FOK
    elif fm & 2:  # IOC supported
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN





class TradeExecutor:
    """
    Deterministic Trade Executor.
    Strictly translates standardized JSON signals into actual MT5 market orders.
    NO LLMs are allowed in this layer to ensure complete safety and control.
    """
    
    def __init__(self):
        self.max_lot_size = CONFIG.MAX_LOT_SIZE
        
    async def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executes a validated trading signal.
        Returns trade details dict on success, None on failure.
        
        Expected signal format:
        {
            "action": "BUY" | "SELL",
            "symbol": "EURUSD",
            "lot_size": 0.1,
            "sl_distance": 0.0030,
            "tp_distance": 0.0060
        }
        """
        if not mcp_client.connected and not mcp_client.connect():
            logger.error("TradeExecutor cannot connect to MT5.")
            return None
            
        action = signal.get("action")
        symbol = signal.get("symbol")
        lot_size = float(signal.get("lot_size", 0.01))
        
        # 1. Validate inputs
        if action not in ("BUY", "SELL"):
            logger.error(f"Invalid action: {action}")
            return None
        
        if lot_size > self.max_lot_size:
            logger.warning(f"Lot size {lot_size} exceeds max {self.max_lot_size}. Clamping.")
            lot_size = self.max_lot_size
        if lot_size <= 0:
            logger.error("Lot size must be > 0")
            return None

        # 2. Get Symbol Info
        if not await mt5_call(mt5.symbol_select, symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
            
        symbol_info = await mt5_call(mt5.symbol_info, symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
                
        # 3. Get price
        tick = await mt5_call(mt5.symbol_info_tick, symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        price = tick.ask if action == "BUY" else tick.bid
        
        point = symbol_info.point if symbol_info.point > 0 else 0.00001
        spread_points = (tick.ask - tick.bid) / point
        
        if price <= 0:
            logger.error(f"Invalid price {price} for {symbol}. Market may be closed.")
            return None
        
        # 4. Calculate SL / TP using raw price distance
        sl_distance = float(signal.get("sl_distance", 0))
        if sl_distance <= 0:
            logger.error("Signal has no stop loss distance (sl_distance <= 0). Rejecting trade.")
            return None
            
        # Sanity check: if sl_distance is > 10% of price, LLM hallucinated pips instead of price points
        if sl_distance > price * 0.1:
            logger.error(f"sl_distance {sl_distance} is absurdly large for price {price}. Probable LLM pip hallucination. Rejecting.")
            return None
            
        tp_distance = float(signal.get("tp_distance", 0))
        if tp_distance > price * 0.5:
            logger.error(f"tp_distance {tp_distance} is absurdly large for price {price}. Probable LLM pip hallucination. Rejecting.")
            return None
        
        if action == "BUY":
            sl = round(price - sl_distance, symbol_info.digits)
            tp = round(price + tp_distance, symbol_info.digits) if tp_distance > 0 else 0.0
        else:
            sl = round(price + sl_distance, symbol_info.digits)
            tp = round(price - tp_distance, symbol_info.digits) if tp_distance > 0 else 0.0
        
        
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        filling_mode = _get_filling_mode(symbol_info)
            
        # 5. Prepare MT5 Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI Auto Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        # 6. Send Order
        # 6. Send Order with Retry Loop
        logger.info(f"Sending order: {action} {lot_size} {symbol} @ {price} (SL={sl}, TP={tp}) | Spread: {spread_points:.1f} pts")
        
        RECOVERABLE_RETCODES = {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS,
            mt5.TRADE_RETCODE_CONNECTION,
        }
        
        max_retries = 3
        backoff = 0.1
        result = None
        
        for attempt in range(max_retries):
            result = await mt5_call(mt5.order_send, request)
            
            if result is None:
                err = await mt5_call(mt5.last_error)
                logger.error(f"Order FAILED (attempt {attempt+1}): mt5.order_send returned None. Last error: {err}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 1.0)
                continue
                
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                break
                
            if result.retcode not in RECOVERABLE_RETCODES:
                logger.error(f"Order FAILED FATAL: retcode={result.retcode}, comment={result.comment}")
                return None
                
            logger.warning(f"Order retry {attempt+1}/{max_retries} due to retcode {result.retcode}. Waiting {backoff}s...")
            
            if result.retcode in (mt5.TRADE_RETCODE_PRICE_OFF, mt5.TRADE_RETCODE_PRICE_CHANGED):
                new_tick = await mt5_call(mt5.symbol_info_tick, symbol)
                if new_tick:
                    request["price"] = new_tick.ask if action == "BUY" else new_tick.bid
            
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 1.0)
            
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed after {max_retries} retries.")
            return None
            
        actual_price = result.price
        slippage_points = abs(actual_price - price) / point
        logger.info(f"Order SUCCESS! Ticket: {result.order} | Fill Price: {actual_price} | Slippage: {slippage_points:.1f} pts")
        
        asyncio.create_task(self._send_slippage_telemetry(symbol, action, slippage_points, spread_points))
            
        trade_details = {
            "ticket": result.order,
            "action": action,
            "symbol": symbol,
            "lot_size": lot_size,
            "price": price,
            "sl": sl,
            "tp": tp
        }
        logger.info(f"Order SUCCESS! Ticket: {result.order}")
        asyncio.create_task(broadcaster.send_trade_execution(
            symbol=symbol,
            action=action,
            lot_size=lot_size,
            price=price,
            ticket=result.order
        ))
        return trade_details

    async def close_position(self, ticket: int) -> bool:
        """
        Closes an open position by its ticket number.
        Returns True if successful, False otherwise.
        """
        if not mcp_client.connected and not mcp_client.connect():
            logger.error("TradeExecutor cannot connect to MT5.")
            return False
            
        position = await mt5_call(mt5.positions_get, ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found to close.")
            return False
            
        pos = position[0]
        symbol = pos.symbol
        action = "SELL" if pos.type == mt5.ORDER_TYPE_BUY else "BUY"
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = await mt5_call(mt5.symbol_info_tick, symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol} to close position.")
            return False
            
        price = tick.bid if action == "SELL" else tick.ask
        symbol_info = await mt5_call(mt5.symbol_info, symbol)
        filling_mode = _get_filling_mode(symbol_info)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI Auto Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        logger.info(f"Sending order to CLOSE position {ticket}: {action} {pos.volume} {symbol} @ {price}")
        
        RECOVERABLE_RETCODES = {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS,
            mt5.TRADE_RETCODE_CONNECTION,
        }
        
        max_retries = 3
        backoff = 0.1
        result = None
        
        for attempt in range(max_retries):
            result = await mt5_call(mt5.order_send, request)
            
            if result is None:
                logger.error(f"Close Order FAILED (attempt {attempt+1}): mt5.order_send returned None. Connection lost.")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 1.0)
                continue
                
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                break
                
            if result.retcode not in RECOVERABLE_RETCODES:
                logger.error(f"Close Order FAILED FATAL: retcode={result.retcode}, comment={result.comment}")
                return False
                
            logger.warning(f"Close Order retry {attempt+1}/{max_retries} due to retcode {result.retcode}. Waiting {backoff}s...")
            
            if result.retcode in (mt5.TRADE_RETCODE_PRICE_OFF, mt5.TRADE_RETCODE_PRICE_CHANGED):
                new_tick = await mt5_call(mt5.symbol_info_tick, symbol)
                if new_tick:
                    request["price"] = new_tick.bid if action == "SELL" else new_tick.ask
            
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 1.0)
            
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close Order failed after {max_retries} retries.")
            return False
            
        logger.info(f"Position {ticket} CLOSED successfully!")
        
        asyncio.create_task(broadcaster.send_trade_execution(
            symbol=symbol,
            action="CLOSE",
            lot_size=pos.volume,
            price=price,
            ticket=ticket
        ))
        
        return True

    async def _send_slippage_telemetry(self, symbol: str, action: str, slippage_points: float, spread_points: float):
        """Sends execution quality metrics to QuestDB."""
        payload = f"execution_quality,symbol={symbol} action=\"{action}\",slippage_pts={slippage_points},spread_pts={spread_points}"
        
        headers = {}
        if CONFIG.QUESTDB_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {CONFIG.QUESTDB_AUTH_TOKEN}"
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{CONFIG.QUESTDB_HOST}:9000/write", data=payload, headers=headers, timeout=2.0) as resp:
                    if resp.status not in (200, 204):
                        logger.warning(f"QuestDB returned {resp.status} for slippage telemetry: {await resp.text()}")
        except Exception as e:
            logger.debug(f"QuestDB slippage telemetry failed: {e}")

trade_executor = TradeExecutor()

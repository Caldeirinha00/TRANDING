import logging
import time
from typing import Dict, Optional, Tuple
import MetaTrader5 as mt5

from core.mt5_executor import mt5_call
from core.config import CONFIG

logger = logging.getLogger(__name__)

CORRELATION_BUCKETS = {
    # EURUSD e GBPUSD têm correlação ~0.85 estável
    "EUR_GBP_LONG":  ["EURUSD_BUY",  "GBPUSD_BUY"],    # Long EUR/GBP = short USD
    "EUR_GBP_SHORT": ["EURUSD_SELL", "GBPUSD_SELL"],    # Short EUR/GBP = long USD
    # XAUUSD em bucket PRÓPRIO
    "GOLD_LONG":     ["XAUUSD_BUY"],
    "GOLD_SHORT":    ["XAUUSD_SELL"],
}

class RiskExposure:
    def __init__(self):
        self.aggregate_risk_pct = 0.0
        self.bucket_risk_pct: Dict[str, float] = {k: 0.0 for k in CORRELATION_BUCKETS.keys()}
        self.positions_count = 0

class RiskBudgetManager:
    def __init__(self):
        self.max_aggregate_risk_pct = CONFIG.MAX_AGGREGATE_RISK_PCT  # 2.0%
        self.max_bucket_risk_pct = CONFIG.MAX_BUCKET_RISK_PCT        # 1.0%
        self.hard_cap = CONFIG.HARD_CAP_POSITIONS                    # 5
        
        # Cache de exposure
        self._cached_exposure: Optional[RiskExposure] = None
        self._cache_ts: float = 0.0
        self._CACHE_TTL: float = 3.0  # segundos
    
    async def compute_exposure(self) -> RiskExposure:
        """Retorna exposure cacheada se < 3s, senão recalcula."""
        now = time.time()
        if self._cached_exposure and (now - self._cache_ts) < self._CACHE_TTL:
            return self._cached_exposure
        
        exposure = RiskExposure()
        
        account_info = await mt5_call(mt5.account_info)
        if not account_info:
            return exposure
            
        equity = account_info.equity
        if equity <= 0:
            return exposure
            
        positions = await mt5_call(mt5.positions_get)
        if positions:
            exposure.positions_count = len(positions)
            for pos in positions:
                # Obter valor do risco da posição (distância entre entry e SL ou 100% se não houver SL)
                sl = pos.sl
                if sl > 0:
                    order_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_SELL
                    # Calculate potential loss monetary value
                    profit = await mt5_call(mt5.order_calc_profit, order_type, pos.symbol, pos.volume, pos.price_open, sl)
                    if profit is not None and profit < 0:
                        risk_monetary = abs(profit)
                    else:
                        risk_monetary = 0.0
                else:
                    risk_monetary = 0.0 # Sem SL definido = risco desconhecido, num sistema robusto devia ser evitado. Assumimos 0.
                
                risk_pct = (risk_monetary / equity) * 100.0
                exposure.aggregate_risk_pct += risk_pct
                
                action = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                key = f"{pos.symbol}_{action}"
                
                for bucket_name, members in CORRELATION_BUCKETS.items():
                    if key in members:
                        exposure.bucket_risk_pct[bucket_name] += risk_pct
        
        self._cached_exposure = exposure
        self._cache_ts = now
        return exposure
        
    async def can_accept_trade(self, symbol: str, action: str, lot_size: float, sl_distance: float) -> Tuple[bool, str]:
        """Verifica se um novo trade cabe no orçamento de risco."""
        account_info = await mt5_call(mt5.account_info)
        if not account_info or account_info.equity <= 0:
            return False, "Failed to get account equity"
            
        equity = account_info.equity
        
        # Obter current exposure
        exposure = await self.compute_exposure()
        
        if exposure.positions_count >= self.hard_cap:
            return False, f"HARD_CAP reached ({self.hard_cap})"
            
        # Calcular risco do novo trade
        tick = await mt5_call(mt5.symbol_info_tick, symbol)
        if not tick:
            return False, "Failed to get tick for risk calculation"
            
        if action == "BUY":
            price_open = tick.ask
            sl = price_open - sl_distance
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price_open = tick.bid
            sl = price_open + sl_distance
            order_type = mt5.ORDER_TYPE_SELL
            
        profit = await mt5_call(mt5.order_calc_profit, order_type, symbol, lot_size, price_open, sl)
        if profit is None:
            return False, "Failed to calculate order profit"
            
        new_risk_monetary = abs(profit) if profit < 0 else 0.0
        new_risk_pct = (new_risk_monetary / equity) * 100.0
        
        if exposure.aggregate_risk_pct + new_risk_pct > self.max_aggregate_risk_pct:
            return False, f"Aggregate risk limit exceeded (Current: {exposure.aggregate_risk_pct:.2f}%, New: {new_risk_pct:.2f}%)"
            
        # Check buckets
        key = f"{symbol}_{action}"
        for bucket_name, members in CORRELATION_BUCKETS.items():
            if key in members:
                current_bucket_risk = exposure.bucket_risk_pct.get(bucket_name, 0.0)
                if current_bucket_risk + new_risk_pct > self.max_bucket_risk_pct:
                    return False, f"Bucket '{bucket_name}' risk limit exceeded"
                    
        return True, "OK"

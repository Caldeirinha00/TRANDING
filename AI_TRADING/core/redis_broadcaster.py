import json
import logging
from redis.asyncio import Redis
from core.config import CONFIG

logger = logging.getLogger(__name__)

class RedisBroadcaster:
    """Handles publishing real-time messages to DragonflyDB/Redis for the Frontend Dashboard."""
    
    def __init__(self):
        self.redis_url = CONFIG.redis_url
        self._client = None
        
    async def _get_client(self) -> Redis:
        """Lazy initialization of the Redis async client."""
        if self._client is None:
            try:
                self._client = Redis.from_url(self.redis_url, decode_responses=True)
                await self._client.ping()
            except Exception as e:
                logger.error(f"Failed to connect to Redis broadcaster: {e}")
                self._client = None
                raise
        return self._client
        
    async def broadcast(self, channel: str, data: dict):
        """
        Publishes a JSON payload to a specific Redis channel.
        Fails silently to ensure trading is never interrupted if Dashboard is down.
        """
        try:
            client = await self._get_client()
            payload = json.dumps(data)
            await client.publish(channel, payload)
        except Exception as e:
            logger.debug(f"Redis Broadcast failed on {channel}: {e}")
            self._client = None # Reset client to force reconnect next time

    async def send_decision(self, symbol: str, action: str, confidence: int, reasoning: str, sl: float = 0.0, tp: float = 0.0):
        """Broadcasts CIO decisions."""
        await self.broadcast("ai_decisions", {
            "type": "CIO_DECISION",
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "sl_distance": sl,
            "tp_distance": tp
        })

    async def send_news_alert(self, symbol: str, sentiment: str, confidence: int, summary: str):
        """Broadcasts breaking news analysis."""
        await self.broadcast("news_sentiment", {
            "type": "NEWS_ALERT",
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "summary": summary
        })

    async def send_market_update(self, symbol: str, trend: str, confidence: int):
        """Broadcasts technical market analysis."""
        await self.broadcast("market_ticks", {
            "type": "TECH_ANALYSIS",
            "symbol": symbol,
            "trend": trend,
            "confidence": confidence
        })
        
    async def send_trade_execution(self, symbol: str, action: str, lot_size: float, price: float, ticket: int):
        """Broadcasts executed trades."""
        await self.broadcast("trade_executions", {
            "type": "EXECUTION",
            "symbol": symbol,
            "action": action,
            "lot_size": lot_size,
            "price": price,
            "ticket": ticket
        })
        
    async def cache_trade_context(self, ticket: int, context_summary: str, confidence: int = 0, setup_type: str = "UNKNOWN"):
        """
        Temporarily stores the reasoning, confidence and setup_type for a trade in Redis so the memory agent
        can fetch it when the trade closes. Key expires after 168h (7 days) to survive weekends.
        """
        try:
            client = await self._get_client()
            key = f"trade_context:{ticket}"
            data = json.dumps({
                "context": context_summary,
                "confidence": confidence,
                "setup_type": setup_type
            })
            await client.set(key, data, ex=604800) # 7 days expiry
        except Exception as e:
            logger.debug(f"Failed to cache context for ticket {ticket}: {e}")
            self._client = None # Reset client to force reconnect next time

broadcaster = RedisBroadcaster()

import asyncio
import json
import logging
import redis.asyncio as redis
from typing import Any, Callable, Dict

from core.config import CONFIG

logger = logging.getLogger(__name__)

class DragonflyBus:
    def __init__(self):
        self.redis_url = f"redis://{CONFIG.DRAGONFLY_HOST}:{CONFIG.DRAGONFLY_PORT}/0"
        self.redis = None
        self.pubsub = None

    async def connect(self):
        """Establish connection to DragonflyDB"""
        if not self.redis:
            try:
                self.redis = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                await self.redis.ping()
                logger.info(f"Connected to DragonflyDB at {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to DragonflyDB: {e}")
                self.redis = None
                raise

    async def disconnect(self):
        """Close connection to DragonflyDB"""
        if self.pubsub:
            await self.pubsub.aclose()
            self.pubsub = None
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            logger.info("Disconnected from DragonflyDB")

    async def publish(self, channel: str, message: Dict[str, Any]):
        """Publish a JSON message to a channel"""
        await self.connect()
        try:
            msg_str = json.dumps(message)
            await self.redis.publish(channel, msg_str)
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            try:
                await self.disconnect()
            except Exception:
                pass

    async def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], Any]):
        """Subscribe to a channel and process messages with a callback"""
        backoff = 5.0
        
        while True:
            try:
                await self.connect()
                if not self.pubsub:
                    self.pubsub = self.redis.pubsub()
                await self.pubsub.subscribe(channel)
                logger.info(f"Subscribed/Re-subscribed to channel: {channel}")
                
                # Reset backoff on successful connection
                backoff = 5.0
                
                async for message in self.pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode message from {channel}: {message['data']}")
                        except Exception as e:
                            logger.error(f"Error in callback for {channel}: {e}")
                            
            except asyncio.CancelledError:
                logger.info(f"Subscription to {channel} cancelled")
                if self.pubsub:
                    await self.pubsub.unsubscribe(channel)
                raise
            except Exception as e:
                logger.error(f"Redis connection lost in subscribe loop for {channel}: {e}")
                logger.info(f"Reconnecting in {backoff} seconds...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                
                # Force cleanup of old connection before retry
                try:
                    await self.disconnect()
                except Exception:
                    pass

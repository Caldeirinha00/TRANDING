import uuid
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timezone
from qdrant_client import models

from memory.embeddings import memory_db

logger = logging.getLogger(__name__)

class TradeJournal:
    def __init__(self):
        self.collection = memory_db.collection_name
        self.client = memory_db.client

    async def log_trade(self, context_summary: str, action: str, outcome: str, profit: float, symbol: str, setup_type: str = "UNKNOWN"):
        """
        Logs a completed trade into the Qdrant vector database asynchronously.
        
        Args:
            context_summary: A text description of the market conditions and news at the time.
            action: BUY or SELL
            outcome: WIN, LOSS or BREAKEVEN
            profit: Financial result
            symbol: The asset traded (e.g. EURUSD)
            setup_type: The technical setup type used (e.g. TREND_PULLBACK)
        """
        if not self.client:
            logger.error("Cannot log trade: Qdrant client not connected.")
            return
            
        # Force types for Qdrant compatibility
        try:
            profit_float = float(profit)
        except (ValueError, TypeError):
            profit_float = 0.0
            
        safe_context = str(context_summary)[:2000] if context_summary else ""
            
        doc_text = f"Action: {action} | Context: {safe_context}"
        
        metadata = {
            "action": action,
            "outcome": outcome,
            "profit": profit_float,
            "setup_type": setup_type,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        def _sync_add():
            self.client.add(
                collection_name=self.collection,
                documents=[doc_text],
                metadata=[metadata],
                ids=[uuid.uuid4().hex]
            )
            
        try:
            # Run the synchronous Qdrant call in a thread pool to avoid blocking the event loop
            await asyncio.to_thread(_sync_add)
            logger.info(f"Trade successfully logged to memory. Outcome: {outcome} ({profit})")
        except Exception as e:
            logger.error(f"Failed to log trade to memory: {e}")

    async def find_similar_context(self, current_context: str, symbol: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches Qdrant for similar past market conditions to learn from them asynchronously.
        """
        if not self.client:
            return []
            
        def _sync_query():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return self.client.query(
                    collection_name=self.collection,
                    query_text=current_context,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="symbol", match=models.MatchValue(value=symbol))]
                    ),
                    limit=limit
                )
            
        try:
            results = await asyncio.to_thread(_sync_query)
            
            past_trades = []
            for hit in results:
                past_trades.append({
                    "score": hit.score,
                    "document": hit.document,
                    "metadata": hit.metadata
                })
                
            return past_trades
            
        except Exception as e:
            logger.error(f"Failed to search similar trades: {e}")
            return []

trade_journal = TradeJournal()

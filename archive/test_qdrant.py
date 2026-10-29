import asyncio
from memory.embeddings import memory_db
from qdrant_client import models

client = memory_db.client
print("Collection exists:", client.collection_exists("trades_journal"))

try:
    print("Testing query_points with string...")
    res = client.query_points(
        collection_name="trades_journal",
        query="Symbol: XAGUSD, Market: BEARISH, News: NEUTRAL",
        limit=1
    )
    print("Success:", res)
except Exception as e:
    print("query_points string failed:", type(e), e)

try:
    print("Testing client.query with query_text...")
    res = client.query(
        collection_name="trades_journal",
        query_text="Symbol: XAGUSD, Market: BEARISH, News: NEUTRAL",
        limit=1
    )
    print("Success:", res)
except Exception as e:
    print("client.query failed:", type(e), e)

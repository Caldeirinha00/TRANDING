import asyncio
from memory.embeddings import memory_db
from qdrant_client import models

client = memory_db.client

try:
    print("Testing query_points with Document...")
    res = client.query_points(
        collection_name="trades_journal",
        query=models.Document(text="Symbol: XAGUSD", model="BAAI/bge-small-en-v1.5"),
        limit=1
    )
    print("Success:", res)
except Exception as e:
    print("query_points Document failed:", type(e), e)

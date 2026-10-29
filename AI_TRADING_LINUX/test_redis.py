import asyncio
import redis.asyncio as redis
from core.config import CONFIG

async def main():
    try:
        r = redis.from_url(CONFIG.redis_url, decode_responses=True)
        ping_res = await asyncio.wait_for(r.ping(), timeout=3.0)
        print("Ping:", ping_res)
        await r.close()
    except Exception as e:
        print("Error connecting to server:", e)

if __name__ == "__main__":
    asyncio.run(main())

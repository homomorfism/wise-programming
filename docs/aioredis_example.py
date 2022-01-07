import asyncio

import aioredis
import numpy as np


async def main():
    redis = aioredis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
    await redis.set("2134", np.asarray([1, 2, 3]).tobytes())
    val = await redis.get("2134")

    print(np.frombuffer(val.encode(), dtype=int))


if __name__ == '__main__':
    asyncio.run(main())

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def measure_time() -> AsyncGenerator[callable, None]:
    """
    Async context manager to measure elapsed time.
    Usage:
        async with measure_time() as elapsed:
            await some_async_func()
        print(elapsed())
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

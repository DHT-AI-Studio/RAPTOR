"""Keep the CUDA permit until a worker completes, even if its caller disconnects."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class GpuLock:
    """One permit shared by STT, TTS, diarization and cloning in one process."""

    def __init__(self) -> None:
        self.semaphore = asyncio.Semaphore(1)

    async def run(self, function: Callable[..., T], *args, **kwargs) -> T:
        """Run blocking inference off the event loop without overlapping GPU work."""
        from app.metrics import inf_voice_gpu_active_tasks
        
        await self.semaphore.acquire()
        inf_voice_gpu_active_tasks.inc()
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))

        def done(task):
            inf_voice_gpu_active_tasks.dec()
            self.semaphore.release()
            # Retrieve orphaned errors after request cancellation.
            if not task.cancelled():
                task.exception()

        worker.add_done_callback(done)
        return await asyncio.shield(worker)

"""
GPU serialization within the engine event loop.

Only one GPU-timing-sensitive subprocess (tuner or NCU) runs at a time
inside one engine process. Cross-process coordination is the API server's
responsibility (see the RunEntry registry in api/).
"""

import asyncio
import contextlib

_gpu_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def acquire_gpu_lock():
    """Async context manager: serializes GPU-sensitive subprocesses."""
    async with _gpu_lock:
        yield

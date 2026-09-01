"""Per-MSME async locks.

Two concurrent ``POST /healthcard/{id}/refresh`` calls would otherwise both score
the firm and race on the version row, producing two cards claiming to be version
N. With this lock the second caller waits and reads back the card the first just
wrote.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class EntityLocks:
    """A keyed lock registry with reference counting."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = {}
        self._guard = asyncio.Lock()

    async def _acquire_lock(self, key: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            self._waiters[key] = self._waiters.get(key, 0) + 1
            return lock

    async def _release_lock(self, key: str) -> None:
        async with self._guard:
            remaining = self._waiters.get(key, 1) - 1
            if remaining <= 0:
                self._waiters.pop(key, None)
                self._locks.pop(key, None)
            else:
                self._waiters[key] = remaining

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        """Hold the lock for ``key`` for the duration of the block."""
        lock = await self._acquire_lock(key)
        try:
            async with lock:
                yield
        finally:
            await self._release_lock(key)

    def held(self) -> int:
        """Number of live locks. Used by tests to assert the registry is pruned."""
        return len(self._locks)

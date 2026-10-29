"""
Thread-Safe Async Wrapper for MetaTrader5.

The MetaTrader5 Python package is NOT thread-safe — it must be called from
the same thread that ran mt5.initialize().  This module provides a
single-thread executor and convenience helpers so every `async def` in the
project can call MT5 without blocking the asyncio event loop.

Usage:
    from core.mt5_executor import mt5_call, mt5_exec

    # Simple call  — returns the same value mt5.positions_get() would:
    positions = await mt5_call(mt5.positions_get)

    # Call with args:
    tick = await mt5_call(mt5.symbol_info_tick, "EURUSD")

    # Arbitrary sync block that touches MT5:
    async def example():
        def _work():
            info = mt5.account_info()
            return info.balance if info else 0.0
        balance = await mt5_exec(_work)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable, Any

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Single-thread executor — guarantees all MT5 calls happen on one OS thread,
# the same thread that will run mt5.initialize() at startup.
_mt5_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")


async def mt5_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a single MetaTrader5 function on the dedicated MT5 thread.

    Strictly separates positional (*args) from keyword (**kwargs) calls because the
    MetaTrader5 C-extension raises (-2, 'Unnamed arguments not allowed') if kwargs
    are passed (even as empty dict) alongside positional arguments.
    """
    loop = asyncio.get_running_loop()
    
    def _invoke():
        if kwargs and not args:
            return func(**kwargs)
        elif kwargs and args:
            return func(*args, **kwargs)
        else:
            return func(*args)

    return await loop.run_in_executor(_mt5_executor, _invoke)


async def mt5_exec(block: Callable[[], T]) -> T:
    """Run an arbitrary synchronous callable on the dedicated MT5 thread.

    Useful when you need to make several MT5 calls in sequence without
    releasing the thread between them:

        def _work():
            info = mt5.symbol_info("EURUSD")
            tick  = mt5.symbol_info_tick("EURUSD")
            return info, tick
        info, tick = await mt5_exec(_work)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_mt5_executor, block)


def get_mt5_executor() -> ThreadPoolExecutor:
    """Return the raw executor for low-level usage (e.g. initializing MT5)."""
    return _mt5_executor

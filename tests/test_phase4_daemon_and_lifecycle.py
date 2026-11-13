"""
Phase 4 Daemon, Continuous Lifecycle & Shutdown Tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.mt5_connector import BrokerAccountSnapshot
from main import TradingDaemon


@pytest.mark.asyncio
async def test_daemon_preflight_blocks_invalid_or_large_accounts() -> None:
    daemon = TradingDaemon(mode="demo")

    # 1. Simular conta errada de 1.000.000 EUR
    mock_wrong_acc = BrokerAccountSnapshot(
        login=99999999,
        server="PepperstoneUK-Demo",
        currency="EUR",
        balance_eur=1_000_000.0,
        equity_eur=1_000_000.0,
        margin_eur=0.0,
        margin_free_eur=1_000_000.0,
        leverage=30,
        is_demo=True,
    )
    with (
        patch.object(daemon.storage, "is_tcp_reachable", return_value=True),
        patch.object(daemon.storage, "ping_dragonfly", return_value=True),
        patch.object(daemon.connector, "connect", return_value=True),
        patch.object(daemon.connector, "get_account_snapshot", return_value=mock_wrong_acc),
    ):
        ret = await daemon.run()
        assert ret == 1

    # 2. Simular conta Real bloqueada por defeito
    mock_real_acc = BrokerAccountSnapshot(
        login=1234567,
        server="PepperstoneUK-Live",
        currency="EUR",
        balance_eur=1000.0,
        equity_eur=1000.0,
        margin_eur=0.0,
        margin_free_eur=1000.0,
        leverage=30,
        is_demo=False,
    )
    with (
        patch.object(daemon.storage, "is_tcp_reachable", return_value=True),
        patch.object(daemon.storage, "ping_dragonfly", return_value=True),
        patch.object(daemon.connector, "connect", return_value=True),
        patch.object(daemon.connector, "get_account_snapshot", return_value=mock_real_acc),
    ):
        ret = await daemon.run()
        assert ret == 1


@pytest.mark.asyncio
async def test_daemon_graceful_shutdown_on_event() -> None:
    daemon = TradingDaemon(mode="dry-run", poll_interval_watchdog=0.1)

    mock_demo_acc = BrokerAccountSnapshot(
        login=62143460,
        server="PepperstoneUK-Demo",
        currency="EUR",
        balance_eur=1000.0,
        equity_eur=1000.0,
        margin_eur=0.0,
        margin_free_eur=1000.0,
        leverage=30,
        is_demo=True,
    )

    async def trigger_shutdown_soon():
        await asyncio.sleep(0.15)
        daemon._shutdown_event.set()

    with (
        patch.object(daemon.storage, "is_tcp_reachable", return_value=True),
        patch.object(daemon.storage, "ping_dragonfly", return_value=True),
        patch.object(daemon.connector, "connect", return_value=True),
        patch.object(daemon.connector, "shutdown", return_value=None),
        patch.object(daemon.connector, "ensure_connected", return_value=True),
        patch.object(daemon.connector, "get_account_snapshot", return_value=mock_demo_acc),
        patch.object(daemon.connector, "get_open_positions", return_value=[]),
        patch.object(daemon.notifier, "send_message_sync", return_value=True),
        patch.object(daemon.clock, "wait_for_next_h4_close", side_effect=asyncio.CancelledError),
    ):
        shutdown_task = asyncio.create_task(trigger_shutdown_soon())
        ret = await daemon.run()
        await shutdown_task
        assert ret == 0
        assert daemon._shutdown_event.is_set()

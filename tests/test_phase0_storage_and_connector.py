from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from config.settings import Settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient, SymbolCostSnapshot
from data.cost_model import CostModel
from tests.test_esma_risk_math import make_mock_spec


def test_cost_model_swap_modes():
    cost_model = CostModel()

    # Mode 0: disabled swap always yields 0.0
    spec_disabled = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, swap_long=-5.0)
    spec_disabled = SymbolContractSpec_with(spec_disabled, swap_mode=0)
    assert cost_model.calculate_daily_swap_eur(spec_disabled, side=+1, lots=0.01) == 0.0

    # Mode 4: deposit currency directly per 1.00 lot
    spec_deposit = SymbolContractSpec_with(spec_disabled, swap_mode=4, swap_long=-12.0)
    assert cost_model.calculate_daily_swap_eur(spec_deposit, side=+1, lots=0.01) == -0.12

    # Unsupported mode (e.g. 5 = interest current) safely returns 0.0 without crashing
    spec_unsupported = SymbolContractSpec_with(spec_disabled, swap_mode=5, swap_long=-12.0)
    assert cost_model.calculate_daily_swap_eur(spec_unsupported, side=+1, lots=0.01) == 0.0


def SymbolContractSpec_with(spec, **overrides):
    from dataclasses import replace

    return replace(spec, **overrides)


def test_storage_client_offline_fails_fast():
    client = StorageClient(
        questdb_host="127.0.0.1",
        questdb_rest_port=59991,
        questdb_ilp_port=59992,
        dragonfly_host="127.0.0.1",
        dragonfly_port=59993,
    )
    snap = SymbolCostSnapshot(
        symbol="EURUSD",
        status="APROVADO (<=1%)",
        margin_001_eur=33.33,
        atr14_h4_pips=28.0,
        sl_pips=56.0,
        risk_1r_eur=5.15,
        spread_points=8,
        friction_pct=1.5,
        swap_mode=1,
        swap_long_raw=-1.5,
        swap_short_raw=0.4,
        swap_4d_long_eur=-0.09,
        swap_4d_short_eur=0.02,
        captured_at=datetime(2025, 3, 3, 12, 0, tzinfo=timezone.utc),
    )
    ok, msg = client.write_symbol_cost_snapshots([snap], timeout_sec=0.1)
    assert ok is False
    assert "offline/unreachable" in msg.lower()
    assert client.ping_dragonfly(timeout_sec=0.1) is False


def test_mt5_connector_ensure_connected_triggers_reconnect():
    conn = MT5Connector()
    conn._connected = True

    with patch("core.mt5_connector.mt5.terminal_info", return_value=None), patch(
        "core.mt5_connector.mt5.shutdown"
    ) as mock_shutdown, patch("core.mt5_connector.mt5.initialize", return_value=True), patch(
        "core.mt5_connector.mt5.account_info",
        return_value=SimpleNamespace(
            login=12345,
            server="PepperstoneUK-Demo",
            currency="EUR",
            balance=1000.0,
            equity=1000.0,
            leverage=30,
        ),
    ):
        assert conn.ensure_connected() is True
        mock_shutdown.assert_called_once()


def test_settings_restores_dotenv_windows_control_escapes():
    # Simulate python-dotenv turning \t inside double-quoted Windows path into a tab character
    corrupted_path = "C:\\Program Files\\MetaTrader 5\terminal64.exe"
    assert "\t" in corrupted_path
    restored = Settings._restore_dotenv_windows_escapes(corrupted_path)
    assert restored == r"C:\Program Files\MetaTrader 5\terminal64.exe"

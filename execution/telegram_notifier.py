"""
Asynchronous Non-Blocking Telegram Alert Notifier (Fase 3).

Cleanly degrades to a structured no-op when bot credentials are unconfigured or
alerts are explicitly disabled (e.g., during automated verification tests).
Never logs or embeds secret bot tokens in tracebacks or return messages.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import urllib.request
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Non-blocking Telegram Bot API client using aiohttp with urllib fallback."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._bot_token = (
            bot_token if bot_token is not None else settings.telegram_bot_token
        ).strip()
        self._chat_id = (
            chat_id if chat_id is not None else settings.telegram_chat_id
        ).strip()
        self.enabled = bool(enabled and self._bot_token and self._chat_id)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._background_tasks: set[asyncio.Task[bool]] = set()

    @property
    def is_configured(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    def _endpoint(self) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/sendMessage"

    def _post_urllib(self, payload: dict[str, str]) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            return 200 <= int(getattr(resp, "status", 0)) < 300

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False

        payload = {
            "chat_id": self._chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        }
        try:
            try:
                import aiohttp  # type: ignore[import-untyped]

                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self._endpoint(), json=payload) as resp:
                        ok = 200 <= resp.status < 300
                        if not ok:
                            logger.warning(
                                "Telegram sendMessage failed with HTTP %s", resp.status
                            )
                        return ok
            except ImportError:
                ok = await asyncio.to_thread(self._post_urllib, payload)
                return bool(ok)
        except Exception as exc:
            # Never emit full URL / bot token in error logs.
            logger.warning("Telegram alert delivery error: %s", type(exc).__name__)
            return False

    def send_message_non_blocking(self, text: str, parse_mode: str = "HTML") -> None:
        """
        Schedules an async send when inside a running event loop without blocking
        the H4 execution path, or executes synchronously in a best-effort thread.
        """
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.send_message(text, parse_mode=parse_mode))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            self.send_message_sync(text, parse_mode=parse_mode)

    def send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        payload = {
            "chat_id": self._chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        }
        try:
            return bool(self._post_urllib(payload))
        except Exception as exc:
            logger.warning("Telegram sync delivery error: %s", type(exc).__name__)
            return False

    def notify_order_opened(
        self,
        symbol: str,
        side: int,
        lots: float,
        price: float,
        sl: float,
        tp: float,
        ticket: int,
        mode: str,
    ) -> None:
        side_str = "BUY" if side == 1 else "SELL"
        msg = (
            f"<b>OPEN {html.escape(side_str)} {html.escape(symbol)}</b>\n"
            f"Ticket: <code>{ticket}</code> | Lots: <code>{lots:.2f}</code> | Mode: <code>{html.escape(mode)}</code>\n"
            f"Entry: <code>{price:.5f}</code> | SL: <code>{sl:.5f}</code> | TP: <code>{tp:.5f}</code>"
        )
        self.send_message_non_blocking(msg)

    def notify_position_updated(
        self,
        symbol: str,
        ticket: int,
        action: str,
        new_sl: float,
        ref_price: float,
    ) -> None:
        msg = (
            f"<b>{html.escape(action)} {html.escape(symbol)}</b>\n"
            f"Ticket: <code>{ticket}</code> | New SL: <code>{new_sl:.5f}</code> | Ref: <code>{ref_price:.5f}</code>"
        )
        self.send_message_non_blocking(msg)

    def notify_position_closed(
        self,
        symbol: str,
        ticket: int,
        reason: str,
        pnl_eur: float,
    ) -> None:
        msg = (
            f"<b>CLOSED {html.escape(symbol)} ({html.escape(reason)})</b>\n"
            f"Ticket: <code>{ticket}</code> | Est PnL: <code>{pnl_eur:+.2f} EUR</code>"
        )
        self.send_message_non_blocking(msg)

    def notify_h4_cycle_summary(
        self,
        closed_bar_open: datetime,
        signals_detected: int,
        approved_count: int,
        open_positions_count: int,
        equity_eur: float,
        next_wakeup_server: datetime,
        mode: str,
    ) -> None:
        msg = (
            f"<b>BARRA H4 CONCLUÍDA ({html.escape(closed_bar_open.strftime('%Y-%m-%d %H:%M'))})</b>\n"
            f"Modo: <code>{html.escape(mode)}</code> | Equity: <code>{equity_eur:.2f} EUR</code>\n"
            f"Sinais: <code>{signals_detected}</code> | Aprovadas: <code>{approved_count}</code> | Posições Abertas: <code>{open_positions_count}</code>\n"
            f"Próximo Fecho: <code>{html.escape(next_wakeup_server.strftime('%Y-%m-%d %H:%M:%S'))}</code>"
        )
        self.send_message_non_blocking(msg)

    def notify_daily_summary(
        self,
        server_date: str,
        balance_eur: float,
        equity_eur: float,
        daily_pnl_eur: float,
        open_positions_count: int,
        total_dd_pct: float,
    ) -> None:
        msg = (
            f"<b>RELATÓRIO DIÁRIO MT5 ({html.escape(server_date)})</b>\n"
            f"Balance: <code>{balance_eur:.2f} EUR</code> | Equity: <code>{equity_eur:.2f} EUR</code>\n"
            f"PnL Dia: <code>{daily_pnl_eur:+.2f} EUR</code> | Max Total DD: <code>{total_dd_pct*100:.2f}%</code>\n"
            f"Posições Ativas: <code>{open_positions_count}</code>"
        )
        self.send_message_non_blocking(msg)

    def notify_risk_alert(self, title: str, details: str) -> None:
        msg = (
            f"<b>RISK ALERT: {html.escape(title)}</b>\n"
            f"<pre>{html.escape(details[:3500])}</pre>"
        )
        self.send_message_non_blocking(msg)

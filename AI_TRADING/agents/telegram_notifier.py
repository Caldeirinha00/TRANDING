"""
agents/telegram_notifier.py
Notificador assíncrono e resiliente para Telegram.
Envia alertas operacionais e de risco em tempo real sem bloquear o loop de execução.
Caso não existam credenciais configuradas, opera em modo silencioso (no-op).
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

from core.config import CONFIG

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or getattr(CONFIG, "TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or getattr(CONFIG, "TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        if not self.enabled:
            logger.info("Telegram Notifier: Credenciais não configuradas no .env. Notificações desativadas (no-op).")
        else:
            logger.info(f"Telegram Notifier: Ativo para chat_id={self.chat_id}.")

    def _send_sync(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning(f"Falha ao enviar mensagem Telegram: {exc}")
            return False

    async def send_message(self, text: str) -> bool:
        """Envia mensagem assíncrona em thread separada com timeout."""
        if not self.enabled:
            return False
        try:
            return await asyncio.to_thread(self._send_sync, text)
        except Exception as exc:
            logger.warning(f"Erro assíncrono Telegram: {exc}")
            return False

    async def notify_trade_approved(
        self,
        symbol: str,
        direction: int,
        lots: float,
        price: float,
        meta_prob: float,
        sl_mult: float,
        tp_mult: float,
    ) -> None:
        dir_str = "🟢 BUY" if direction > 0 else "🔴 SELL"
        msg = (
            f"<b>[AI_TRADING V2] Novo Trade Aprovado</b>\n"
            f"Ativo: <b>{symbol}</b> ({dir_str})\n"
            f"Volume: <code>{lots:.2f} lotes</code>\n"
            f"Preço Exec: <code>{price:.5f}</code>\n"
            f"Probabilidade ONNX: <code>{meta_prob:.1%}</code>\n"
            f"Ajuste SL/TP: <code>{sl_mult:.2f}x / {tp_mult:.2f}x</code>\n"
            f"Conta Base: <code>1.000 EUR (ESMA 1:30)</code>"
        )
        await self.send_message(msg)

    async def notify_trade_closed(
        self,
        symbol: str,
        pnl_eur: float,
        pnl_r: float,
        reason: str,
    ) -> None:
        icon = "🎉 LUCRO" if pnl_eur >= 0 else "🛑 STOP"
        msg = (
            f"<b>[AI_TRADING V2] Posição Encerrada</b>\n"
            f"Resultado: <b>{icon}</b>\n"
            f"Ativo: <b>{symbol}</b>\n"
            f"PnL: <b>{pnl_eur:+.2f} EUR</b> ({pnl_r:+.2f}R)\n"
            f"Motivo: <code>{reason}</code>"
        )
        await self.send_message(msg)

    async def notify_risk_alert(
        self,
        level: str,
        message: str,
    ) -> None:
        icon = "⚠️" if level in ("YELLOW", "ORANGE") else "🚨🚨🚨"
        msg = (
            f"<b>{icon} ALERTA DE RISCO: {level}</b>\n"
            f"{message}\n"
            f"Capital Base: 1.000 EUR"
        )
        await self.send_message(msg)


telegram_notifier = TelegramNotifier()
